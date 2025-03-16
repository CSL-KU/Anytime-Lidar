import sys
import os
import json
import math
import copy
import random
import uuid
import numpy as np
import pcdet.utils.res_pred_utils as res_pred_utils
from alive_progress import alive_bar

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

import nuscenes.utils.splits
all_scenes = set(nuscenes.utils.splits.train + nuscenes.utils.splits.val)
nusc = None


map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}

classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']


def read_nusc():
    global nusc
    nusc = NuScenes(version='v1.0-trainval', dataroot='../data/nuscenes/v1.0-trainval', verbose=True)
    #nusc.list_scenes()

# atan2 to quaternion:
# Quaternion(axis=[0, 0, 1], radians=atan2results)

def generate_pose_dict():
    print('generating pose dict')
    global nusc
    token_to_cs_and_pose = {}

    #global all_scenes
    for scene in nusc.scene:
        #if scene['name'] not in all_scenes:
            #print(f'Skipping {scene["name"]}')
        #    continue
        tkn = scene['first_sample_token']
        while tkn != "":
            #print('token:',tkn)
            sample = nusc.get('sample', tkn)
            #print('timestamp:', sample['timestamp'])
            sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs = nusc.get('calibrated_sensor',
                    sample_data['calibrated_sensor_token'])
            #print('calibrated sensor translation:', cs['translation'])
            #print('calibrated sensor rotation:', cs['rotation'])
            pose = nusc.get('ego_pose',
                sample_data['ego_pose_token'])
            #print('ego pose translation:', pose['translation'])
            #print('ego pose rotation:', pose['rotation'])
            scene_name = nusc.get('scene', sample['scene_token'])['name']
            token_to_cs_and_pose[tkn] = {
                    'timestamp' : sample['timestamp'],
                    'scene' : sample['scene_token'],
                    'scene_name': scene_name,
                    'cs_translation' : cs['translation'],
                    'cs_rotation' : cs['rotation'],
                    'ep_translation' : pose['translation'],
                    'ep_rotation' : pose['rotation'],
            }
            tkn = sample['next']

    print('Dict size:', sys.getsizeof(token_to_cs_and_pose)/1024/1024, ' MB')

    with open('token_to_pos.json', 'w') as handle:
        json.dump(token_to_cs_and_pose, handle, indent=4)


def generate_anns_dict():
    print('generating annotations dict')
    global nusc
    #global all_scenes
    global map_name_from_general_to_detection
    global classes

    token_to_anns = {}

    for scene in nusc.scene:
        #if scene['name'] not in all_scenes:
            #print(f'Skipping {scene["name"]}')
        #    continue
        tkn = scene['first_sample_token']
        #print(scene['name'])
        categories_in_scene = set()
        while tkn != "":
            #print('token:',tkn)
            sample = nusc.get('sample', tkn)
            #print('timestamp:', sample['timestamp'])
            sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs = nusc.get('calibrated_sensor',
                    sample_data['calibrated_sensor_token'])
            #print('calibrated sensor translation:', cs['translation'])
            #print('calibrated sensor rotation:', cs['rotation'])
            pose = nusc.get('ego_pose',
                sample_data['ego_pose_token'])
            #print('ego pose translation:', pose['translation'])
            #print('ego pose rotation:', pose['rotation'])

            annos = np.zeros((len(sample['anns']),9))
            labels = []
            num_ignored = 0
            for i, anno_token in enumerate(sample['anns']):
                anno = nusc.get('sample_annotation', anno_token)
                cn = anno['category_name']
                name = map_name_from_general_to_detection[cn]
                if name == 'ignore':
                    num_ignored += 1
                    continue
                categories_in_scene.add(name)
                labels.append(classes.index(name)+1)
                #print(anno['category_name'])
                anno_vel = nusc.box_velocity(anno_token)
                box = Box(anno['translation'], anno['size'],
                    Quaternion(anno['rotation']), velocity=tuple(anno_vel))
                box.translate(-np.array(pose['translation']))
                box.rotate(Quaternion(pose['rotation']).inverse)
                box.translate(-np.array(cs['translation']))
                box.rotate(Quaternion(cs['rotation']).inverse)

                idx = i - num_ignored
                annos[idx, :3] = box.center
                annos[idx, 3] = box.wlh[1]
                annos[idx, 4] = box.wlh[0]
                annos[idx, 5] = box.wlh[2]
                r, x, y, z = box.orientation.elements
                annos[idx, 6] = 2. * math.atan2(math.sqrt(x*x+y*y+z*z),r)
                annos[idx, 7:] = box.velocity[:2] # this is actually global velocity
            annos = annos[:annos.shape[0]-num_ignored]

            labels = np.array(labels)
            indices = labels.argsort()
            labels.sort()
            annos = annos[indices]
            #print('Annos:\n', annos)
            token_to_anns[tkn] = {
                'pred_boxes': annos.tolist(),
                'pred_scores': [1.0] * annos.shape[0],
                'pred_labels': labels.tolist(),
            }
            tkn = sample['next']
        print(len(categories_in_scene), categories_in_scene)

    #print('Dict size:', sys.getsizeof(token_to_anns)/1024/1024, ' MB')

    with open('token_to_anns.json', 'w') as handle:
        json.dump(token_to_anns, handle, indent=4)

def gen_new_token(table_name):
    # Generate a unique anno token
    # each token is 32 chars
    global nusc
    
    while True:
        new_token = uuid.uuid4().hex
        if new_token not in nusc._token2ind[table_name]:
            nusc._token2ind[table_name][new_token] = -1 # enough for now
            break

    return new_token

# step defines the time between populated annotations in milliseconds
# step 50ms, 100ms, 150ms, ...
def populate_annos_v2(step):
    print('populating annotations')
    global nusc
    step = step//50
    scene_to_sd = {}
    scene_to_sd_cam = {}
    for i, sd_rec in enumerate(nusc.sample_data):
        for channel, dct in zip(['LIDAR_TOP', 'CAM_FRONT'], \
                [scene_to_sd, scene_to_sd_cam]):
            if sd_rec['channel'] == channel:
                scene_tkn = nusc.get('sample', sd_rec['sample_token'])['scene_token']
                if scene_tkn not in dct:
                    dct[scene_tkn] = []
                dct[scene_tkn].append(sd_rec)

    for dct in [scene_to_sd, scene_to_sd_cam]:
        for k, v in dct.items():
            dct[k] = sorted(v, key=lambda item: item['timestamp'])

    scene_to_kf_indexes = {}
    for k, v in scene_to_sd.items():
        # Filter based on time, also filter the ones which cannot
        # be interpolated
        is_kf_arr = [sd['is_key_frame'] for sd in v]
        kf_indexes = [i for i in range(len(is_kf_arr)) if is_kf_arr[i]]
        scene_to_kf_indexes[k] = kf_indexes

    all_new_sample_datas = []
    all_new_samples = []
    all_new_annos = []
    #global all_scenes
    for scene in nusc.scene:
        #if scene['name'] not in all_scenes:
            #print(f'Skipping {scene["name"]}')
        #    continue
        #print('Processing scene', scene['name'])
        sd_records = scene_to_sd[scene['token']]
        sd_records_cam = scene_to_sd_cam[scene['token']]
        kf_indexes = scene_to_kf_indexes[scene['token']]
        for idx in range(len(kf_indexes) - 1):
            # generate sample between these two
            begin_kf_idx = kf_indexes[idx]
            end_kf_idx = kf_indexes[idx+1]
            cur_sample = nusc.get('sample', sd_records[begin_kf_idx]['sample_token'])
            next_sample = nusc.get('sample', sd_records[end_kf_idx]['sample_token'])
            # if these two are equal, this is a problem for interpolation
            assert cur_sample['token'] != next_sample['token']
            sd_rec_indexes = np.arange(begin_kf_idx+step, end_kf_idx-step+1, step)

            new_samples = []
            new_sample_annos = []
            for sd_rec_idx in sd_rec_indexes:
                sd_rec = sd_records[sd_rec_idx]
                new_token = gen_new_token('sample')
                # find the sd_record_cam with closest timestamp
                lidar_ts = sd_rec['timestamp']
                cam_ts_arr = np.asarray([sd_rec_cam['timestamp'] \
                        for sd_rec_cam in sd_records_cam])
                cam_idx = (np.abs(cam_ts_arr - lidar_ts)).argmin()
                sd_rec_cam = sd_records_cam[cam_idx]
                new_samples.append({
                        'token': new_token,
                        'timestamp' : lidar_ts,
                        'prev': "",
                        'next': "",
                        'scene_token': scene['token'],
                        'data': {'LIDAR_TOP': sd_rec['token'],
                            'CAM_FRONT': sd_rec_cam['token']},
                        'anns': [],
                })

                # update sample data record
                sd_rec['sample_token'] = new_samples[-1]['token']
                sd_rec['is_key_frame'] = True # not sure this is right
                if not sd_rec_cam['is_key_frame']:
                    sd_rec_cam['sample_token'] = new_samples[-1]['token']
                    sd_rec_cam['is_key_frame'] = True # not sure this is right
                else:
                    # Fabricate an sd_rec_cam with a new token
                    # because we cannot override this one as it is a keyframe
                    new_sd_rec_cam = copy.deepcopy(sd_rec_cam)
                    new_token = gen_new_token('sample_data')
                    new_sd_rec_cam['token'] = new_token
                    new_sd_rec_cam['sample_token'] = new_samples[-1]['token'] 
                    # I am not sure whether this one should be befor or after
                    # sd_rec_cam, but I will assume it will be after
                    new_sd_rec_cam['prev'] = sd_rec_cam['token']
                    new_sd_rec_cam['next'] = sd_rec_cam['next']
                    if new_sd_rec_cam['next'] != "":
                        nusc.get('sample_data', new_sd_rec_cam['next'])['prev'] = \
                                new_token
                    sd_rec_cam['next'] = new_token

                    # Do I need to generate a corresponding ego_pose_rec? I hope not
                    all_new_sample_datas.append(new_sd_rec_cam)

            # link the samples
            if not new_samples:
                continue

            cur_sample['next'] = new_samples[0]['token']
            assert cur_sample['timestamp'] < new_samples[0]['timestamp']
            new_samples[0]['prev'] = cur_sample['token']
            for i in range(1, len(new_samples)):
                new_samples[i-1]['next'] = new_samples[i]['token']
                new_samples[i]['prev'] = new_samples[i-1]['token']
            new_samples[-1]['next'] = next_sample['token']
            next_sample['prev'] = new_samples[-1]['token']

            # Generate annotations
            # For each anno in the cur_sample, find its corresponding anno
            # in the next sample. The matching can be done via instance_token
            total_time_diff = next_sample['timestamp'] - cur_sample['timestamp']
            for cur_anno_tkn in cur_sample['anns']:
                cur_anno = nusc.get('sample_annotation', cur_anno_tkn)
                next_anno_tkn = cur_anno['next']
                if next_anno_tkn == "":
                    continue
                next_anno = nusc.get('sample_annotation', next_anno_tkn)

                new_annos = []
                # Interpolate this anno for all new samples
                for new_sample in new_samples:
                    new_token = gen_new_token('sample_annotation')
                    new_anno = copy.deepcopy(cur_anno)

                    new_anno['token'] = new_token
                    new_anno['sample_token'] = new_sample['token']
                    new_sample['anns'].append(new_token)

                    time_diff = new_sample['timestamp'] - cur_sample['timestamp']
                    rratio = time_diff / total_time_diff
                    new_anno['translation'] = (1.0 - rratio) * \
                            np.array(cur_anno['translation'], dtype=float) + \
                            rratio * np.array(next_anno['translation'], dtype=float)
                    new_anno['translation'] = new_anno['translation'].tolist()
                    new_anno['rotation'] = Quaternion.slerp(
                            q0=Quaternion(cur_anno['rotation']),
                            q1=Quaternion(next_anno['rotation']),
                            amount=rratio
                    ).elements.tolist()
                    new_anno['prev'] = ''
                    new_anno['next'] = ''
                    new_annos.append(new_anno)

                # link the annos
                cur_anno['next'] = new_annos[0]['token']
                new_annos[0]['prev'] = cur_anno_tkn
                for i in range(1, len(new_annos)):
                    new_annos[i-1]['next'] = new_annos[i]['token']
                    new_annos[i]['prev'] = new_annos[i-1]['token']
                new_annos[-1]['next'] = next_anno_tkn
                next_anno['prev'] = new_annos[-1]['token']

                all_new_annos.extend(new_annos)
                # increase the number of annos in the instance table
                nusc.get('instance', cur_anno['instance_token'])['nbr_annotations'] += \
                        len(new_annos)

            all_new_samples.extend(new_samples)

            scene['nbr_samples'] += len(new_samples)

    nusc.sample.extend(all_new_samples)
    nusc.sample_annotation.extend(all_new_annos)
    nusc.sample_data.extend(all_new_sample_datas)


def prune_annos(step):
    print('pruning annotations')
    #num_lidar_sample_data = sum([sd['channel'] == 'LIDAR_TOP' for sd in nusc.sample_data])
    # make sure the number of samples are equal to number of sample_datas
    #assert len(nusc.sample) == num_lidar_sample_data, \
    #        "{len(nusc.sample)}, {num_lidar_sample_data}"

    step = step//50
    print('step', step)
    num_skips=step-1
    new_nusc_samples = []
    discarded_nusc_samples = []
    global nusc
    #global all_scenes
    for scene in nusc.scene:
        #if scene['name'] not in all_scenes:
        #    print(f'Skipping {scene["name"]}')
        #    continue
        # skip skip skip get, skip skip skip get...
        samples_to_del = [] # sample token : replacement sample
        samples_to_connect = []
        sample_tkn = scene['first_sample_token']
        for i in range(num_skips):
            if sample_tkn != '':
                sample = nusc.get('sample', sample_tkn)
                samples_to_del.append(sample)
                sample_tkn = sample['next']
        while sample_tkn != '':
            sample = nusc.get('sample', sample_tkn)
            samples_to_connect.append(sample)
            sample_tkn = sample['next']
            for i in range(num_skips):
                if sample_tkn != '':
                    sample = nusc.get('sample', sample_tkn)
                    samples_to_del.append(sample)
                    sample_tkn = sample['next']

        # Update the scene
        scene['first_sample_token'] = samples_to_connect[0]['token']
        scene['last_sample_token'] = samples_to_connect[-1]['token']

        assert len(set([s['scene_token'] for s in samples_to_connect])) == 1

        #update samples
        samples_to_connect[0]['prev'] = ''
        for i in range(len(samples_to_connect)-1):
            s1, s2 = samples_to_connect[i], samples_to_connect[i+1]
            s1['next'] = s2['token']
            s2['prev'] = s1['token']
        samples_to_connect[-1]['next'] = ''

        #delete the samples
        new_nusc_samples.extend(samples_to_connect)
        discarded_nusc_samples.extend(samples_to_del)

    tokens_c = set([s['token'] for s in new_nusc_samples])
    ts_arr_c = np.array([s['timestamp'] for s in new_nusc_samples])
    assert (ts_arr_c != 0).all()
    new_samples_scene_tokens = np.array([s['scene_token'] for s in new_nusc_samples])
    tokens_d = set([s['token'] for s in discarded_nusc_samples])

    new_nusc_sample_datas = []
    for sd in nusc.sample_data:
        tkn = sd['sample_token']
        if tkn in tokens_c:
            sd['is_key_frame'] = True
            new_nusc_sample_datas.append(sd)
        elif tkn in tokens_d:
            sd['is_key_frame'] = False
            new_nusc_sample_datas.append(sd)
            # point to the sample with closest timestamp

            sd_ts = sd['timestamp']
            assert sd_ts != 0
            scene_token_of_sd = nusc.get('sample', tkn)['scene_token']
            mask = (new_samples_scene_tokens != scene_token_of_sd)
            diffs = np.abs(ts_arr_c - sd_ts) + (999999 * mask)
            min_idx = np.argmin(diffs)
            s = new_nusc_samples[min_idx]
            assert scene_token_of_sd == s['scene_token']
            sd['sample_token'] = s['token']

    new_nusc_sample_annos=[]
    new_nusc_instances=[]
    # Go through all instances and prune deleted sample annotations
    num_removed_instances = 0
    for inst in nusc.instance:
        sa_tkn = inst['first_annotation_token']
        sa = nusc.get('sample_annotation', sa_tkn)
        while sa_tkn != '' and sa['sample_token'] not in tokens_c:
            sa_tkn = sa['next']
            if sa_tkn != '':
                sa = nusc.get('sample_annotation', sa_tkn)
        if sa_tkn == '':
            #whoops, need to remove this instance!
            num_removed_instances += 1
            continue

        inst['first_annotation_token'] = sa_tkn
        sa['prev'] = ''
        new_nusc_sample_annos.append(sa)
        cnt = 1

        # find next and connect
        sa_tkn = sa['next']
        while sa_tkn != '':
            sa = nusc.get('sample_annotation', sa_tkn)
            while sa_tkn != '' and sa['sample_token'] not in tokens_c:
                sa_tkn = sa['next']
                if sa_tkn != '':
                    sa = nusc.get('sample_annotation', sa_tkn)
            if sa_tkn != '':
                new_nusc_sample_annos[-1]['next'] = sa_tkn
                sa['prev'] = new_nusc_sample_annos[-1]['token']
                new_nusc_sample_annos.append(sa)
                cnt += 1
                sa_tkn = sa['next']

        new_nusc_sample_annos[-1]['next'] = ''
        inst['last_annotation_token'] = new_nusc_sample_annos[-1]['token']
        inst['nbr_annotations'] = cnt
        new_nusc_instances.append(inst)

    print('Num prev instances:', len(nusc.instance))
    print('Num new instances:', len(new_nusc_instances))
    print('Num removed instances:', num_removed_instances)
    nusc.sample_annotation = new_nusc_sample_annos
    nusc.sample = new_nusc_samples
    nusc.sample_data = new_nusc_sample_datas
    nusc.instances = new_nusc_instances


def calc_scene_velos():
    global nusc
    global train
    global map_name_from_general_to_detection
    global classes

    scene_to_fp={}

    visualize=False
    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    with alive_bar(len(nusc.sample), force_tty=True, max_cols=160) as bar:
        for sample_idx, sample in enumerate(nusc.sample):
            # Only use train scenes
            #if nusc.get('scene', sample['scene_token'])['name'] not in train:
            #    continue

            sample_tkn = sample['token']
            ep, egovel = res_pred_utils.get_egopose_and_egovel(nusc, sample_tkn)

            sd_tkn = sample['data']['LIDAR_TOP']
            sample_data = nusc.get('sample_data', sd_tkn)
            cs = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            boxes = nusc.get_boxes(sample['data']['LIDAR_TOP']) # annos
            good_boxes = []
            for box in boxes:
                box.name = map_name_from_general_to_detection[box.name]
                if box.name != 'ignore':
                    box.velocity = nusc.box_velocity(box.token)
                    # Move box to ego vehicle coord system
                    box.translate(-np.array(ep['translation']))
                    box.rotate(Quaternion(ep['rotation']).inverse)
                    #  Move box to sensor coord system
                    box.translate(-np.array(cs['translation']))
                    box.rotate(Quaternion(cs['rotation']).inverse)

                    if np.isnan(box.velocity).any():
                        continue

                    good_boxes.append(box)

            if len(good_boxes) == 0:
                continue

            coords = np.array([box.center for box in good_boxes])
            velos = np.array([box.velocity for box in good_boxes])
            labels = [classes.index(box.name) for box in good_boxes]
            rel_velos = velos  - egovel

            if visualize and sample_idx % 100 == 0:
                ax.clear()
                nusc.render_sample_data(sd_tkn, nsweeps=1, axes_limit=60,
                    use_flat_vehicle_coordinates=False, underlay_map=False,
                    ax=ax, verbose=False)
                for relvel, coord in zip(rel_velos, coords):
                    # Relative velos
                    ax.arrow(coord[0], coord[1], relvel[0], relvel[1],
                    #ax.arrow(coord[0], coord[1], vel[0], vel[1],
                         head_width=0.9, head_length=0.7, fc='red', ec='red')

                # Egovel
                ax.arrow(0, 0, egovel[0], egovel[1],
                         head_width=0.9, head_length=0.7, fc='red', ec='red')
                plt.savefig(f"visualized_samples/{sample_idx}.jpg")

            st = sample['scene_token']
            if st not in scene_to_fp:
                scene_to_fp[st] = 0.

            total_fp = 0
            for tdiff_sec in [0.1, 0.2, 0.3]:
                total_fp += res_pred_utils.calc_falsepos_when_shifted(tdiff_sec, \
                        coords, rel_velos, labels)

            scene_to_fp[st] += total_fp
            bar()

    scene_tuples = []
    for scene_tkn, all_fp in scene_to_fp.items():
        scene = nusc.get('scene', scene_tkn)
        scene_tuples.append((scene['name'], all_fp, scene['description']))
        #print(scene['name'], sum(velos), scene['description'])
    scene_tuples = sorted(scene_tuples, key=lambda x: x[1])
    names = []
    for i, t in enumerate(scene_tuples):
        if i < 30:
            names.append(t[0])

        print('VAL' if t[0] in val else 'TRAIN' , t[0], t[1], t[2])

    print('You can copy the following to your splits.py file:')
    print("train = [\'" + "\',\'".join(names) + "\']")

def prune_training_data_from_tables():
    global nusc
    global all_scenes

    #NOTE the splits.py should be modified before you run this

    keep_indexes = {nm: set() for nm in nusc.table_names}

    for scene in nusc.scene:
        if scene['name'] in all_scenes:
            keep_indexes['scene'].add(nusc.getind('scene', scene['token']))
            #keep_indexes['log'].add(nusc.getind('log', scene['log_token'])

            sample_tkn = scene['first_sample_token']
            while sample_tkn != '':
                keep_indexes['sample'].add(nusc.getind('sample', sample_tkn))
                sample = nusc.get('sample', sample_tkn)
                sample_tkn = sample['next']

    for sample_data in nusc.sample_data:
        sample_tkn = sample_data['sample_token']
        if nusc.getind('sample', sample_tkn) in keep_indexes['sample']:
            keep_indexes['sample_data'].add(nusc.getind('sample_data', sample_data['token']))
            keep_indexes['ego_pose'].add(nusc.getind('ego_pose', sample_data['ego_pose_token']))
            #keep_indexes['calibrated_sensor'].add(nusc.getind('calibrated_sensor',
            #    sample_data['calibrated_sensor_token']))

    for sample_anno in nusc.sample_annotation:
        sample_tkn = sample_anno['sample_token']
        if nusc.getind('sample', sample_tkn) in keep_indexes['sample']:
            keep_indexes['sample_annotation'].add(nusc.getind('sample_annotation',
                sample_anno['token']))
            keep_indexes['instance'].add(nusc.getind('instance',
                sample_anno['instance_token']))
    
    for k, indexes in keep_indexes.items():
        table = getattr(nusc, k)
        setattr(nusc, k, [table[i] for i in indexes])


def dump_data(dumpdir='.'):
    global nusc

    indent_num=0
    print('Dumping the tables')
    if dumpdir != '.':
        os.makedirs(dumpdir, exist_ok=True)
        curdir = os.getcwd()
        os.chdir(dumpdir)

    with open('scene.json', 'w') as handle:
        json.dump(nusc.scene, handle, indent=indent_num)
    
    for sd in nusc.sample:
        del sd['anns']
        del sd['data']
    with open('sample.json', 'w') as handle:
        json.dump(nusc.sample, handle, indent=indent_num)

    for sd in nusc.sample_data:
        del sd['sensor_modality']
        del sd['channel']
    with open('sample_data.json', 'w') as handle:
        json.dump(nusc.sample_data, handle, indent=indent_num)

    with open('ego_pose.json', 'w') as handle:
        json.dump(nusc.ego_pose, handle, indent=indent_num)

    for sd in nusc.sample_annotation:
        del sd['category_name']
    with open('sample_annotation.json', 'w') as handle:
        json.dump(nusc.sample_annotation, handle, indent=indent_num)

    with open('instance.json', 'w') as handle:
        json.dump(nusc.instance, handle, indent=indent_num)

    if dumpdir != '.':
        os.chdir(curdir)


def main():
    read_nusc()
    if len(sys.argv) == 3 and sys.argv[1] == 'populate_annos_v2':
        step = int(sys.argv[2])
        populate_annos_v2(step)
        dump_data()
    elif len(sys.argv) == 3 and sys.argv[1] == 'prune_annos':
        step = int(sys.argv[2])
        prune_annos(step)
        dump_data()
    elif len(sys.argv) == 2 and sys.argv[1] == 'generate_dicts':
        #generate_anns_dict()
        generate_pose_dict()
    elif len(sys.argv) == 2 and sys.argv[1] == 'calc_velos':
        calc_scene_velos()
    elif len(sys.argv) == 2 and sys.argv[1] == 'prune_training_data_from_tables':
        prune_training_data_from_tables()
        dump_data("./pruned_tables")
    else:
        print('Usage error, doing nothing.')

if __name__ == "__main__":
    main()
