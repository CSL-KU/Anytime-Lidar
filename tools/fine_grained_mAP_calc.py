import os, re
import glob
import sys
import copy
import json
import math
from multiprocessing import Process
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pickle
import numba
from matplotlib import pyplot as plt
from collections import OrderedDict
from itertools import cycle

#"fields": [
#        "scene",
#        "time_segment",
#        "dist_th",
#        "class",
#        "deadline_ms",
#        "seg_sample_stats"
#    ],

#@numba.jit(nopython=True)
def gen_features(bboxes, scores, labels, coords_2d=True):
    #bboxes = det_annos['boxes_lidar']
    #TODO, consider duplicated feature coords
    if coords_2d:
       feature_coords = (bboxes[:, :2] + 57.6).astype(int) # since pc range is -57.6 +57+6
    else:
       feature_coords = (bboxes[:, :3] + 57.6)

     # sizes(3), heading(1), vel(2), score(1), label(1)
    features = np.empty((bboxes.shape[0], 8), dtype=float)
    features[:, :3] = bboxes[:, 3:6] / np.array([40., 10., 15.]) # max sizes x y z
    features[:, 3] = bboxes[:, 6] / 3.14
    #TODO use relative velo
    # assuming max vel is 15 meters per second 
    features[:, 4:6] = bboxes[:, 7:9] / 15.0
    features[:, 6] = scores
    features[:, 7] = (labels-1) / 10.

    return feature_coords, features

def create_bev_tensor(feature_coords, features):
    bev_tensor = np.zeros((128, 128, 8), dtype=float)
    bev_tensor[feature_coords[:,0], feature_coords[:,1]] = features
    return bev_tensor

class SegInfo:
    def __init__(self, path_list, eval_path_list = [], dist_th=None, class_name=None):
        self.class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
                      'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.dist_thresholds = ['0.5', '1.0', '2.0', '4.0'] # hope this will be ok

        self.dataset_dict = {} # key: sample_token val: (deadline, bev_tensor)
        for path in eval_path_list:
            print('Loading', path)
            with open(path, 'rb') as handle:
                d = pickle.load(handle)
                deadline_ms = d['calib_deadline_ms']
                coords_glob = d['annos_in_glob_coords']
                assert not coords_glob

                det_annos_all = d['det_annos']
                last_scores = np.zeros(1, dtype=float)
                for det_annos in det_annos_all: #[:10]:
                    scores = det_annos['score']
                    if len(scores) > 0 and not np.array_equal(scores, last_scores):
                        last_scores = scores
                        feature_coords, features = gen_features(det_annos['boxes_lidar'], scores, 
                                det_annos['pred_labels'], coords_2d=False)
                        sample_token = det_annos['metadata']['token']
                        tpl = (feature_coords, features, deadline_ms) # in in out
                        if sample_token in self.dataset_dict:
                            self.dataset_dict[sample_token].append(tpl)
                        else:
                            self.dataset_dict[sample_token] = [tpl]

        self.seg_info_tuples = []
        for path in path_list:
            print('Loading', path)
            with open(path, 'r') as handle:
                seg_info = json.load(handle)
                seg_info_fields = seg_info['fields']
                self.seg_info_tuples += seg_info['tuples']

        print(f'All files loaded.')

        for i, f in enumerate(seg_info_fields):
            self.__setattr__(f+'_idx', i)

        if dist_th is not None:
            self.seg_info_tuples = [t for t in self.seg_info_tuples \
                    if t[self.dist_th_idx] == dist_th]

        if class_name is not None:
            self.seg_info_tuples = [t for t in self.seg_info_tuples \
                    if t[self.class_idx] == class_name]
        
        self.scene_to_idx={}
        self.scene_to_idx_counter=0

        # turn time segs into ints
        seg = self.seg_info_tuples[0][self.time_segment_idx]
        seg_len = seg[1] - seg[0] # assume all segments have the same length
        print(f'Time segment length: {seg_len}')
        for t in self.seg_info_tuples:
            t[self.time_segment_idx] = int(t[self.time_segment_idx][0] / seg_len)

        # each value of this dict holds eval data of same scene, time, dist_th and class 
        # but different deadlines
        seg_eval_data_dict = {}
        # this one on the other hand merges the classes and dist thresholds
        seg_prec_dict = {}
        # this one will be used to build the dataset
        seg_sample_tokens_dict = {}
        all_deadlines = set()
        for i, tpl in enumerate(self.seg_info_tuples):
            seg_sample_stats = tpl[self.seg_sample_stats_idx] # list of dicts
            num_gt_seg, tp_arr, scr_arr = 0, [], []
            # these segments were inserted considering cls scores, from high to low
            for sample in seg_sample_stats: # all samples in the segment
                num_gt_seg += sample['num_gt']
                #sample['egopose_translation_xy']
                #sample['egovel_xy']
                predictions = sample['pred_data']
                if len(predictions) > 0:
                    if isinstance(predictions[0], dict):
                        tp_arr += [p['is_true_pos'] for p in predictions]
                        scr_arr += [p['detection_score'] for p in predictions]
                    else:
                        tp_arr += [p[0] for p in predictions]
                        scr_arr += [p[1] for p in predictions]

            tp_arr = np.array(tp_arr, dtype=int)
            scr_arr = np.array(scr_arr, dtype=float)

            deadline = int(tpl[self.deadline_ms_idx])
            all_deadlines.add(deadline)
            data = [deadline, tp_arr, scr_arr, num_gt_seg]

            key = self.get_key(i, use_cls=True, use_dist_th=True, use_dl=False)
            if key not in seg_eval_data_dict:
                seg_eval_data_dict[key] = []
            seg_eval_data_dict[key].append(data)

#            key = self.get_key(i, use_cls=False, use_dist_th=False, use_dl=True)
#            if key not in seg_prec_dict:
#                seg_prec_dict[key] = dict()
#            if deadline not in seg_prec_dict[key]:
#                seg_prec_dict[key][deadline] = []
#            seg_prec_dict[key][deadline].append(

            key = self.get_key(i, use_cls=False, use_dist_th=False, use_dl=False)
            if key not in seg_sample_tokens_dict:
                seg_sample_tokens_dict[key] = [s['sample_token'] for s in seg_sample_stats]
        self.seg_eval_data_dict = seg_eval_data_dict
        self.seg_sample_tokens_dict = seg_sample_tokens_dict
        self.all_segments = list(seg_sample_tokens_dict.keys())
        self.all_deadlines = list(all_deadlines)


#        def merge_datas(datas):
#            return (datas[0][0],
#                    np.concatenate([d[1] for d in datas]),
#                    np.concatenate([d[2] for d in datas]),
#                    sum([d[3] for d in datas]))

#        # for each time segment of each scene, this dict holds the best deadline
#        new_seg_prec_dict = {}
#        for k,v in seg_prec_dict.items():
#            tokens = k.split('=')
#            new_key = '='.join(tokens[:-1])
#            best_deadline = int(tokens[-1])
#
#            if new_key not in new_seg_prec_dict:
#                new_seg_prec_dict[new_key] = []
#            new_seg_prec_dict[new_key].append((best_deadline, ap))
#        self.seg_prec_dict = new_seg_prec_dict

#TODO dataset stuff
#        if len(self.dataset_dict) > 0:
#            dataset_tuples=[]
#            for seg_key, (best_deadline, prec) in self.seg_prec_dict.items():
#                sample_tokens = seg_sample_tokens_dict[seg_key]
#                for sample_tkn in sample_tokens:
#                    if sample_tkn in self.dataset_dict:
#                        inout_list = self.dataset_dict[sample_tkn]
#                        deadlines = [l[-1] for l in inout_list]
#                        if best_deadline in deadlines:
#                            best_one = deadlines.index(best_deadline)
#                            dataset_tuples.append(inout_list[best_one])
#
#            # dump the tuples
#            print(f'Dumping {len(dataset_tuples)} samples as dataset')
#            with open('deadline_dataset.pkl', 'wb') as f:
#                pickle.dump(dataset_tuples, f)
#
    def do_eval(self, deadline_ms=None):
        #if deadline is none, it will pick the deadline that gives the best result

        def get_dl_data(datas, dl):
            for d in datas:
                if d[0] == dl:
                    return d
            return datas[0]


        #calculate AP for each class and dist_th
        if deadline_ms is None: # calculate upper bound
#            chosen_deadlines = []
#            num_inconsistent = 0
#            for key, datas in self.seg_eval_data_dict.items():
#                # Get the best deadline
#                tokens = key.split('=')
#                seg_key = tokens[0] + '=' + tokens[1]
#                best_dl = self.seg_prec_dict[seg_key][0]
#                deadlines = [d[0] for d in datas]
#                if best_dl in deadlines:
#                    precs = [self.get_seg_prec_recall(d)[0] for d in datas]
#                    best_dl_cls = deadlines[np.argmax(precs)]
#                    if best_dl != best_dl_cls:
#                        num_inconsistent += 1
#                        #print('best_dls are not consistent', best_dl, best_dl_cls)
#                    seg_eval_dict[key] = datas[deadlines.index(best_dl)]
#                    #seg_eval_dict[key] = datas[deadlines.index(best_dl_cls)]
#                else:
#                    #print(f'Deadline {best_dl} not found for {key} in {deadlines}, using first')
#                    seg_eval_dict[key] = datas[0]
#                chosen_deadlines.append(seg_eval_dict[key][0])
#            print('Num inconsistent best segment deadline choices:', num_inconsistent)
#            deadlines, occurances =  np.unique(chosen_deadlines, return_counts=True)
            # for each segment, try all deadlines and find the best deadline
            # that gives the most boost to the mAP.
            max_mAP = 0.
            cur_seg_dls = {seg:self.all_deadlines[0] for seg in self.all_segments}
            for i, seg in enumerate(self.all_segments):
                print(f'Processing {seg}, progress: {int(i/len(self.all_segments)*100)}')
                for new_dl in self.all_deadlines:
                    seg_eval_dict = {}
                    old_dl = cur_seg_dls[seg]
                    cur_seg_dls[seg] = new_dl
                    for key, datas in self.seg_eval_data_dict.items():
                        seg_i = '='.join(key.split('=')[:2])
                        seg_eval_dict[key] = get_dl_data(datas, cur_seg_dls[seg_i])
                    mAP = self.calc_mAP(seg_eval_dict, False)
                    if mAP > max_mAP:
                        max_mAP = mAP
                        print('max mAP:', max_mAP)
                    else:
                        cur_seg_dls[seg] = old_dl
            print('Upper bound mAP:', max_mAP)
            print('Deadline stats:')
            deadlines, occurances = np.unique(np.array(list(cur_seg_dls.values())), return_counts=True)
            print(deadlines)
            print(occurances)
            print(np.round(occurances / np.sum(occurances), 2))

        else:
            seg_eval_dict = {}
            for key, datas in self.seg_eval_data_dict.items():
                seg_eval_dict[key] = get_dl_data(datas, deadline_ms)
            mAP = self.calc_mAP(seg_eval_dict)
            print('mAP:', mAP)

    def calc_mAP(self, seg_eval_dict, print_detailed=True):
        average_precs = []
        for cls_nm in self.class_names:
            for dist_th in self.dist_thresholds:
                all_tp, all_scr, all_num_gt = [], [], 0
                for key, data in seg_eval_dict.items():
                    tokens = key.split('=')
                    if tokens[2] == cls_nm and tokens[3] == dist_th:
                        #if len(data) != 4:
                        #    print(data)
                        deadline_ms, tp_arr, scr_arr, num_gt_seg = data
                        all_tp.append(tp_arr)
                        all_scr.append(scr_arr)
                        all_num_gt += num_gt_seg

                if all_num_gt == 0:
                    ap = 0.
                else:
                    all_tp = np.concatenate(all_tp)
                    all_scr = np.concatenate(all_scr)
                    ap = self.calc_ap(all_tp, all_scr, all_num_gt)

                average_precs.append((cls_nm, dist_th, ap))

        def print_fixed_width(s, width):
            # Format the string to a fixed width, with spaces padded to the right
            formatted_str = f"{s:<{width}}"
            print(formatted_str, end='')

        all_aps = []
        for ap in average_precs:
            if print_detailed:
                print_fixed_width(ap[0], 30)
                print_fixed_width(ap[1], 5)
                print(round(ap[2], 3))
            all_aps.append(ap[2])
        mAP = np.mean(all_aps)
        return mAP

    def calc_ap(self, all_tp, all_scr, all_num_gt) -> float:
        """ Calculated average precision. """
        sort_inds = np.argsort(-all_scr) # descending
        tp = all_tp[sort_inds]
        scr =  all_scr[sort_inds] # this is confidence score
        fp = np.logical_not(tp)
        #print('class:', cls_nm, 'dist thr:', dist_th, 'num dets:', tp.shape)

        tp = np.cumsum(tp).astype(float)
        fp = np.cumsum(fp).astype(float)

        prec = tp / (fp + tp)
        rec = tp / float(all_num_gt)
        nelem = 101
        rec_interp = np.linspace(0, 1, nelem)
        precision = np.interp(rec_interp, rec, prec, right=0)
        #conf = np.interp(rec_interp, rec, scr, right=0)
        #rec = rec_interp

        min_recall = 0.1
        min_precision = 0.1

        prec = np.copy(precision)
        prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
        prec -= min_precision  # Clip low precision
        prec[prec < 0] = 0
        return float(np.mean(prec)) / (1.0 - min_precision)


    def get_seg_prec_recall(self, eval_data_tpl):
        deadline_ms, tp_arr, scr_arr, num_gt_seg = eval_data_tpl

        if num_gt_seg == 0: # dont consider these, ignore false positives
            return -1, -1

        tp = np.sum(tp_arr)
        fp = len(tp_arr) - tp
        if fp == 0. and tp == 0.:
            return 0., 0.
        else:
            return (tp / (fp + tp)).item(), (tp / float(num_gt_seg)).item()


    #def __getitem__(self, idx_field_tuple):
    #    return self.seg_info_tuples[idx_field_tuple[0]][self.field_inds[idx_field_tuple[1]]]

    def __len__(self):
        return len(self.seg_info_tuples)

    def get_key(self, tpl_idx, use_cls=True, use_dist_th=True, use_dl=True):
        tpl = self.seg_info_tuples[tpl_idx]
        key = str(tpl[self.scene_idx]) + '=' + \
                str(tpl[self.time_segment_idx]) + '='
        if use_cls:
            key += str(tpl[self.class_idx]) + '='
        if use_dist_th:
            key += str(tpl[self.dist_th_idx]) +  '='
        if use_dl:
            key += str(tpl[self.deadline_ms_idx]) + '='
        return key[:-1]

    def get_keyidx(self, tpl_idx):
        t = self.seg_info_tuples[tpl_idx]

        scene = t[self.scene_idx]
        if scene not in self.scene_to_idx:
            self.scene_to_idx[scene] = self.scene_to_idx_counter
            self.scene_to_idx_counter += 1

        num_time_seg_in_scene = 20
        return self.scene_to_idx[scene] * num_time_seg_in_scene + t[fi['time_segment']]

out_dir = './seg_plots'
in_dir = './streaming_spi/'
seg_info = SegInfo( # takes a list
#        [in_dir+'cp_pp.json',
#        in_dir+'cp_voxel01.json',
#        in_dir+'cp_voxel0075.json'],
#        [in_dir+'cp_pp_valo.json'],
#        [in_dir+'cp_voxel01_valo.json'],
#        ['segment_precision_info_centerpoint_voxel01_valo.json']
        ['segment_precision_info.json'],
#        [f'eval_data_{dl}ms.pkl' for dl in (45.0,70.0,95.0)]
#        ['./cp_pp_valo_eval_data/'+f'eval_data_{dl}ms.pkl' for dl in (40.0,45.0,50.0,55.0,60.0)]
)
#VALO
for dl in seg_info.all_deadlines:
    print('Deadline', dl)
    seg_info.do_eval(dl)
seg_info.do_eval()

#baseline
#for dl in (75, 125, 175):
#    print('deadline', dl)
#    seg_info.do_eval(dl)
#seg_info.do_eval()

#sys.exit()
#
#keys = [seg_info.get_key(i) for i in range(len(seg_info))]
#key_inds = np.array([seg_info.get_keyidx(i) for i in range(len(seg_info))], dtype=int)
#vels = [seg_info[i, 'mean_ego_vel'] for i in range(len(seg_info))]
##vels = [np.mean(seg_info[i, 'vel_diffs']) for i in range(len(seg_info))]
#deadlines = np.array([seg_info[i, 'deadline_ms'] for i in range(len(seg_info))], dtype=float)
#precs = np.array([seg_info.get_precision(i) for i in range(len(seg_info))], dtype=float)
#recalls = np.array([seg_info.get_recall(i) for i in range(len(seg_info))], dtype=float)
#
#all_deadlines= (45, 95, 145, 195) #  245, 295)
#
#dl_prec_dict=OrderedDict()
#fig, axes = plt.subplots(len(all_deadlines), 1, figsize=(6, 12), constrained_layout=True)
#for dl, ax in zip(all_deadlines, axes):
#    mask = (deadlines == dl)
#    y = precs[mask]
#    x = key_inds[mask] * 2
#    sort_inds = np.argsort(x)
#    dl_prec_dict[dl] = (x[sort_inds], y[sort_inds])
#    ax.scatter(x, y)
#    ax.set_xlabel('Time', fontsize='large')
#    ax.set_ylabel('Precision', fontsize='large')
#    ax.set_ylim(-0.05, 1.05)
#    ax.set_title(f'Deadline={dl}ms')
#plt.savefig(out_dir + "/precs_over_time.jpg")
#
#dl_recall_dict=OrderedDict()
#fig, axes = plt.subplots(len(all_deadlines), 1, figsize=(6, 12), constrained_layout=True)
#for dl, ax in zip(all_deadlines, axes):
#    mask = (deadlines == dl)
#    y = recalls[mask]
#    x = key_inds[mask] * 2
#    sort_inds = np.argsort(x)
#    dl_recall_dict[dl] = (x[sort_inds], y[sort_inds])
#    ax.scatter(x, y)
#    ax.set_xlabel('Time', fontsize='large')
#    ax.set_ylabel('Recall', fontsize='large')
#    ax.set_ylim(-0.05, 1.05)
#    ax.set_title(f'Deadline={dl}ms')
#plt.savefig(out_dir + "/recalls_over_time.jpg")
#
##precs_merged = np.stack([v[1] for v in dl_prec_dict.values()])
##recalls_merged = np.stack([v[1] for v in dl_recall_dict.values()])
##max_prec_masks = np.zeros(precs_merged.shape, dtype=bool)
##max_rec_masks = np.zeros(precs_merged.shape, dtype=bool)
##for i in range(precs_merged.shape[1]):
##    max_idx = np.argmax(precs_merged[:,i])
##    max_prec_masks[max_idx,i] = True
##    max_idx = np.argmax(recalls_merged[:,i])
##    max_rec_masks[max_idx,i] = True
#
#colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
#fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
##for pmask, rmask, dl in zip(max_prec_masks, max_rec_masks, all_deadlines):
#for dl in all_deadlines:
#    clr = next(colors)
#    #y_precs = dl_prec_dict[dl][1][pmask]
#    #x_precs = dl_prec_dict[dl][0][pmask]
#    #axes[0,0].scatter(x_precs, y_precs, label=f'DL={dl}ms', color=clr)
#
#    y_precs = dl_prec_dict[dl][1][:40]
#    x_precs = dl_prec_dict[dl][0][:40]
#    axes[0].scatter(x_precs, y_precs, label=f'DL={dl}ms', color=clr)
#    axes[0].plot(x_precs, y_precs, color=clr)
#
#    #y_recalls = dl_recall_dict[dl][1][rmask]
#    #x_recalls = dl_recall_dict[dl][0][rmask]
#    #axes[1,0].scatter(x_recalls, y_recalls, label=f'DL={dl}ms', color=clr)
#
#    y_recalls = dl_recall_dict[dl][1][:40]
#    x_recalls = dl_recall_dict[dl][0][:40]
#    axes[1].scatter(x_recalls, y_recalls, label=f'DL={dl}ms', color=clr)
#    axes[1].plot(x_recalls, y_recalls, color=clr)
#
##axes[0,0].set_title('Maximums only', fontsize='x-large')
#axes[0].set_title('Recall and precision for different deadlines', fontsize='x-large')
#axes[0].set_ylabel('Precision', fontsize='large')
##axes[0,1].set_ylabel('Precision', fontsize='x-large')
#axes[1].set_ylabel('Recall', fontsize='large')
##axes[1,1].set_ylabel('Recall', fontsize='x-large')
#axes[1].set_xlabel('Time (sec)', fontsize='large')
##axes[1,1].set_xlabel('Time (sec)', fontsize='large')
#for ax in np.ravel(axes):
#    ax.set_ylim(-0.05, 1.05)
#    ax.legend()
##ax.set_title(f'Deadline={dl}ms')
#plt.savefig(out_dir + "/precs_recalls_merged.jpg")
#
#vel_dict_p = {}
#vel_dict_r = {}
#for key, vel, dl, p, r in zip (keys, vels, deadlines, precs, recalls):
#    if p > 0. and (key not in vel_dict_p or vel_dict_p[key][2] < p): # best precision
#        vel_dict_p[key] = (vel, dl, p, r)
#    if r > 0. and (key not in vel_dict_r or vel_dict_r[key][3] < r): # best recall
#        vel_dict_r[key] = (vel, dl, p, r)
#
#fig, axes = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
#data_p = np.array([v[:2] for v in vel_dict_p.values()], dtype=float)
#data_r = np.array([v[:2] for v in vel_dict_r.values()], dtype=float)
#for ax, data, typ in zip(axes, (data_p, data_r), ('precision', 'recall')):
#    ax.scatter(data[:,0], data[:,1])
#    ax.set_xlabel('Velocity (m/s)', fontsize='x-large')
#    ax.set_ylabel(f'Best {typ} deadline (ms)', fontsize='x-large')
#    ax.grid('True', ls='--')
#plt.savefig(out_dir + "/vel_vs_prec_and_recalls.jpg")
#
#
