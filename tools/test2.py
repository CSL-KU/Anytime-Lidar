#!/usr/bin/env python
# coding: utf-8
import os
import _init_path
import sys
import datetime
import time
import json
import math
from pathlib import Path

import torch
import gc
import sys
import pickle
import numpy as np
from alive_progress import alive_bar

from eval_utils import eval_utils
from visual_utils.bev_visualizer import visualize_bev_detections
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.model_utils.tensorrt_utils.trtwrapper import TRTWrapper
from nuscenes import NuScenes
#from eval_utils.res_pred_utils import get_2d_egovel
from pcdet.models.detectors.mural_calibrator import get_stats

import matplotlib.pyplot as plt
import nuscenes
import importlib
# import numba
import concurrent.futures

from eval_utils.centerpoint_tracker import CenterpointTracker as Tracker

def get_dataset(cfg):
    log_file = './tmp_results/log_eval_%s' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = log_file + str(np.random.randint(0, 9999)) + '.txt'
    logger = common_utils.create_logger(log_file, rank=0)
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
        dist=False, workers=0, logger=logger, training=False
    )

    return logger, test_set, test_loader, sampler

def calc_tail_ms(cur_time_point_ms, data_period_ms):
    return cur_time_point_ms - math.floor(cur_time_point_ms / data_period_ms) * data_period_ms

def build_model(cfg_file, ckpt_file, default_deadline_sec, method):
    cfg_from_yaml_file(cfg_file, cfg)
    
    set_cfgs = ['MODEL.METHOD', method,
            'MODEL.DEADLINE_SEC', str(default_deadline_sec),
            'MODEL.DENSE_HEAD.NAME', 'CenterHeadInf',
            'OPTIMIZATION.BATCH_SIZE_PER_GPU', '1']
#            'MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH', '0.1']
    cfg_from_list(set_cfgs, cfg)
    logger, test_set, test_loader, sampler = get_dataset(cfg)
    print(f'Loaded dataset with {len(test_set)} samples')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=ckpt_file, logger=logger, to_cpu=False)
    # model.pre_hook_handle.remove()
    # model.post_hook_handle.remove()
    model.eval() # should be run with @torch.no_grad
    model.cuda()

    return model

def get_latest_exec_time(model):
    pp_ms =  model._time_dict['PreProcess'][-1]
    sched_ms =  model._time_dict['Sched'][-1]
    preprocess_ms = pp_ms + sched_ms
    vfe_ms = model._time_dict['VFE'][-1]
    bb3d_ms = model._time_dict.get('Backbone3D', [0])[-1]
    dense_ops_ms = float(model._time_dict['DenseOps'][-1])
    genbox_ms =  model._time_dict['CenterHead-GenBox'][-1]
    postp_ms =  model._time_dict['PostProcess'][-1]
    postprocess_ms = postp_ms + genbox_ms

    return np.array([preprocess_ms, vfe_ms, bb3d_ms, dense_ops_ms, postprocess_ms])


def gen_gt_database(model):
    num_samples = len(model.dataset)
    cur_sample_idx = 0
    model.calibrate()
    visualize = False

    gt_and_timepred_tuples = []

    with alive_bar(num_samples, force_tty=True, max_cols=160, manual=False) as bar:
        with torch.no_grad():
            while cur_sample_idx < num_samples:
                #pred_dicts, ret_dict = model([cur_sample_idx])
                data_dict = model.dataset[cur_sample_idx]
                sample_tkn = data_dict['metadata']['token']
                gt_boxes = data_dict['gt_boxes']

                batch_dict = model.dataset.collate_batch([data_dict])
                points = torch.from_numpy(batch_dict['points']).float().cuda()
                points = common_utils.pc_range_filter(points, model.filter_pc_range)
                all_pred_res_latencies, _ = model.pred_all_res_times(points)
                gt_and_timepred_tuples.append((sample_tkn,
                                              gt_boxes, all_pred_res_latencies))

                if visualize:
                    pboxes = pred_dicts[0]['pred_boxes'][:, :7].cpu().numpy()
                    gtboxes = batch_dict['gt_boxes'][0, :, :7].cpu().numpy()
                    fname = f"images/sample{cur_sample_idx}.png"
                    print(pboxes.shape, gtboxes.shape)
                    visualize_bev_detections(pboxes, gtboxes, fname, swap_wh=True)
                    print(f'Image saved to {fname}')

                cur_sample_idx += 1
                bar()

    calib_id = os.environ.get("CALIBRATION", "0")
    with open(f'sampled_dets/respred_gt_database_{calib_id}.pkl', 'wb') as f:
        pickle.dump(gt_and_timepred_tuples, f)


def run_test(model, resolution_idx=0, streaming=True, forecasting=False, simulate_exec_time=False):
    if hasattr(model, 'res_idx'):
        print('***********************')
        print(f'***RESOLUTION INDEX {resolution_idx}**')
        print('***********************')

    data_period_ms = int(os.environ["DATASET_PERIOD"])
    num_samples = len(model.dataset)

    cur_sample_idx = 0
    last_exec_time_ms = 100.
    target_sched_time_ms = 0.
    exec_times_ms = []
    time_pred_diffs = []
    # sample_tokens = []
    model.simulate_exec_time = simulate_exec_time
    model.enable_forecasting_to_fut = forecasting
    model.calibrate()
    resolution_stats = [0] * model.num_res if 'num_res' in dir(model) else [0]
    sampled_exec_times_ms = [None] * num_samples
    scene_begin_inds = []

    sched_algo = 'dynamic'
    #sched_algo = 'periodic'
    #sched_algo = 'dynamic'

    with alive_bar(num_samples, force_tty=True, max_cols=160, manual=True) as bar:
        while cur_sample_idx < num_samples:
            # Check if we are in a new scene, reset if we are
            if streaming:
                potential_sample_tkn = model.dataset.infos[cur_sample_idx]['token']
                scene_token = model.token_to_scene[potential_sample_tkn]
                if cur_sample_idx > 0 and model.prev_scene_token != scene_token:
                    target_sched_time_ms = 0.
                    while model.prev_scene_token != scene_token:
                        cur_sample_idx -= 1
                        potential_sample_tkn = model.dataset.infos[cur_sample_idx]['token']
                        scene_token = model.token_to_scene[potential_sample_tkn]
                    cur_sample_idx += 1
                    model.sim_cur_time_ms = float(cur_sample_idx * data_period_ms)
                    scene_begin_inds.append(cur_sample_idx)

            with torch.no_grad():
                lbd = model.latest_batch_dict # save bef its modified
                pred_dicts, ret_dict = model([cur_sample_idx])

            # Predict the execution time as if the DNN were to be executed on target platform
            batch_dict = model.latest_batch_dict
            last_exec_time_ms = model.last_elapsed_time_musec * 1e-3

            if hasattr(model, 'calibrators'):
                predicted = model.calibrators[model.res_idx].last_pred
                orig = get_latest_exec_time(model)
                time_pred_diffs.append(predicted - orig)

            sample_tkn = batch_dict['metadata'][0]['token']

            if not streaming:
                # model.sim_cur_time_ms += data_period_ms # unnecessary
                model.sampled_dets[cur_sample_idx] = pred_dicts

            exec_times_ms.append((sample_tkn, last_exec_time_ms))
            sampled_exec_times_ms[cur_sample_idx] = (last_exec_time_ms, model.sim_cur_time_ms)

            if hasattr(model, 'res_idx'):
                resolution_stats[model.res_idx] += 1

            if streaming and sched_algo == 'periodic':
                # The model is assumed to be executed in a periodic manner
                # If exec time is more than the period, it is assumed that
                # the task was aborted with no result
                #model.sim_cur_time_ms = int(len(exec_times_ms) * \
                #        (model._default_deadline_sec * 1000))
                next_sample_idx = int(model.sim_cur_time_ms / data_period_ms)
            elif streaming and last_exec_time_ms > data_period_ms:
                cur_tail = calc_tail_ms(model.sim_cur_time_ms, data_period_ms)
                if sched_algo == 'closer':
                    if cur_tail > data_period_ms / 2:
                        model.sim_cur_time_ms += data_period_ms - cur_tail
                elif sched_algo == 'dynamic':
                    #Dynamic scheduling
                    pred_finish_time = model.sim_cur_time_ms + last_exec_time_ms
                    next_tail = calc_tail_ms(pred_finish_time, data_period_ms)
                    if next_tail < cur_tail:
                        model.sim_cur_time_ms += data_period_ms - cur_tail

                next_sample_idx = int(round(model.sim_cur_time_ms / data_period_ms))
            else:
                next_sample_idx = cur_sample_idx + 1
                model.sim_cur_time_ms = next_sample_idx * data_period_ms

            if cur_sample_idx == next_sample_idx:
                print(f'ERROR, trying to process already processed sample {next_sample_idx}')
            cur_sample_idx = next_sample_idx
            bar(cur_sample_idx / num_samples)

    # ignore preprocessing time prediction since it happens prior to scheduling
    if hasattr(model, 'calibrators'):
        tpred_diffs = np.array(time_pred_diffs)[:, 1:]

        e2e_diffs = tpred_diffs.sum(1)
        print('E2E execution time prediction error is below.')
        print('If number is positive, then finished earlier then predicted.')
        get_stats(e2e_diffs)
        print('Other time prediction errors:')
        for i in range(tpred_diffs.shape[1]):
            get_stats(tpred_diffs[:, i])

    util = sum([t[1] for t in exec_times_ms]) / (num_samples*data_period_ms)
    print(f'Utilization: %{util:.2f}')
    exec_times_musec = {st:(et*1000) for st, et in exec_times_ms}

    #with open(f'tmp_results/detdata_res{model.res_idx}.pkl', 'wb') as f:
    #    pickle.dump([sampled_dets, exec_times_musec, resolution_stats], f)

    print(f'Sampled {len(model.sampled_dets)} objects')
    return model.sampled_dets, exec_times_musec, resolution_stats, model.sampled_egovels, sampled_exec_times_ms, scene_begin_inds

def do_eval(sampled_dets, resolution_idx, dataset, streaming=True,
            exec_times_musec=None, loaded_nusc=None, egovels=None,
            sampled_exec_times_ms=None, scene_begin_inds=None):

    det_annos = []
    num_ds_elems = len(dataset)
    for i in range(num_ds_elems):
        data_dict = dataset.get_metadata_dict(i)
        for k, v in data_dict.items():
            data_dict[k] = [v] # make it a batch dict
        pred_dicts = sampled_dets[i]

        if pred_dicts is None:
            if i > 0 and sampled_dets[i-1] is not None:
                sampled_dets[i] = sampled_dets[i-1]
                pred_dicts = sampled_dets[i]
            else:
                pred_dicts = [{
                    'pred_boxes': torch.empty((0, 9)),
                    'pred_scores': torch.empty(0),
                    'pred_labels': torch.empty(0, dtype=torch.long)
                }]
        data_dict['final_box_dicts'] = pred_dicts
        det_annos += dataset.generate_prediction_dicts(
            data_dict, data_dict['final_box_dicts'], dataset.class_names, output_path=None
        )

    #nusc_annos = {} # not needed but keep it anyway
    print('STREAMING EVAL' if streaming else 'OFFLINE EVAL')
    nusc_annos = {}
    result_str, result_dict = dataset.evaluation(
        det_annos, dataset.class_names,
        eval_metric='kitti', #model.model_cfg.POST_PROCESSING.EVAL_METRIC,
        output_path='./tmp_results',
        boxes_in_global_coords=False,
        loaded_nusc=loaded_nusc,
        det_elapsed_musec=None,
        nusc_annos_outp=nusc_annos,
        #(None if streaming else exec_times_musec)
    )

    do_tracking=False
    if do_tracking:
        ## NUSC TRACKING START
        tracker = Tracker(max_age=6, hungarian=False)
        predictions = nusc_annos['results']
        with open('frames/frames_meta.json', 'rb') as f:
            frames=json.load(f)['frames']

        nusc_trk_annos = {
            "results": {},
            "meta": None,
        }
        size = len(frames)

        print("Begin Tracking\n")
        start = time.time()
        for i in range(size):
            token = frames[i]['token']

            # reset tracking after one video sequence
            if frames[i]['first']:
                # use this for sanity check to ensure your token order is correct
                # print("reset ", i)
                tracker.reset()
                last_time_stamp = frames[i]['timestamp']

            time_lag = (frames[i]['timestamp'] - last_time_stamp)
            last_time_stamp = frames[i]['timestamp']

            preds = predictions[token]

            outputs = tracker.step_centertrack(preds, time_lag)
            annos = []

            for item in outputs:
                if item['active'] == 0:
                    continue
                nusc_anno = {
                    "sample_token": token,
                    "translation": item['translation'],
                    "size": item['size'],
                    "rotation": item['rotation'],
                    "velocity": item['velocity'],
                    "tracking_id": str(item['tracking_id']),
                    "tracking_name": item['detection_name'],
                    "tracking_score": item['detection_score'],
                }
                annos.append(nusc_anno)
            nusc_trk_annos["results"].update({token: annos})
        end = time.time()
        second = (end-start)
        speed=size / second
        print("The speed is {} FPS".format(speed))
        nusc_trk_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        with open('tmp_results/tracking_result.json', "w") as f:
            json.dump(nusc_trk_annos, f)

        #result is nusc_annos
        dataset.tracking_evaluation(
            output_path='tmp_results',
            res_path='tmp_results/tracking_result.json'
        )
        ## NUSC TRACKING END

        with open('tmp_results/metrics_summary.json', 'rb') as f:
            tracking_res=json.load(f)

    calib_id = int(os.environ.get('CALIBRATION', '0'))
    #if calib_id > 0:
    ridx = int(os.environ.get('FIXED_RES_IDX', -1))
    eval_d = {
           'cfg': cfg,
           'scene_begin_inds': scene_begin_inds,
           'exec_times_ms': sampled_exec_times_ms,
           'objects': sampled_dets,
           'egovels': egovels,
           'resolution': ridx,
           'result_str': result_str,
           #'tracking_result': tracking_res,
    }
    with open(f'sampled_dets/res{ridx}_calib{calib_id}.pkl', 'wb') as f:
        pickle.dump(eval_d, f)

    if do_tracking:
        for k, v in tracking_res.items():
            if isinstance(v, str) or isinstance(v, float) or isinstance(v, int):
                result_str += k + ': ' + str(v) + '\n'

    return result_str

LOADED_NUSC = None
def load_dataset():
    global LOADED_NUSC
    if LOADED_NUSC is None:
        dataset_version = 'v1.0-trainval'
        root_path = "../data/nuscenes/" + dataset_version
        LOADED_NUSC = NuScenes(version=dataset_version, dataroot=root_path, verbose=True)
    return LOADED_NUSC

if __name__ == "__main__":
    chosen_method = sys.argv[1]
    default_deadline_sec = sys.argv[2] if len(sys.argv) > 2 else 100.0
    mode = sys.argv[3] if len(sys.argv) > 3 else "streaming"
    forecasting = sys.argv[4] if len(sys.argv) > 4 else "noforecast"
    streaming = (mode == "streaming") # otherwise offline
    build_gt_database = (mode == "build_gt_database")
    forecasting = (forecasting == "forecast")

    num_res = 1
    method = 0
    if chosen_method == 'Pillarnet0100':
        cfg_file  = "./cfgs/nuscenes_models/pillarnet0100.yaml"
        ckpt_file = "../models/pillarnet0100_e20.pth"
    elif chosen_method == 'Pillarnet0128':
        cfg_file  = "./cfgs/nuscenes_models/pillarnet0128.yaml"
        ckpt_file = "../models/pillarnet0128_e20.pth"
    elif chosen_method == 'Pillarnet0160':
        cfg_file  = "./cfgs/nuscenes_models/pillarnet0160.yaml"
        ckpt_file = "../models/pillarnet0160_e20.pth"
    elif chosen_method == 'Pillarnet0200':
        cfg_file  = "./cfgs/nuscenes_models/pillarnet0200.yaml"
        ckpt_file = "../models/pillarnet0200_e20.pth"
    elif chosen_method == 'VALO': # VALO Pillarnet 01
        cfg_file  = "./cfgs/nuscenes_models/valo_pillarnet_0100.yaml"
        ckpt_file = "../models/pillarnet0100_e20.pth"
        method = 5
    elif chosen_method == 'MURAL_0075_3res':
        cfg_file  = "./cfgs/nuscenes_models/mural_pillarnet_0075_0100_0150.yaml"
        ckpt_file = "../models/mural_pillarnet_0075_0100_0150_e20.pth"
        num_res = 3
        method = 12
    elif chosen_method == 'MURAL_0100_0128_0200':
        cfg_file  = "./cfgs/nuscenes_models/mural_pillarnet_0100_0128_0200.yaml"
        ckpt_file = "../models/mural_pillarnet_0100_0128_0200_e20.pth"
        num_res = 3
        method = 12
    elif chosen_method == 'MURAL_CenterPointPP':
        cfg_file="./cfgs/nuscenes_models/mural_pp_centerpoint_0200_0256_0400.yaml"
        ckpt_file="../models/mural_pp_centerpoint_0200_0256_0400_e20.pth"
        num_res = 3
        method = 12
    else:
        print('Unknown method, exiting.')
        sys.exit()

    sim_exec_time = False # Only VALOR supports it
    skip_eval = False

    results = []
    with open(f"evalres_{chosen_method}_{int(time.time())}.txt", "w") as fw:
        for resolution_idx in range(1): #num_res):
            # os.environ["FINE_GRAINED_EVAL"] = ("1" if resolution_idx >= 0 else "0")
            t1 = time.time()
            model = build_model(cfg_file, ckpt_file, default_deadline_sec, method)

            if build_gt_database:
                # FOR DATASET GEN ONLY
                gen_gt_database(model)
                sys.exit()

            sampled_dets, exec_times_musec, resolution_stats, egovels, sampled_exec_times_ms, sbi = \
                    run_test(model, resolution_idx,
                    streaming=streaming,
                    forecasting=forecasting,
                    simulate_exec_time=sim_exec_time)
            dl_misses = model._eval_dict['deadlines_missed']
            for outf in (fw, sys.stdout):
                print(f'Method:           {chosen_method}\n'
                      f'Config file:      {cfg_file}\n'
                      f'Checkpoint file:  {ckpt_file}\n'
                      f'Default deadline: {default_deadline_sec} seconds\n'
                      f'Power mode:       {os.environ["PMODE"]}\n'
                      f'Streaming:        {streaming}\n'
                      f'Forecasting:      {forecasting}\n'
                      f'Resolution stats: {resolution_stats}\n'
                      f'Deadline misses:  {dl_misses}\n'
                      f'Latency stats:', file=outf)
                model.print_time_stats(outfile=outf)

            if not skip_eval:
                # fname = f'tmp_results/detdata_res{resolution_idx}.pkl'
                # with open(fname, 'rb') as f:
                #     sampled_dets, exec_times_musec, resolution_stats = pickle.load(f)
                #     print(f'Loaded {len(sampled_dets)} objects from {fname}')

                dataset = model.dataset
                del model
                loaded_nusc = load_dataset()
                result_str = do_eval(sampled_dets, resolution_idx, dataset, streaming,
                                     exec_times_musec=exec_times_musec,
                                     loaded_nusc=loaded_nusc,
                                     egovels=egovels, sampled_exec_times_ms=sampled_exec_times_ms,
                                     scene_begin_inds=sbi)

                for outf in (fw, sys.stdout):
                    print(result_str, file=outf)

            t2 = time.time()
            print('Time passed (seconds):', t2-t1)
