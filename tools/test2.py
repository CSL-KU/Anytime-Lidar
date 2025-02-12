#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
#os.environ["DATASET_PERIOD"] = "500"
#os.environ["PMODE"] = "pmode_0003" # same as jetson orin
#os.environ["CALIBRATION"] = "1"
#os.environ["PCDET_PATH"] = os.environ["HOME"] + "/shared/Anytime-Lidar"

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
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models.model_utils.tensorrt_utils.trtwrapper import TRTWrapper
from nuscenes import NuScenes
#from eval_utils.res_pred_utils import get_2d_egovel
from pcdet.models.detectors.valor_calibrator import get_stats

import matplotlib.pyplot as plt
import nuscenes
import importlib
# import numba
import concurrent.futures

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

def build_model(cfg_file, ckpt_file, default_deadline_sec):
    cfg_from_yaml_file(cfg_file, cfg)
    
    set_cfgs = ['MODEL.METHOD', '0',
            'MODEL.DEADLINE_SEC', str(default_deadline_sec),
            'MODEL.DENSE_HEAD.NAME', 'CenterHeadInf',
            'OPTIMIZATION.BATCH_SIZE_PER_GPU', '1']
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

def get_lastest_exec_time(model):
    pp_ms =  model._time_dict['PreProcess'][-1]
    sched_ms =  model._time_dict['Sched'][-1]
    preprocess_ms = pp_ms + sched_ms
    vfe_ms = model._time_dict['VFE'][-1]
    bb3d_ms = model._time_dict['Backbone3D'][-1]
    dense_ops_ms = float(model._time_dict['DenseOps'][-1])
    genbox_ms =  model._time_dict['CenterHead-GenBox'][-1]
    postp_ms =  model._time_dict['PostProcess'][-1]
    postprocess_ms = postp_ms + genbox_ms

    return np.array([preprocess_ms, vfe_ms, bb3d_ms, dense_ops_ms, postprocess_ms])

def run_test(model, resolution_idx=0, streaming=True, forecasting=False, simulate_exec_time=False):
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
    model.enable_forecasting = forecasting
    model.calibrate()
    resolution_stats = [0] * model.num_res if 'num_res' in dir(model) else [0]
    model.prev_scene_token = model.token_to_scene[model.dataset.infos[cur_sample_idx]['token']]

    with alive_bar(num_samples, force_tty=True, max_cols=160, manual=True) as bar:
        while cur_sample_idx < num_samples:
            # Check if we are in a new scene, reset if we are
            if streaming:
                potential_sample_tkn = model.dataset.infos[cur_sample_idx]['token']
                scene_token = model.token_to_scene[potential_sample_tkn]
                if model.prev_scene_token != scene_token:
                    target_sched_time_ms = 0.
                    while model.prev_scene_token != scene_token:
                        cur_sample_idx -= 1
                        potential_sample_tkn = model.dataset.infos[cur_sample_idx]['token']
                        scene_token = model.token_to_scene[potential_sample_tkn]
                    cur_sample_idx += 1
                    model.sim_cur_time_ms = cur_sample_idx * data_period_ms

            with torch.no_grad():
                lbd = model.latest_batch_dict # save bef its modified
                pred_dicts, ret_dict = model([cur_sample_idx])

            # Predict the execution time as if the DNN were to be executed on target platform
            batch_dict = model.latest_batch_dict
            xlen = batch_dict['tensor_slice_inds'][1] - batch_dict['tensor_slice_inds'][0]
            if simulate_exec_time:
                num_points = batch_dict['points'].size(0)
                num_voxels = np.array([batch_dict['bb3d_num_voxels']])
                last_exec_time_ms = model.calibrators[model.res_idx].pred_exec_time_ms(
                   num_points, num_voxels, xlen, consider_prep_time=True)
            else:
                last_exec_time_ms = model.last_elapsed_time_musec * 1e-3

            predicted = model.calibrators[model.res_idx].last_pred
            orig = get_lastest_exec_time(model)
            time_pred_diffs.append(predicted - orig)

            sample_tkn = batch_dict['metadata'][0]['token']

            if not streaming:
                # model.sim_cur_time_ms += data_period_ms # unnecessary
                model.sampled_dets[cur_sample_idx] = pred_dicts

            exec_times_ms.append((sample_tkn, last_exec_time_ms))

            resolution_stats[model.res_idx] += 1

            sched_algo = 'closer'
            #sched_algo = 'dynamic'
            if streaming and last_exec_time_ms > data_period_ms:
                cur_tail = calc_tail_ms(model.sim_cur_time_ms, data_period_ms)
                if sched_algo == 'closer':
                    if cur_tail > data_period_ms / 2:
                        model.sim_cur_time_ms += data_period_ms - cur_tail + 0.1
                elif sched_algo == 'dynamic':
                    #Dynamic scheduling
                    pred_finish_time = model.sim_cur_time_ms + last_exec_time_ms
                    next_tail = calc_tail_ms(pred_finish_time, data_period_ms)
                    if next_tail < cur_tail:
                        model.sim_cur_time_ms += data_period_ms - cur_tail + 0.1

                next_sample_idx = int(model.sim_cur_time_ms / data_period_ms)
            else:
                next_sample_idx = cur_sample_idx + 1
                model.sim_cur_time_ms = next_sample_idx * data_period_ms

            if cur_sample_idx == next_sample_idx:
                print(f'ERROR, trying to process already processed sample {next_sample_idx}')
            cur_sample_idx = next_sample_idx
            bar(cur_sample_idx / num_samples)

    # ignore preprocessing time prediction since it happens prior to scheduling
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
    return model.sampled_dets, exec_times_musec, resolution_stats

def do_eval(sampled_objects, resolution_idx, dataset, streaming=True,
            exec_times_musec=None, dump_eval_dict=True, loaded_nusc=None):

    det_annos = []
    num_ds_elems = len(dataset)
    for i in range(num_ds_elems):
        data_dict = dataset.get_metadata_dict(i)
        for k, v in data_dict.items():
            data_dict[k] = [v] # make it a batch dict
        pred_dicts = sampled_objects[i]

        if pred_dicts is None:
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
    result_str, result_dict = dataset.evaluation(
        det_annos, dataset.class_names,
        eval_metric='kitti', #model.model_cfg.POST_PROCESSING.EVAL_METRIC,
        output_path='./tmp_results',
        boxes_in_global_coords=False,
        loaded_nusc=loaded_nusc,
        det_elapsed_musec=None,
        #(None if streaming else exec_times_musec)
    )

    if dump_eval_dict:
        eval_d = {
                'cfg': cfg,
                'exec_times_musec': exec_times_musec,
                'det_annos': det_annos,
                'annos_in_glob_coords': False,
                'resolution': resolution_idx,
                'result_str': result_str,
        }

        with open(f'sampled_dets_res{resolution_idx}.pkl', 'wb') as f:
            pickle.dump(eval_d, f)
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
    forecasting = (forecasting == "forecast")

    num_res = 1
    if chosen_method == 'Pillarnet010':
        cfg_file  = "./cfgs/nuscenes_models/pillarnet010.yaml"
        ckpt_file = "../models/pillarnet010_epoch20.pth"
    elif chosen_method == 'Pillarnet015':
        cfg_file  = "./cfgs/nuscenes_models/pillarnet015.yaml"
        ckpt_file = "../models/pillarnet015_epoch20.pth"
    elif chosen_method == 'Pillarnet020':
        cfg_file  = "./cfgs/nuscenes_models/pillarnet020.yaml"
        ckpt_file = "../models/pillarnet020_epoch20.pth"
    elif chosen_method == 'Pillarnet024':
        cfg_file  = "./cfgs/nuscenes_models/pillarnet024.yaml"
        ckpt_file = "../models/pillarnet024_epoch20.pth"
    elif chosen_method == 'Pillarnet030':
        cfg_file  = "./cfgs/nuscenes_models/pillarnet030.yaml"
        ckpt_file = "../models/pillarnet030_epoch20.pth"
    elif chosen_method == 'VALO': # VALO Pillarnet 01
        cfg_file  = "./cfgs/nuscenes_models/cbgs_dyn_pillar015_res2d_centerpoint_valo.yaml"
        ckpt_file = "../models/pillarnet015_epoch20.pth"
    elif chosen_method == 'VALOR': # VALOR Pillarnet 5 res
        cfg_file  = "./cfgs/nuscenes_models/pillar01_015_02_024_03_valor.yaml"
        ckpt_file = "../models/pillar01_015_02_024_03_valor_epoch30.pth"
        #ckpt_file = "../output/nuscenes_models/pillar01_015_02_024_03_valor/default/ckpt/checkpoint_epoch_25.pth"
        num_res = 5
    elif chosen_method == 'VALOR2': # VALOR Pillarnet LS 5res
        cfg_file  = "./cfgs/nuscenes_models/pillar01_01125_01285_016_0225_valor.yaml"
        ckpt_file = "../models/pillar01_01125_01285_016_0225_valor_e30.pth"
        num_res = 5
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
            model = build_model(cfg_file, ckpt_file, default_deadline_sec)
            sampled_objects, exec_times_musec, resolution_stats = run_test(model, resolution_idx,
                                                                        streaming=streaming,
                                                                        forecasting=forecasting,
                                                                        simulate_exec_time=sim_exec_time)
            for outf in (fw, sys.stdout):
                print(f'Method:           {chosen_method}\n'
                      f'Config file:      {cfg_file}\n'
                      f'Checkpoint file:  {ckpt_file}\n'
                      f'Default deadline: {default_deadline_sec} seconds\n'
                      f'Power mode:       {os.environ["PMODE"]}\n'
                      f'Streaming:        {streaming}\n'
                      f'Forecasting:      {forecasting}\n'
                      f'Resolution stats: {resolution_stats}\n'
                      f'Latency stats:', file=outf)
                model.print_time_stats(outfile=outf)

            if not skip_eval:
                # fname = f'tmp_results/detdata_res{resolution_idx}.pkl'
                # with open(fname, 'rb') as f:
                #     sampled_objects, exec_times_musec, resolution_stats = pickle.load(f)
                #     print(f'Loaded {len(sampled_objects)} objects from {fname}')

                dataset = model.dataset
                del model
                loaded_nusc = load_dataset()
                result_str = do_eval(sampled_objects, resolution_idx, dataset, streaming,
                                     exec_times_musec=exec_times_musec,
                                     dump_eval_dict=False, loaded_nusc=loaded_nusc)

                for outf in (fw, sys.stdout):
                    print(result_str, file=outf)

            #model.dump_eval_dict(ret_dict) # not necessary
            t2 = time.time()
            print('Time passed (seconds):', t2-t1)
