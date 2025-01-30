#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("/root/shared/Anytime-Lidar/tools")
os.environ["DATASET_PERIOD"] = "50"
os.environ["PMODE"] = "pmode_0003" # same as jetson orin
os.environ["CALIBRATION"] = "0"
os.environ["PCDET_PATH"] = "/root/shared/Anytime-Lidar"

import _init_path
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

import matplotlib.pyplot as plt
import res_pred_utils
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

def build_model():
    cfg_file = "./cfgs/nuscenes_models/pillar01_015_02_024_03_valor.yaml"
    cfg_from_yaml_file(cfg_file, cfg)
    
    set_cfgs = ['MODEL.METHOD', '0', 'MODEL.DEADLINE_SEC', '100.0', 'MODEL.DENSE_HEAD.NAME', 'CenterHeadInf',
                'OPTIMIZATION.BATCH_SIZE_PER_GPU', '1']
    cfg_from_list(set_cfgs, cfg)
    logger, test_set, test_loader, sampler = get_dataset(cfg)
    print(f'Loaded dataset with {len(test_set)} samples')

    ckpt_file="../output/nuscenes_models/pillar01_015_02_024_03_valor/default/ckpt/checkpoint_epoch_30.pth"

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=ckpt_file, logger=logger, to_cpu=False)
    # model.pre_hook_handle.remove()
    # model.post_hook_handle.remove()
    model.eval() # should be run with @torch.no_grad
    model.cuda()

    return model

@torch.jit.script
def move_bounding_boxes(bboxes, egovel, time_diffs_sec):
    outp_shape = (time_diffs_sec.shape[0], bboxes.shape[0], bboxes.shape[1])
    outp_bboxes = torch.empty(outp_shape, dtype=bboxes.dtype)
    outp_bboxes[:, :, 2:] = bboxes[:, 2:]

    for t in range(time_diffs_sec.shape[0]):
        outp_bboxes[t, :, :2] = bboxes[:, :2] + (bboxes[:, 7:9] - egovel) * time_diffs_sec[t]

    return outp_bboxes

def run_test(model, resolution_idx, streaming=True, forecasting=False, sched_period_ms=2000):
    print('***********************')
    print(f'***RESOLUTION INDEX {resolution_idx}**')
    print('***********************')

    data_period_ms = int(os.environ["DATASET_PERIOD"])
    num_samples = len(model.dataset)

    cur_sample_idx = 0
    sim_cur_time_ms = 0.
    last_exec_time_ms = 100.
    target_sched_time_ms = 0.
    sampled_dets = [None] * num_samples
    exec_times_ms = []
    # sample_tokens = []
    resolution_stats = [0] * model.num_res

    model.calibrate()
    do_res_sched = (resolution_idx == -1)
    model.res_idx = 0 if do_res_sched else resolution_idx

    if do_res_sched:
        trt_path = f"./deploy_files/trt_engines/pmode_0000/resolution_pred_mdl.engine"
        print('Trying to load trt engine at', trt_path)
        res_pred_trt = TRTWrapper(trt_path, ['objcount_and_egovel'], ['res_scores'])
        res_pred_out_buf = None

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
                    sim_cur_time_ms = cur_sample_idx * data_period_ms

            with torch.no_grad():
                lbd = model.latest_batch_dict # save bef its modified
                pred_dicts, ret_dict = model([cur_sample_idx])

            # Predict the execution time as if the DNN were to be executed on target platform
            batch_dict = model.latest_batch_dict
            num_points = batch_dict['points'].size(0)
            num_voxels = np.array([batch_dict['bb3d_num_voxels']])
            xlen = batch_dict['tensor_slice_inds'][1] - batch_dict['tensor_slice_inds'][0]
            last_exec_time_ms = model.calibrators[model.res_idx].pred_exec_time_ms(
               num_points, num_voxels, xlen)

            sample_tkn = batch_dict['metadata'][0]['token']
            if lbd is not None and not batch_dict['scene_reset']:
                prev_sample_tkn = lbd['metadata'][0]['token']
                egovel = res_pred_utils.get_2d_egovel(
                        model.token_to_ts[prev_sample_tkn],
                        model.token_to_pose[prev_sample_tkn],
                        model.token_to_ts[sample_tkn],
                        model.token_to_pose[sample_tkn])
            else: # assume its zero
                egovel = np.zeros(2)

            exec_times_ms.append((sample_tkn, last_exec_time_ms))
            if not streaming:
                # sim_cur_time_ms += data_period_ms # unnecessary
                sampled_dets[cur_sample_idx] = pred_dicts
            else:
                # the sampled_dets can be overwritten, which is okay
                sim_cur_time_ms += last_exec_time_ms
                num_to_forecast = 500 // data_period_ms
                future_sample_inds = [(sim_cur_time_ms+(i*data_period_ms))//data_period_ms for i in range(1,num_to_forecast+1)]
                future_sample_inds = torch.tensor([ind for ind in future_sample_inds if ind < num_samples]).int()
                if forecasting: # NOTE consider the overhead here
                    # Forecast for next 500 ms
                    time_diffs_sec = (future_sample_inds * data_period_ms - (sim_cur_time_ms - last_exec_time_ms)) * 1e-3
                    outp_bboxes_all = move_bounding_boxes(pred_dicts[0]['pred_boxes'], torch.from_numpy(egovel), time_diffs_sec)
                    for outp_bboxes, sample_ind_f in zip(outp_bboxes_all, future_sample_inds.tolist()):
                        forecasted_pd = {k : pred_dicts[0][k] for k in ('pred_scores', 'pred_labels')}
                        forecasted_pd['pred_boxes'] = outp_bboxes
                        sampled_dets[sample_ind_f] = [forecasted_pd]
                else:
                    for sample_ind_f in future_sample_inds.tolist():
                        sampled_dets[sample_ind_f] = pred_dicts

            if do_res_sched and sim_cur_time_ms >= target_sched_time_ms:
                lbl_dist = torch.bincount(pred_dicts[0]['pred_labels'] - 1, minlength=10).float() / 100.0
                inp_tensor = torch.tensor(lbl_dist.tolist() + [np.linalg.norm(egovel).item()/15.0], dtype=torch.float).unsqueeze(0)
                inp_tensor[torch.isnan(inp_tensor)] = 0.
                res_pred_out_buf = res_pred_trt({'objcount_and_egovel': inp_tensor.cuda()},
                    res_pred_out_buf) 
                res_scores = res_pred_out_buf['res_scores'].cpu()

                _, chosen_res = torch.max(res_scores, 1)
                model.res_idx = chosen_res.item()

                #NOTE I need to consider the sched time as well and add to sim cur time ms
                target_sched_time_ms += sched_period_ms
                resolution_stats[model.res_idx] += 1

            #Dynamic scheduling
            if streaming:
                cur_tail = calc_tail_ms(sim_cur_time_ms, data_period_ms)
                pred_finish_time = sim_cur_time_ms + last_exec_time_ms #NOTE I can also use mean exec time
                next_tail = calc_tail_ms(pred_finish_time, data_period_ms)
                if next_tail < cur_tail:
                    # Sleep, extra 1 ms is added to make sure sleep time is enough
                    sim_cur_time_ms += data_period_ms - cur_tail + 1

                next_sample_idx = int(sim_cur_time_ms / data_period_ms)
            else:
                next_sample_idx = cur_sample_idx + 1

            if cur_sample_idx == next_sample_idx:
                print(f'ERROR, trying to process already processed sample {next_sample_idx}')

            cur_sample_idx = next_sample_idx
            bar(cur_sample_idx / num_samples)

    if do_res_sched:
        model.res_idx = -1
    model.print_time_stats()
    print('Resolution selection stats:')
    print(resolution_stats)

    exec_times_musec = {st:(et*1000) for st, et in exec_times_ms}

    with open(f'tmp_results/detdata_res{model.res_idx}.pkl', 'wb') as f:
        pickle.dump([sampled_dets, exec_times_musec, resolution_stats], f)

    print(f'Sampled {len(sampled_dets)} objects')
    return sampled_dets, exec_times_musec, resolution_stats

def do_eval(sampled_objects, resolution_idx, dataset, exec_times_musec=None, dump_eval_dict=True, loaded_nusc=None):
    #Convert them to openpcdet format
    os.environ["RESOLUTION_IDX"] = str(resolution_idx)

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
    streaming = (len(exec_times_musec) != len(dataset))
    print('STREAMING EVAL' if streaming else 'OFFLINE EVAL')
    result_str, result_dict = dataset.evaluation(
        det_annos, dataset.class_names,
        eval_metric='kitti', #model.model_cfg.POST_PROCESSING.EVAL_METRIC,
        output_path='./tmp_results',
        boxes_in_global_coords=False,
        loaded_nusc=loaded_nusc,
        det_elapsed_musec=None, #(None if streaming else exec_times_musec)
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



from nuscenes import NuScenes

dataset_version = 'v1.0-trainval'
root_path = "../data/nuscenes/" + dataset_version
loaded_nusc = NuScenes(version=dataset_version, dataroot=root_path, verbose=True)

# Run single test
# resolution_idx = 0
# streaming = False
# forecasting = False
# model = build_model()
# res_pred_mdl = res_pred_utils.ResolutionPredictor(model.num_res)
# sampled_objects, exec_times_musec, resolution_stats = run_test(model, resolution_idx, loaded_nusc,
                            # streaming=streaming, forecasting=forecasting, sched_period_ms=2000, collect_res_pred_data=True)
# result_str = do_eval(sampled_objects, resolution_idx, model.dataset, exec_times_musec=exec_times_musec,
#                              dump_eval_dict=False, loaded_nusc=loaded_nusc)
# print(result_str)

# Run batch test

streaming = True
offline = not streaming
results = []
skip_eval=False
forecasting=False # ignored if offline
num_res = 5
for resolution_idx in range(num_res):
    # os.environ["FINE_GRAINED_EVAL"] = ("1" if resolution_idx >= 0 else "0")
    t1 = time.time()
    model = build_model()

    sampled_objects, exec_times_musec, resolution_stats = run_test(model, resolution_idx, 
                                                                streaming=streaming, 
                                                                forecasting=forecasting, sched_period_ms=2000)
    if not skip_eval:
        # fname = f'tmp_results/detdata_res{resolution_idx}.pkl'
        # with open(fname, 'rb') as f:
        #     sampled_objects, exec_times_musec, resolution_stats = pickle.load(f)
        #     print(f'Loaded {len(sampled_objects)} objects from {fname}')

        dataset = model.dataset
        del model
        result_str = do_eval(sampled_objects, resolution_idx, dataset, exec_times_musec=exec_times_musec,
                             dump_eval_dict=True, loaded_nusc=loaded_nusc)
        results.append([resolution_idx, forecasting, resolution_stats, result_str])
        result = results[-1]
        print(f'Resolution index: {result[0]}')
        print(f'Forecasting: {forecasting}')
        print(f'Resolution stats: {result[2]}')
        print(result[3])
    t2 = time.time()
    print('Time passed (seconds):', t2-t1)
if not skip_eval:
    with open(f"output_streaming_streaming{streaming}_forecasting{forecasting}.txt", "w") as f:
        for resolution_idx, forecasting, resolution_stats, result_str in results:
            if forecasting:
                f.write('FORECASTING WAS UTILIZED\n')
            f.write(f'{resolution_stats}\n')
            f.write(result_str)
            f.write('\n')

