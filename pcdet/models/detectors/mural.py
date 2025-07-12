from .detector3d_template import Detector3DTemplate
from .mural_calibrator import  MURALCalibrator
from ..model_utils.tensorrt_utils.trtwrapper import TRTWrapper, create_trt_engine
import torch
import time
import onnx
import json
import pickle
import os
import sys
import numpy as np
import platform
from typing import Dict, List, Tuple, Optional, Final
from .forecaster import split_dets, Forecaster
from .mural_utils import *
from ...utils import common_utils, vsize_calc

import ctypes
pth = os.environ['PCDET_PATH']
pth = os.path.join(pth, "pcdet/trt_plugins/slice_and_batch_nhwc/build/libslice_and_batch_lib.so")
ctypes.CDLL(pth, mode=ctypes.RTLD_GLOBAL)

class MURAL(Detector3DTemplate):
    def method_num_to_str_list(method_num):
        if method_num == 6:
            # dynamic scheduling, dense conv opt, res interpolation, forecasting
            return ("DS", "RI", "DCO", "FRC")
        elif method_num == 7:
            return ("DS", "RI", "DCO")
        elif method_num == 8:
            return ("DS", "RI")
        elif method_num == 9:
            return ("DS",)
        elif method_num == 10:
            return ("WS",) # wcet scheduling
        elif method_num == 11:
            return ("WS", "RI", "DCO", "FRC")
        elif method_num == 12:
            return ("DS", "DCO")
        else:
            return tuple()

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.method = int(self.model_cfg.METHOD)
        self.method_str = MURAL.method_num_to_str_list(self.method)
        self.dettype = self.model_cfg.DETECTOR_TYPE
        assert self.dettype == 'PillarNet' or self.dettype == 'PointPillarsCP'

        rd = model_cfg.get('RESOLUTION_DIV', [1.0])
        pc_range = self.dataset.point_cloud_range
        self.max_grid_l = self.dataset.grid_size[0]
        grid_slice_sz = 32 if self.dettype == 'PillarNet' else 8
        if "RI" in self.method_str:
            pc_range_l = pc_range.tolist()
            all_pc_ranges, all_pillar_sizes, all_grid_lens, new_resdivs, resdiv_mask = \
                    vsize_calc.interpolate_pillar_sizes(self.max_grid_l, rd, pc_range_l,
                    step=grid_slice_sz)
            #Additional resolution lower than lowest trained resolution
            area_l_mm = int((pc_range_l[3] - pc_range_l[0]) * 1000000)
            area_min_l_mm = area_l_mm - 1500000
            area_max_l_mm = area_l_mm + 1500000
            min_grid_len = all_grid_lens[-1]
            if model_cfg.get('EXTRAPOLATE_RES', True):
                diffs = (32, 64, 96) if self.dettype == 'PointPillarsCP' else (128,)
                for diff in diffs:
                    opt = vsize_calc.calc_area_and_pillar_sz(min_grid_len-diff, area_min_l_mm,
                                                             area_max_l_mm)
                    new_pc_range, new_psize, new_grid_l = vsize_calc.option_to_params(opt,
                                                                                      pc_range_l,
                                                                                      pillar_h=0.2)
                    all_pc_ranges.append(new_pc_range)
                    all_pillar_sizes.append(new_psize)
                    all_grid_lens.append(new_grid_l)
                    resdiv_mask.append(False)
            new_resdivs = [all_grid_lens[0]/gl for gl in all_grid_lens]
            rd = new_resdivs
            all_pillar_sizes = torch.tensor(all_pillar_sizes)
        else:
            t = torch.tensor(rd) * self.dataset.voxel_size[0]
            all_pillar_sizes = t.repeat_interleave(2).reshape(-1, 2)
            all_pc_ranges = [pc_range.tolist()] * len(rd)
            resdiv_mask = [True] * len(rd)

        self._eval_dict['resolution_selections'] = [0] * len(rd)

        self.model_cfg.VFE.RESOLUTION_DIV = rd
        self.model_cfg.VFE.RESDIV_MASK = resdiv_mask
        self.model_cfg.VFE.ALL_PC_RANGES = all_pc_ranges
        if self.dettype == 'PillarNet':
            self.model_cfg.BACKBONE_3D.RESOLUTION_DIV = rd
            self.model_cfg.BACKBONE_3D.RESDIV_MASK = resdiv_mask
        elif self.dettype == 'PointPillarsCP':
            self.model_cfg.MAP_TO_BEV.RESOLUTION_DIV = rd
            self.model_cfg.MAP_TO_BEV.RESDIV_MASK = resdiv_mask
        self.model_cfg.BACKBONE_2D.RESOLUTION_DIV = rd
        self.model_cfg.BACKBONE_2D.RESDIV_MASK = resdiv_mask
        self.model_cfg.DENSE_HEAD.RESOLUTION_DIV = rd
        self.model_cfg.DENSE_HEAD.RESDIV_MASK = resdiv_mask
        self.model_cfg.DENSE_HEAD.ALL_PC_RANGES = all_pc_ranges
        self.resolution_dividers = rd
        self.resdiv_mask = resdiv_mask
        self.all_pc_ranges = torch.tensor(all_pc_ranges)

        self.interpolate_batch_norms = ("RI" in self.method_str)
        self.dense_conv_opt_on = ("DCO" in self.method_str)
        self.enable_forecasting_to_fut = ("FRC" in self.method_str) and self.ignore_dl_miss
        self.enable_forecasting = ("FRC" in self.method_str) and not self.ignore_dl_miss
        self.wcet_scheduling = ("WS" in self.method_str)

        self.model_cfg.DENSE_HEAD.OPTIMIZE_ATTR_CONVS = self.dense_conv_opt_on

        self.deadline_based_selection = not self.ignore_dl_miss
        if self.deadline_based_selection:
            print('Deadline scheduling is enabled!')

        torch.backends.cudnn.benchmark = True
        if torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark_limit = 0

        allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        torch.cuda.manual_seed(0)
        self.module_list = self.build_networks()

        self.model_name = self.model_cfg.NAME + '_' + self.model_cfg.NAME_POSTFIX

        self.update_time_dict({
            'Sched' : [],
            'VFE' : [],
            ('Backbone3D' if self.dettype == 'PillarNet' else 'MapToBEV'): [],
            'DenseOps':[],
            'CenterHead-GenBox': [],
        })

        if self.dettype == 'PillarNet':
            self.vfe, self.backbone_3d, self.backbone_2d, self.dense_head = self.module_list
            self.map_to_bev = None
        if self.dettype == 'PointPillarsCP':
            self.vfe, self.map_to_bev, self.backbone_2d, self.dense_head = self.module_list
            self.map_to_bev_scrpt = torch.jit.script(self.map_to_bev)
            self.backbone_3d = None
        print('Model size is:', self.get_model_size_MB(), 'MB')

        self.num_res = len(self.resolution_dividers)
        self.res_aware_1d_batch_norms, self.res_aware_2d_batch_norms = get_all_resawarebn(self)
        self.res_idx = 0

        self.inf_stream = torch.cuda.Stream()
        self.optimization_done = [False] * self.num_res
        self.trt_outputs = [None] * self.num_res # Since output size of trt is fixed, use buffered
        self.fused_convs_trt = [None] * self.num_res

        fms = self.model_cfg.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
        self.target_hw = [sz // fms for sz in self.dataset.grid_size[:2]]
        self.opt_dense_convs = None

        # narrow it a little bit to avoid numerical errors
        filter_min = torch.max(self.all_pc_ranges[:, :3], dim=0).values
        filter_max = torch.min(self.all_pc_ranges[:, 3:], dim=0).values
        filter_range = torch.cat((filter_min, filter_max))
        self.filter_pc_range =  filter_range + \
                torch.tensor([0.01, 0.01, 0.01, -0.01, -0.01, -0.01])
        self.filter_pc_range = self.filter_pc_range.cuda()
        print('Point cloud filtering range:')
        print(self.filter_pc_range)
        self.calib_pc_range = self.filter_pc_range.clone()

        mpc = MultiPillarCounter(all_pillar_sizes, torch.tensor(all_pc_ranges), grid_slice_sz)
        mpc.eval()
        self.mpc_script = torch.jit.script(mpc)

        self.dense_head_scrpt = None
        self.inp_tensor_sizes = [np.ones(4, dtype=int)] * self.num_res
        self.dense_inp_slice_sz = 4 if self.dettype == 'PillarNet' else 8
        self.calibrators = [MURALCalibrator(self, ri, self.mpc_script.num_slices[ri]) \
                for ri in range(self.num_res)]

        self.use_oracle_res_predictor = False
        self.res_predictor = None
        if self.use_oracle_res_predictor:
            with open('oracle_test.json', 'r') as f:
                self.oracle_respred = json.load(f)
            print('USING ORACLE RESOLUTION PREDICTOR!')
        else:
            try:
                with open('random_forest_model.pkl', 'rb') as f:
                    self.res_predictor = pickle.load(f)
                    print('Loaded resolution predictor.')
            except:
                pass
        self.sched_time_point_ms = 0
        self.batch_norm_interpolated = False
        self.x_minmax = torch.empty((self.num_res, 2), dtype=torch.int)
        self.e2e_min_times_ms = None

        if self.enable_forecasting:
            pc_range, tcount = tuple(all_pc_ranges[0]), 1
            self.forecaster = torch.jit.script(Forecaster(pc_range, tcount,
                    self.score_thresh, num_det_heads=self.dense_head.num_det_heads,
                    cls_id_to_det_head_idx_map=self.dense_head.cls_id_to_det_head_idx_map))

        self.traced_vfe = [None] * self.num_res

    def forward(self, batch_dict):
        if self.training:
            assert False # not supported
        else:
            return self.forward_eval(batch_dict)

    def forward_eval(self, batch_dict):
        if self.interpolate_batch_norms and not self.batch_norm_interpolated:
            interpolate_batch_norms(self.res_aware_1d_batch_norms, self.max_grid_l)
            interpolate_batch_norms(self.res_aware_2d_batch_norms, self.max_grid_l) #, True)
            self.batch_norm_interpolated = True
            print('Model size is:', self.get_model_size_MB(), 'MB')

        scene_reset = batch_dict['scene_reset']

        with torch.cuda.stream(self.inf_stream):
            # The time before this is measured as preprocess
            self.measure_time_start('Sched')
            if self.is_calibrating() and self.calibrators[self.res_idx].repeat_points > 1:
                pts = batch_dict['points']
                batch_dict['points'] = pts.repeat(self.calibrators[self.res_idx].repeat_points, 1)
            points = common_utils.pc_range_filter(batch_dict['points'],
                                self.calib_pc_range if self.is_calibrating() else
                                self.filter_pc_range)
            batch_dict['points'] = points

            fixed_res_idx = int(os.environ.get('FIXED_RES_IDX', -1))
            # Schedule by calculating the exec time of all resolutions
            for i in range(self.num_res):
                self.x_minmax[i, 0] = 0
                self.x_minmax[i, 1] = self.mpc_script.num_slices[i] - 1
            x_minmax_calculated = False

            if fixed_res_idx > -1:
                pred_res_idx = fixed_res_idx
            elif self.deadline_based_selection:
                conf_found = False
                abs_dl_sec = batch_dict['abs_deadline_sec']
                if self.wcet_scheduling:
                    t = time.time()
                    time_passed = (t - batch_dict['start_time_sec']) * 1000
                    time_left = (abs_dl_sec - time.time()) * 1000
                    for i in range(self.num_res):
                        pred_latency = self.calibrators[i].get_e2e_wcet_ms() - time_passed
                        if pred_latency < time_left:
                            pred_res_idx = i
                            conf_found = True
                            break
                else:
                    if self.e2e_min_times_ms is not None:
                        keepmask = (self.e2e_min_times_ms <= (batch_dict['deadline_sec']*1000))
                        first_res_idx = torch.where(keepmask)[0]
                        if first_res_idx.size(0) <= 1: # no need for scheduling
                            first_res_idx = self.num_res - 1
                        else:
                            first_res_idx = first_res_idx[0].item()
                    else:
                        first_res_idx = 0
                    if first_res_idx < self.num_res - 1:
                        if self.backbone_3d is not None:
                            points_xy = points[:, 1:3]
                            pc0, all_pillar_counts = self.mpc_script(points_xy, first_res_idx)
                            if self.dense_conv_opt_on:
                                x_minmax_tmp = torch.from_numpy(get_xminmax_from_pc0(pc0.cpu().numpy()))
                                self.x_minmax[first_res_idx:] = x_minmax_tmp
                                x_minmax_calculated = True
                            all_pillar_counts = all_pillar_counts.cpu()
                        else:
                            if self.dense_conv_opt_on:
                                self.x_minmax = self.mpc_script.get_minmax_inds(points[:, 1])
                                x_minmax_calculated = True
                        num_points = points.size(0)
                        for i in range(first_res_idx, self.num_res):
                            pillar_counts = all_pillar_counts[i-first_res_idx].numpy() \
                                    if self.backbone_3d is not None else None
                            pred_latency = self.calibrators[i].pred_exec_time_ms(num_points,
                                    pillar_counts,
                                    (self.x_minmax[i, 1] - self.x_minmax[i, 0] + 1).item(),
                                    consider_prep_time=self.simulate_exec_time)
                            if self.simulate_exec_time:
                                if pred_latency < batch_dict['deadline_sec'] * 1000:
                                    pred_res_idx = i
                                    conf_found = True
                                    break
                            else:
                                time_left = (abs_dl_sec - time.time()) * 1000
                                if pred_latency < time_left:
                                    pred_res_idx = i
                                    conf_found = True
                                    break

                if not conf_found:
                    pred_res_idx = self.num_res - 1
            elif self.use_oracle_res_predictor:
                sample_token = batch_dict['metadata'][0]['token']
                pred_res_idx = self.oracle_respred[sample_token]
            elif self.res_predictor is not None:
                if self.latest_batch_dict is not None:
                    # If you want to sched periodically rel to scene start, uncomment
                    tdiff = int(self.sim_cur_time_ms - self.sched_time_point_ms)
                    if self.is_calibrating() or tdiff >= 2000:
                        # takes 13 ms for 3 resolutions
                        self.sched_time_point_ms = self.sim_cur_time_ms

                        pd = self.latest_batch_dict['final_box_dicts'][0]

                        # Random forest based prediction
                        ev = self.sampled_egovels[self.dataset_indexes[0]] # current one

                        boxes = pd['pred_boxes'].numpy()
                        if len(boxes) > 0:
                            obj_velos = boxes[:, 7:9]
                            velmask = np.isnan(obj_velos).any(1)
                            obj_velos[velmask] = 0.
                            rel_velos = obj_velos - ev
                            #obj_velos = np.linalg.norm(obj_velos, axis=1)
                            rel_velos = np.linalg.norm(rel_velos, axis=1)
                            relvel_mean = np.mean(rel_velos)
                            relvel_perc5, relvel_perc95 = np.percentile(rel_velos, (5, 95))

                            objpos = boxes[:, :2]
                            objpos = np.linalg.norm(objpos, axis=1)
                            objpos_mean = np.mean(objpos)
                            objpos_perc5, objpos_perc95 = np.percentile(objpos, (5, 95))
                        else:
                            objpos_perc5, objpos_mean, objpos_perc95 = 0., 0., 0.
                            relvel_perc5, relvel_mean, relvel_perc95 = 0., 0., 0.

                        #Override x_minmax
                        pred_exec_times, self.x_minmax = self.pred_all_res_times(points)
                        x_minmax_calculated = True

                        inp_tuple = np.array([[*pred_exec_times,
                                    objpos_perc5, objpos_mean, objpos_perc95,
                                    np.linalg.norm(ev), relvel_perc5, relvel_perc95, relvel_mean,
                        ]])
                        pred_res_idx = self.res_predictor.predict(inp_tuple)[0] # takes 5 ms
                    else:
                        pred_res_idx = self.res_idx # keep the same one
                elif not self.is_calibrating():
                    self.sched_time_point_ms = -2000 #enforce scheduling next time
                    pred_res_idx = 2 # random forest was trained with its input
            else:
                pred_res_idx = self.num_res - 1
            if not self.is_calibrating():
                self.res_idx = pred_res_idx

            if self.dense_conv_opt_on and not x_minmax_calculated:
                self.x_minmax = self.mpc_script.get_minmax_inds(points[:, 1])

            self._eval_dict['resolution_selections'][self.res_idx] += 1
            xmin, xmax = self.x_minmax[self.res_idx] # must do this!
            batch_dict['tensor_slice_inds'] = (xmin, xmax)

            resdiv = self.resolution_dividers[self.res_idx]
            batch_dict['resolution_divider'] = resdiv

            set_bn_resolution(self.res_aware_1d_batch_norms, self.res_idx)
            if self.fused_convs_trt[self.res_idx] is None:
                set_bn_resolution(self.res_aware_2d_batch_norms, self.res_idx)

            if self.traced_vfe[self.res_idx] is None:
                self.vfe.adjust_voxel_size_wrt_resolution(self.res_idx)
                self.traced_vfe[self.res_idx] = torch.jit.trace(self.vfe, points)

            self.measure_time_end('Sched')
            self.measure_time_start('VFE')
            points = batch_dict['points']
            batch_dict['pillar_coords'], batch_dict['pillar_features'] = \
                    self.traced_vfe[self.res_idx](points)

            if self.map_to_bev is not None:
                self.measure_time_start('MapToBEV')
                self.map_to_bev_scrpt.adjust_grid_size_wrt_resolution(self.res_idx)
                conv_inp = self.map_to_bev_scrpt(batch_dict['pillar_features'],
                        batch_dict['pillar_coords'],
                        batch_dict['batch_size'])
                self.measure_time_end('MapToBEV')
            self.measure_time_end('VFE')

            if self.backbone_3d is not None:
                self.measure_time_start('Backbone3D')
                if self.is_calibrating():
                    batch_dict['record_time'] = True # returns bb3d_layer_time_events
                batch_dict['record_int_vcounts'] = True # returns bb3d_num_voxels, no overhead
                batch_dict = self.backbone_3d.forward_up_to_dense(batch_dict)
                conv_inp = batch_dict['x_conv4_out']
                self.measure_time_end('Backbone3D')

            if not self.optimization_done[self.res_idx]:
                self.optimize(conv_inp)

            self.measure_time_start('DenseOps')
            forecasted_dets = self.do_forecast(batch_dict)
            if self.dense_conv_opt_on:
                lim1 = xmin*self.dense_inp_slice_sz
                lim2 = (xmax+1)*self.dense_inp_slice_sz
                conv_inp = conv_inp[..., lim1:lim2].contiguous()
            pred_dicts, topk_outputs = self.forward_eval_dense(conv_inp)

            self.dense_head_scrpt.adjust_voxel_size_wrt_resolution(self.res_idx)
            if not self.dense_conv_opt_on:
                topk_outputs = self.dense_head_scrpt.forward_topk(pred_dicts)
            self.measure_time_end('DenseOps')

            self.measure_time_start('CenterHead-GenBox')
            if self.dense_conv_opt_on:
                if self.dettype == 'PointPillarsCP':
                    lim1 //= 4
                for i, topk_out in enumerate(topk_outputs):
                    topk_out[-1] += lim1

            if forecasted_dets is not None and self.enable_forecasting:
                forecasted_dets = torch.jit.wait(forecasted_dets)
                if len(forecasted_dets) != self.dense_head.num_det_heads:
                    forecasted_dets = None # NOTE this appears to be a bug
            batch_dict['final_box_dicts'] = self.dense_head_scrpt.forward_genbox(
                    batch_dict['batch_size'], pred_dicts,
                    topk_outputs, forecasted_dets)

            self.measure_time_end('CenterHead-GenBox')

            return batch_dict

    def do_forecast(self, batch_dict):
        lbd, forecasted_dets = self.latest_batch_dict, None
        if self.enable_forecasting_to_fut:
            # Pickup the already forecasted data which happened at post processing
            forecasted_dets = self.sampled_dets[self.dataset_indexes[0]]
            if forecasted_dets is not None:
                forecasted_pd = forecasted_dets[0]
                # Deprioritize the forecasted for NMS
                scores = forecasted_pd['pred_scores']
                forecasted_pred_scores = torch.full(scores.shape,
                        self.score_thresh * 0.9, dtype=scores.dtype)
                # Split
                forecasted_dets = split_dets(
                        self.dense_head_scrpt.cls_id_to_det_head_idx_map,
                        self.dense_head_scrpt.num_det_heads,
                        forecasted_pd['pred_boxes'],
                        forecasted_pred_scores,
                        forecasted_pd['pred_labels'] - 1,
                        False) # moves results to gpu if true
        elif self.enable_forecasting and lbd is not None:
            # Fully on cpu
            last_pred_dict = lbd['final_box_dicts'][0]
            last_token = lbd['metadata'][0]['token']
            last_pose = self.token_to_pose[last_token]
            last_ts = self.token_to_ts[last_token] - self.scene_init_ts
            cur_token = batch_dict['metadata'][0]['token']
            cur_pose = self.token_to_pose[cur_token]
            cur_ts = self.token_to_ts[cur_token] - self.scene_init_ts

            # NOTE beware that that is returned here is a future object !
            forecasted_dets = self.forecaster.fork_forward_1tile(last_pred_dict,
                    last_pose, last_ts, cur_pose, cur_ts)

        return forecasted_dets

    # takes already sliced input
    def forward_eval_dense(self, conv_inp):
        if self.dense_conv_opt_on:
            if self.fused_convs_trt[self.res_idx] is not None:
                self.trt_outputs[self.res_idx] = self.fused_convs_trt[self.res_idx](
                        {'conv_inp': conv_inp}, self.trt_outputs[self.res_idx])
                pred_dicts, topk_outputs = self.convert_trt_outputs(self.trt_outputs[self.res_idx])
            else:
                outputs = self.opt_dense_convs(conv_inp)
                out_dict = {name:outp for name, outp in zip(self.opt_outp_names, outputs)}
                pred_dicts, topk_outputs = self.convert_trt_outputs(out_dict)
        else:
            if self.fused_convs_trt[self.res_idx] is not None:
                self.trt_outputs[self.res_idx] = self.fused_convs_trt[self.res_idx](
                        {'conv_inp': conv_inp}, self.trt_outputs[self.res_idx])
                outputs = [self.trt_outputs[self.res_idx][nm] for nm in self.opt_outp_names]
            else:
                outputs = self.opt_dense_convs(conv_inp)
            pred_dicts = self.dense_head.convert_out_to_pred_dicts(outputs)
            topk_outputs = None

        return pred_dicts, topk_outputs

    def optimize(self, fwd_data):
        optimize_start = time.time()

        self.inp_tensor_sizes[self.res_idx] = fwd_data.shape
        assert fwd_data.shape[-3] % self.dense_inp_slice_sz == 0

        if self.dense_head_scrpt is None:
            if self.dense_conv_opt_on:
                self.dense_head.instancenorm_mode()
            self.dense_head_scrpt = torch.jit.script(self.dense_head)

        # Not necessary but its ok
        self.dense_head.adjust_voxel_size_wrt_resolution(self.res_idx)

        if self.opt_dense_convs is None:
            self.opt_dense_convs = DenseConvsPipeline(self.backbone_3d,
                    self.backbone_2d, self.dense_head, self.dettype)
            self.opt_dense_convs.eval()

        input_names = ['conv_inp']
        print('Resolution idx:', self.res_idx, 'Input:', input_names[0], fwd_data.size())
        if self.dense_conv_opt_on:
            self.opt_dense_convs_output_names_pd = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                    for name in self.dense_head.ordered_outp_names(False)]
            self.topk_outp_names = ('scores', 'class_ids', 'xs', 'ys')
            self.opt_dense_convs_output_names_topk = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                    for name in self.topk_outp_names]
            self.opt_outp_names = self.opt_dense_convs_output_names_pd + self.opt_dense_convs_output_names_topk
        else:
            self.opt_outp_names = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                    for name in self.dense_head.ordered_outp_names()]

        # Create a onnx and tensorrt file for each resolution

        opt = ("_opt"  if self.dense_conv_opt_on else "")
        grid_sz = fwd_data.size(2)
        onnx_path = self.model_cfg.ONNX_PATH + opt + f'_H{grid_sz}.onnx'
        if not os.path.exists(onnx_path):
            dynamic_axes = {
                "conv_inp": {
                    3: "width",
                },
            } if self.dense_conv_opt_on else {}

            torch.onnx.export(
                    self.opt_dense_convs,
                    fwd_data.detach(),
                    onnx_path,
                    input_names=input_names,
                    output_names=self.opt_outp_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=17,
                    custom_opsets={"cuda_slicer": 17}
            )

        power_mode = os.getenv('PMODE', 'UNKNOWN_POWER_MODE')
        if power_mode == 'UNKNOWN_POWER_MODE':
            print('WARNING! Power mode is unknown. Please export PMODE.')

        fname = onnx_path.split('/')[-1].split('.')[0]
        trt_path = f'./deploy_files/trt_engines/{power_mode}/{fname}.engine'
        print('Trying to load trt engine at', trt_path)
        try:
            self.fused_convs_trt[self.res_idx] = TRTWrapper(trt_path, input_names, self.opt_outp_names)
        except:
            print('TensorRT wrapper for fused_convs throwed exception, building the engine')
            if self.dense_conv_opt_on:
                N, C, H, W = (int(s) for s in fwd_data.shape)
                # NOTE assumes the point cloud range is a square H == max W
                max_W = H
                min_shape = (N, C, H, self.dense_inp_slice_sz)
                opt_shape = (N, C, H, max_W  - self.dense_inp_slice_sz)
                max_shape = (N, C, H, max_W)
                create_trt_engine(onnx_path, trt_path, input_names[0], min_shape, opt_shape, max_shape)
            else:
                create_trt_engine(onnx_path, trt_path, input_names[0])
            self.fused_convs_trt[self.res_idx] = TRTWrapper(trt_path, input_names, self.opt_outp_names)

        optimize_end = time.time()
        print(f'Optimization took {optimize_end-optimize_start} seconds.')

        self.optimization_done[self.res_idx] = True

    def convert_trt_outputs(self, out_tensors):
        pred_dicts = [dict() for i in range(self.dense_head.num_det_heads)]
        topk_outputs = [[None] * 4 for i in range(self.dense_head.num_det_heads)]
        for k, v in out_tensors.items():
            idx = int(k[-1]) # assumes the number at the end is 1 digit
            name = k[:-1]
            if k in self.opt_dense_convs_output_names_pd:
                pred_dicts[idx][name] = v
            else:
                topk_outputs[idx][self.topk_outp_names.index(name)] = v

        return pred_dicts, topk_outputs

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
                'loss_rpn': loss_rpn.item(),
                **tb_dict
                }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing_pre(self, batch_dict):
        return (batch_dict,)

    def post_processing_post(self, pp_args):
        batch_dict = pp_args[0]
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                    box_preds=pred_boxes.cuda(),
                    recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST
                    )

        return final_pred_dict, recall_dict

    def calibrate(self, batch_size=1):
        if self.training:
            super().calibrate(1)
            self.res_idx = 0
            return None

        cur_res_idx = self.res_idx
        collect_calib_data = [False] * self.num_res
        calib_fnames = [""] * self.num_res
        for res_idx in range(self.num_res):
            self.res_idx = res_idx
            power_mode = os.getenv('PMODE', 'UNKNOWN_POWER_MODE')
            name = self.dettype + self.model_name + '_m' + str(self.method)
            calib_fnames[res_idx] = f"calib_files/{name}_{power_mode}_res{res_idx}.json"
            print('Trying to load calib file:', calib_fnames[res_idx])
            try:
                self.calibrators[res_idx].read_calib_data(calib_fnames[res_idx])
            except OSError:
                collect_calib_data[res_idx] = True

            self.calibration_on()
            print(f'Calibrating resolution {res_idx}')
            super().calibrate(1)

            if collect_calib_data[res_idx]:
                self.calibrators[res_idx].collect_data(calib_fnames[res_idx])
            self.calibration_off()
            self.sched_time_point_ms = 0
        self.clear_stats()
        self.res_idx = cur_res_idx

        self.e2e_min_times_ms = torch.tensor([c.get_e2e_min_ms() for c in self.calibrators])

        if any(collect_calib_data):
            sys.exit()
        return None

    def pred_all_res_times(self, points):
        points_xy = points[:, 1:3]
        num_points = points_xy.size(0)
        pc0, all_pillar_counts = self.mpc_script(points_xy, 0)
        if self.dense_conv_opt_on:
            x_minmax = torch.from_numpy(get_xminmax_from_pc0(pc0.cpu().numpy()))
        else:
            x_minmax = torch.empty((self.num_res, 2), dtype=torch.int)
            for i in range(self.num_res):
                x_minmax[i, 0] = 0
                x_minmax[i, 1] = self.mpc_script.num_slices[i] - 1

        latencies = [0.] * self.num_res
        all_pillar_counts = all_pillar_counts.cpu().numpy()
        for i in range(self.num_res):
            pillar_counts = all_pillar_counts[i]
            latencies[i] = self.calibrators[i].pred_exec_time_ms(num_points,
                    pillar_counts, (x_minmax[i,1] - x_minmax[i,0] + 1).item())

        return latencies, x_minmax
