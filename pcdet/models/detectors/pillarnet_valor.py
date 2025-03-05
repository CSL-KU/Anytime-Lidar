from .detector3d_template import Detector3DTemplate
from .valor_calibrator import ValorCalibrator
from ..model_utils.tensorrt_utils.trtwrapper import TRTWrapper, create_trt_engine
import torch
import time
import onnx
import pickle
import os
import sys
import numpy as np
import platform
from typing import Dict, List, Tuple, Optional, Final
from .forecaster import split_dets
from .valor_utils import *
from ...utils import common_utils

import ctypes
pth = os.environ['PCDET_PATH']
pth = os.path.join(pth, "pcdet/trt_plugins/slice_and_batch_nhwc/build/libslice_and_batch_lib.so")
ctypes.CDLL(pth, mode=ctypes.RTLD_GLOBAL)

VALO_OPT_ON = 6
VALO_OPT_OFF = 7 # no dense conv opt, no forecasting
VALO_OPT_WO_FORECASTING = 8 # yes dense conv opt, no forecasting
VALO_OPT_OFF_WCET_SCHED = 9 # no dense conv opt, no forecasting, wcet scheduling

class PillarNetVALOR(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        rd = model_cfg.get('RESOLUTION_DIV', [1.0])
        self.model_cfg.VFE.RESOLUTION_DIV = rd
        self.model_cfg.BACKBONE_3D.RESOLUTION_DIV = rd
        self.model_cfg.BACKBONE_2D.RESOLUTION_DIV = rd
        self.model_cfg.DENSE_HEAD.RESOLUTION_DIV = rd
        self.resolution_dividers = rd

        self.method = int(self.model_cfg.METHOD)
        self.valo_opt_on = (self.method == VALO_OPT_ON or \
                self.method == VALO_OPT_WO_FORECASTING)
        self.model_cfg.DENSE_HEAD.OPTIMIZE_ATTR_CONVS = self.valo_opt_on
        self.enable_forecasting_to_fut = (self.method == VALO_OPT_ON)

        self.deadline_based_selection = True
        if self.deadline_based_selection:
            print('Deadline scheduling is enabled!')

        self.enable_data_sched = False
        if self.enable_data_sched:
            print('Data scheduling is enabled!')

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
            'Backbone3D': [],
            'DenseOps':[],
            'CenterHead-GenBox': [],
        })

        self.vfe, self.backbone_3d, self.backbone_2d, self.dense_head = self.module_list
        print('Model size is:', self.get_model_size_MB(), 'MB')

        self.num_res = len(self.resolution_dividers)
        self.latest_losses = [0.] * self.num_res
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
        self.filter_pc_range = self.vfe.point_cloud_range + \
                torch.tensor([0.01, 0.01, 0.01, -0.01, -0.01, -0.01]).cuda()
        self.calib_pc_range = self.filter_pc_range.clone()

        #NOTE, this seems to work but I am not absolute
        t = torch.tensor(self.resolution_dividers) * self.vfe.voxel_size.cpu()[0]
        pillar_sizes = t.repeat_interleave(2).reshape(-1, 2)
        mpc = MultiPillarCounter(pillar_sizes, self.vfe.point_cloud_range.cpu())
        mpc.eval()
        self.mpc_script = torch.jit.script(mpc)
        self.shrink_flip = False

        self.dense_head_scrpt = None
        self.inp_tensor_sizes = [np.ones(4, dtype=int)] * self.num_res
        self.dense_inp_slice_sz = 4
        self.calibrators = [ValorCalibrator(self, ri, self.mpc_script.num_slices[ri]) \
                for ri in range(self.num_res)]

        self.res_predictor = None
        try:
            with open('random_forest_model.pkl', 'rb') as f:
                self.res_predictor = pickle.load(f)
                print('Loaded resolution predictor.')
        except:
            pass
        self.sched_step = 0
        self.sched_time_point_ms = 0

    def forward(self, batch_dict):
        if self.training:
            batch_dict['points'] = common_utils.pc_range_filter(batch_dict['points'],
                                self.vfe.point_cloud_range)
            resdiv = self.resolution_dividers[self.res_idx]
            batch_dict['resolution_divider'] = resdiv
            self.vfe.adjust_voxel_size_wrt_resolution(self.res_idx)
            set_bn_resolution(self.res_aware_1d_batch_norms, self.res_idx)
            set_bn_resolution(self.res_aware_2d_batch_norms, self.res_idx)

            points = batch_dict['points']
            batch_dict['pillar_coords'], batch_dict['pillar_features'] = self.vfe(points)
            batch_dict = self.backbone_3d(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)

            self.dense_head.adjust_voxel_size_wrt_resolution(resdiv)
            batch_dict = self.dense_head.forward_pre(batch_dict)
            batch_dict = self.dense_head.forward_post(batch_dict)

            batch_dict = self.dense_head.forward_assign_targets(batch_dict)

            loss, tb_dict, disp_dict = self.get_training_loss()

            self.latest_losses[self.res_idx] = loss.item()
            self.res_idx = (self.res_idx +1) % self.num_res

            ret_dict = {
               'loss': loss
            }

            disp_dict.update({f"loss_resdiv{self.resolution_dividers[i]}": l \
                    for i, l in enumerate(self.latest_losses)})

            return ret_dict, tb_dict, disp_dict
        else:
            return self.forward_eval(batch_dict)

    def forward_eval(self, batch_dict):
        with torch.cuda.stream(self.inf_stream):
            # The time before this is measured as preprocess
            self.measure_time_start('Sched')
            points = batch_dict['points']
            points = common_utils.pc_range_filter(points,
                                self.calib_pc_range if self.is_calibrating() else
                                self.filter_pc_range)
            batch_dict['points'] = points
            fixed_res_idx = int(os.environ.get('FIXED_RES_IDX', -1))

            # Schedule by calculating the exec time of all resolutions
            conf_found, shrink = False, False
            x_minmax = torch.empty((self.num_res, 2), dtype=torch.int)
            for i in range(self.num_res):
                x_minmax[i, 0] = 0
                x_minmax[i, 1] = self.mpc_script.num_slices[i] - 1

            if fixed_res_idx > -1:
                if not self.is_calibrating():
                    self.res_idx = fixed_res_idx
                if self.valo_opt_on:
                    points_xy_s = batch_dict['points'][:, 1:3] - self.mpc_script.pc_range_min
                    pillar_counts = self.mpc_script.forward_one_res(points_xy_s, self.res_idx)
                    nz_slice_inds = pillar_counts[0].cpu().nonzero()
                    xmin, xmax = nz_slice_inds[0, 0], nz_slice_inds[-1, 0]
                    x_minmax[self.res_idx, 0] = xmin
                    x_minmax[self.res_idx, 1] = xmax
            elif self.deadline_based_selection:
                abs_dl_sec = batch_dict['abs_deadline_sec']
                if self.method == VALO_OPT_OFF_WCET_SCHED:
                    t = time.time()
                    time_passed = (t - batch_dict['start_time_sec']) * 1000
                    time_left = (abs_dl_sec - time.time()) * 1000
                    for i in range(self.num_res):
                        pred_latency = self.calibrators[i].get_e2e_wcet_ms() - time_passed
                        if not self.is_calibrating() and pred_latency < time_left:
                            self.res_idx = i
                            conf_found = True
                            break
                else:
                    points = batch_dict['points']
                    points_xy = points[:, 1:3].contiguous()
                    num_points = points_xy.size(0)
                    #start_time = batch_dict['start_time_sec']
                    #deadline_ms = batch_dict['deadline_sec'] * 1e3
                    # This is needed in case we did not start when input arrived
                    #deadline_ms -= (int(self.sim_cur_time_ms) % self.data_period_ms)
                    all_pillar_counts = self.mpc_script(points_xy).int().cpu()
                    all_pillar_counts = self.mpc_script.split_pillar_counts(all_pillar_counts)
                    for i in range(self.num_res):
                        pillar_counts = all_pillar_counts[i]
                        if self.valo_opt_on:
                            nz_slice_inds = pillar_counts[0].nonzero()
                            #time_passed_ms = (time.time() - start_time) * 1e3
                            #time_left = deadline_ms - time_passed_ms
                            xmin, xmax = nz_slice_inds[0, 0], nz_slice_inds[-1, 0]
                            x_minmax[i, 0] = xmin
                            x_minmax[i, 1] = xmax
                        else:
                            xmin, xmax = x_minmax[i, 0], x_minmax[i, 1]
                        time_left = (abs_dl_sec - time.time()) * 1000

                        if self.enable_data_sched:
                            pred_latency, new_xmin, new_xmax = self.calibrators[i]. \
                                    find_config_to_meet_dl(num_points,
                                    pillar_counts.numpy(),
                                    xmin.item(),
                                    xmax.item(),
                                    time_left,
                                    self.shrink_flip)
                                    # set a shrinking limit

                            if not self.is_calibrating() and pred_latency < time_left:
                                self.res_idx = i
                                conf_found = True
                                shrink = (new_xmin > xmin) or (new_xmax < xmax)
                                xmin, xmax = new_xmin, new_xmax
                                x_minmax[i, 0] = xmin
                                x_minmax[i, 1] = xmax
                                break
                        else:
                            pred_latency = self.calibrators[i].pred_exec_time_ms(num_points,
                                    pillar_counts.numpy(), (xmax - xmin + 1).item())
                            if not self.is_calibrating() and pred_latency < time_left:
                                self.res_idx = i
                                conf_found = True
                                break

                if not self.is_calibrating() and not conf_found:
                    self.res_idx = self.num_res - 1
            elif self.res_predictor is not None:
                assert batch_dict['scene_reset'] == (self.latest_batch_dict is None) # just to make sure
                if self.latest_batch_dict is not None:
                    # If you want to sched periodically rel to scene start, uncomment
                    #tdiff = int(self.sim_cur_time_ms - self.sched_time_point_ms)
                    #if tdiff >= 5000:
                    #    self.sched_step += 1
                    #    self.sched_time_point_ms = self.sim_cur_time_ms

                    if self.sched_step > 0:
                        self.sched_step -= 1
                        pd = self.latest_batch_dict['final_box_dicts'][0]

                        # Random forest based prediction
                        ev = self.sampled_egovels[self.dataset_indexes[0]] # current one

                        obj_velos = pd['pred_boxes'][:, 7:9].numpy()
                        mask = np.logical_not(np.isnan(obj_velos).any(1))
                        obj_velos = obj_velos[mask]
                        rel_velos = obj_velos - ev
                        obj_velos = np.linalg.norm(obj_velos, axis=1)
                        rel_velos = np.linalg.norm(rel_velos, axis=1)

                        NUM_BINS=10
                        MAX_CAR_VEL = 15
                        MAX_REL_VEL = 2*MAX_CAR_VEL
                        objvel_dist = np.bincount((obj_velos/MAX_CAR_VEL*NUM_BINS).astype(int),
                                                  minlength=NUM_BINS)[:NUM_BINS]
                        relvel_dist = np.bincount((rel_velos/MAX_REL_VEL*NUM_BINS).astype(int),
                                                  minlength=NUM_BINS)[:NUM_BINS]

                        exec_time_ms = self.last_elapsed_time_musec / 1000
                        inp_tuple = np.concatenate((objvel_dist, relvel_dist, [exec_time_ms]))
                        inp_tuple = np.expand_dims(inp_tuple, 0)
                        if not self.is_calibrating():
                            self.res_idx = self.res_predictor.predict(inp_tuple)[0] # takes 5 ms
                elif not self.is_calibrating():
                    self.res_idx = 2 # middle, since random forest was trained with its input

            self.sched_step += batch_dict['scene_reset']
            if self.sched_step > 0:
                self.sched_time_point_ms = self.sim_cur_time_ms

            self._eval_dict['resolution_selections'][self.res_idx] += 1
            xmin, xmax = x_minmax[self.res_idx] # must do this!

            resdiv = self.resolution_dividers[self.res_idx]
            batch_dict['resolution_divider'] = resdiv

            self.vfe.adjust_voxel_size_wrt_resolution(self.res_idx)
            set_bn_resolution(self.res_aware_1d_batch_norms, self.res_idx)
            if self.fused_convs_trt[self.res_idx] is None:
                set_bn_resolution(self.res_aware_2d_batch_norms, self.res_idx)

            if shrink:
                self.shrink_flip = not self.shrink_flip
                pc_filter_lims = self.mpc_script.slice_inds_to_point_cloud_range(
                        self.res_idx, xmin, xmax)
                points_x = points_xy[:, 0]
                mask = (points_x >= pc_filter_lims[0]) & (points_x <= pc_filter_lims[1])
                points = points[mask]
                batch_dict['points'] = points

            self.measure_time_end('Sched')
            self.measure_time_start('VFE')
            points = batch_dict['points']
            batch_dict['pillar_coords'], batch_dict['pillar_features'] = self.vfe(points)
            self.measure_time_end('VFE')

            self.measure_time_start('Backbone3D')
            if self.is_calibrating():
                batch_dict['record_time'] = True # returns bb3d_layer_time_events
            batch_dict['record_int_vcounts'] = True # returns bb3d_num_voxels, no overhead
            batch_dict = self.backbone_3d.forward_up_to_dense(batch_dict)
            x_conv4 = batch_dict['x_conv4_out']
            self.measure_time_end('Backbone3D')

            if not self.optimization_done[self.res_idx]:
                self.optimize(x_conv4)

            self.measure_time_start('DenseOps')
            if fixed_res_idx == -1 and self.valo_opt_on:
                lim1 = xmin*self.dense_inp_slice_sz
                lim2 = (xmax+1)*self.dense_inp_slice_sz
                x_conv4 = x_conv4[..., lim1:lim2].contiguous()
            batch_dict['tensor_slice_inds'] = (xmin, xmax)
            pred_dicts, topk_outputs = self.forward_eval_dense(x_conv4)

            self.dense_head_scrpt.adjust_voxel_size_wrt_resolution(resdiv)
            if not self.valo_opt_on:
                topk_outputs = self.dense_head_scrpt.forward_topk(pred_dicts)
            self.measure_time_end('DenseOps')

            self.measure_time_start('CenterHead-GenBox')
            if fixed_res_idx == -1 and self.valo_opt_on:
                for i, topk_out in enumerate(topk_outputs):
                    topk_out[-1] += lim1 # NOTE assume the tensor resolution is same

            forecasted_dets = None
            if self.enable_forecasting_to_fut:
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

            batch_dict['final_box_dicts'] = self.dense_head_scrpt.forward_genbox(
                    batch_dict['batch_size'], pred_dicts,
                    topk_outputs, forecasted_dets)
            self.measure_time_end('CenterHead-GenBox')

            return batch_dict

    # takes already sliced input
    def forward_eval_dense(self, x_conv4):
        if self.valo_opt_on:
            if self.fused_convs_trt[self.res_idx] is not None:
                self.trt_outputs[self.res_idx] = self.fused_convs_trt[self.res_idx](
                        {'x_conv4': x_conv4}, self.trt_outputs[self.res_idx])
                pred_dicts, topk_outputs = self.convert_trt_outputs(self.trt_outputs[self.res_idx])
            else:
                outputs = self.opt_dense_convs(x_conv4)
                out_dict = {name:outp for name, outp in zip(self.opt_outp_names, outputs)}
                pred_dicts, topk_outputs = self.convert_trt_outputs(out_dict)
        else:
            if self.fused_convs_trt[self.res_idx] is not None:
                self.trt_outputs[self.res_idx] = self.fused_convs_trt[self.res_idx](
                        {'x_conv4': x_conv4}, self.trt_outputs[self.res_idx])
                outputs = [self.trt_outputs[self.res_idx][nm] for nm in self.opt_outp_names]
            else:
                outputs = self.opt_dense_convs(x_conv4)
            pred_dicts = self.dense_head.convert_out_to_pred_dicts(outputs)
            topk_outputs = None

        return pred_dicts, topk_outputs

    def optimize(self, fwd_data):
        optimize_start = time.time()

        self.inp_tensor_sizes[self.res_idx] = fwd_data.shape
        assert fwd_data.shape[-3] % self.dense_inp_slice_sz == 0

        if self.dense_head_scrpt is None:
            if self.valo_opt_on:
                self.dense_head.instancenorm_mode()
            self.dense_head_scrpt = torch.jit.script(self.dense_head)

        # Not necessary but its ok
        self.dense_head.adjust_voxel_size_wrt_resolution(self.resolution_dividers[self.res_idx]) 

        if self.opt_dense_convs is None:
            self.opt_dense_convs = DenseConvsPipeline(self.backbone_3d, self.backbone_2d, self.dense_head)
            self.opt_dense_convs.eval()

        input_names = ['x_conv4']
        print('Resolution idx:', self.res_idx, 'Input:', input_names[0], fwd_data.size())
        if self.valo_opt_on:
            self.opt_dense_convs_output_names_pd = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                    for name in self.dense_head.ordered_outp_names(False)]
            self.topk_outp_names = ('scores', 'class_ids', 'xs', 'ys')
            self.opt_dense_convs_output_names_topk = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                    for name in self.topk_outp_names]
            self.opt_outp_names = self.opt_dense_convs_output_names_pd + self.opt_dense_convs_output_names_topk
        else:
            self.opt_outp_names = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                    for name in self.dense_head.ordered_outp_names()]


        #print('Fused operations output names:', self.opt_outp_names)

        # Create a onnx and tensorrt file for each resolution
        onnx_path = self.model_cfg.ONNX_PATH + '_m' + str(self.method) + f'_res{self.res_idx}.onnx'
        if not os.path.exists(onnx_path):
            dynamic_axes = {
                "x_conv4": {
                    3: "width",
                },
            } if self.valo_opt_on else {}

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
            if self.valo_opt_on:
                N, C, H, W = (int(s) for s in fwd_data.shape)
                # NOTE assumes the point cloud range is a square H == max W
                max_W = H
                min_shape = (N, C, H, 16)
                opt_shape = (N, C, H, max_W  - 16)
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
            name = self.model_name + '_m' + str(self.method)
            calib_fnames[res_idx] = f"calib_files/{name}_{power_mode}_res{res_idx}.json"
            try:
                self.calibrators[res_idx].read_calib_data(calib_fnames[res_idx])
            except OSError:
                collect_calib_data[res_idx] = True

            self.calibration_on()
            print(f'Calibrating resolution {res_idx}')
            super().calibrate(1)

            if collect_calib_data[res_idx]:
                self.calibrators[res_idx].collect_data(calib_fnames[res_idx])
                # After this, the calibration data should be processed with dynamic deadline
            self.calibration_off()
            self.sched_step = 0
            self.sched_time_point_ms = 0
        self.clear_stats()
        self.res_idx = cur_res_idx
        #self.res_idx = 4 # DONT SET THIS WHEN USING THE NOTEBOOK TO COLLECT DATA
        return None
