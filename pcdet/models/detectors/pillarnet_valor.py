from .detector3d_template import Detector3DTemplate
#from ..dense_heads.center_head_inf import scatter_sliced_tensors
#from ..backbones_2d.base_bev_backbone_sliced import prune_spatial_features
from .valor_calibrator import ValorCalibrator
from ..model_utils.tensorrt_utils.trtwrapper import TRTWrapper, create_trt_engine
#from .forecaster import Forecaster
import torch
import time
import onnx
import os
import sys
import numpy as np
import platform
from pcdet.ops.norm_funcs.res_aware_bnorm import ResAwareBatchNorm1d, ResAwareBatchNorm2d
from typing import List, Tuple

import ctypes
pth = os.environ['PCDET_PATH']
pth = os.path.join(pth, "pcdet/trt_plugins/slice_and_batch_nhwc/build/libslice_and_batch_lib.so")
ctypes.CDLL(pth, mode=ctypes.RTLD_GLOBAL)

def set_bn_resolution(resawarebns, res_idx):
    for rabn in resawarebns:
        rabn.setResIndex(res_idx)

def get_all_resawarebn(model):
    resawarebns = []
    for module in model.modules():
        if isinstance(module, ResAwareBatchNorm1d) or isinstance(module, ResAwareBatchNorm2d):
            resawarebns.append(module)
    return resawarebns


@torch.jit.script
def get_slice_range(down_scale_factor : int, x_min: int, x_max: int, maxsz: int) \
        -> Tuple[int, int]:
    dsf = down_scale_factor
    x_min, x_max = x_min // dsf, x_max // dsf + 1
    denom = 4 # denom is dependent on strides within the dense covs
    minsz = 16

    rng = (x_max - x_min)
    if rng < minsz:
        diff = minsz - rng
        if x_max + diff <= maxsz:
            x_max += diff
        elif x_min - diff >= 0:
            x_min -= diff
        #else: # very unlikely
        #    pass
        rng = (x_max - x_min)

    pad = denom - (rng % denom)
    if pad == denom:
        pass
    elif x_min >= pad: # pad from left
        x_min -= pad
    elif (maxsz - x_max) >= pad: # pad from right
        x_max += pad
    else: # don't slice
        x_min, x_max = 0 , maxsz
    return x_min, x_max

@torch.jit.script
def slice_tensor(down_scale_factor : int, x_min: int, x_max: int, inp : torch.Tensor) \
        -> Tuple[torch.Tensor, int, int]:
    x_min, x_max = get_slice_range(down_scale_factor, x_min, x_max, inp.size(3))
    return inp[..., x_min:x_max].contiguous(), x_min, x_max

# This will be used to generate the onnx
class DenseConvsPipeline(torch.nn.Module):
    def __init__(self, backbone_3d, backbone_2d, dense_head):
        super().__init__()
        self.backbone_3d = backbone_3d
        self.backbone_2d = backbone_2d
        self.dense_head = dense_head

    def forward(self, x_conv4 : torch.Tensor) -> List[torch.Tensor]:
        x_conv5 = self.backbone_3d.forward_dense(x_conv4)
        data_dict = self.backbone_2d({"multi_scale_2d_features" : 
            {"x_conv4": x_conv4, "x_conv5": x_conv5}})

        outputs = self.dense_head.forward_pre(data_dict['spatial_features_2d'])
        shr_conv_outp = outputs[0]
        heatmaps = outputs[1:]

        topk_outputs = self.dense_head.forward_topk_trt(heatmaps)

        ys_all = [topk_outp[2] for topk_outp in topk_outputs]
        xs_all = [topk_outp[3] for topk_outp in topk_outputs]

        sliced_inp = self.dense_head.slice_shr_conv_outp(shr_conv_outp, ys_all, xs_all)
        outputs = self.dense_head.forward_sliced_inp_trt(sliced_inp)
        for topk_output in topk_outputs:
            outputs += topk_output

        return outputs

class MultiPillarCounter(torch.nn.Module):
    # Pass the args in cpu , pillar sizes should be Nx2, pc_range should be [6]
    def __init__(self, pillar_sizes : torch.Tensor, pc_range : torch.Tensor):
        super().__init__()

        xy_length = pc_range[[3,4]] - pc_range[[0,1]]
        self.grid_sizes = torch.empty((pillar_sizes.size(0), 2), dtype=torch.int) # cpu
        for i, ps in enumerate(pillar_sizes):
            self.grid_sizes[i] = torch.round(xy_length / ps)

        self.pillar_sizes = pillar_sizes.cuda()
        self.pc_range_min = pc_range[[0,1]].cuda()

    def forward(self, points_xy : torch.Tensor) -> torch.Tensor:
        # store minx max pillar_count
        mpc_out = torch.empty((3, self.pillar_sizes.size(0)), device=points_xy.device, dtype=torch.int)
        for i, (ps, grid_sz) in enumerate(zip(self.pillar_sizes, self.grid_sizes)):
            point_coords = torch.floor((points_xy - self.pc_range_min) / ps).int()
            grid = torch.zeros((grid_sz[0], grid_sz[1]), device=points_xy.device, dtype=torch.int)
            grid[point_coords[:, 0], point_coords[:, 1]] = 1
            mpc_out[0, i] = grid.sum()

            xmin, xmax = torch.aminmax(point_coords[:, 0])
            mpc_out[1, i] = xmin
            mpc_out[2, i] = xmax

        return mpc_out

class PillarNetVALOR(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        torch.backends.cudnn.benchmark = True
        if torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark_limit = 0

        is_x86 = (platform.machine() in ['x86_64', 'AMD64', 'x86'])

        torch.backends.cuda.matmul.allow_tf32 = is_x86
        torch.backends.cudnn.allow_tf32 = is_x86
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

        self.resolution_dividers = self.model_cfg.BACKBONE_3D.get('RESOLUTION_DIV', [1.0])
        self.num_res = len(self.resolution_dividers)
        self.latest_losses = [0.] * self.num_res
        self.res_aware_batch_norms = get_all_resawarebn(self)
        self.res_idx = 0

        self.inf_stream = torch.cuda.Stream()
        self.optimization_done = [False] * self.num_res
        self.trt_outputs = [None] * self.num_res # Since output size of trt is fixed, use buffered
        self.fused_convs_trt = [None] * self.num_res

        fms = self.model_cfg.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
        self.target_hw = [sz // fms for sz in self.dataset.grid_size[:2]]
        self.opt_dense_convs = None

        #NOTE later, integrate periodic forecasting
        # Force forecasting to be disabled
        #self.keep_forecasting_disabled = False
        #if not self.keep_forecasting_disabled:
        #    self.forecaster = torch.jit.script(Forecaster(tuple(self.dataset.point_cloud_range.tolist()), 
        #            self.tcount, self.score_thresh, self.forecasting_coeff, self.dense_head.num_det_heads,
        #            self.dense_head.cls_id_to_det_head_idx_map))

        self.dense_head_scrpt = None
        self.calibrators = [None] * self.num_res
        self.calib_pc_range = torch.tensor(self.dataset.point_cloud_range).cuda()
        self.mpc_optimized = False
        self.mpc_trt = None
        self.mpc_outp = None

    def forward(self, batch_dict):
        if self.training:
            batch_dict = self.vfe.range_filter(batch_dict)
            resdiv = self.resolution_dividers[self.res_idx]
            batch_dict['resolution_divider'] = resdiv
            self.vfe.adjust_voxel_size_wrt_resolution(self.res_idx)
            set_bn_resolution(self.res_aware_batch_norms, self.res_idx)

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
            batch_dict = self.vfe.range_filter(batch_dict, self.calib_pc_range \
                    if self.is_calibrating() else None)

            points_xy = batch_dict['points'][:, 1:3].contiguous()
            if not self.mpc_optimized:
                self.optimize_mpc(points_xy)
            if self.mpc_trt is None:
                mpc_out = self.mpc(points_xy).cpu()
            else:
                self.mpc_outp = self.mpc_trt({'points_xy':points_xy}, self.mpc_outp)
                mpc_out = self.mpc_outp['counts'].cpu()

            resdiv = self.resolution_dividers[self.res_idx]
            batch_dict['resolution_divider'] = resdiv

            self.vfe.adjust_voxel_size_wrt_resolution(self.res_idx)
            set_bn_resolution(self.res_aware_batch_norms, self.res_idx)
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
            x_minmax = mpc_out[1:, self.res_idx]
            x_conv4, x_min, x_max= slice_tensor(self.backbone_3d.sparse_outp_downscale_factor(),
                    x_minmax[0], x_minmax[1], x_conv4)
            batch_dict['tensor_slice_inds'] = (x_min, x_max)

            if self.fused_convs_trt[self.res_idx] is not None:
                self.trt_outputs[self.res_idx] = self.fused_convs_trt[self.res_idx](
                        {'x_conv4': x_conv4}, self.trt_outputs[self.res_idx])
                pred_dicts, topk_outputs = self.convert_trt_outputs(self.trt_outputs[self.res_idx])
            else:
                outputs = self.opt_dense_convs(x_conv4)
                out_dict = {name:outp for name, outp in zip(self.opt_outp_names, outputs)}
                pred_dicts, topk_outputs = self.convert_trt_outputs(out_dict)
            self.measure_time_end('DenseOps')

            self.measure_time_start('CenterHead-GenBox')

            for i, topk_out in enumerate(topk_outputs):
                topk_out[-1] += x_min  # NOTE assume the tensor resolution is same

            forecasted_dets = None
            self.dense_head_scrpt.adjust_voxel_size_wrt_resolution(resdiv)
            batch_dict['final_box_dicts'] = self.dense_head_scrpt.forward_genbox(
                    batch_dict['batch_size'], pred_dicts,
                    topk_outputs, forecasted_dets)

            self.measure_time_end('CenterHead-GenBox')

            return batch_dict

    def optimize(self, fwd_data):
        optimize_start = time.time()

        if self.dense_head_scrpt is None:
            self.dense_head.instancenorm_mode()
            self.dense_head_scrpt = torch.jit.script(self.dense_head)

        # Not necessary but its ok
        self.dense_head.adjust_voxel_size_wrt_resolution(self.resolution_dividers[self.res_idx]) 

        if self.opt_dense_convs is None:
            self.opt_dense_convs = DenseConvsPipeline(self.backbone_3d, self.backbone_2d, self.dense_head)
            self.opt_dense_convs.eval()

        input_names = ['x_conv4']
        print('Resolution idx:', self.res_idx, 'Input:', input_names[0], fwd_data.size())
        self.opt_dense_convs_output_names_pd = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                for name in self.dense_head.ordered_outp_names(False)]

        self.topk_outp_names = ('scores', 'class_ids', 'xs', 'ys')
        self.opt_dense_convs_output_names_topk = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                for name in self.topk_outp_names]

        self.opt_outp_names = self.opt_dense_convs_output_names_pd + self.opt_dense_convs_output_names_topk
        #print('Fused operations output names:', self.opt_outp_names)

        # Create a onnx and tensorrt file for each resolution
        onnx_path = self.model_cfg.ONNX_PATH + f'_res{self.res_idx}.onnx'
        if not os.path.exists(onnx_path):
            dynamic_axes = {
                "x_conv4": {
                    3: "width",
                },
            }

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
            N, C, H, W = (int(s) for s in fwd_data.shape)
            # NOTE assumes the point cloud range is a square H == max W
            max_W = H
            min_shape = (N, C, H, 16)
            opt_shape = (N, C, H, max_W  - 16)
            max_shape = (N, C, H, max_W)
            create_trt_engine(onnx_path, trt_path, input_names[0], min_shape, opt_shape, max_shape)
            self.fused_convs_trt[self.res_idx] = TRTWrapper(trt_path, input_names, self.opt_outp_names)

        optimize_end = time.time()
        print(f'Optimization took {optimize_end-optimize_start} seconds.')

        self.optimization_done[self.res_idx] = True

    def optimize_mpc(self, points_xy):
        #TODO, set pillar sizes programmatically
        pillar_sizes = torch.tensor([[0.1, 0.1], [0.15, 0.15], [0.2, 0.2], [0.24, 0.24], [0.30, 0.30]])
        self.mpc = MultiPillarCounter(pillar_sizes, self.calib_pc_range.cpu())
        self.mpc.eval()

        # Create a onnx and tensorrt file for each resolution
        onnx_path = 'deploy_files/onnx_files/mpc.onnx'
        input_names = ['points_xy']
        outp_names = ['counts']
        if not os.path.exists(onnx_path):
            torch.onnx.export(
                self.mpc,
                points_xy.detach(),
                onnx_path,
                input_names=input_names,
                output_names=outp_names,
                dynamic_axes={"points_xy": {0: 'num_points'}},
                opset_version=17,
            )

        power_mode = os.getenv('PMODE', 'UNKNOWN_POWER_MODE')
        if power_mode == 'UNKNOWN_POWER_MODE':
            print('WARNING! Power mode is unknown. Please export PMODE.')

        trt_path = f'./deploy_files/trt_engines/{power_mode}/mpc.engine'
        print('Trying to load trt engine at', trt_path)
        try:
            self.mpc_trt = TRTWrapper(trt_path, input_names, outp_names)
        except:
            print('TensorRT wrapper for mpc throwed exception, building the engine')
            min_shape, opt_shape, max_shape = (10000,2), (250000,2), (350000,2) # num points
            create_trt_engine(onnx_path, trt_path, input_names[0], min_shape, opt_shape, max_shape)
            self.mpc_trt = TRTWrapper(trt_path, input_names, outp_names)

        #self.mpc_trt = None
        self.mpc_optimized = True

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
            self.calibrators[res_idx] = ValorCalibrator(self)
            power_mode = os.getenv('PMODE', 'UNKNOWN_POWER_MODE')
            calib_fnames[res_idx] = f"calib_files/{self.model_name}_{power_mode}_res{res_idx}.json"
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

        self.res_idx = cur_res_idx
        #self.res_idx = 4 # DONT SET THIS WHEN USING THE NOTEBOOK TO COLLECT DATA
        return None
