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
from pcdet.models.backbones_3d.spconv_backbone_2d import PillarRes18BackBone8x_pillar_calc
from typing import Dict, List, Tuple, Optional, Final
from .forecaster import split_dets

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
    # Pass the args in cpu , pillar sizes should be [N,2], pc_range should be [6]
    grid_sizes: Final[List[List[int]]]
    num_slices: Final[List[int]]
    pillar_sizes : torch.Tensor
    pc_range_min: torch.Tensor
    pc_range_cpu: torch.Tensor

    def __init__(self, pillar_sizes : torch.Tensor, pc_range : torch.Tensor,
                 slice_sz: int = 32):
        super().__init__()
        xy_length = pc_range[[3,4]] - pc_range[[0,1]]
        grid_sizes = torch.empty((pillar_sizes.size(0), 2), dtype=torch.int) # cpu
        self.num_slices = [0] * pillar_sizes.size(0)
        for i, ps in enumerate(pillar_sizes):
            grid_sizes[i] = torch.round(xy_length / ps)
            self.num_slices[i] = (grid_sizes[i, 0] // slice_sz).item()
        self.grid_sizes = grid_sizes.tolist()

        self.pillar_sizes = pillar_sizes.cuda()
        self.pc_range_cpu = pc_range
        self.pc_range_min = pc_range[[0,1]].cuda()

        print('num_slices', self.num_slices)
        print('grid_sizes', self.grid_sizes)
        print('pillar_sizes', self.pillar_sizes)

    def forward_one_res(self, points_xy : torch.Tensor, res_idx : int) -> Tuple[torch.Tensor, torch.Tensor]:
        ps = self.pillar_sizes[res_idx]
        grid_sz = self.grid_sizes[res_idx]
        grid = torch.zeros([1, 1, grid_sz[0], grid_sz[1]], device=points_xy.device)
        point_coords = torch.floor((points_xy - self.pc_range_min) / ps).long()
        grid[:, :, point_coords[:, 0], point_coords[:, 1]] = 1.
        pillar_counts = PillarRes18BackBone8x_pillar_calc(grid, self.num_slices[res_idx]).cpu()
        nz_slice_inds = pillar_counts[0].nonzero()

        #return the nonzero slice inds
        return pillar_counts.int(), nz_slice_inds

    def forward(self, points_xy : torch.Tensor) -> torch.Tensor:
        all_pillar_counts = []
        for res_idx, grid_sz in enumerate(self.grid_sizes):
            grid = torch.zeros([1, 1, grid_sz[0], grid_sz[1]], device=points_xy.device)
            point_coords = torch.floor((points_xy - self.pc_range_min) / self.pillar_sizes[res_idx]).long()
            grid[:, :, point_coords[:, 0], point_coords[:, 1]] = 1.
            pillar_counts = PillarRes18BackBone8x_pillar_calc(grid, self.num_slices[res_idx])
            all_pillar_counts.append(pillar_counts)
        all_pillar_counts = torch.cat(all_pillar_counts, dim=1)
        return all_pillar_counts # later make it int

    @torch.jit.export
    def split_pillar_counts(self, all_pillar_counts : torch.Tensor) -> List[torch.Tensor]:
        chunks, bgn = [], 0
        for num_slice in self.num_slices:
            chunks.append(all_pillar_counts[:, bgn:bgn+num_slice])
            bgn+=num_slice
        return chunks

    @torch.jit.export
    def slice_inds_to_point_cloud_range(self, res_idx : int, minx : torch.Tensor, maxx : torch.Tensor):
        ns = self.num_slices[res_idx]
        minx_perc = minx / ns
        maxx_perc = (maxx+1) / ns
        rng = (self.pc_range_cpu[3] - self.pc_range_cpu[0])
        minmax = torch.empty(2)
        minmax[0] = (minx_perc * rng) + self.pc_range_cpu[0]
        minmax[1] = (maxx_perc * rng) + self.pc_range_cpu[0]
        return minmax.cuda()

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

        self.calib_pc_range = torch.tensor(self.dataset.point_cloud_range).cuda()

        #NOTE, this seems to work but I am not absolute
        t = torch.tensor(self.resolution_dividers) * self.vfe.voxel_size.cpu()[0]
        pillar_sizes = t.repeat_interleave(2).reshape(-1, 2)
        self.mpc = MultiPillarCounter(pillar_sizes, self.calib_pc_range.cpu())
        self.mpc.eval()
        #self.mpc_script = torch.jit.script(mpc)
        self.shrink_flip = False

        self.dense_head_scrpt = None
        self.mpc_optimized = False
        self.mpc_trt = None
        self.mpc_outp = None
        self.inp_tensor_sizes = [np.ones(4, dtype=int)] * self.num_res
        self.dense_inp_slice_sz = 4
        self.calibrators = [ValorCalibrator(self, ri, self.mpc.num_slices[ri]) \
                for ri in range(self.num_res)]

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

            fixed_res_idx = int(os.environ.get('FIXED_RES_IDX', -1))
            if self.is_calibrating():
                fixed_res_idx = -1 # enforce to calculate wcet

            # Schedule by calculating the exec time of all resolutions
            conf_found, shrink = False, False
            x_minmax = torch.empty((self.num_res, 2), dtype=torch.int)
            if fixed_res_idx > -1:
                self.res_idx = fixed_res_idx
                xmin, xmax = 0, (self.mpc.num_slices[self.res_idx] - 1).item()
            else:
                points = batch_dict['points']
                points_xy = points[:, 1:3].contiguous()
                num_points = points_xy.size(0)
                start_time = batch_dict['start_time_sec']
                deadline_ms = batch_dict['deadline_sec'] * 1e3
                if not self.mpc_optimized:
                    self.optimize_mpc(points_xy)
                if self.mpc_trt is not None:
                    self.mpc_outp = self.mpc_trt({'points_xy': points_xy}, self.mpc_outp)
                    all_pillar_counts = self.mpc_outp['counts'].int().cpu()
                else:
                    all_pillar_counts = self.mpc(points_xy).int().cpu()
                all_pillar_counts = self.mpc.split_pillar_counts(all_pillar_counts)
                for i in range(self.num_res): # - 1, -1, -1):
                    pillar_counts = all_pillar_counts[i]
                    nz_slice_inds = pillar_counts[0].nonzero()
                    time_passed_ms = (time.time() - start_time) * 1e3
                    time_left = deadline_ms - time_passed_ms
                    xmin, xmax = nz_slice_inds[0, 0], nz_slice_inds[-1, 0]
                    x_minmax[i, 0] = xmin
                    x_minmax[i, 1] = xmax

                    pred_latency, new_xmin, new_xmax = self.calibrators[i].find_config_to_meet_dl(num_points,
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

                if not self.is_calibrating() and not conf_found:
                    self.res_idx = self.num_res - 1

            xmin, xmax = x_minmax[self.res_idx] # must do this!

            resdiv = self.resolution_dividers[self.res_idx]
            batch_dict['resolution_divider'] = resdiv

            self.vfe.adjust_voxel_size_wrt_resolution(self.res_idx)
            set_bn_resolution(self.res_aware_batch_norms, self.res_idx)

            if shrink:
                self.shrink_flip = not self.shrink_flip
                pc_filter_lims = self.mpc.slice_inds_to_point_cloud_range(self.res_idx, xmin, xmax)
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
            if fixed_res_idx == -1:
                lim1 = xmin*self.dense_inp_slice_sz
                lim2 = (xmax+1)*self.dense_inp_slice_sz
                x_conv4 = x_conv4[..., lim1:lim2].contiguous()
            batch_dict['tensor_slice_inds'] = (xmin, xmax)
            pred_dicts, topk_outputs = self.forward_eval_dense(x_conv4)
            self.measure_time_end('DenseOps')

            self.measure_time_start('CenterHead-GenBox')
            if fixed_res_idx == -1:
                for i, topk_out in enumerate(topk_outputs):
                    topk_out[-1] += lim1 # NOTE assume the tensor resolution is same

            forecasted_dets = None
            if self.enable_forecasting:
                forecasted_dets = self.sampled_dets[self.dataset_indexes[0]]
                if forecasted_dets is not None:
                    # filter those which were forecasted already
                    forecasted_pd = forecasted_dets[0]
                    ps = forecasted_pd['pred_scores']
                    mask = (ps >= self.score_thresh)
                    for k in ('pred_boxes', 'pred_labels'):
                        forecasted_pd[k] = forecasted_pd[k][mask]
                    # Deprioritize the forecasted
                    forecasted_pd['pred_scores'] = torch.full(forecasted_pd['pred_labels'].shape,
                            self.score_thresh * 0.9, dtype=ps.dtype)
                    # Split
                    forecasted_dets = split_dets(
                            self.dense_head_scrpt.cls_id_to_det_head_idx_map,
                            self.dense_head_scrpt.num_det_heads,
                            forecasted_pd['pred_boxes'],
                            forecasted_pd['pred_scores'],
                            forecasted_pd['pred_labels'] - 1,
                            False) # moves results to gpu if true
            self.dense_head_scrpt.adjust_voxel_size_wrt_resolution(resdiv)
            batch_dict['final_box_dicts'] = self.dense_head_scrpt.forward_genbox(
                    batch_dict['batch_size'], pred_dicts,
                    topk_outputs, forecasted_dets)
            self.measure_time_end('CenterHead-GenBox')

            return batch_dict

    # takes already sliced input
    def forward_eval_dense(self, x_conv4):
        if self.fused_convs_trt[self.res_idx] is not None:
            self.trt_outputs[self.res_idx] = self.fused_convs_trt[self.res_idx](
                    {'x_conv4': x_conv4}, self.trt_outputs[self.res_idx])
            pred_dicts, topk_outputs = self.convert_trt_outputs(self.trt_outputs[self.res_idx])
        else:
            outputs = self.opt_dense_convs(x_conv4)
            out_dict = {name:outp for name, outp in zip(self.opt_outp_names, outputs)}
            pred_dicts, topk_outputs = self.convert_trt_outputs(out_dict)

        return pred_dicts, topk_outputs

    def optimize(self, fwd_data):
        optimize_start = time.time()

        self.inp_tensor_sizes[self.res_idx] = fwd_data.shape
        assert fwd_data.shape[-3] % self.dense_inp_slice_sz == 0

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
        self.sim_cur_time_ms = 0.
        return None
