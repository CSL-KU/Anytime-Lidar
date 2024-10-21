from .anytime_template_v2 import AnytimeTemplateV2
from ..dense_heads.center_head_inf import scatter_sliced_tensors
from ..backbones_2d.base_bev_backbone_sliced import prune_spatial_features
from ..model_utils.tensorrt_utils.trtwrapper import TRTWrapper
from .forecaster import Forecaster
import torch
import time
import onnx
import os
import sys
import numpy as np
import platform
from typing import List

import ctypes
ctypes.CDLL("../pcdet/trt_plugins/slice_and_batch_nhwc/build/libslice_and_batch_lib.so",
        mode=ctypes.RTLD_GLOBAL)

class DenseConvsPipeline(torch.nn.Module):
    def __init__(self, backbone_3d, backbone_2d, dense_head, tcount):
        super().__init__()
        self.backbone_3d = backbone_3d
        self.backbone_2d = backbone_2d
        self.dense_head = dense_head
        self.tcount = tcount

    def forward(self, x_conv4 : torch.Tensor, set_out_tile_sizes : bool = False) -> List[torch.Tensor]:
        x_conv5 = self.backbone_3d.forward_dense(x_conv4)
        data_dict = self.backbone_2d({"multi_scale_2d_features" : 
            {"x_conv4": x_conv4, "x_conv5": x_conv5}})

        outputs = self.dense_head.forward_pre(data_dict['spatial_features_2d'])
        shr_conv_outp = outputs[0]
        heatmaps = outputs[1:]

        # onnx export should ignore this while tracing
        if set_out_tile_sizes:
            # Assumes the spatial_features is full sized input with all tiles
            self.out_tile_sizes= [hm.size(-1) // self.tcount for hm in heatmaps]

        topk_outputs = self.dense_head.forward_topk_trt(heatmaps)

        ys_all = [topk_outp[2] for topk_outp in topk_outputs]
        xs_all = [topk_outp[3] for topk_outp in topk_outputs]

        sliced_inp = self.dense_head.slice_shr_conv_outp(shr_conv_outp, ys_all, xs_all)
        outputs = self.dense_head.forward_sliced_inp_trt(sliced_inp)
        for topk_output in topk_outputs:
            outputs += topk_output

        return outputs

    def get_out_tile_sizes(self):
        return self.out_tile_sizes

# correct the topk xs, this can be faster if xs's are catted
@torch.jit.script
def fix_topk_outputs(tile_sizes : List[int], mapping : torch.Tensor,
        topk_outputs : List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
    for i, topk_out in enumerate(topk_outputs):
        tile_sz =  tile_sizes[i] #// mapping.size(0)
        xs = topk_out[-1].int()
        xs_tile_inds = torch.div(xs, tile_sz, rounding_mode='trunc')
        topk_out[-1] = xs + tile_sz * (mapping[xs_tile_inds] - xs_tile_inds)
    return topk_outputs

class PillarNetVALO(AnytimeTemplateV2):
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

        self.update_time_dict({
            'Sched1' : [],
            'VFE' : [],
            'Sched2' : [],
            'Backbone3D': [],
            'FusedOps':[],
            'CenterHead-GenBox': [],
        })

        self.vfe, self.backbone_3d, self.backbone_2d, self.dense_head = self.module_list

        self.inf_stream = torch.cuda.Stream()
        self.optimization1_done = False
        self.trt_outputs = None # Since output size of trt is fixed, use buffered

        self.resolution_dividers = self.model_cfg.BACKBONE_3D.get('RESOLUTION_DIV', [1])
        self.res_idx = 0
        self.latest_losses = [0.] * len(self.resolution_dividers)

        # Force forecasting to be disabled
        self.keep_forecasting_disabled = False
        if not self.keep_forecasting_disabled:
            self.forecaster = torch.jit.script(Forecaster(tuple(self.dataset.point_cloud_range.tolist()), 
                    self.tcount, self.score_thresh, self.forecasting_coeff, self.dense_head.num_det_heads,
                    self.dense_head.cls_id_to_det_head_idx_map))
        
        self.calc_ult_heatmap = False

    def forward(self, batch_dict):
        if self.training:
            batch_dict = self.vfe.range_filter(batch_dict)
            points = batch_dict['points']
            batch_dict['voxel_coords'], batch_dict['voxel_features'] = self.vfe(points)
            batch_dict['pillar_features'] = batch_dict['voxel_features']
            batch_dict['pillar_coords'] = batch_dict['voxel_coords']

            resdiv = self.resolution_dividers[self.res_idx]
            batch_dict['resolution_divider'] = resdiv
            self.res_idx = (self.res_idx +1) % len(self.resolution_dividers)

            batch_dict = self.backbone_3d(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)

            self.dense_head.adjust_voxel_size_wrt_resolution(resdiv)
            batch_dict = self.dense_head.forward_pre(batch_dict)
            batch_dict = self.dense_head.forward_post(batch_dict)

            fms = self.model_cfg.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
            target_hw = [sz // fms // resdiv for sz in self.dataset.grid_size[:2]]
            batch_dict = self.dense_head.forward_assign_targets(batch_dict, feature_map_size=target_hw)
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            self.latest_losses[self.res_idx] = loss.item()

            disp_dict.update({f"loss_resdiv{self.resolution_dividers[i]}": l \
                    for i, l in enumerate(self.latest_losses)})

            return ret_dict, tb_dict, disp_dict
        else:
            return self.forward_eval(batch_dict)

    def forward_eval(self, batch_dict):
        with torch.cuda.stream(self.inf_stream):
            batch_dict = self.vfe.range_filter(batch_dict)
            self.measure_time_start('Sched1')
            batch_dict = self.schedule1(batch_dict)
            self.measure_time_end('Sched1')

            if self.is_calibrating():
                e1 = torch.cuda.Event(enable_timing=True)
                e1.record()

            self.measure_time_start('VFE')
            points = batch_dict['points']
            batch_dict['voxel_coords'], batch_dict['voxel_features'] = self.vfe(points)
            batch_dict['pillar_features'] = batch_dict['voxel_features']
            batch_dict['pillar_coords'] = batch_dict['voxel_coords']
            self.measure_time_end('VFE')

            if self.is_calibrating():
                e2 = torch.cuda.Event(enable_timing=True)
                e2.record()
                batch_dict['vfe_layer_time_events'] = [e1, e2]

            if batch_dict['pillar_coords'].size(0) == 1:
                # Can't infer anything out of this, use random data to prevent instancenorm error
                pc = batch_dict['pillar_coords']
                pf = batch_dict['pillar_features']

                num_rand_pillars = 64
                xlim, ylim, _ = self.dataset.grid_size
                pc = torch.randint(0, min(xlim, ylim), (num_rand_pillars, pc.size(1)),
                        dtype=pc.dtype, device=pc.device)
                pc[:, 0] = 0 # batch size 1
                batch_dict['pillar_coords'] = pc
                pf = pf.repeat(num_rand_pillars, 1)
                batch_dict['pillar_features'] = pf

            batch_dict['resolution_divider'] = 1

            self.measure_time_start('Backbone3D')
            batch_dict = self.backbone_3d.forward_up_to_dense(batch_dict)
            x_conv4 = batch_dict['x_conv4_out']
            self.measure_time_end('Backbone3D')

            if self.is_calibrating():
                e3 = torch.cuda.Event(enable_timing=True)
                e3.record()

            self.measure_time_start('Sched2')
            batch_dict = self.schedule2(batch_dict)
            lbd = self.latest_batch_dict
            if self.enable_forecasting and lbd is not None:
                # Takes 1.2 ms, fully on cpu
                last_pred_dict = lbd['final_box_dicts'][0]
                last_ctc = torch.from_numpy(lbd['chosen_tile_coords']).long()
                last_token = lbd['metadata'][0]['token']
                last_pose = self.token_to_pose[last_token]
                last_ts = self.token_to_ts[last_token] - self.scene_init_ts
                cur_token = batch_dict['metadata'][0]['token']
                cur_pose = self.token_to_pose[cur_token]
                cur_ts = self.token_to_ts[cur_token] - self.scene_init_ts

                fcdets_fut = self.forecaster.fork_forward(last_pred_dict, last_ctc,
                        last_pose, last_ts, cur_pose, cur_ts, batch_dict['scene_reset'])
            else:
                fcdets_fut = None

            if not self.optimization1_done:
                self.optimize1(x_conv4)

            batch_dict['spatial_features'] = x_conv4
            batch_dict['tcount'] = self.tcount
            batch_dict = prune_spatial_features(batch_dict)
            x_conv4 = batch_dict['spatial_features'] # sliced
            self.measure_time_end('Sched2')

            self.measure_time_start('FusedOps')
            if self.fused_convs_trt is not None:
                self.trt_outputs = self.fused_convs_trt({'x_conv4': x_conv4}, self.trt_outputs)
                pred_dicts, topk_outputs = self.convert_trt_outputs(self.trt_outputs)
            else:
                outputs = self.opt_dense_convs(x_conv4)
                out_dict = {name:outp for name, outp in zip(self.opt_outp_names, outputs)}
                pred_dicts, topk_outputs = self.convert_trt_outputs(out_dict)
            self.measure_time_end('FusedOps')

            if self.is_calibrating():
                e4 = torch.cuda.Event(enable_timing=True)
                e4.record()
                batch_dict['bb2d_time_events'] = [e3, e4]

            self.measure_time_start('CenterHead-GenBox')

            tile_sizes = self.opt_dense_convs.get_out_tile_sizes()
            topk_outputs = fix_topk_outputs(tile_sizes, batch_dict['tile_mapping'], topk_outputs)

            if self.calc_ult_heatmap:
                fms = self.model_cfg.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
                target_hw = [sz // fms for sz in self.dataset.grid_size[:2]]
                ult_heatmap = torch.zeros((1, 1, target_hw[0], target_hw[1]),
                        dtype=topk_outputs[0][0].dtype,
                        device=topk_outputs[0][0].device)
                all_scores = torch.cat([t[0] for t in topk_outputs])
                all_ys = torch.cat([t[2] for t in topk_outputs]).long()
                all_xs = torch.cat([t[3] for t in topk_outputs]).long()
                # NOTE, better is to do scatter_max, but its ok
                ult_heatmap[:, :, all_ys, all_xs] = all_scores
                batch_dict['ult_heatmap'] = ult_heatmap

            forecasted_dets = torch.jit.wait(fcdets_fut) if fcdets_fut is not None else None
            if forecasted_dets is not None and len(forecasted_dets) != self.dense_head.num_det_heads:
                forecasted_dets = None # NOTE this appears to be a bug, but dont know how to fix it
            self.dense_head_scrpt.adjust_voxel_size_wrt_resolution(batch_dict['resolution_divider'])
            batch_dict['final_box_dicts'] = self.dense_head_scrpt.forward_genbox(
                    batch_dict['batch_size'], pred_dicts,
                    topk_outputs, forecasted_dets)
            self.measure_time_end('CenterHead-GenBox')

            return batch_dict

    def optimize1(self, fwd_data):
        optimize_start = time.time()

        self.opt_dense_convs = DenseConvsPipeline(self.backbone_3d, self.backbone_2d, self.dense_head, self.tcount)
        self.opt_dense_convs.eval()

        input_names = ['x_conv4']
        print(input_names[0], fwd_data.size())
        self.opt_dense_convs_output_names_pd = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                for name in self.dense_head.ordered_outp_names(False)]

        self.topk_outp_names = ('scores', 'class_ids', 'xs', 'ys')
        self.opt_dense_convs_output_names_topk = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                for name in self.topk_outp_names]

        self.opt_outp_names = self.opt_dense_convs_output_names_pd + self.opt_dense_convs_output_names_topk
        print('Fused operations output names:', self.opt_outp_names)

        self.dense_head.instancenorm_mode()

        self.opt_dense_convs(fwd_data, True) # do this so tile sizes are determined

        generated_onnx=False
        onnx_path = self.model_cfg.ONNX_PATH + '.onnx'
        if not os.path.exists(onnx_path):
            dynamic_axes = {
                "x_conv4": {
                    3: "width",
                },
            }

            torch.onnx.export(
                    self.opt_dense_convs,
                    fwd_data.requires_grad_(False),
                    onnx_path,
                    input_names=input_names,
                    output_names=self.opt_outp_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=17,
                    custom_opsets={"cuda_slicer": 17}
            )
            generated_onnx=True

        power_mode = os.getenv('PMODE', 'UNKNOWN_POWER_MODE')
        if power_mode == 'UNKNOWN_POWER_MODE':
            print('WARNING! Power mode is unknown. Please export PMODE.')

        if generated_onnx:
            print('ONNX files created, please run again after creating TensorRT engines.')
            sys.exit(0)

        tokens = self.model_cfg.ONNX_PATH.split('/')
        trt_path = '/'.join(tokens[:-2]) + f'/trt_engines/{power_mode}/{tokens[-1]}.engine'
        print('Trying to load trt engine at', trt_path)
        try:
            self.fused_convs_trt = TRTWrapper(trt_path, input_names, self.opt_outp_names)
        except:
            print('TensorRT wrapper for fused_convs throwed exception, using eager mode')
            self.fused_convs_trt = None

        optimize_end = time.time()
        print(f'Optimization took {optimize_end-optimize_start} seconds.')

        self.dense_head_scrpt = torch.jit.script(self.dense_head)
        self.optimization1_done = True

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
        return super().calibrate(1)

