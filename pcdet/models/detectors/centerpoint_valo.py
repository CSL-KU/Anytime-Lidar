from .anytime_template_v2 import AnytimeTemplateV2
from ..dense_heads.center_head_inf import scatter_sliced_tensors
from ..model_utils.tensorrt_utils.trtwrapper import TRTWrapper
from .forecaster import Forecaster
import torch
import time
import onnx
import os
import sys
from typing import List

class DenseConvsPipeline(torch.nn.Module):
    def __init__(self, backbone_2d, dense_head):
        super().__init__()
        self.backbone_2d = backbone_2d
        self.dense_head = dense_head

    def forward(self, spatial_features : torch.Tensor) -> List[torch.Tensor]:
        spatial_features_2d = self.backbone_2d(spatial_features)

        if self.dense_head.optimize_attr_convs:
            return self.dense_head.forward_pre(spatial_features_2d)
        else:
            return self.dense_head.forward_up_to_topk(spatial_features_2d)

class SlicedConvsPipeline(torch.nn.Module):
    def __init__(self, dense_head):
        super().__init__()
        self.dense_head = dense_head

    def forward(self, slices_per_head: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.dense_head.forward_sliced_inp_trt(slices_per_head)

class CenterPointVALO(AnytimeTemplateV2):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        torch.backends.cudnn.benchmark = True
        if torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark_limit = 0
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        torch.cuda.manual_seed(0)
        self.module_list = self.build_networks()

        self.update_time_dict( {
            'Sched1' : [],
            'VFE' : [],
            'MapToBEV' : [],
            'Sched2' : [],
        })
        self.use_pillars = (self.model_cfg.get('BACKBONE_3D', None) is None)
        if not self.use_pillars:
            self.update_time_dict({'Backbone3D': []})
        self.update_time_dict( {
            'FusedOps2':[],
            'CenterHead-Topk': [],
            'CenterHead-AttrConvs': [],
            'CenterHead-GenBox': [],
        })

        if self.use_pillars:
            self.vfe, self.map_to_bev, self.backbone_2d, self.dense_head = self.module_list
            self.map_to_bev_scrpt = torch.jit.script(self.map_to_bev)
        else:
            self.vfe, self.backbone_3d, self.map_to_bev, self.backbone_2d, \
                    self.dense_head = self.module_list
            self.map_to_bev_scrpt = self.map_to_bev # no need to script

        self.inf_stream = torch.cuda.Stream()
        self.optimization1_done = False
        self.optimization2_done = False

        # Force forecasting to be disabled
        self.keep_forecasting_disabled = False
        if not self.keep_forecasting_disabled:
            self.forecaster = torch.jit.script(Forecaster(tuple(self.dataset.point_cloud_range.tolist()), 
                    self.tcount, self.score_thresh, self.forecasting_coeff, self.dense_head.num_det_heads,
                    self.dense_head.cls_id_to_det_head_idx_map))

    def forward(self, batch_dict):
        with torch.cuda.stream(self.inf_stream):
            # VFE doesn't take much of a time (5 ms), do not schedule its input
            self.measure_time_start('VFE')
            batch_dict = self.vfe.range_filter(batch_dict)
            points = batch_dict['points']
            batch_dict['voxel_coords'], batch_dict['voxel_features'] = self.vfe(points)
            batch_dict['pillar_features'] = batch_dict['voxel_features']
            self.measure_time_end('VFE')

            self.measure_time_start('Sched1')
            batch_dict = self.schedule1(batch_dict)
            self.measure_time_end('Sched1')

            if not self.use_pillars:
                self.measure_time_start('Backbone3D')
                batch_dict = self.backbone_3d(batch_dict)
                self.measure_time_end('Backbone3D')

            if self.is_calibrating():
                e1 = torch.cuda.Event(enable_timing=True)
                e1.record()

            self.measure_time_start('MapToBEV')
            if not self.use_pillars:
                batch_dict = self.map_to_bev(batch_dict)
            else:
                batch_dict['spatial_features'] = self.map_to_bev_scrpt(
                        batch_dict['pillar_features'],
                        batch_dict['voxel_coords'],
                        batch_dict['batch_size'])
            self.measure_time_end('MapToBEV')

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

            batch_dict = self.backbone_2d.prune_spatial_features(batch_dict)
            self.measure_time_end('Sched2')

            if not self.optimization1_done:
                self.optimize1(batch_dict['spatial_features'])

            self.measure_time_start('FusedOps2')
            sf = batch_dict['spatial_features']
            if self.fused_convs_trt is not None:
                outputs = self.fused_convs_trt({'spatial_features': sf})
                outputs = [outputs[nm] for nm in self.opt_dense_convs_output_names]
            else:
                outputs = self.opt_dense_convs(sf)

            ctc = batch_dict['chosen_tile_coords'].tolist()
            outputs = scatter_sliced_tensors(ctc, outputs, self.tcount)
            if self.dense_head.optimize_attr_convs:
                shr_conv_outp = outputs[0]
                pred_dicts = [{'hm':hm} for hm in outputs[1:]]
            else:
                pred_dicts = self.dense_head.convert_out_to_pred_dicts(outputs)
            self.measure_time_end('FusedOps2')

            self.measure_time_start('CenterHead-Topk')
            topk_outputs = self.dense_head_scrpt.forward_topk(pred_dicts)
            self.measure_time_end('CenterHead-Topk')

            if self.dense_head.optimize_attr_convs:
                self.measure_time_start('CenterHead-AttrConvs')
                sliced_inp = self.dense_head_scrpt.slice_shr_conv_outp(shr_conv_outp, pred_dicts,
                        topk_outputs)

                if not self.optimization2_done:
                    self.optimize2(sliced_inp)

                if self.sliced_convs_trt is not  None:
                    inp = {nm:sinp for nm,sinp in zip(self.sliced_input_names, sliced_inp)}
                    outputs = self.sliced_convs_trt(inp)
                    for k,v in outputs.items():
                        pd_idx = int(k[-1]) # assumes the number at the end is 1 digit
                        pred_dicts[pd_idx][k[:-1]] = v
                else:
                    pred_dicts = self.dense_head_scrpt.forward_sliced_inp(sliced_inp, pred_dicts)
                self.measure_time_end('CenterHead-AttrConvs')

            batch_dict["pred_dicts"] = pred_dicts

            if self.is_calibrating():
                e2 = torch.cuda.Event(enable_timing=True)
                e2.record()
                batch_dict['bb2d_time_events'] = [e1, e2]

            self.measure_time_start('CenterHead-GenBox')
            forecasted_dets = torch.jit.wait(fcdets_fut) if fcdets_fut is not None else None
            if forecasted_dets is not None and len(forecasted_dets) != self.dense_head.num_det_heads:
                forecasted_dets = None # NOTE this appears to be a bug, but dont know how to fix it
            batch_dict['final_box_dicts'] = self.dense_head_scrpt.forward_genbox(
                    batch_dict['batch_size'], batch_dict["pred_dicts"],
                    topk_outputs, forecasted_dets)
            self.measure_time_end('CenterHead-GenBox')

            if self.training:
                loss, tb_dict, disp_dict = self.get_training_loss()

                ret_dict = {
                        'loss': loss
                        }
                return ret_dict, tb_dict, disp_dict
            else:
                # let the hooks of parent class handle this
                return batch_dict

    def optimize1(self, fwd_data):
        optimize_start = time.time()

        input_names = ['spatial_features']
        print(input_names[0], fwd_data.size())

        if self.dense_head.optimize_attr_convs:
            self.opt_dense_convs_output_names = ['shr_conv_outp'] + ['hm' + str(i) \
                    for i in range(self.dense_head.num_det_heads)]
        else:
            self.opt_dense_convs_output_names = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                    for name in self.dense_head.ordered_outp_names()]
        print('Fused operations output names:', self.opt_dense_convs_output_names)

        self.opt_dense_convs = DenseConvsPipeline(self.backbone_2d, self.dense_head)
        self.opt_dense_convs.eval()

        generated_onnx=False
        opt_path = self.model_cfg.BACKBONE_2D.OPT_PATH
        if self.dense_head.optimize_attr_convs:
            opt_path += '_dhopt'
        onnx_path = opt_path + '.onnx'
        if not os.path.exists(onnx_path):
            dynamic_axes = {
                "spatial_features": {
                    3: "width",
                },
            }
            for nm in self.opt_dense_convs_output_names:
                dynamic_axes[nm] = {3 : "out_width"}

            torch.onnx.export(
                    self.opt_dense_convs,
                    fwd_data,
                    onnx_path, input_names=input_names,
                    output_names=self.opt_dense_convs_output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=17,
                    #custom_opsets={"kucsl": 17}
            )
            generated_onnx=True

        trt_path = opt_path + '.engine'
        try:
            self.fused_convs_trt = TRTWrapper(trt_path, input_names, self.opt_dense_convs_output_names)
        except:
            print('TensorRT wrapper for fused_convs throwed exception, using eager mode')
            eager_outputs = self.opt_dense_convs(fwd_data) # for calibration
            self.fused_convs_trt = None

        optimize_end = time.time()
        print(f'Optimization took {optimize_end-optimize_start} seconds.')

        self.dense_head_scrpt = torch.jit.script(self.dense_head)
        self.optimization1_done = True

    def optimize2(self, fwd_data):
        optimize_start = time.time()

        self.sliced_input_names = [f'sliced_inp_{i}' for i in range(self.dense_head.num_det_heads)]
        #print(input_names[0], fwd_data.size())
        outp_names = self.dense_head.get_sliced_forward_outp_names()
        self.sliced_output_names = [nm + str(i) \
                    for i in range(self.dense_head.num_det_heads) for nm in outp_names]
        print('Sliced convs output names:', self.sliced_output_names)

        self.dense_head.instancenorm_mode()
        self.opt_sliced_convs = SlicedConvsPipeline(self.dense_head)
        self.opt_sliced_convs.eval()

        generated_onnx=False
        opt_path = self.model_cfg.DENSE_HEAD.OPT_PATH
        onnx_path = opt_path + '.onnx'
        if not os.path.exists(onnx_path):
            torch.onnx.export(
                    self.opt_sliced_convs,
                    fwd_data,
                    onnx_path, input_names=self.sliced_input_names,
                    output_names=self.sliced_output_names,
                    opset_version=17,
                    #custom_opsets={"kucsl": 17}
            )
            generated_onnx=True

        trt_path = opt_path + '.engine'
        try:
            self.sliced_convs_trt = TRTWrapper(trt_path, self.sliced_input_names, self.sliced_output_names)
        except:
            print('TensorRT wrapper for fused_convs throwed exception, using eager mode')
            eager_outputs = self.opt_sliced_convs(fwd_data) # for calibration
            self.sliced_convs_trt = None

        optimize_end = time.time()
        print(f'Optimization took {optimize_end-optimize_start} seconds.')
        if generated_onnx:
            print('ONNX files created, please run again after creating TensorRT engines.')
            sys.exit(0)

        self.optimization2_done = True

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

