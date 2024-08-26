from .detector3d_template import Detector3DTemplate
import torch
import numpy as np
import numba
import time
import onnx
#import onnxruntime as ort
import onnx_graphsurgeon as gs
import os
import struct
import sys
from typing import List
#import torch_tensorrt
from ..model_utils.tensorrt_utils.trtwrapper import TRTWrapper

class OptimizedFwdPipeline2(torch.nn.Module):
    def __init__(self, backbone_2d, dense_head):
        super().__init__()
        self.backbone_2d = backbone_2d
        self.dense_head = dense_head

    def forward(self, spatial_features : torch.Tensor) -> List[torch.Tensor]:
        spatial_features_2d = self.backbone_2d(spatial_features)
        return self.dense_head.forward_up_to_topk(spatial_features_2d)

class DSVT_CenterHead_Opt(Detector3DTemplate):
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

        self.is_voxel_enc=True
        self.vfe, self.backbone_3d, self.map_to_bev, self.backbone_2d, \
                self.dense_head = self.module_list

        self.update_time_dict( {
            'VFE' : [],
            'Backbone3D-IL': [],
            'Backbone3D-Fwd':[],
            'FusedOps2':[],
            'CenterHead-Topk': [],
            'CenterHead-GenBox': [],
        })

        self.map_to_bev_scrpt = torch.jit.script(self.map_to_bev)
        self.inf_stream = torch.cuda.Stream()
        self.optimization1_done = False
        self.optimization2_done = False


    def forward(self, batch_dict):
        with torch.cuda.stream(self.inf_stream):
            self.measure_time_start('VFE')
            batch_dict = self.vfe.range_filter(batch_dict)
            points = batch_dict['points']
            batch_dict['voxel_coords'], batch_dict['voxel_features'] = self.vfe(points)
            batch_dict['pillar_features'] = batch_dict['voxel_features']
            self.measure_time_end('VFE')

            self.measure_time_start('Backbone3D-IL')
            vinfo = self.backbone_3d.get_voxel_info(batch_dict['voxel_features'],
                    batch_dict['voxel_coords'])
            self.measure_time_end('Backbone3D-IL')

            if not self.optimization1_done:
                self.optimize1(vinfo[:-1])
                self.dense_head_scrpt = torch.jit.script(self.dense_head)

            self.measure_time_start('Backbone3D-Fwd')
            inputs_dict = {'voxel_feat': vinfo[0],
                    'set_voxel_inds_tensor_shift_0': vinfo[1],
                    'set_voxel_inds_tensor_shift_1': vinfo[2],
                    'set_voxel_masks_tensor_shift_0': vinfo[3],
                    'set_voxel_masks_tensor_shift_1': vinfo[4],
                    'pos_embed_tensor' : vinfo[5],
            }
            if self.backbone_3d_trt is not None:
                output = self.backbone_3d_trt(inputs_dict, vinfo[0].size())['output']
            else:
                vinfo_ = vinfo[:-1]
                output = self.backbone_3d(*vinfo_)

            batch_dict['spatial_features'] = self.map_to_bev_scrpt(output, vinfo[-1]) # 1 is batch size
            self.measure_time_end('Backbone3D-Fwd')

            if not self.optimization2_done:
                self.optimize2(batch_dict['spatial_features'])

            self.measure_time_start('FusedOps2')
            sf = batch_dict['spatial_features']

            if self.fused_ops2_trt is not None:
                outputs = self.fused_ops2_trt({'spatial_features': sf})
                outputs = [outputs[nm] for nm in self.opt_fwd2_output_names]
            else:
                outputs = self.opt_fwd2(sf)
            batch_dict["pred_dicts"] = self.dense_head.convert_out_to_batch_dict(outputs)
            self.measure_time_end('FusedOps2')

            self.measure_time_start('CenterHead-Topk')
            topk_outputs = self.dense_head_scrpt.forward_topk(batch_dict["pred_dicts"])
            self.measure_time_end('CenterHead-Topk')
            self.measure_time_start('CenterHead-GenBox')
            batch_dict['final_box_dicts'] = self.dense_head_scrpt.forward_genbox(batch_dict['batch_size'], batch_dict["pred_dicts"],
                    topk_outputs, None)
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

        input_names = [
                'voxel_feat',
                'set_voxel_inds_tensor_shift_0',
                'set_voxel_inds_tensor_shift_1',
                'set_voxel_masks_tensor_shift_0',
                'set_voxel_masks_tensor_shift_1',
                'pos_embed_tensor',
        ]

        output_names = ['output']

        opt_fwd = self.backbone_3d
        opt_fwd.eval()

        generated_onnx=False
        onnx_path = self.model_cfg.BACKBONE_3D.OPT_PATH + '.onnx'
        if not os.path.exists(onnx_path):
            dynamic_axes = {
                "voxel_feat": {
                    0: "voxel_number",
                },
                "set_voxel_inds_tensor_shift_0": {
                    1: "set_number_shift_0",
                },
                "set_voxel_inds_tensor_shift_1": {
                    1: "set_number_shift_1",
                },
                "set_voxel_masks_tensor_shift_0": {
                    1: "set_number_shift_0",
                },
                "set_voxel_masks_tensor_shift_1": {
                    1: "set_number_shift_1",
                },
                "pos_embed_tensor": {
                    2: "voxel_number",
                },
            }

            torch.onnx.export(
                    opt_fwd,
                    fwd_data,
                    onnx_path, input_names=input_names,
                    output_names=output_names, dynamic_axes=dynamic_axes,
                    opset_version=17,
                    custom_opsets={"kucsl": 17}
            )

        trt_path = self.model_cfg.BACKBONE_3D.OPT_PATH + '.engine'
        try:
            self.backbone_3d_trt = TRTWrapper(trt_path, input_names, output_names)
        except:
            print('TensorRT wrapper for backbone3d throwed exception, using eager mode')
            self.backbone_3d_trt = None
        optimize_end = time.time()
        print(f'Optimization took {optimize_end-optimize_start} seconds.')
        self.optimization1_done = True

    def optimize2(self, fwd_data):
        optimize_start = time.time()

        input_names = ['spatial_features']

        self.opt_fwd2_output_names = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                for name in self.dense_head.ordered_outp_names()]
        print('Fused operations output names:', self.opt_fwd2_output_names)

        self.opt_fwd2 = OptimizedFwdPipeline2(self.backbone_2d, self.dense_head)
        self.opt_fwd2.eval()

        generated_onnx=False
        onnx_path = self.model_cfg.BACKBONE_2D.OPT_PATH + '.onnx'
        if not os.path.exists(onnx_path):
            torch.onnx.export(
                    self.opt_fwd2,
                    fwd_data,
                    onnx_path, input_names=input_names,
                    output_names=self.opt_fwd2_output_names,
                    opset_version=17,
                    #custom_opsets={"kucsl": 17}
            )
            generated_onnx=True

        sf = fwd_data 
        trt_path = self.model_cfg.BACKBONE_2D.OPT_PATH + '.engine'
        try:
            self.fused_ops2_trt = TRTWrapper(trt_path, input_names, self.opt_fwd2_output_names)
        except:
            print('TensorRT wrapper for fused_ops2 throwed exception, using eager mode')
            eager_outputs = self.opt_fwd2(fwd_data) # for calibration
            self.fused_ops2_trt = None

        optimize_end = time.time()
        print(f'Optimization took {optimize_end-optimize_start} seconds.')
        self.optimization2_done = True
        if generated_onnx:
            print('ONNX files created, please run again after creating TensorRT engines.')
            sys.exit(0)


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

############SOME CODE HERE TO USE LATER IF NEEDED ##################
#        trt_path = self.model_cfg.BACKBONE_3D.trt_engine
#        self.compiled_bb3d = TRTWrapper(trt_path, input_names, output_names)
#        print('Outputs after trt inference:')
#        inputs_dict = {'voxel_feat': vinfo[0],
#                'set_voxel_inds_tensor_shift_0': vinfo[1],
#                'set_voxel_inds_tensor_shift_1': vinfo[2],
#                'set_voxel_masks_tensor_shift_0': vinfo[3],
#                'set_voxel_masks_tensor_shift_1': vinfo[4],
#               'pos_embed_tensor' : vinfo[5]}
#        out = self.compiled_bb3d(inputs_dict)['output']
#        print(out.size(), out)

#            inp1, inp2, inp3, inp4, inp5, inp6 = vinfo
#            torch._dynamo.mark_dynamic(inp1, 0, min=3000, max=30000)
#            torch._dynamo.mark_dynamic(inp2, 1, min=60, max=400)
#            torch._dynamo.mark_dynamic(inp3, 1, min=60, max=400)
#            torch._dynamo.mark_dynamic(inp4, 1, min=60, max=400)
#            torch._dynamo.mark_dynamic(inp5, 1, min=60, max=400)
#            torch._dynamo.mark_dynamic(inp6, 2, min=3000, max=30000)
#            export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
#            torch.onnx.enable_fake_mode()
#            onnx_program = torch.onnx.dynamo_export(self.backbone_3d,
#                    inp1, inp2, inp3, inp4, inp5, inp6, export_options=export_options)
#            onnx_program.save(onnx_path)

#            sm = torch.jit.script(self.backbone_3d, example_inputs=[vinfo])
#            output = sm(*vinfo)
#            print('Outputs after torchscript inference:')
#            print(output.size(), output)

         ####TensorRT ONNX conf
#        def get_shapes_str(d1, d2, d3):
#            return f'voxel_feat:{d1}x128,set_voxel_inds_tensor_shift_0:2x{d2}x90,' \
#        f'set_voxel_inds_tensor_shift_1:2x{d3}x90,set_voxel_masks_tensor_shift_0:2x{d2}x90,' \
#        f'set_voxel_masks_tensor_shift_1:2x{d3}x36,pos_embed_tensor:4x2x{d1}x128'
#        
#        tensorrt_conf = {
#                'device_id': torch.cuda.current_device(),
#                "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
#                #'trt_max_workspace_size': 2147483648,
#                'trt_profile_min_shapes': get_shapes_str(3000, 60, 60),
#                'trt_profile_opt_shapes': get_shapes_str(11788,190,205),
#                'trt_profile_max_shapes': get_shapes_str(30000, 400, 400),
#                'trt_fp16_enable': False,
#                'trt_layer_norm_fp32_fallback': True,
#        }

#                #TensorRT
#                inputs_dict = {'voxel_feat': vinfo[0],
#                        'set_voxel_inds_tensor_shift_0': vinfo[1],
#                        'set_voxel_inds_tensor_shift_1': vinfo[2],
#                        'set_voxel_masks_tensor_shift_0': vinfo[3],
#                        'set_voxel_masks_tensor_shift_1': vinfo[4],
#                        'pos_embed_tensor' : vinfo[5]}
#                output = self.compiled_bb3d(inputs_dict)['output']


