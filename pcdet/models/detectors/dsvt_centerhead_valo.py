from .anytime_template_v2 import AnytimeTemplateV2
from ..dense_heads.center_head_inf import scatter_sliced_tensors
import torch
import numpy as np
import numba
import time
import onnx
import onnxruntime as ort
import onnx_graphsurgeon as gs
import os
import struct
import sys
#import torch_tensorrt
from ..model_utils.tensorrt_utils.trtwrapper import TRTWrapper

class OptimizedFwdPipeline2(torch.nn.Module):
    def __init__(self, backbone_2d, dense_head):
        super().__init__()
        self.backbone_2d = backbone_2d
        self.dense_head = dense_head

    def forward(self, spatial_features : torch.Tensor) -> torch.Tensor:
        spatial_features_2d = self.backbone_2d(spatial_features)
        #hm, center, center_z, dim, rot, vel, iou = \
        hm, center, center_z, dim, rot, vel = \
                self.dense_head.forward_up_to_topk(spatial_features_2d)
        return hm, center, center_z, dim, rot, vel

# Optimized with tensorrt
class DSVT_CenterHead_VALO(AnytimeTemplateV2):
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
            'Sched1': [],
            'VFE' : [],
            'Backbone3D-IL': [],
            'Backbone3D-Fwd':[],
            'Sched2': [],
            'FusedOps2':[],
            'CenterHead-Topk': [],
            'CenterHead-GenBox': [],
        })

        self.inf_stream = torch.cuda.Stream()
        self.optimization1_done = False
        self.optimization2_done = False

        # Force forecasting to be disabled
        self.keep_forecasting_disabled = False

    def forward(self, batch_dict):
        with torch.cuda.stream(self.inf_stream):

            self.measure_time_start('Sched1')
            batch_dict = self.vfe.range_filter(batch_dict)
            batch_dict = self.schedule1(batch_dict)
            #batch_dict['chosen_tile_coords'] = np.arange(6)
            self.measure_time_end('Sched1')

            if self.is_calibrating():
                e1 = torch.cuda.Event(enable_timing=True)
                e1.record()

            self.measure_time_start('VFE')
            points = batch_dict['points']
            batch_dict['voxel_coords'], batch_dict['voxel_features'] = self.vfe(points)
            batch_dict['pillar_features'] = batch_dict['voxel_features']
            self.measure_time_end('VFE')

            if self.is_calibrating():
                e2 = torch.cuda.Event(enable_timing=True)
                e2.record()
                batch_dict['vfe_layer_time_events'] = [e1, e2]

            self.measure_time_start('Backbone3D-IL')
            vinfo = self.backbone_3d.get_voxel_info(batch_dict['voxel_features'],
                    batch_dict['voxel_coords'])
            self.measure_time_end('Backbone3D-IL')

            if not self.optimization1_done:
                self.optimize1(vinfo[:-1])
                self.map_to_bev_scrpt = torch.jit.script(self.map_to_bev)

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

            if self.is_calibrating():
                e3 = torch.cuda.Event(enable_timing=True)
                e3.record()
                batch_dict['bb3d_layer_time_events'] = [e2, e3]

            if not self.optimization2_done:
                self.optimize2(batch_dict['spatial_features'])

            self.measure_time_start('Sched2')
            batch_dict = self.schedule2(batch_dict)

            lbd = self.latest_batch_dict
            fcdets_fut = None
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
                #if fcdets[0]['pred_boxes'].size(0) > 0:

            batch_dict = self.backbone_2d.prune_spatial_features(batch_dict)
            self.measure_time_end('Sched2')

            self.measure_time_start('FusedOps2')
            sf = batch_dict['spatial_features']

            if self.fused_ops2_trt is not None:
                outputs = self.fused_ops2_trt({'spatial_features': sf})
                outputs = [outputs[nm] for nm in self.dense_head.ordered_outp_names()]
            else:
                outputs = self.opt_fwd2(sf)

            outputs = scatter_sliced_tensors(batch_dict['chosen_tile_coords'], outputs,
                    self.sched_algo, self.tcount)
            out_dict = self.dense_head.convert_out_to_batch_dict(outputs)
            batch_dict["pred_dicts"] = [out_dict]
            self.measure_time_end('FusedOps2')

            if self.is_calibrating():
                e4 = torch.cuda.Event(enable_timing=True)
                e4.record()
                batch_dict['bb2d_time_events'] = [e3, e4]

            #TODO , use the optimized cuda code for the rest available in autoware
            self.measure_time_start('CenterHead-Topk')
            batch_dict = self.dense_head.forward_topk(batch_dict)
            self.measure_time_end('CenterHead-Topk')
            self.measure_time_start('CenterHead-GenBox')
            if fcdets_fut is not None:
                batch_dict['forecasted_dets'] = torch.jit.wait(fcdets_fut) # this is a fut obj!
            batch_dict = self.dense_head.forward_genbox(batch_dict)
            self.measure_time_end('CenterHead-GenBox')

            # Move this to detector3d_template post hook
            #if self.is_calibrating():
            #    e5 = torch.cuda.Event(enable_timing=True)
            #    e5.record()
            #    batch_dict['detheadpost_time_events'] = [e4, e5]

#        if self.training:
#            loss, tb_dict, disp_dict = self.get_training_loss()
#
#            ret_dict = {
#                    'loss': loss
#                    }
#            return ret_dict, tb_dict, disp_dict
#        else:
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
            self.backbone_3d_trt = None
        optimize_end = time.time()
        print(f'Optimization took {optimize_end-optimize_start} seconds.')
        self.optimization1_done = True

    def optimize2(self, fwd_data):
        optimize_start = time.time()

        input_names = ['spatial_features']

        output_names = self.dense_head.ordered_outp_names()
        print('Fused operations output names:', output_names)

        self.opt_fwd2 = OptimizedFwdPipeline2(self.backbone_2d, self.dense_head)
        self.opt_fwd2.eval()
        eager_outputs = self.opt_fwd2(fwd_data)

        generated_onnx=False
        onnx_path = self.model_cfg.BACKBONE_2D.OPT_PATH + '.onnx'
        if not os.path.exists(onnx_path):
            dynamic_axes = {
                "spatial_features": {
                    3: "width",
                },
            }
            for nm in output_names:
                dynamic_axes[nm] = {3 : "out_width"}

            torch.onnx.export(
                    self.opt_fwd2,
                    fwd_data,
                    onnx_path, input_names=input_names,
                    output_names=output_names, dynamic_axes=dynamic_axes,
                    opset_version=17,
                    #custom_opsets={"kucsl": 17}
            )
            generated_onnx=True

        sf = fwd_data 
        trt_path = self.model_cfg.BACKBONE_2D.OPT_PATH + '.engine'
        try:
            self.fused_ops2_trt = TRTWrapper(trt_path, input_names, output_names)
        except:
            self.fused_ops2_trt = None

        optimize_end = time.time()
        print(f'Optimization took {optimize_end-optimize_start} seconds.')
        self.optimization2_done = True
        if generated_onnx:
            print('ONNX files created, please run again after building TensorRt engines.')
            sys.exit(0)

    def get_iobinding(self, ort_session, inp_tensors, outp_tensors):
        typedict = {torch.float: np.float32,
                torch.bool: np.bool_,
                torch.int: np.int32,
                torch.long: np.int64}
        io_binding = ort_session.io_binding()
        for inp, tensor in zip(ort_session.get_inputs(), inp_tensors):
            io_binding.bind_input(
                    name=inp.name,
                    device_type='cuda',
                    device_id=0,
                    element_type=typedict[tensor.dtype],
                    shape=tuple(tensor.shape),
                    buffer_ptr=tensor.data_ptr()
                    )
        for outp, tensor in zip(ort_session.get_outputs(), outp_tensors):
            io_binding.bind_output(
                    name=outp.name,
                    device_type='cuda',
                    device_id=0,
                    element_type=typedict[tensor.dtype],
                    shape=tuple(tensor.shape),
                    buffer_ptr=tensor.data_ptr()
                    )

        return io_binding

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

    def dump_tensors(self, inputs_dict):
        for name, t in inputs_dict.items():
            print('torch', name, t.shape, t.dtype)
            data = t.cpu().contiguous().numpy() # t is a pytorch tensor
            print('numpy', name, data.shape, data.dtype)
            with open(f"deploy_files/{name}.bin", "wb") as f:
                data = np.ravel(data)
                if data.dtype == np.float32:
                    for d in data:
                        f.write(struct.pack("<f", d.item()))
                elif data.dtype == np.int64:
                    for d in data:
                        f.write(struct.pack("<q", d.item()))  # 'q' is for int64
                elif data.dtype == np.int32:
                    for d in data:
                        f.write(struct.pack("<i", d.item()))  # 'i' is for int32
                elif data.dtype == np.bool_:
                    for d in data:
                        f.write(struct.pack("<B", d.item()))  # 'B' is for unsigned char
                else:
                    raise ValueError(f"Unsupported data type: {data.dtype}")
        print('Done saving tensors.')

    def infer_ort(self, ort_session, inputs, sizes_of_outps, sync=False):
        # ONNX
        ro = ort.RunOptions()
        ro.add_run_config_entry("disable_synchronize_execution_providers", "1")
        outputs = [torch.empty(sz, device='cuda', dtype=torch.float) \
                for sz in sizes_of_outps]
        io_binding = self.get_iobinding(ort_session, inputs, outputs)
        ort_session.run_with_iobinding(io_binding, run_options=ro)
        if sync:
            torch.cuda.synchronize() # Do I have to? Doesn't matter a lot since rest is short
        return outputs



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
#
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

# ONNX RT conf
        #self.ort_out1_sizes = [eager_outputs.shape]
        #print('Optimized forward pipeline 1 output sizes:\n', self.ort_out1_sizes)

        #so = ort.SessionOptions()
        #so.log_severity_level = 1
        #so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        #so.optimized_model_filepath = f"{base_dir}/dsvt1_optimized.onnx" # speeds up initialization

        #if os.path.exists(so.optimized_model_filepath):
        #    onnx_path=so.optimized_model_filepath
        #self.ort_session1 = ort.InferenceSession(onnx_path, providers=self.ort_EP_list, sess_options=so)
        #self.ort_session1.enable_fallback()

        # Invoke JIT optimization
        #self.infer_ort(self.ort_session1, fwd_data, self.ort_out1_sizes, True)

#        self.ort_out2_sizes = [tuple(out.shape) for out in eager_outputs]
#        print('Optimized forward pipeline 2 output sizes:\n', self.ort_out2_sizes)
#        self.ort_out2_scales = [fwd_data.size(3) / sz[3] for sz in self.ort_out2_sizes]
#
#        so = ort.SessionOptions()
#        #so.log_severity_level = 1
#        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#        so.optimized_model_filepath = f"{base_dir}/dsvt2_optimized.onnx" # speeds up initialization
#        if os.path.exists(so.optimized_model_filepath):
#            onnx_path=so.optimized_model_filepath
#        self.ort_session2 = ort.InferenceSession(onnx_path, providers=self.ort_EP_list, sess_options=so)
#        self.ort_session2.enable_fallback()
#
#        # Invoke JIT optimization for all sizes, needed for cudnn benchmarking
#        for i in range(1, self.tcount+1):
#            dummy_dict = {'batch_size':1, 'spatial_features': fwd_data.detach().clone(),
#                    'chosen_tile_coords': torch.arange(i)}
#            dummy_dict = self.backbone_2d.prune_spatial_features(dummy_dict)
#            inp = dummy_dict['spatial_features']
#            out2_sizes = [(sz[0], sz[1], sz[2], int(inp.size(3) / scl)) \
#                    for sz, scl in zip(self.ort_out2_sizes, self.ort_out2_scales)]
#            self.infer_ort(self.ort_session2, [inp], out2_sizes, True)

        #self.ort_session1 = None
        #self.ort_out1_sizes = None
        #self.ort_session2 = None
        #self.ort_out2_sizes = None
        #self.ort_out2_scales = None

#        cuda_conf = {
#                'device_id': torch.cuda.current_device(),
#                'user_compute_stream': str(torch.cuda.current_stream().cuda_stream),
#                'arena_extend_strategy': 'kNextPowerOfTwo',
#                #'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
#                'cudnn_conv_algo_search': 'HEURISTIC',  #'EXHAUSTIVE',
#                'do_copy_in_default_stream': True,
#                'use_tf32': 1,
#        }
#
#        self.ort_EP_list= [#('TensorrtExecutionProvider', tensorrt_conf),
#                ('CUDAExecutionProvider', cuda_conf),
#                'CPUExecutionProvider']


#class OptimizedFwdPipeline1(torch.nn.Module):
#    def __init__(self, backbone_3d, map_to_bev):
#        super().__init__()
#        self.backbone_3d = backbone_3d
#        self.map_to_bev = map_to_bev
#
#    def forward(self,
#            voxel_feat : torch.Tensor, 
#            set_voxel_inds_tensor_shift_0 : torch.Tensor,
#            set_voxel_inds_tensor_shift_1 : torch.Tensor,
#            set_voxel_masks_tensor_shift_0 : torch.Tensor, 
#            set_voxel_masks_tensor_shift_1: torch.Tensor,
#            pos_embed_tensor : torch.Tensor,
#            voxel_coords : torch.Tensor) -> torch.Tensor:
#        output = self.backbone_3d(
#                voxel_feat, set_voxel_inds_tensor_shift_0, 
#                set_voxel_inds_tensor_shift_1, set_voxel_masks_tensor_shift_0,
#                set_voxel_masks_tensor_shift_1, pos_embed_tensor)
#        spatial_features = self.map_to_bev(1, output, voxel_coords) # 1 is batch size
#        return spatial_features

            #ORT
            #out2_sizes = [(sz[0], sz[1], sz[2], int(sf.size(3) / scl)) \
            #        for sz, scl in zip(self.ort_out2_sizes, self.ort_out2_scales)]
            #outputs = self.infer_ort(self.ort_session2, [sf], out2_sizes, True)

            #BASELINE
            #spatial_features_2d = self.backbone_2d(sf)
            #outputs = self.dense_head.forward_up_to_topk(spatial_features_2d)


