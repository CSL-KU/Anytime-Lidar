from .detector3d_template import Detector3DTemplate
from .mural_calibrator import  MURALCalibrator
import torch
import time
import onnx
import os
import sys
import platform
from typing import List
from ..model_utils.tensorrt_utils.trtwrapper import TRTWrapper, create_trt_engine
from ...utils import common_utils

class DenseOps(torch.nn.Module):
    def __init__(self, backbone_3d, backbone_2d, dense_head):
        super().__init__()
        self.backbone_3d = backbone_3d
        self.backbone_2d = backbone_2d
        self.dense_head = dense_head

    def forward(self, x_conv4 : torch.Tensor) -> List[torch.Tensor]:
        x_conv5 = self.backbone_3d.forward_dense(x_conv4)
        data_dict = self.backbone_2d({"multi_scale_2d_features" : 
            {"x_conv4": x_conv4, "x_conv5": x_conv5}})
        return self.dense_head.forward_up_to_topk(data_dict['spatial_features_2d'])

class PillarNetOpt(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
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

        self.update_time_dict({
            'Sched': [],
            'VFE' : [],
            'Backbone3D': [],
            'DenseOps':[],
            'CenterHead-Topk': [],
            'CenterHead-GenBox': [],
        })

        self.vfe, self.backbone_3d, self.backbone_2d, self.dense_head = self.module_list
        print('Model size is:', self.get_model_size_MB(), 'MB')
        self.inf_stream = torch.cuda.Stream()
        self.trt_outputs = None # Since output size of trt is fixed, use buffered
        self.optimization1_done = False
        self.dense_convs_trt = None
        self.filter_pc_range =  self.vfe.point_cloud_range + \
                torch.tensor([0.01, 0.01, 0.01, -0.01, -0.01, -0.01]).cuda()
        self.calib_pc_range = self.filter_pc_range.clone()
        self.traced_vfe = None

        self.resolution_dividers = [1.0]
        self.res_idx = 0
        num_slices = 1
        self.dense_conv_opt_on = False
        self.model_name = self.model_cfg.NAME + '_' + self.model_cfg.NAME_POSTFIX
        self.dense_inp_slice_sz = int(self.dataset.grid_size[0] /  \
                self.backbone_3d.sparse_outp_downscale_factor())
        self.calibrators = [MURALCalibrator(self, self.res_idx, num_slices)]

    def forward(self, batch_dict):
        if self.training:
            batch_dict['points'] = common_utils.pc_range_filter(batch_dict['points'],
                                self.calib_pc_range if self.is_calibrating() else
                                self.filter_pc_range)
            points = batch_dict['points']
            batch_dict['voxel_coords'], batch_dict['voxel_features'] = self.vfe(points)
            batch_dict['pillar_features'] = batch_dict['voxel_features']
            batch_dict['pillar_coords'] = batch_dict['voxel_coords']

            batch_dict = self.backbone_3d(batch_dict)
            batch_dict = self.backbone_2d(batch_dict)
            batch_dict = self.dense_head(batch_dict)
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }

            return ret_dict, tb_dict, disp_dict
        else:
            return self.forward_eval(batch_dict)

    def forward_eval(self, batch_dict):
        with torch.cuda.stream(self.inf_stream):
            self.measure_time_start('Sched')
            if self.is_calibrating() and self.calibrators[self.res_idx].repeat_points > 1:
                pts = batch_dict['points']
                batch_dict['points'] = pts.repeat(self.calibrators[self.res_idx].repeat_points, 1)
            batch_dict['points'] = common_utils.pc_range_filter(batch_dict['points'],
                                self.calib_pc_range if self.is_calibrating() else
                                self.filter_pc_range)
            self.measure_time_end('Sched')
            self.measure_time_start('VFE')
            points = batch_dict['points']
            if self.traced_vfe is None:
                self.traced_vfe = torch.jit.trace(self.vfe, points)
            batch_dict['voxel_coords'], batch_dict['voxel_features'] = self.traced_vfe(points)
            batch_dict['pillar_features'] = batch_dict['voxel_features']
            batch_dict['pillar_coords'] = batch_dict['voxel_coords']
            self.measure_time_end('VFE')

            self.measure_time_start('Backbone3D')
            if self.is_calibrating():
                batch_dict['record_time'] = True # returns bb3d_layer_time_events
            batch_dict['record_int_vcounts'] = True # returns bb3d_num_voxels, no overhead
            batch_dict = self.backbone_3d.forward_up_to_dense(batch_dict)
            x_conv4 = batch_dict['x_conv4_out']
            self.measure_time_end('Backbone3D')

            if not self.optimization1_done:
                self.optimize1(x_conv4)
                self.dense_head_scrpt = torch.jit.script(self.dense_head)

            self.measure_time_start('DenseOps')

            if self.dense_convs_trt is not None:
                #outputs = self.dense_convs_trt({'x_conv4': x_conv4})
                self.trt_outputs = self.dense_convs_trt({'x_conv4': x_conv4}, self.trt_outputs)
                outputs = [self.trt_outputs[nm] for nm in self.opt_dense_convs_output_names]
            else:
                outputs = self.opt_dense_convs(x_conv4)
            batch_dict["pred_dicts"] = self.dense_head.convert_out_to_pred_dicts(outputs)
            self.measure_time_end('DenseOps')

            self.measure_time_start('CenterHead-Topk')
            topk_outputs = self.dense_head_scrpt.forward_topk(batch_dict["pred_dicts"])
            self.measure_time_end('CenterHead-Topk')
            self.measure_time_start('CenterHead-GenBox')
            batch_dict['final_box_dicts'] = self.dense_head_scrpt.forward_genbox(
                    batch_dict['batch_size'], batch_dict["pred_dicts"], topk_outputs, None)
            self.measure_time_end('CenterHead-GenBox')


            batch_dict['tensor_slice_inds'] = [0, 0] # we need this just for time prediction

            # let the hooks of parent class handle this
            return batch_dict

    def optimize1(self, fwd_data):
        optimize_start = time.time()

        input_names = ['x_conv4']

        self.opt_dense_convs_output_names = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                for name in self.dense_head.ordered_outp_names()]
        print('Fused operations output names:', self.opt_dense_convs_output_names)

        self.opt_dense_convs = DenseOps(self.backbone_3d, self.backbone_2d, self.dense_head)
        self.opt_dense_convs.eval()

        onnx_path = self.model_cfg.ONNX_PATH + '.onnx'
        if not os.path.exists(onnx_path):
            torch.onnx.export(
                    self.opt_dense_convs,
                    fwd_data,
                    onnx_path, input_names=input_names,
                    output_names=self.opt_dense_convs_output_names,
                    opset_version=17,
            )

        power_mode = os.getenv('PMODE', 'UNKNOWN_POWER_MODE')
        if power_mode == 'UNKNOWN_POWER_MODE':
            print('WARNING! Power mode is unknown. Please export PMODE.')

        tokens = self.model_cfg.ONNX_PATH.split('/')
        trt_path = '/'.join(tokens[:-2]) + f'/trt_engines/{power_mode}/{tokens[-1]}.engine'
        print('Trying to load trt engine at', trt_path)
        try:
            self.dense_convs_trt = TRTWrapper(trt_path, input_names, self.opt_dense_convs_output_names)
        except:
            print('TensorRT wrapper for fused_ops throwed exception, creating the engine')
            create_trt_engine(onnx_path, trt_path, input_names[0])
            self.dense_convs_trt = TRTWrapper(trt_path, input_names, self.opt_dense_convs_output_names)

        optimize_end = time.time()
        print(f'Optimization took {optimize_end-optimize_start} seconds.')

        self.optimization1_done = True

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

    #def calibrate(self, batch_size=1):
    #    return super().calibrate(1)

    def calibrate(self, batch_size=1):
        collect_calib_data = False
        calib_fname = ""

        power_mode = os.getenv('PMODE', 'UNKNOWN_POWER_MODE')
        calib_fname = f"calib_files/{self.model_name}_{power_mode}.json"
        print('Trying to load calib file:', calib_fname)
        try:
            self.calibrators[0].read_calib_data(calib_fname)
        except OSError:
            collect_calib_data = True

        self.calibration_on()
        print(f'Calibrating baseline model...')
        super().calibrate(1)

        if collect_calib_data:
            self.calibrators[0].collect_data(calib_fname)
        self.calibration_off()

        self.clear_stats()

        if collect_calib_data:
            print('Collected calib data, exiting...')
            sys.exit()

        return None
