from .detector3d_template import Detector3DTemplate
import torch
import time
import onnx
import os
import sys
from typing import List
from ..model_utils.tensorrt_utils.trtwrapper import TRTWrapper, create_trt_engine
from ...utils import common_utils

class DenseConvsPipeline(torch.nn.Module):
    def __init__(self, backbone_2d, dense_head):
        super().__init__()
        self.backbone_2d = backbone_2d
        self.dense_head = dense_head

    def forward(self, spatial_features : torch.Tensor) -> List[torch.Tensor]:
        spatial_features_2d = self.backbone_2d(spatial_features)
        return self.dense_head.forward_up_to_topk(spatial_features_2d)

class CenterPointOpt(Detector3DTemplate):
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

        self.update_time_dict( {
            'VFE' : [],
            'MapToBEV' : [],
        })
        self.use_pillars = (self.model_cfg.get('BACKBONE_3D', None) is None)
        if not self.use_pillars:
            self.update_time_dict({'Backbone3D': []})
        self.update_time_dict( {
            'DenseConvsPipeline':[],
            'CenterHead-Topk': [],
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
        self.opt_done = False
        self.dense_convs_trt = None
        self.trt_outputs = None # Since output size of trt is fixed, use buffered
        print('Model size is:', self.get_model_size_MB(), 'MB')
        self.filter_pc_range =  self.vfe.point_cloud_range + \
                torch.tensor([0.01, 0.01, 0.01, -0.01, -0.01, -0.01]).cuda()
        self.traced_vfe = None

    def forward(self, batch_dict):
        assert not self.training
        with torch.cuda.stream(self.inf_stream):
            self.measure_time_start('VFE')
            batch_dict['points'] = common_utils.pc_range_filter(batch_dict['points'],
                                self.filter_pc_range)
            points = batch_dict['points']
            if self.traced_vfe is None:
                self.traced_vfe = torch.jit.trace(self.vfe, points)
            batch_dict['voxel_coords'], batch_dict['voxel_features'] = self.traced_vfe(points)
            batch_dict['pillar_features'] = batch_dict['voxel_features']
            batch_dict['pillar_coords'] = batch_dict['voxel_coords']
            self.measure_time_end('VFE')

            if not self.use_pillars:
                self.measure_time_start('Backbone3D')
                batch_dict = self.backbone_3d(batch_dict)
                self.measure_time_end('Backbone3D')
                self.measure_time_start('MapToBEV')
                batch_dict = self.map_to_bev(batch_dict)
                self.measure_time_end('MapToBEV')
            else:
                self.measure_time_start('MapToBEV')
                batch_dict['spatial_features'] = self.map_to_bev_scrpt(batch_dict['pillar_features'],
                        batch_dict['voxel_coords'], batch_dict['batch_size'])
                self.measure_time_end('MapToBEV')

            if not self.opt_done:
                self.optimize(batch_dict['spatial_features'])
                self.dense_head_scrpt = torch.jit.script(self.dense_head)

            self.measure_time_start('DenseConvsPipeline')
            sf = batch_dict['spatial_features']

            if self.dense_convs_trt is not None:
                self.trt_outputs = self.dense_convs_trt({'spatial_features': sf}, self.trt_outputs)
                outputs = [self.trt_outputs[nm] for nm in self.opt_dense_convs_output_names]
            else:
                outputs = self.opt_fwd(sf)
            batch_dict["pred_dicts"] = self.dense_head.convert_out_to_pred_dicts(outputs)
            self.measure_time_end('DenseConvsPipeline')

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

    def optimize(self, fwd_data):
        optimize_start = time.time()

        input_names = ['spatial_features']

        self.opt_dense_convs_output_names = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                for name in self.dense_head.ordered_outp_names()]
        print('Fused operations output names:', self.opt_dense_convs_output_names)

        self.opt_fwd = DenseConvsPipeline(self.backbone_2d, self.dense_head)
        self.opt_fwd.eval()

        onnx_path = self.model_cfg.ONNX_PATH + '.onnx'
        if not os.path.exists(onnx_path):
            torch.onnx.export(
                    self.opt_fwd,
                    fwd_data,
                    onnx_path,
                    input_names=input_names,
                    output_names=self.opt_dense_convs_output_names,
                    opset_version=17,
                    #custom_opsets={"kucsl": 17}
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

        self.opt_done = True

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

