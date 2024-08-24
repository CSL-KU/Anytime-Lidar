from .detector3d_template import Detector3DTemplate
import torch
import time
import onnx
import os
import sys
from typing import List
from ..model_utils.tensorrt_utils.trtwrapper import TRTWrapper

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
        return self.dense_head.forward_up_to_topk(data_dict['spatial_features_2d'])

class PillarNetOpt(Detector3DTemplate):
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

        self.update_time_dict({
            'VFE' : [],
            'Backbone3D': [],
            'DenseConvsPipeline':[],
            'CenterHead-Topk': [],
            'CenterHead-GenBox': [],
        })

        self.vfe, self.backbone_3d, self.backbone_2d, self.dense_head = self.module_list
        self.inf_stream = torch.cuda.Stream()
        self.optimization1_done = False

    def forward(self, batch_dict):
        with torch.cuda.stream(self.inf_stream):
            self.measure_time_start('VFE')
            batch_dict = self.vfe.range_filter(batch_dict)
            points = batch_dict['points']
            batch_dict['voxel_coords'], batch_dict['voxel_features'] = self.vfe(points)
            batch_dict['pillar_features'] = batch_dict['voxel_features']
            batch_dict['pillar_coords'] = batch_dict['voxel_coords']
            self.measure_time_end('VFE')

            self.measure_time_start('Backbone3D')
            x_conv4 = self.backbone_3d.forward_up_to_dense(batch_dict)
            self.measure_time_end('Backbone3D')

            if not self.optimization1_done:
                self.optimize1(x_conv4)
                self.dense_head_scrpt = torch.jit.script(self.dense_head)

            self.measure_time_start('DenseConvsPipeline')

            if self.dense_convs_trt is not None:
                outputs = self.dense_convs_trt({'x_conv4': x_conv4})
                outputs = [outputs[nm] for nm in self.opt_dense_convs_output_names]
            else:
                outputs = self.opt_dense_convs(sf)
            batch_dict["pred_dicts"] = self.dense_head.convert_out_to_batch_dict(outputs)
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

    def optimize1(self, fwd_data):
        optimize_start = time.time()

        input_names = ['x_conv4']

        self.opt_dense_convs_output_names = [name + str(i) for i in range(self.dense_head.num_det_heads) \
                for name in self.dense_head.ordered_outp_names()]
        print('Fused operations output names:', self.opt_dense_convs_output_names)

        self.opt_dense_convs = DenseConvsPipeline(self.backbone_3d, self.backbone_2d, self.dense_head)
        self.opt_dense_convs.eval()

        generated_onnx=False
        onnx_path = self.model_cfg.BACKBONE_2D.OPT_PATH + '.onnx'
        if not os.path.exists(onnx_path):
            torch.onnx.export(
                    self.opt_dense_convs,
                    fwd_data,
                    onnx_path, input_names=input_names,
                    output_names=self.opt_dense_convs_output_names,
                    opset_version=17,
                    #custom_opsets={"kucsl": 17}
            )
            generated_onnx=True

        sf = fwd_data 
        trt_path = self.model_cfg.BACKBONE_2D.OPT_PATH + '.engine'
        try:
            self.dense_convs_trt = TRTWrapper(trt_path, input_names, self.opt_dense_convs_output_names)
        except:
            print('TensorRT wrapper for fused_ops2 throwed exception, using eager mode')
            eager_outputs = self.opt_dense_convs(fwd_data) # for calibration
            self.dense_convs_trt = None

        optimize_end = time.time()
        print(f'Optimization took {optimize_end-optimize_start} seconds.')
        if generated_onnx:
            print('ONNX files created, please run again after creating TensorRT engines.')
            sys.exit(0)

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

    def calibrate(self, batch_size=1):
        return super().calibrate(1)

