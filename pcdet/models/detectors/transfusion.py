from .detector3d_template import Detector3DTemplate
import torch
from easydict import EasyDict as edict
from ..model_utils import model_nms_utils

class TransFusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        torch.backends.cudnn.benchmark = False
        if torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark_limit = 0
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed(0)
        self.module_list = self.build_networks()

        self.vfe, self.backbone_3d, self.map_to_bev, self.backbone_2d, \
                self.dense_head = self.module_list
        self.update_time_dict( {
            'VFE': [],
            'Backbone3D':[],
            'MapToBEV': [],
            'Backbone2D': [],
            'TransfusionHead': [],
            'ProjectionNMS': []})

        self.enable_projection = False
        self.enable_splitting_projections = False

    def forward(self, batch_dict):
        self.measure_time_start('VFE')
        batch_dict = self.vfe(batch_dict)
        self.measure_time_end('VFE')

        self.measure_time_start('Backbone3D')
        batch_dict = self.backbone_3d(batch_dict)
        self.measure_time_end('Backbone3D')

        self.measure_time_start('MapToBEV')
        batch_dict = self.map_to_bev(batch_dict)
        self.measure_time_end('MapToBEV')

        self.measure_time_start('Backbone2D')
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.do_projection(batch_dict) # run in parallel with bb2d and dethead
        self.measure_time_end('Backbone2D')

        self.measure_time_start('TransfusionHead')
        batch_dict = self.dense_head(batch_dict)
        self.measure_time_end('TransfusionHead')

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                    'loss': loss
                    }
            return ret_dict, tb_dict, disp_dict
        else:
            #pred_dicts, recall_dicts = self.post_processing(batch_dict)
            #return pred_dicts, recall_dicts

            # ? merge the detections and forecasted objects ?
            final_pred_dicts = batch_dict['final_box_dicts']
            #for d in final_pred_dicts: # per batch
            #    for k,v in d.items():
            #        print(k, v.size())

            self.measure_time_start('ProjectionNMS')
            if 'projections_nms' in batch_dict:
                proj_dict = batch_dict['projections_nms']

                boxes = torch.cat((final_pred_dicts[0]['pred_boxes'], proj_dict['pred_boxes']))
                scores = torch.cat((final_pred_dicts[0]['pred_scores'], proj_dict['pred_scores']))
                labels = torch.cat((final_pred_dicts[0]['pred_labels']-1, proj_dict['pred_labels']))

                nms_config = edict({
                    'NMS_PRE_MAXSIZE': 1000,
                    'NMS_POST_MAXSIZE': 200,
                    'NMS_THRESH': 0.5,
                    'NMS_TYPE': 'nms_gpu'
                })    
                
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=scores, box_preds=boxes,
                    nms_config=nms_config,
                    score_thresh=None
                )

                final_pred_dicts[0]['pred_boxes'] = boxes[selected]
                final_pred_dicts[0]['pred_scores'] = selected_scores
                final_pred_dicts[0]['pred_labels'] = labels[selected]+1

            self.measure_time_end('ProjectionNMS')

            return batch_dict

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
                'loss_trans': loss_trans.item(),
                **tb_dict
                }

        loss = loss_trans
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


