from .anytime_template_v2 import AnytimeTemplateV2
import torch
from easydict import EasyDict as edict
from ..model_utils import model_nms_utils

class TransFusionAnytime(AnytimeTemplateV2):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        torch.backends.cudnn.benchmark = True
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
            'Sched1': [],
            'Backbone3D':[],
            'Sched2': [],
            'MapToBEV': [],
            'Backbone2D': [],
            'TransfusionHead': [],
            'ProjectionNMS': []})

        self.enable_projection = True
        self.enable_splitting_projections = False
        self.cudnn_calibrated = False

    def forward(self, batch_dict):
        # We are going to do projection earlier so the
        # dense head can use its results for NMS
        if self.training:
            return self.forward_train(batch_dict)
        else:
            return self.forward_eval(batch_dict)

    def forward_eval(self, batch_dict):
        self.measure_time_start('VFE')
        batch_dict = self.vfe(batch_dict, model=self)
        self.measure_time_end('VFE')

        self.measure_time_start('Sched1')
        batch_dict = self.schedule1(batch_dict)
        self.measure_time_end('Sched1')

        self.measure_time_start('Backbone3D')
        batch_dict = self.backbone_3d(batch_dict)
        self.measure_time_end('Backbone3D')

        if self.is_calibrating():
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()

        self.measure_time_start('Sched2')
        batch_dict = self.schedule2(batch_dict)
        self.measure_time_end('Sched2')

        self.measure_time_start('MapToBEV')
        batch_dict = self.map_to_bev(batch_dict)
        self.measure_time_end('MapToBEV')

        if not self.cudnn_calibrated and torch.backends.cudnn.benchmark:
            self.calibrate_for_cudnn_benchmarking(batch_dict)

        self.measure_time_start('Backbone2D')
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.do_projection(batch_dict) # run in parallel with bb2d and dethead
        self.measure_time_end('Backbone2D')

        if self.is_calibrating():
            e2 = torch.cuda.Event(enable_timing=True)
            e2.record()
            batch_dict['bb2d_time_events'] = [e1, e2]

        # what if I consider this one as 2d backbone?
        self.measure_time_start('TransfusionHead')
        batch_dict['tcount'] = self.tcount
        batch_dict['sched_algo'] = self.sched_algo
        batch_dict = self.dense_head(batch_dict)
        self.measure_time_end('TransfusionHead')

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

        if self.is_calibrating():
            e3 = torch.cuda.Event(enable_timing=True)
            e3.record()
            batch_dict['detheadpost_time_events'] = [e2, e3]

        return batch_dict

    def forward_train(self, batch_dict):
        batch_dict = self.vfe(batch_dict, model=self)
        batch_dict = self.schedule1(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        loss, tb_dict, disp_dict = self.get_training_loss()

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
                'loss_trans': loss_trans.item(),
                **tb_dict
                }

        loss = loss_trans
        return loss, tb_dict, disp_dict


    def calibrate_for_cudnn_benchmarking(self, batch_dict):
        print('Calibrating bb2d and det head pre for cudnn benchmarking, max num tiles is',
                self.tcount)
        # Try out all different chosen tile sizes
        dummy_dict = {'batch_size':1, 'spatial_features': batch_dict['spatial_features']}
        print('Tensor size is:', batch_dict['spatial_features'].size())
        for i in range(1, self.tcount+1):
            dummy_dict['chosen_tile_coords'] = torch.arange(i)
            dummy_dict = self.backbone_2d(dummy_dict)
            inputs = dummy_dict['spatial_features_2d']
            self.dense_head.shared_conv(inputs)
        print('done.')
        self.cudnn_calibrated = True
