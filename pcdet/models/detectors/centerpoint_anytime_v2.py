from .anytime_template_v2 import AnytimeTemplateV2

import torch

class CenterPointAnytimeV2(AnytimeTemplateV2):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        if self.model_cfg.get('BACKBONE_3D', None) is None:
            #pillar
            self.is_voxel_enc=False
            self.vfe, self.map_to_bev, self.backbone_2d, \
                    self.dense_head = self.module_list
            self.update_time_dict( {
                    'VFE': [],
                    'Sched': [],
                    'MapToBEV': [],
                    'Backbone2D': [],
                    'CenterHead': [],
                    'Projection': []})
        else:
            #voxel
            self.is_voxel_enc=True
            self.vfe, self.backbone_3d, self.map_to_bev, self.backbone_2d, \
                    self.dense_head = self.module_list
            self.update_time_dict( {
                    'VFE': [],
                    'Sched': [],
                    'Backbone3D':[],
                    'MapToBEV': [],
                    'Backbone2D': [],
                    'CenterHead': [],
                    'Projection': []})
        self.calibrated = False

    def forward(self, batch_dict):
        # We are going to do projection earlier so the
        # dense head can use its results for NMS
        if self.training:
            return self.forward_train(batch_dict)
        else:
            return self.forward_eval(batch_dict)

    def forward_eval(self, batch_dict):
        self.measure_time_start('Projection')
        batch_dict = self.projection(batch_dict)
        self.measure_time_end('Projection')
        self.measure_time_start('VFE')
        batch_dict = self.vfe(batch_dict, model=self)
        self.measure_time_end('VFE')
        batch_dict = self.schedule1(batch_dict)
        if self.is_voxel_enc:
            self.measure_time_start('Backbone3D')
            batch_dict = self.backbone_3d(batch_dict)
            self.measure_time_end('Backbone3D')
        batch_dict = self.schedule2(batch_dict)
        self.measure_time_start('MapToBEV')
        batch_dict = self.map_to_bev(batch_dict)
        self.measure_time_end('MapToBEV')

        if not self.calibrated:
            self.calibrate_for_cudnn_benchmarking(batch_dict)

        self.measure_time_start('Backbone2D')
        batch_dict = self.backbone_2d(batch_dict)
        self.measure_time_end('Backbone2D')
        self.measure_time_start('CenterHead')
        batch_dict = self.dense_head.forward_eval_pre(batch_dict)
        #torch.cuda.synchronize()
        batch_dict = self.schedule3(batch_dict)
        batch_dict = self.dense_head.forward_eval_post(batch_dict)
        self.measure_time_end('CenterHead')

        return batch_dict

    def forward_train(self, batch_dict):
        batch_dict = self.vfe(batch_dict, model=self)
        if self.is_voxel_enc:
            batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        loss, tb_dict, disp_dict = self.get_training_loss()

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict


    def calibrate_for_cudnn_benchmarking(self, batch_dict):
        print('Calibrating bb2d and det head pre for cudnn benchmarking, max num tiles is',
                self.tcount, ' ...')
        # Try out all different chosen tile sizes
        dummy_dict = {'batch_size':1, 'spatial_features': batch_dict['spatial_features']}
        for i in range(1, self.tcount+1):
            dummy_dict['chosen_tile_coords'] = torch.arange(i)
            dummy_dict = self.backbone_2d(dummy_dict)
            self.dense_head.forward_eval_pre(dummy_dict)
        print('done.')
        self.calibrated = True
