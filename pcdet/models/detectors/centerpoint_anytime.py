from .anytime_template import AnytimeTemplate

import torch
from sbnet.layers import ReduceMask

class CenterPointAnytime(AnytimeTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

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

    def produce_reduce_mask(self, data_dict):
        tile_coords = data_dict['chosen_tile_coords']
        batch_idx = torch.div(tile_coords, self.total_num_tiles, rounding_mode='trunc').short()
        row_col_idx = tile_coords - batch_idx * self.total_num_tiles
        row_idx = torch.div(row_col_idx, self.tcount[0], rounding_mode='trunc').short()
        col_idx = (row_col_idx - row_idx * self.tcount[1]).short()
        inds = torch.stack((batch_idx, col_idx, row_idx), dim=1)
        counts = torch.full((1,), inds.size(0), dtype=torch.int32)
        return ReduceMask(inds, counts)

    def forward(self, batch_dict):
        self.projection_init(batch_dict)

        self.measure_time_start('VFE')
        batch_dict = self.vfe(batch_dict, model=self)
        self.measure_time_end('VFE')

        # Produce the reduce mask in parallel in a seperate stream
        #with torch.cuda.stream(self.reduce_mask_stream):
        #    batch_dict['reduce_mask'] = self.produce_reduce_mask(batch_dict)

        self.measure_time_start('Backbone3D')
        batch_dict = self.backbone_3d(batch_dict)
        self.measure_time_end('Backbone3D')

        self.measure_time_start('MapToBEV')
        batch_dict = self.map_to_bev(batch_dict)
        #self.reduce_mask_stream.synchronize()
        self.measure_time_end('MapToBEV')

        self.measure_time_start('Backbone2D')
        batch_dict['reduce_mask'] = self.produce_reduce_mask(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        self.measure_time_end('Backbone2D')

        self.measure_time_start('CenterHead')
        batch_dict = self.dense_head(batch_dict)
        self.measure_time_end('CenterHead')

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            self.measure_time_start('Projection')
            batch_dict = self.projection(batch_dict)
            self.measure_time_end('Projection')

            return batch_dict

    def calibrate(self):
        super().calibrate("calib_raw_data_centerpoint.json")
        return None
