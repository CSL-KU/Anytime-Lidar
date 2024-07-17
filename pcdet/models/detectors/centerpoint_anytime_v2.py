from .anytime_template_v2 import AnytimeTemplateV2

import torch
import torch_tensorrt

# This class is created for tensorrt.compile to optimize both modules together
class DenseConvsToHeatmap(torch.nn.Module):
    def __init__(self, backbone2d, dense_head):
        super().__init__()
        self.backbone2d = backbone2d
        self.dense_head = dense_head

    def forward(self, spatial_features):
        spatial_features_2d = self.backbone2d(spatial_features)
        shr_conv_outp, heatmaps = self.dense_head.forward_eval_conv(spatial_features_2d)
        return shr_conv_outp, heatmaps

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
                    'Sched1': [],
                    'Sched2': [],
                    'MapToBEV': [],
                    #'Backbone2D': [],
                    #'CenterHead-Pre': [],
                    'BB2D-and-CH-pre': [],
                    'CenterHead-Topk': [],
                    'CenterHead-Post': [],
                    'CenterHead-GenBox': [],
                    'CenterHead': []})
        else:
            #voxel
            self.is_voxel_enc=True
            self.vfe, self.backbone_3d, self.map_to_bev, self.backbone_2d, \
                    self.dense_head = self.module_list
            self.update_time_dict( {
                    'VFE': [],
                    'Sched1': [],
                    'Backbone3D':[],
                    'Sched2': [],
                    'MapToBEV': [],
                    #'Backbone2D': [],
                    #'CenterHead-Pre': [],
                    'BB2D-and-CH-pre': [],
                    'CenterHead-Topk': [],
                    'CenterHead-Post': [],
                    'CenterHead-GenBox': [],
                    'CenterHead': []})

        self.combined_dense_convs = None
        self.compiled_bb2ds = [None] * self.tcount
        self.inf_stream = torch.cuda.Stream()

    def forward(self, batch_dict):
        if self.training:
            return self.forward_train(batch_dict)
        else:
            # Tensorrt doesn't like the default stream, so I
            # am using another one
            with torch.cuda.stream(self.inf_stream):
                return self.forward_eval(batch_dict)

    def forward_eval(self, batch_dict):
        self.measure_time_start('VFE')
        batch_dict = self.vfe(batch_dict, model=self)
        self.measure_time_end('VFE')
        self.measure_time_start('Sched1')
        batch_dict = self.schedule1(batch_dict)
        self.measure_time_end('Sched1')
        if self.is_voxel_enc:
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

        if self.combined_dense_convs is None:
            self.compile_models(batch_dict)

        self.measure_time_start('BB2D-and-CH-pre')
        if not streaming_eval:
            batch_dict = self.do_projection(batch_dict)
        batch_dict = self.backbone_2d.prune_spatial_features(batch_dict)
        nt = batch_dict['num_tiles_in_sf']
        batch_dict['shr_conv_outp'], batch_dict['heatmaps'] = self.compiled_bb2ds[nt-1](
                batch_dict['spatial_features'])

        batch_dict = self.dense_head.scat_heatmaps_and_gen_pred_dicts(batch_dict)
        self.measure_time_end('BB2D-and-CH-pre')
        if self.is_calibrating():
            e2 = torch.cuda.Event(enable_timing=True)
            e2.record()
            batch_dict['bb2d_time_events'] = [e1, e2]
        streaming_eval = self.model_cfg.STREAMING_EVAL
        self.measure_time_start('CenterHead')
        self.measure_time_start('CenterHead-Topk')
        batch_dict = self.dense_head.forward_eval_topk(batch_dict)
        self.measure_time_end('CenterHead-Topk')

        self.measure_time_start('CenterHead-Post')
        batch_dict = self.dense_head.forward_eval_post(batch_dict)
        self.measure_time_end('CenterHead-Post')
        self.measure_time_start('CenterHead-GenBox')
        pred_dicts = self.dense_head.generate_predicted_boxes_eval(
            batch_dict['batch_size'], batch_dict['pred_dicts'],
            batch_dict.get('projections_nms', None),
            do_nms=(not streaming_eval)
        )
        batch_dict['final_box_dicts'] = pred_dicts
        if streaming_eval:
            batch_dict = self.do_projection(batch_dict) # Project ALL
            batch_dict = self.dense_head.nms_after_gen(batch_dict)
        self.measure_time_end('CenterHead-GenBox')
        self.measure_time_end('CenterHead')

        if self.is_calibrating():
            e3 = torch.cuda.Event(enable_timing=True)
            e3.record()
            batch_dict['detheadpost_time_events'] = [e2, e3]

        return batch_dict

    def forward_train(self, batch_dict):
        batch_dict = self.vfe(batch_dict, model=self)
        batch_dict = self.schedule1(batch_dict)
        if self.is_voxel_enc:
            batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        batch_dict['spatial_features_2d'] = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        loss, tb_dict, disp_dict = self.get_training_loss()

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict


    def compile_models(self, batch_dict):
        print('Compiling bb2d and det head pre convs, max num tiles is' ,self.tcount, ' ...')
        # Try out all different chosen tile sizes

        bb2d_name = self.model_cfg.get('BACKBONE_2D').NAME
        # maybe add centerhead name?
        self.combined_dense_convs = DenseConvsToHeatmap(self.backbone_2d, self.dense_head)
        sf = batch_dict['spatial_features']
        for i in range(1, self.tcount+1):
            dummy_dict = {'batch_size':1, 'spatial_features': sf.detach().clone(),
                    'chosen_tile_coords': torch.arange(i)}
            dummy_dict = self.backbone_2d.prune_spatial_features(dummy_dict)
            inp = (dummy_dict['spatial_features'],)
            #load model if available, otherwise compile
            sz = "x".join([str(s) for s in inp[0].size()])
            fname = f"trt-models/cp-valo-{bb2d_name}-{sz}.ep"
            try:
                self.compiled_bb2ds[i-1] = torch.export.load(fname).module()
                print(f'Loaded compiled 2DBB model for size {sz} ...')
            except:
                s = '#' * 64
                print(f'{s}\nCompiling for size {sz} ...\n{s}')

                # 1 << 22 is 4 GB
                self.compiled_bb2ds[i-1] = torch_tensorrt.compile(self.combined_dense_convs,
                        ir='dynamo', inputs=inp) #, workspace_size=1 << 22)
                torch_tensorrt.save(self.compiled_bb2ds[i-1], fname, inputs=inp)
