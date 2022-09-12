from .detector3d_template import Detector3DTemplate, pre_forward_hook
import torch
from nuscenes.nuscenes import NuScenes
from sbnet.layers import ReduceMask
import time
import sys
import pickle
import json
import numpy as np
#from sklearn.model_selection import train_test_split

from ...ops.cuda_projection import cuda_projection
from ...ops.cuda_point_tile_mask import cuda_point_tile_mask

# Define the neural network
class PostSchedWCETPred(torch.nn.Module):
    def __init__(self):
        super(PostSchedWCETPred, self).__init__()
        self.fc1 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

#https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40
class StandardScaler:
    def __init__(self, mean=0., std=0., epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions.
	The module does not expect the
        tensors to be of any specific shape; as long as the features are the last
	dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features.
		The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = torch.tensor(mean, dtype=torch.float)
        self.std = torch.tensor(std, dtype=torch.float)
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, values):
        return values * (self.std + self.epsilon) + self.mean

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std= self.std.cuda()
        return self

    def get_params(self):
        return (self.mean, self.std, self.epsilon)

    def set_params(self, params):
        self.mean, self.std, self.epsilon = params

class CenterPointAnytime(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.model_cfg.BACKBONE_2D.TILE_COUNT = self.model_cfg.TILE_COUNT
        self.model_cfg.DENSE_HEAD.TILE_COUNT = self.model_cfg.TILE_COUNT
        self.module_list = self.build_networks()
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed(0)

        if self.model_cfg.get('BACKBONE_3D', None) is None:
            #pillar
            self.is_voxel_enc=False
            self.vfe, self.map_to_bev, self.backbone_2d, \
                    self.dense_head = self.module_list
            self.update_time_dict( {
                    'VFE': [],
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
                    'Backbone3D':[],
                    'MapToBEV': [],
                    'Backbone2D': [],
                    'CenterHead': [],
                    'Projection': []})

        ################################################################################
        #self.tcount= torch.tensor(self.model_cfg.TILE_COUNT).long().cuda()
        self.tcount = self.model_cfg.TILE_COUNT
        self.total_num_tiles = self.tcount[0] * self.tcount[1]

        #Tile prios are going to be updated dynamically, initially all tiles have equal priority
        self.tile_prios = torch.full((self.total_num_tiles,), \
                self.total_num_tiles//2, dtype=torch.long, device='cuda')
        #self.tile_prios = torch.randint(0, self.total_num_tiles, (self.total_num_tiles,), \
        #        dtype=torch.long, device='cuda')

        # This number will be determined by the scheduling algorithm initially for each input
        self.last_tile_coord = 0
        self.reduce_mask_stream = torch.cuda.Stream()
        self.tile_size_voxels = torch.from_numpy(self.dataset.grid_size[:2] / self.tcount).cuda().long()

        ####Projection###
        self.enable_projection = True
        self.token_to_scene = {}
        self.token_to_ts = {}
        with open('token_to_pos.json', 'r') as handle:
            self.token_to_pose = json.load(handle)

        for k, v in self.token_to_pose.items():
            cst, csr, ept, epr = v['cs_translation'],  v['cs_rotation'], \
                    v['ep_translation'], v['ep_rotation']
            # convert time stamps to seconds
            # 3 4 3 4
            self.token_to_pose[k] = torch.tensor((*cst, *csr, *ept, *epr), dtype=torch.float)
            self.token_to_ts[k] = torch.tensor((v['timestamp'],), dtype=torch.long)
            self.token_to_scene[k] = v['scene']

        self.past_detections = {'num_dets': []}
        # Poses include [cst(3) csr(4) ept(3) epr(4)]
        self.past_poses = torch.zeros([0, 14], dtype=torch.float)
        self.past_ts = torch.zeros([0], dtype=torch.long)
        self.det_timeout_limit = int(0.5 * 1000000) # in microseconds
        self.prev_scene_token = ''
        ################################################################################

        self.calibrating_now = False
        self.calib_tile_idx_start = 1
        self.add_dict = self._eval_dict['additional']
        self.add_dict['total_num_tiles'] = self.total_num_tiles
        for k in ('num_voxels', 'num_tiles', 'PostSched'):
            self.add_dict[k] = []

        # these values needs calibration
        self.post_sched_pred_net = PostSchedWCETPred().cuda()
        self.projection_wcet = 0.002
        self.pred_net_scaler_in = StandardScaler()
        #self.pred_net_scaler_out = StandardScaler()
        self.pred_net_time_stats = {'99perc':0.0}

        print(self)

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
        self.latest_token = batch_dict['metadata'][0]['token']
        scene_token = self.token_to_scene[self.latest_token]
        if scene_token != self.prev_scene_token:
            if self.enable_projection:
                self.projection_reset()
            self.last_tile_coord = 0
            self.prev_scene_token = scene_token

        if self.calibrating_now:
            return self.calib_forward(batch_dict)
            #batch_dict['final_box_dicts'] = [self.get_empty_det_dict()]
            #return batch_dict

        if self.enable_projection and batch_dict['batch_size'] == 1:
            self.cur_pose = self.token_to_pose[self.latest_token]
            self.cur_ts = self.token_to_ts[self.latest_token]

        self.measure_time_start('VFE')
        batch_dict = self.vfe(batch_dict, model=self)
        self.measure_time_end('VFE')

        # Produce the reduce mask in parallel in a seperate stream
        with torch.cuda.stream(self.reduce_mask_stream):
            batch_dict['reduce_mask'] = self.produce_reduce_mask(batch_dict)

        if self.is_voxel_enc:
            self.measure_time_start('Backbone3D')
            batch_dict = self.backbone_3d(batch_dict)
            self.measure_time_end('Backbone3D')

        self.measure_time_start('MapToBEV')
        batch_dict = self.map_to_bev(batch_dict)
        self.reduce_mask_stream.synchronize()
        self.measure_time_end('MapToBEV')

        self.measure_time_start('Backbone2D')
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
            if self.enable_projection:
                torch.cuda.synchronize()
                self.measure_time_start('Projection')
                batch_dict = self.projection(batch_dict)
                self.measure_time_end('Projection')


            return batch_dict

    #TODO det_timeout_limit needs to be calibrated
    def projection(self, batch_dict):
        # First, remove the outdated detections
        num_dets, dets_to_rm = self.past_detections['num_dets'], []
        pred_dicts = batch_dict['final_box_dicts']
        while num_dets:
            # timestamp comparison
            if (self.cur_ts[0] - self.past_ts[len(dets_to_rm)]) <= self.det_timeout_limit:
                break
            dets_to_rm.append(num_dets.pop(0))
        if dets_to_rm:
            self.past_poses = self.past_poses[len(dets_to_rm):]
            self.past_ts = self.past_ts[len(dets_to_rm):]
            dets_to_rm_sum = sum(dets_to_rm)
            for k in ('pred_boxes', 'pred_scores', 'pred_labels', 'pose_idx', 'tile_inds'):
                self.past_detections[k] = self.past_detections[k][dets_to_rm_sum:]
            self.past_detections['pose_idx'] -= len(dets_to_rm)

        projected_boxes=None
        if self.past_poses.size(0) > 0:

            mask, projected_boxes = cuda_projection.project_past_detections(
                    batch_dict['chosen_tile_coords'],
                    self.past_detections['tile_inds'],
                    self.past_detections['pred_boxes'],
                    self.past_detections['pose_idx'],
                    self.past_poses.cuda(),
                    self.cur_pose.cuda(),
                    self.past_ts.cuda(),
                    self.cur_ts.item())

            projected_boxes = projected_boxes[mask]
            projected_scores = self.past_detections['pred_scores'][mask]
            projected_labels = self.past_detections['pred_labels'][mask]

        # Second, append new detections
        num_dets = pred_dicts[0]['pred_labels'].size(0)
        self.past_detections['num_dets'].append(num_dets)
        # Append the current pose
        self.past_poses = torch.cat((self.past_poses, self.cur_pose.unsqueeze(0)))
        self.past_ts = torch.cat((self.past_ts, self.cur_ts))
        # Append the pose idx for the detection that will be added
        past_poi = self.past_detections['pose_idx']
        poi = torch.full((num_dets,), self.past_poses.size(0)-1,
            dtype=past_poi.dtype, device=past_poi.device)
        self.past_detections['pose_idx'] = torch.cat((past_poi, poi))
        for k in ('pred_boxes', 'pred_scores', 'pred_labels', 'tile_inds'):
            self.past_detections[k] = torch.cat((self.past_detections[k], pred_dicts[0][k]))

        # append the projected detections
        if projected_boxes is not None:
            pred_dicts[0]['pred_boxes'] = torch.cat((pred_dicts[0]['pred_boxes'],
                projected_boxes))
            pred_dicts[0]['pred_scores'] = torch.cat((pred_dicts[0]['pred_scores'],
                projected_scores))
            pred_dicts[0]['pred_labels'] = torch.cat((pred_dicts[0]['pred_labels'],
                projected_labels))

            batch_dict['final_box_dicts'] = pred_dicts

        return batch_dict


    def calib_forward(self, batch_dict):
        points = batch_dict['points']
        # First, learn the number of nonempty tiles
        batch_dict['calib_num_tiles'] = 0
        batch_dict = self.vfe(batch_dict, model=self)
        max_num_tiles = batch_dict['chosen_tile_coords'].size(0)

        # Now try out all possilibities
        #self.calib_tile_idx_start = (self.calib_tile_idx_start + 1) % 10
        for num_tiles in range(self.calib_tile_idx_start, max_num_tiles+1, 1):
            batch_dict['points'] = points
            #batch_dict['point_coords'] = point_coords
            batch_dict['calib_num_tiles'] = num_tiles
            batch_dict = self.vfe(batch_dict, model=self)
            # Produce the reduce mask in parallel in a seperate stream
            with torch.cuda.stream(self.reduce_mask_stream):
                batch_dict['reduce_mask'] = self.produce_reduce_mask(batch_dict)

            self.add_dict['num_voxels'].append(
                    batch_dict['voxel_coords'].size(0))
            self.add_dict['num_tiles'].append(
                    batch_dict['chosen_tile_coords'].size(0))

            if self.is_voxel_enc:
                batch_dict = self.backbone_3d(batch_dict)

            batch_dict = self.map_to_bev(batch_dict)
            self.reduce_mask_stream.synchronize()

            batch_dict = self.backbone_2d(batch_dict)
            batch_dict = self.dense_head(batch_dict)

            torch.cuda.synchronize()

            self.add_dict['PostSched'].append(
                    time.time() - batch_dict['PostSched_start'])
            #print(self.add_dict['num_tiles'][-1], self.add_dict['PostSched'][-1])

        return batch_dict


    # This method is being called inside VFE
    def schedule(self, batch_dict):
        # float32, long
        points, point_coords = batch_dict['points'], batch_dict['point_coords']

        # Calculate where each voxel resides in which tile
        voxel_coords = torch.unique(point_coords, dim=0) # this gives us the total voxels [b x y z]
        voxel_tile_coords = torch.div(voxel_coords[..., 1:3], self.tile_size_voxels,
                rounding_mode='floor')
        voxel_tile_coords = voxel_coords[:, 0] * self.total_num_tiles + \
                voxel_tile_coords[...,0] * self.tcount[1] + voxel_tile_coords[...,1]

        # Get the tiles and number of voxels in them, maybe faster in cpu?
        nonempty_tile_coords, voxel_counts = torch.unique(voxel_tile_coords, \
                sorted=True, return_counts=True)

        if not self.training:
            # Here I need to run the scheduling algorithm

            # supress empty tiles by temporarily increasing the priority of nonempty tiles
            #tile_prios[nonempty_tile_coords] += self.total_num_tiles
            #highest_prios, chosen_tile_coords = \
            #        torch.topk(tile_prios, calib_num_tiles, sorted=False)
            #tile_prios[nonempty_tile_coords] -= self.total_num_tiles

            num_nonempty_tiles = nonempty_tile_coords.size(0)

            # find the index+1 of the last tile that was processed in the previous round
            # if it doesn't exist, find the one that is smaller closest
            tile_begin_idx = \
                    (nonempty_tile_coords > self.last_tile_coord).type(torch.uint8).argmax()
            #print(tile_begin_idx, self.last_tile_coord)

            tl_end = tile_begin_idx + num_nonempty_tiles
            ntc = nonempty_tile_coords.expand(2, num_nonempty_tiles).flatten()
            ntc = ntc[tile_begin_idx:tl_end]

            num_tiles = torch.arange(1, ntc.size(0)+1, device=voxel_counts.device)
            cnts = voxel_counts.expand(2, voxel_counts.size(0)).flatten()
            cnts = cnts[tile_begin_idx:tl_end]
            cnts_cumsum = torch.cumsum(cnts, dim=0)

            # Get execution time predictions
            inputs = torch.stack((cnts_cumsum, num_tiles)).T.float()
            #print(inputs)
            inputs_n = self.pred_net_scaler_in.transform(inputs)
            tpreds = self.post_sched_pred_net(inputs_n)
            #print(tpreds_n.flatten())
            #tpreds = self.pred_net_scaler_out.inverse_transform(tpreds_n).flatten()

            torch.cuda.synchronize()
            batch_dict['PostSched_start'] = time.time()
            #print(tpreds)
            rem_time = batch_dict['abs_deadline_sec'] - batch_dict['PostSched_start']
            rem_time -= self.pred_net_time_stats['99perc'] + self.projection_wcet
            #print(tpreds, rem_time)
            #print('tpreds', tpreds)
            diffs = (tpreds < rem_time)
            #print(rem_time, diffs.all())
            #print(diffs.flatten())

            calib_num_tiles = 0
            if 'calib_num_tiles' in batch_dict:
                # calibration in progress, ignore tpreds and deadline
                calib_num_tiles = batch_dict['calib_num_tiles']

            if (not self.calibrating_now and diffs.all()) or \
                    (self.calibrating_now and calib_num_tiles == 0):
                # Use all tiles
                chosen_tile_coords = nonempty_tile_coords
                self.last_tile_coord = 0
            else:
                # Point filtering is needed
                #print(diffs)
                #TODO check if this is correct
                idx = calib_num_tiles if self.calibrating_now else diffs.to(dtype=torch.uint8).argmin()+1
                chosen_tile_coords = ntc[:idx]
                self.last_tile_coord = 0 if self.calibrating_now else chosen_tile_coords[-1]
                #print('last tile coord:', self.last_tile_coord)

                # Calculate point tile coords
                point_tile_coords = torch.div(point_coords[..., 1:3], self.tile_size_voxels, \
                        rounding_mode='trunc').long()
                point_tile_coords = point_coords[:, 0].long() * self.total_num_tiles + \
                        point_tile_coords[...,0] * self.tcount[1] + point_tile_coords[...,1]

                tile_filter = cuda_point_tile_mask.point_tile_mask(point_tile_coords, \
                        chosen_tile_coords)

                batch_dict['points'] = points[tile_filter]
                batch_dict['point_coords'] = point_coords[tile_filter]
            batch_dict['chosen_tile_coords'] = chosen_tile_coords
            #print(num_nonempty_tiles, chosen_tile_coords.size(0))
        else:
            #torch.cuda.synchronize()
            #batch_dict['PostSched_start'] = time.time()
            # No filtering, process all nonempty tiles
            batch_dict['chosen_tile_coords'] = nonempty_tile_coords

        return batch_dict

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
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict

    def projection_reset(self):
        # Poses include [cst(3) csr(4) ept(3) epr(4)]
        self.past_detections = self.get_empty_det_dict()
        self.past_detections['num_dets'] = []
        self.past_detections['pose_idx'] = torch.zeros([0], dtype=torch.long,
            device=self.past_detections["pred_labels"].device)
        self.past_detections['tile_inds'] = torch.zeros([0], dtype=torch.long,
            device=self.past_detections["pred_labels"].device)
        self.past_poses = torch.zeros([0, 14], dtype=torch.float)
        self.past_ts = torch.zeros([0], dtype=torch.long)

    def calibrate(self, batch_size=1):
        ep = self.enable_projection
        self.enable_projection = False
        super().calibrate()
        pred_dicts = super().calibrate(batch_size)
        self.enable_projection = ep
        self.projection_reset()

        # check if the wcet pred file is there
        fname = f"calib_data.pkl"
        try:
            with open(fname, 'rb') as handle:
                calib_data = pickle.load(handle)
                self.post_sched_pred_net.load_state_dict(calib_data[0])
                self.post_sched_pred_net = self.post_sched_pred_net.cuda()
                self.pred_net_scaler_in.set_params(calib_data[1])
                self.pred_net_scaler_in = self.pred_net_scaler_in.cuda()
        #        self.pred_net_scaler_out = calib_data[2].cuda()
                self.pred_net_time_stats = calib_data[2]

        except FileNotFoundError:
            print(f'Calibration file {fname} not found, running calibration')
            self.calibrating_now = True # time calibration!

        print(self.pred_net_time_stats)

        return None #pred_dicts

    def post_eval(self):
        if self.calibrating_now:
            print('Time calibration Complete')
            with open(f"calib_raw_data.json", 'w') as handle:
                json.dump(self.add_dict, handle, indent=4)
            sys.exit()
