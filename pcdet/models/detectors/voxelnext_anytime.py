from .detector3d_template import Detector3DTemplate, pre_forward_hook
import torch
from nuscenes.nuscenes import NuScenes
import time
import sys
import json
import random
import numpy as np
import scipy
#from sklearn.model_selection import train_test_split

from ...ops.cuda_projection import cuda_projection
from ...ops.cuda_point_tile_mask import cuda_point_tile_mask

class VoxelNeXtAnytime(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed(0)

        self.vfe, self.backbone_3d, self.dense_head = self.module_list
        self.update_time_dict( {
                'VFE': [],
                'Sched': [],
                'Backbone3D':[],
                'VoxelHead': [],
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
        self.tile_size_voxels = torch.from_numpy(\
                self.dataset.grid_size[:2] / self.tcount).cuda().long()

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
        self.add_dict['tcount'] = self.tcount
        for k in ('voxel_counts', 'chosen_tile_coords', 'PostSched'):
            self.add_dict[k] = []

        # these values needs calibration
        self.time_pred_coeffs = [1.,1.,1.]
        self.projection_wcet = 0.002
        self.pred_net_time_stats = {'99perc':0.0, 'max': 0.0}

        grid_size = self.dataset.grid_size
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        #print(self)


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

        self.measure_time_start('Backbone3D')
        batch_dict = self.backbone_3d(batch_dict)
        self.measure_time_end('Backbone3D')

        self.measure_time_start('VoxelHead')
        batch_dict = self.dense_head(batch_dict)
        self.measure_time_end('VoxelHead')

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

        ctc = batch_dict['chosen_tile_coords']
        pcr = self.dataset.point_cloud_range
        for pd in pred_dicts:
            x = pd['pred_boxes'][:, 0]
            y = pd['pred_boxes'][:, 1]
            # NOTE I am not sure about the indices of pcr ant tcount
            # but it is fine since tcount[0] == tcount[1]
            x_inds =  ((x - pcr[0]) / (pcr[3] - pcr[0]) * self.tcount[0]).trunc().long()
            y_inds =  ((y - pcr[1]) / (pcr[4] - pcr[1]) * self.tcount[1]).trunc().long()
            inds = x_inds * self.tcount[1] + y_inds
            #tile_filter = cuda_point_tile_mask.point_tile_mask(inds, ctc)
            #inds = inds[tile_filter]
            #for k,v in pd.items():
            #    if k != 'pred_ious':
            #        pd[k] = v[tile_filter]
            pd['tile_inds'] = inds

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
        self.calib_tile_idx_start = (self.calib_tile_idx_start % 10) + 1
        for num_tiles in range(self.calib_tile_idx_start, max_num_tiles+1, 10):
            times, vcs, tiles = [], [], []
            for rep in range(3):
                batch_dict['points'] = points
                #batch_dict['point_coords'] = point_coords
                batch_dict['calib_num_tiles'] = num_tiles
                batch_dict = self.vfe(batch_dict, model=self)
                batch_dict = self.backbone_3d(batch_dict)
                batch_dict = self.dense_head(batch_dict)
                torch.cuda.synchronize()

                times.append(time.time() - batch_dict['PostSched_start'])
                vcs.append(batch_dict['voxel_counts'])
                tiles.append(batch_dict['chosen_tile_coords'])

            idx = times.index(max(times))
            self.add_dict['PostSched'].append(times[idx])
            self.add_dict['voxel_counts'].append(vcs[idx])
            self.add_dict['chosen_tile_coords'].append(tiles[idx])

        return batch_dict


    # This method is being called inside VFE
    # NOTE Assumes batch size of 1
    def schedule(self, batch_dict):
        self.measure_time_start('Sched')
        # float32, long
        points, point_coords = batch_dict['points'], batch_dict['point_coords']

        # Calculate where each voxel resides in which tile
        merge_coords = point_coords[:, 1] * self.scale_yz + \
                        point_coords[:, 2] * self.scale_z + \
                        point_coords[:, 3]
        unq_coords = torch.unique(merge_coords)
        voxel_x = torch.div(unq_coords, self.scale_yz, rounding_mode='trunc')
        voxel_y = torch.div((unq_coords % self.scale_yz), self.scale_z, rounding_mode='trunc')
        tile_x = torch.div(voxel_x, self.tile_size_voxels[0], rounding_mode='trunc')
        tile_y = torch.div(voxel_y, self.tile_size_voxels[1], rounding_mode='trunc')
        voxel_tile_coords = tile_x * self.tcount[0] + tile_y

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

            tl_end = tile_begin_idx + num_nonempty_tiles
            ntc = nonempty_tile_coords.expand(2, num_nonempty_tiles).flatten()
            ntc = ntc[tile_begin_idx:tl_end]

            num_tiles = torch.arange(1, ntc.size(0)+1, device=voxel_counts.device).float()
            cnts = voxel_counts.expand(2, voxel_counts.size(0)).flatten()
            cnts = cnts[tile_begin_idx:tl_end]
            cnts_cumsum = torch.cumsum(cnts, dim=0).float()

            # Get execution time predictions
            #inputs = torch.stack((cnts_cumsum, num_tiles)).T.float()

            C = self.time_pred_coeffs
            tpreds= C[0]*cnts_cumsum + C[1]*num_tiles + C[2]

            torch.cuda.synchronize()
            batch_dict['PostSched_start'] = time.time()
            rem_time = batch_dict['abs_deadline_sec'] - batch_dict['PostSched_start']
            rem_time -= self.pred_net_time_stats['max'] + self.projection_wcet
            diffs = (tpreds < rem_time)
            #print(diffs)

            # calibration in progress, ignore tpreds and deadline
            calib_num_tiles = batch_dict['calib_num_tiles'] if self.calibrating_now else -1

            # diffs.all() is the when we can meet all deadlines

            all_true = diffs.all()
            if (not self.calibrating_now and all_true) or (calib_num_tiles == 0):
                # Use all tiles
                chosen_tile_coords = nonempty_tile_coords
                self.last_tile_coord = 0
                batch_dict['voxel_counts'] = cnts
            else:
                # Point filtering is needed
                idx = diffs.to(dtype=torch.uint8).argmin()+1
                if self.calibrating_now:
                    idx = calib_num_tiles
                chosen_tile_coords = ntc[:idx]

                self.last_tile_coord = random.randint(0, num_nonempty_tiles-1) \
                        if self.calibrating_now else chosen_tile_coords[-1]

                point_tile_coords = torch.div(point_coords[..., 1:3], self.tile_size_voxels, \
                        rounding_mode='trunc').long()
                point_tile_coords = point_coords[:, 0].long() * self.total_num_tiles + \
                        point_tile_coords[...,0] * self.tcount[1] + point_tile_coords[...,1]

                tile_filter = cuda_point_tile_mask.point_tile_mask(point_tile_coords, \
                        chosen_tile_coords)

                batch_dict['points'] = points[tile_filter]
                batch_dict['point_coords'] = point_coords[tile_filter]
                batch_dict['voxel_counts'] = cnts[:idx]
            batch_dict['chosen_tile_coords'] = chosen_tile_coords

            self.measure_time_end('Sched')
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
        fname = f"calib_raw_data.json"
        try:
            with open(fname, 'r') as handle:
                calib_dict = json.load(handle)

            voxel_counts = calib_dict["voxel_counts"]
            num_voxels = [sum(vc) for vc in voxel_counts]

            tile_coords = calib_dict["chosen_tile_coords"]
            num_tiles = [len(tc) for tc in tile_coords]

            psched_time = calib_dict["PostSched"]
            X = np.array(num_voxels, dtype=np.float32)
            Y = np.array(num_tiles, dtype=np.float32)
            Z = np.array(psched_time, dtype=np.float32)
            x1, y1, z1 = X.flatten(), Y.flatten(), Z.flatten()

            A = np.c_[x1, y1, np.ones(x1.shape[0])]
            C,_,_,_ = scipy.linalg.lstsq(A, z1)    # coefficients

            self.time_pred_coeffs = C
            plane_z = C[0]*x1 + C[1]*y1 + C[2]

            diff = z1 - plane_z
            perc95 = np.percentile(diff, 95, method='lower')
            perc99 = np.percentile(diff, 99, method='lower')
            self.pred_net_time_stats = {
                'min': float(min(diff)),
                'avrg': float(sum(diff)/len(diff)),
                '95perc': float(perc95),
                '99perc': float(perc99),
                'max': float(max(diff))}
            print('Time prediction stats:')

        except FileNotFoundError:
            print(f'Calibration file {fname} not found, running calibration')
            self.calibrating_now = True # time calibration!

        print(self.pred_net_time_stats)

        return None #pred_dicts

    def post_eval(self):
        if self.calibrating_now:
            print('Time calibration Complete')
            for k in ('voxel_counts', 'chosen_tile_coords'):
                for i, t in enumerate(self.add_dict[k]):
                    self.add_dict[k][i] = t.cpu().tolist()
            with open(f"calib_raw_data.json", 'w') as handle:
                json.dump(self.add_dict, handle, indent=4)
            sys.exit()
        print(f"Deadlines missed: {self._eval_dict['deadlines_missed']}")
