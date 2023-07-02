from .detector3d_template import Detector3DTemplate
import torch
from nuscenes.nuscenes import NuScenes
import time
import sys
import json
import numpy as np
import scipy
import gc
import copy
import numba

from ..model_utils import model_nms_utils
from ...ops.cuda_projection import cuda_projection
from ...ops.cuda_point_tile_mask import cuda_point_tile_mask
from .. import load_data_to_gpu

#os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
def transform(x, m, s):
    x -= m
    x /= s

    return x

@numba.jit(nopython=True)
def get_num_tiles(ctc): # chosen tile coords
    ctc_s, ctc_e = ctc[0], ctc[-1]
    if ctc_s <= ctc_e:
        num_tiles = ctc_e - ctc_s + 1
    else:
        j = 0
        while ctc[j] < ctc[j+1]:
            j += 1
        num_tiles = ctc[j] - ctc_s + 1 + ctc_e - ctc[j+1] + 1

    return num_tiles

@numba.jit(nopython=True)
def round_robin_sched_helper(netc, last_tile_coord, tcount, netc_vcounts):
    num_nonempty_tiles = netc.shape[0]
    for i in range(num_nonempty_tiles):
        if netc[i] > last_tile_coord:
            tile_begin_idx = i
            break

    netc = np.concatenate((netc[tile_begin_idx:], netc[:tile_begin_idx]))
    net_vcounts = np.concatenate((netc_vcounts[tile_begin_idx:],
        netc_vcounts[:tile_begin_idx]))

    vcounts_all = np.zeros((netc.shape[0], tcount), dtype=float)
    num_tiles = np.empty((netc.shape[0],), dtype=float)

    for i in range(vcounts_all.shape[0]):
        ctc = netc[:i+1]
        num_tiles[i] = get_num_tiles(ctc)
        for j in range(i+1):
            vcounts_all[i, ctc[j]] = netc_vcounts[j]

    return num_tiles, vcounts_all


class AnytimeTemplateV2(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        if 'BACKBONE_2D' in self.model_cfg:
            self.model_cfg.BACKBONE_2D.TILE_COUNT = self.model_cfg.TILE_COUNT
        if 'DENSE_HEAD' in self.model_cfg:
            self.model_cfg.DENSE_HEAD.TILE_COUNT = self.model_cfg.TILE_COUNT
        self.module_list = self.build_networks()
        torch.backends.cudnn.benchmark = False # TODO, we can allow this
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed(0)
#        torch.use_deterministic_algorithms(True)

        ################################################################################
        self.tcount = self.model_cfg.TILE_COUNT
        self.tcount_cuda = torch.tensor(self.model_cfg.TILE_COUNT).long().cuda()
        self.total_num_tiles = self.tcount

        # This number will be determined by the scheduling algorithm initially for each input
        self.last_tile_coord = -1
        #self.reduce_mask_stream = torch.cuda.Stream()

        # divide the tiles in X axis only
        self.tile_size_voxels = torch.tensor(\
                self.dataset.grid_size[1] / self.tcount).cuda().long()

        ####Projection###
        self.enable_projection = False
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
        ################################################################################

        self.calibrating_now = False
        self.add_dict = self._eval_dict['additional']
        for k in ('voxel_counts', 'num_tiles', 'PostSched'):
            self.add_dict[k] = []


        self.bb3d_time_pred_model = self.define_bb3d_time_pred_model(self.tcount)
        # TODO assign the right numbers to these two using json
        self.bb3d_model_mean = 0.0
        self.bb3d_model_std = 0.0

        self.proj_time_limit_musec = 1000000 # 1 sec

        self.RoundRobin = 1
        self.ProjectionOnly, self.projLastNth = 2, 1

        self.sched_algo = self.model_cfg.METHOD

        self.past_detections = {'num_dets': []}
        self.prev_scene_token = ''
        if self.sched_algo == self.ProjectionOnly:
            self.past_poses = []
            self.past_ts = []
        else:
            # Poses include [cst(3) csr(4) ept(3) epr(4)]
            self.past_poses = torch.zeros([0, 14], dtype=torch.float)
            self.past_ts = torch.zeros([0], dtype=torch.long)

        # Needs to be calibrated
        self.post_bb3d_times_ms = torch.tensor([9999.9] * self.tcount, dtype=torch.float)
        self.score_thresh = self.model_cfg.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH

    def define_bb3d_time_pred_model(self, num_tiles):
        inp_size = num_tiles
        outp_size = 1 # Execution time
        possibilities = inp_size * (inp_size +1) // 2
        model = torch.nn.Sequential(
            torch.nn.Linear(inp_size, possibilities),
            torch.nn.ReLU(),
            torch.nn.Linear(possibilities, outp_size)).cuda()
        return model 

    # The input will have all
    def pred_completion_time(self, vcounts_all, num_tiles):
        vcounts_all -= self.bb3d_model_mean
        vcounts_all /= self.bb3d_model_std
        times = self.bb3d_time_pred_model(vcounts_all).flatten()
        return times + self.post_bb3d_times_ms[num_tiles]

    # When projecting, set the pred scores to a number below 0.3.
    # After running nms, remove the dets that are projected using their
    # pred score when adding the new dets to the past detections.
    # However, output the entire detections.
    def projection(self, batch_dict):
        if self.enable_projection:
            latest_token = batch_dict['metadata'][0]['token']
            scene_token = self.token_to_scene[latest_token]
            if scene_token != self.prev_scene_token:
                self.projection_reset()
                self.prev_scene_token = scene_token

            self.cur_pose = self.token_to_pose[latest_token]
            self.cur_ts = self.token_to_ts[latest_token]
        else:
            return batch_dict

        if self.sched_algo == self.ProjectionOnly:
            return self.projection_for_test(batch_dict)

        # Clear unuseful dets
        if self.cur_ts - self.past_ts[0] > self.proj_time_limit_musec:
            self.past_poses = self.past_poses[1:]
            self.past_ts = self.past_ts[1:]
            nd = self.past_detection['num_dets'].pop(0)
            for k in ('pred_boxes', 'pred_scores', 'pred_labels', 'pose_idx'):
                self.past_detections[k] = self.past_detections[k][nd:]
            self.past_detections['pose_idx'] -= 1

        # Assign the scores in a way to favor fresh objects
        self.past_detections['pred_scores'] = self.score_thresh - \
                (self.score_thresh / (self.past_detections['pose_idx'] + 2))

        if self.past_detections['pred_boxes'].size(0) > 0:
            proj_dict = {}
            proj_dict['pred_boxes'] = cuda_projection.project_past_detections(
                    self.past_detections['pred_boxes'],
                    self.past_detections['pose_idx'],
                    self.past_poses.cuda(),
                    self.cur_pose.cuda(),
                    self.past_ts.cuda(),
                    self.cur_ts.item())

            proj_dict['pred_scores'] = self.past_detections['pred_scores']
            proj_dict['pred_labels'] = self.past_detections['pred_labels'] - 1
            batch_dict['projections'] = proj_dict
        else:
            batch_dict['projections'] = None

        return batch_dict


    def projection_post(self, batch_dict):
        pred_dict = batch_dict['final_box_dicts'][0]

        # Before appending the dets, extract the projected ones
        proj_mask = pred_dict['pred_scores'] > self.score_thresh
        new_dets_dict = {}
        for k in ('pred_boxes', 'pred_scores', 'pred_labels'):
            new_dets_dict[k] = pred_dict[k][proj_mask]

        num_dets = new_dets_dict['pred_boxes'].size(0)
        # Append new detections
        self.past_detections['num_dets'].append(num_dets)
        # Append the current pose and ts
        self.past_poses = torch.cat((self.past_poses, self.cur_pose.unsqueeze(0)))
        self.past_ts = torch.cat((self.past_ts, self.cur_ts))
        # Append the pose idx for the detection that will be added
        past_poi = self.past_detections['pose_idx']
        poi = torch.full((num_dets,), self.past_poses.size(0)-1,
            dtype=past_poi.dtype, device=past_poi.device)
        self.past_detections['pose_idx'] = torch.cat((past_poi, poi))
        for k in ('pred_boxes', 'pred_scores', 'pred_labels'):
            self.past_detections[k] = torch.cat((self.past_detections[k], new_dets_dict[k]))

        return batch_dict

    def get_nonempty_tiles(self, voxel_coords):
        # Calculate where each voxel resides in which tile
        #tile_coords = torch.div(voxel_coords[:, -2:], self.tile_size_voxels, \
        #        rounding_mode='trunc').long()
        voxel_tile_coords = torch.div(voxel_coords[:, -1], self.tile_size_voxels, \
                rounding_mode='trunc').long()

        #voxel_tile_coords = tile_coords[:, 1] * self.tcount[1] + tile_coords[:, 0]
        if self.training:
            nonempty_tile_coords = torch.unique(voxel_tile_coords, sorted=True)
            return nonempty_tile_coords
        else:
            nonempty_tile_coords, voxel_counts = torch.unique(voxel_tile_coords, \
                    sorted=True, return_counts=True)
            return voxel_tile_coords, nonempty_tile_coords, voxel_counts

    def schedule(self, batch_dict):
        voxel_coords = batch_dict['voxel_coords']
        if self.training:
            batch_dict['chosen_tile_coords'] = self.get_nonempty_tiles(voxel_coords)
            return batch_dict
        self.measure_time_start('Sched')
        voxel_tile_coords, netc, netc_vcounts = self.get_nonempty_tiles(voxel_coords)
        netc = netc.cpu() # sync
        batch_dict['mask'] = None

        if self.calibrating_now and self.calib_num_tiles == netc.size(0):
            # Simply process all tiles, no need for scheduling
            chosen_tile_coords = netc
            self.last_tile_coord = -1
        elif self.sched_algo == self.RoundRobin:
            num_tiles, vcounts_all = round_robin_sched_helper(
                    netc.numpy(), self.last_tile_coord, self.tcount,
                    netc_vcounts.cpu().numpy())

            # Let's try no using cuda and see the performance
            vcounts_all = torch.from_numpy(vcounts_all)
            num_tiles = torch.from_numpy(num_tiles)
            tpreds = self.pred_completion_time(vcounts_all, num_tiles)

            self.psched_start_time = time.time()
            rem_time = batch_dict['abs_deadline_sec'] - self.psched_start_time

            # Choose configuration that can meet the deadline, that's it
            # NOTE The following code assumes there is going to be a 
            # second syncronization after bb3d, which will do further scheduling if
            # deadline cannot be met
            diffs = tpreds < rem_time
            tiles_idx = torch.sum(diffs).item()
            if tiles_idx < diffs.size(0):
                # Voxel filtering is needed
                chosen_tile_coords = netc[:tiles_idx]
                self.last_tile_coord = chosen_tile_coords[-1].item()
                tile_filter = cuda_point_tile_mask.point_tile_mask(voxel_tile_coords, \
                        chosen_tile_coords.cuda())
                batch_dict['mask'] = tile_filter
                if 'voxel_features' in batch_dict:
                    batch_dict['voxel_features'] = \
                            batch_dict['voxel_features'][tile_filter].contiguous()
                batch_dict['voxel_coords'] = voxel_coords[tile_filter].contiguous()

        elif self.sched_algo == self.ProjectionOnly:
            batch_dict['chosen_tile_coords'] = netc
            self.measure_time_end('Sched')
            return batch_dict

        batch_dict['chosen_tile_coords'] = chosen_tile_coords
        self.add_dict['voxel_counts'].append(batch_dict['voxel_coords'].size(0))
        self.add_dict['num_tiles'].append(batch_dict['chosen_tile_coords'].size(0))

        self.measure_time_end('Sched')

        return batch_dict

    def schedule_after_bb3d(self, batch_dict):
        ctc = batch_dict['chosen_tile_coords']
        num_tiles = get_num_tiles(ctc)
        torch.cuda.synchronize()
        rem_time = batch_dict['abs_deadline_sec'] - time.time()

        while num_tiles > 1 and self.post_bb3d_times_ms[num_tiles] > rem_time:
            num_tiles -= 1
        batch_dict['chosen_tile_coords'] = ctc[:num_tiles]
        self.last_tile_coord = batch_dict['chosen_tile_coords'][-1]

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
        if self.sched_algo == self.ProjectionOnly:
            for k in ('pred_boxes', 'pred_scores', 'pred_labels', 'pose_idx', 'num_dets'):
                self.past_detections[k] = []
            self.past_poses, self.past_ts = [], []
        else:
            self.past_detections = self.get_empty_det_dict()
            self.past_detections['num_dets'] = []
            self.past_detections['pose_idx'] = torch.zeros([0], dtype=torch.long,
                device=self.past_detections["pred_labels"].device)
            self.past_poses = torch.zeros([0, 14], dtype=torch.float)
            self.past_ts = torch.zeros([0], dtype=torch.long)
        self.last_tile_coord = -1

    def calibrate(self, fname='calib_raw_data.json'):
        super().calibrate(1)
        # Keep projection disabled to see the tracking
        #self.enable_projection = True
        self.projection_reset()

        for l in self.add_dict.values():
            l.clear()

        if self.sched_algo == self.ProjectionOnly:
            print('Projection test is running.')
            return None

        # check if the wcet pred file is there
        try:
            with open(fname, 'r') as handle:
                calib_dict = json.load(handle)

            num_voxels = calib_dict["voxel_counts"]

            if 'num_tiles' in calib_dict and calib_dict['num_tiles']:
                num_tiles = calib_dict['num_tiles']
            else:
                tile_coords = calib_dict["chosen_tile_coords"]
                num_tiles = [len(tc) for tc in tile_coords]

            num_ALL_samples = calib_dict['calib_dataset_len']
            num_voxels_ALL = num_voxels[-num_ALL_samples:]
            num_tiles_ALL = num_tiles[-num_ALL_samples:]

            num_voxels = num_voxels[:-num_ALL_samples]
            num_tiles = num_tiles[:-num_ALL_samples]

            psched_time = calib_dict["PostSched"]
            psched_time_ALL = psched_time[-num_ALL_samples:]
            psched_time = psched_time[:-num_ALL_samples]

            self.time_pred_coeffs_1, self.pred_net_time_stats_1 = \
                    self.calc_time_pred_coeffs(num_voxels, num_tiles, psched_time)
            self.time_pred_coeffs_ALL, self.pred_net_time_stats_ALL = \
                    self.calc_time_pred_coeffs(num_voxels_ALL, num_tiles_ALL, psched_time_ALL)
            self.time_pred_coeffs_ALL = self.time_pred_coeffs_ALL.cpu()

        except FileNotFoundError:
            print(f'Calibration file {fname} not found, running calibration')
            self.calibrating_now = True # time calibration!
            self.calibration_procedure(fname)
            sys.exit()

        return None

    def calibrate_after_bb3d(self, spatial_features):
        self.calibrated = True
        # Create a timing model for different tile sizes
        # taking the worst case into account

        def get_worst_case_tiles(num_tiles, tcount):
            # Now exactly sure if this would be the worst case
            # but probably very close
            if num_tiles == 1:
                return [0]

            num_tiles_l = num_tiles // 2
            num_tiles_r = num_tiles - num_tiles_l

            r_list = list(range(tcount - num_tiles_r, tcount))
            l_list = list(range(num_tiles_l))

            return r_list + l_list

        for num_tiles in range(1, self.tcount+1):
            # NOTE Slice in W dimension, I think it corresponds to x
            # if it is not, when change it to H
            # Warmup first
            self.forward({'spatial_features': spatial_features,
                'chosen_tile_coords': get_worst_case_tiles(num_tiles, self.tcount)})
            reps=3
            torch.cuda.synchronize()
            t0 = time.time()
            for j in range(reps):
                self.forward({'spatial_features': spatial_features,
                    'chosen_tile_coords': get_worst_case_tiles(num_tiles, self.tcount)})
            torch.cuda.synchronize()
            t_elapsed = time.time() - t0
            self.slc_time_ms[num_tiles-1] = round(t_elapsed*1000/reps, 3)
        print('BB2D Pred times:', self.slc_time_ms)


    def calibration_procedure(self, fname="calib_raw_data.json"):
        gc.disable()
        all_max_num_tiles = []

        pc_range = torch.from_numpy(self.dataset.point_cloud_range).cuda()[[0, 1]]
        voxel_size = torch.tensor(self.dataset.voxel_size).cuda()[[0, 1]]
        grid_size = torch.from_numpy(self.dataset.grid_size).cuda()[[0, 1]]
        for i in range(len(self.dataset)):
            data_dict = self.dataset.getitem_pre(i)
            data_dict = self.dataset.getitem_post(data_dict)
            data_dict = self.dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
    
            if 'voxel_coords' not in data_dict:
                points = data_dict['points']
                #print(pc_range, voxel_size, grid_size)
                #print(points[-10:])
                points_coords = torch.floor( \
                    (points[:, [1, 2]] - pc_range) / voxel_size).int()
                mask = ((points_coords >= 0) & (points_coords < grid_size)).all(dim=1)
                #print(points_coords[-10:])
                data_dict['voxel_coords'] = points_coords[mask][:, [1,0]]

            _, nonempty_tile_coords, _ = self.get_nonempty_tiles(data_dict['voxel_coords'])
            max_num_tiles = nonempty_tile_coords.size(0)
            all_max_num_tiles.append(max_num_tiles)

        mit, mat = min(all_max_num_tiles), max(all_max_num_tiles)
        print(f'Min num tiles: {mit}, Max num tiles: {mat}')
        torch.cuda.empty_cache()
        gc.collect()

        # 10 different num of tiles should be enough
        for num_tiles in range(1, mat, mat//10):
            print('Num tiles:', num_tiles)
            for i in range(len(self.dataset)):
                if num_tiles < all_max_num_tiles[i]:
                    self.calib_num_tiles = num_tiles
                    with torch.no_grad():
                        pred_dicts, ret_dict = self([i])
                    gc.collect()

        print('Num tiles: ALL')
        for i in range(len(self.dataset)):
            self.calib_num_tiles = all_max_num_tiles[i]
            with torch.no_grad():
                pred_dicts, ret_dict = self([i])
            gc.collect()

        gc.enable()
        self.add_dict['tcount'] = self.tcount
        self.add_dict['method'] = self.sched_algo
        self.add_dict['exec_times'] = self.get_time_dict()
        self.add_dict['exec_time_stats'] = self.get_time_dict_stats()
        self.add_dict['calib_dataset_len'] = len(self.dataset)
        print('Time calibration Complete')
        with open(fname, 'w') as handle:
            json.dump(self.add_dict, handle, indent=4)

    def post_eval(self):
        self.add_dict['tcount'] = self.tcount
        print(f"\nDeadlines missed: {self._eval_dict['deadlines_missed']}\n")


    def projection_for_test(self, batch_dict):
        pred_dicts = batch_dict['final_box_dicts']

        if self.enable_projection:
            # only keeps the previous detection
            projected_boxes=None
            pb = self.past_detections['pred_boxes']
            if len(pb) >= self.projLastNth and pb[-self.projLastNth].size(0) > 0:

                projected_boxes = cuda_projection.project_past_detections(
                        self.past_detections['pred_boxes'][-self.projLastNth],
                        self.past_detections['pose_idx'][-self.projLastNth],
                        self.past_poses[-self.projLastNth].cuda(),
                        self.cur_pose.cuda(),
                        self.past_ts[-self.projLastNth].cuda(),
                        self.cur_ts.item())

                projected_labels = self.past_detections['pred_labels'][-self.projLastNth]
                projected_scores = self.past_detections['pred_scores'][-self.projLastNth]

            ####USE DETECTION DATA#### START
#            # Second, append new detections
#            num_dets = pred_dicts[0]['pred_labels'].size(0)
#            self.past_detections['num_dets'] = num_dets
#            # Append the current pose
#            self.past_poses = self.cur_pose.unsqueeze(0)
#            self.past_ts = self.cur_ts #.unsqueeze(0)
#            # Append the pose idx for the detection that will be added
#            self.past_detections['pose_idx'] = \
#                    torch.full((num_dets,), 0, dtype=torch.long, device='cuda')
#
#            for k in ('pred_boxes', 'pred_scores', 'pred_labels'):
#                self.past_detections[k] = pred_dicts[0][k]
#
#            # append the projected detections
#            if projected_boxes is not None:
#                pred_dicts[0]['pred_boxes'] = projected_boxes
#                pred_dicts[0]['pred_scores'] = projected_scores
#                pred_dicts[0]['pred_labels'] = projected_labels
            ####USE DETECTION DATA#### END

            ####USE GROUND TRUTH#### START
            self.past_detections['pred_boxes'].append(batch_dict['gt_boxes'][0][..., :9])
            self.past_detections['pred_labels'].append(batch_dict['gt_boxes'][0][...,-1].int())
            self.past_detections['pred_scores'].append(torch.ones_like(\
                    self.past_detections['pred_labels'][-1]))

            num_dets = self.past_detections['pred_scores'][-1].size(0)
            self.past_poses.append(self.cur_pose.unsqueeze(0))
            self.past_ts.append(self.cur_ts)
            self.past_detections['pose_idx'].append( \
                    torch.zeros((num_dets,), dtype=torch.long, device='cuda'))
            ####USE GROUND TRUTH#### END

            while len(self.past_poses) > self.projLastNth:
                for k in ('pred_boxes', 'pred_scores', 'pred_labels', 'pose_idx'):
                    self.past_detections[k].pop(0)
                self.past_poses.pop(0)
                self.past_ts.pop(0)

            # append the projected detections
            if projected_boxes is not None:
                pred_dicts[0]['pred_boxes']  = projected_boxes
                pred_dicts[0]['pred_labels'] = projected_labels
                pred_dicts[0]['pred_scores'] = projected_scores
            else:
                # use groud truth if projection was not possible
                pred_dicts[0]['pred_boxes']  = self.past_detections['pred_boxes'][-1]
                pred_dicts[0]['pred_labels'] = self.past_detections['pred_labels'][-1]
                pred_dicts[0]['pred_scores'] = self.past_detections['pred_scores'][-1]

        return batch_dict


