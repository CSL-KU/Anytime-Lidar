from .detector3d_template import Detector3DTemplate
from .anytime_calibrator import AnytimeCalibrator, get_stats
from .sched_helpers import *
import torch
from nuscenes.nuscenes import NuScenes
import time
import sys
import json
import numpy as np
import scipy
import gc
import copy

from ...ops.cuda_projection import cuda_projection
from ...ops.cuda_point_tile_mask import cuda_point_tile_mask
from .. import load_data_to_gpu

class AnytimeTemplateV2(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.sched_algo = self.model_cfg.METHOD

        if self.sched_algo == SchedAlgo.RoundRobin_NoProj:
            self.keep_projection_disabled=True
            self.sched_algo = SchedAlgo.RoundRobin
        else:
            self.keep_projection_disabled=False

        if 'BACKBONE_2D' in self.model_cfg:
            self.model_cfg.BACKBONE_2D.TILE_COUNT = self.model_cfg.TILE_COUNT
            self.model_cfg.BACKBONE_2D.METHOD = self.sched_algo
        if 'DENSE_HEAD' in self.model_cfg:
            self.model_cfg.DENSE_HEAD.TILE_COUNT = self.model_cfg.TILE_COUNT
            self.model_cfg.DENSE_HEAD.METHOD = self.sched_algo
        torch.backends.cudnn.benchmark = True
        if torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark_limit = 0
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed(0)
        self.module_list = self.build_networks()
#        torch.use_deterministic_algorithms(True)

        ################################################################################
        self.tcount = self.model_cfg.TILE_COUNT
        self.tcount_cuda = torch.tensor(self.model_cfg.TILE_COUNT).long().cuda()
        self.total_num_tiles = self.tcount

        # divide the tiles in X axis only
        self.tile_size_voxels = torch.tensor(\
                self.dataset.grid_size[1] / self.tcount).cuda().long()
        self.tile_size_voxels_int = self.tile_size_voxels.int()

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

        self.add_dict = self._eval_dict['additional']
        self.add_dict['bb3d_preds'] = []
        self.add_dict['nonempty_tiles'] = []
        self.add_dict['chosen_tiles_1'] = []
        self.add_dict['chosen_tiles_2'] = []

        if self.sched_algo == SchedAlgo.RoundRobin or self.sched_algo == SchedAlgo.AdaptiveRR:
            self.init_tile_coord = -1
        elif self.sched_algo == SchedAlgo.MirrorRR:
            self.init_tile_coord = 0
            m2 = self.tcount//2
            m1 = m2 - 1
            self.mtiles = np.array([m1, m2], dtype=np.int32)
        self.last_tile_coord = self.init_tile_coord

        self.past_detections = {}  #{'num_dets': []}
        self.prev_scene_token = ''
        if self.sched_algo == SchedAlgo.ProjectionOnly:
            self.past_poses = []
            self.past_ts = []
        else:
            # Poses include [cst(3) csr(4) ept(3) epr(4)]
            self.past_poses = torch.zeros([0, 14], dtype=torch.float)
            self.past_ts = torch.zeros([0], dtype=torch.long)
            self.num_dets_per_tile = torch.zeros([self.tcount], dtype=torch.long)

        # Needs to be calibrated
        self.score_thresh = self.model_cfg.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH

        total_num_classes = sum([m.size(0) for m in self.dense_head.class_id_mapping_each_head])
        self.cls_id_to_det_head_idx_map = torch.zeros((total_num_classes,), dtype=torch.int)
        self.num_det_heads = len(self.dense_head.class_id_mapping_each_head)
        for i, cls_ids in enumerate(self.dense_head.class_id_mapping_each_head):
            for cls_id in cls_ids:
                self.cls_id_to_det_head_idx_map[cls_id] = i
        #self.cls_id_to_det_head_idx_map = self.cls_id_to_det_head_idx_map.cuda()

        if self.sched_algo == SchedAlgo.AdaptiveRR:
            self.processing_time_limit_sec = 0.650 # Every x ms, reset
            self.sched_reset()

        self.pc_range = self.vfe.point_cloud_range.cpu()
        self.projection_stream = torch.cuda.Stream()

    def initialize(self, batch_dict):
        batch_dict['projections_nms'] = None
        latest_token = batch_dict['metadata'][0]['token']
        scene_token = self.token_to_scene[latest_token]

        if scene_token != self.prev_scene_token:
            self.sched_reset()
            if self.enable_projection:
                self.projection_reset()
            self.prev_scene_token = scene_token

        return batch_dict

    def get_nonempty_tiles(self, voxel_coords):
        # Calculate where each voxel resides in which tile
        voxel_tile_coords = torch.div(voxel_coords[:, -1], self.tile_size_voxels, \
                rounding_mode='trunc').long()

        if self.training:
            nonempty_tile_coords = torch.unique(voxel_tile_coords, sorted=True)
            return nonempty_tile_coords
        else:
            nonempty_tile_coords, voxel_counts = torch.unique(voxel_tile_coords, \
                    sorted=True, return_counts=True)

            netc = nonempty_tile_coords.cpu().numpy()
            voxel_counts = voxel_counts.cpu().numpy()

            return voxel_tile_coords, netc, voxel_counts

    def schedule0(self, batch_dict):
        return batch_dict

    def schedule1(self, batch_dict):
        voxel_coords = batch_dict['voxel_coords']
        if self.training:
            batch_dict['chosen_tile_coords'] = self.get_nonempty_tiles(voxel_coords)
            return batch_dict
        self.measure_time_start('Sched')
        voxel_tile_coords, netc, netc_vcounts = self.get_nonempty_tiles(voxel_coords)
        vcount_area = np.zeros((self.tcount,), dtype=netc_vcounts.dtype)
        vcount_area[netc] = netc_vcounts
        vcount_area = np.expand_dims(vcount_area, 0)
        batch_dict['vcount_area'] = torch.from_numpy(vcount_area).int().cuda()

        if self.sched_algo == SchedAlgo.MirrorRR:
            netc, netc_vcounts= fill_tile_gaps(netc, netc_vcounts)
        elif self.sched_algo == SchedAlgo.AdaptiveRR:
            latest_token = batch_dict['metadata'][0]['token']
            cur_ts = self.token_to_ts[latest_token]

            # upper limit 25%
            if self.num_blacklisted_tiles > netc.shape[0] - int(self.tcount//4):
                self.num_blacklisted_tiles = netc.shape[0] - int(self.tcount//4)

            if self.reset_ts is not None:
                elapsed_time_sec = (cur_ts - self.reset_ts) / 1000000.0
                if elapsed_time_sec > self.processing_time_limit_sec:
                    # Reset
                    if self.processed_area_perc < 1.0:
                        # We need to blacklist tiles
                        self.num_blacklisted_tiles += int(np.ceil(\
                                (netc.shape[0] - self.num_blacklisted_tiles) * \
                                (1.0 - self.processed_area_perc)))
                        self.num_blacklisted_tiles = min(self.num_blacklisted_tiles, \
                                netc.shape[0] - int(self.tcount//4))
                    elif self.processed_area_perc > 1.0:
                        self.num_blacklisted_tiles -= int(np.ceil(\
                                (netc.shape[0] - self.num_blacklisted_tiles) * \
                                (self.processed_area_perc - 1.0)))
                        self.num_blacklisted_tiles = max(self.num_blacklisted_tiles, 0)

                    self.reset_ts = cur_ts
                    self.processed_area_perc = 0.

            else:
                self.reset_ts = cur_ts

            if self.num_blacklisted_tiles > 0:
                lptr, rptr = 0, netc_vcounts.shape[0]-1
                bt = self.num_blacklisted_tiles
                while bt > 0 and rptr - lptr > 0:
                    if netc_vcounts[lptr] < netc_vcounts[rptr]:
                        lptr += 1
                    else:
                        rptr -= 1
                    bt -= 1
                netc = netc[lptr:rptr+1]
                netc_vcounts = netc_vcounts[lptr:rptr+1]

            self.cur_netc_num_tiles = netc.shape[0]

        batch_dict['nonempty_tile_coords'] = netc

        if self.sched_algo == SchedAlgo.RoundRobin or self.sched_algo == SchedAlgo.AdaptiveRR:
            num_tiles, tiles_queue = round_robin_sched_helper(
                    netc, self.last_tile_coord, self.tcount)
        elif self.sched_algo == SchedAlgo.MirrorRR:
            num_tiles, vcounts_all, tiles_queue = mirror_sched_helper(
                    netc, netc_vcounts,
                    self.last_tile_coord, self.tcount)

        if self.sched_algo == SchedAlgo.RoundRobin or self.sched_algo == SchedAlgo.MirrorRR or \
                self.sched_algo == SchedAlgo.AdaptiveRR:
            batch_dict['tiles_queue'] = tiles_queue
            self.add_dict['nonempty_tiles'].append(netc.tolist())
            bb3d_times, post_bb3d_times = self.calibrator.pred_req_times_ms(\
                    vcount_area, tiles_queue, num_tiles)
            batch_dict['post_bb3d_times'] = post_bb3d_times
            tpreds = bb3d_times + post_bb3d_times
            psched_start_time = time.time()
            rem_time_ms = (batch_dict['abs_deadline_sec'] - psched_start_time) * 1000

            # Choose configuration that can meet the deadline, that's it
            diffs = tpreds < rem_time_ms

            ##### MANUAL OVERRIDE
            #tiles_to_run = 4
            #for idx, nt in enumerate(num_tiles):
            #    if nt >= tiles_to_run:
            #        tiles_idx = idx + 1
            #        break
            #####

            if diffs[-1]:
                chosen_tile_coords = netc
                self.add_dict['bb3d_preds'].append(float(bb3d_times[-1]))
                if self.sched_algo == SchedAlgo.MirrorRR:
                    self.last_tile_coord = self.init_tile_coord
                tiles_idx=0
            else:
                tiles_idx=1
                while tiles_idx < diffs.shape[0] and diffs[tiles_idx]:
                    tiles_idx += 1

                self.add_dict['bb3d_preds'].append(float(bb3d_times[tiles_idx-1]))

                # Voxel filtering is needed
                if self.sched_algo == SchedAlgo.RoundRobin or self.sched_algo == SchedAlgo.AdaptiveRR:
                    chosen_tile_coords = tiles_queue[:tiles_idx]
                    self.last_tile_coord = chosen_tile_coords[-1].item()
                else:
                    chosen_tile_coords = np.concatenate((self.mtiles, tiles_queue[:tiles_idx-1]))

                tile_filter = cuda_point_tile_mask.point_tile_mask(voxel_tile_coords, \
                        torch.from_numpy(chosen_tile_coords).cuda())

                if 'voxel_features' in batch_dict:
                    batch_dict['voxel_features'] = \
                            batch_dict['voxel_features'][tile_filter].contiguous()
                batch_dict['voxel_coords'] = voxel_coords[tile_filter].contiguous()

            batch_dict['chosen_tile_coords'] = chosen_tile_coords
            self.add_dict['chosen_tiles_1'].append(chosen_tile_coords.tolist())
        elif self.sched_algo == SchedAlgo.ProjectionOnly:
            batch_dict['chosen_tile_coords'] = netc
        self.measure_time_end('Sched')

        batch_dict['record_int_vcoords'] = True
        batch_dict['tile_size_voxels'] = self.tile_size_voxels_int
        batch_dict['num_tiles'] = self.tcount

        return batch_dict

    # Recalculate chosen tiles based on the time spent on bb3d
    def schedule2(self, batch_dict):
        torch.cuda.synchronize()
        vcoords = batch_dict['bb3d_intermediary_vcoords']
        vcoords.insert(0, batch_dict['vcount_area'])
        voxel_dists = torch.cat(vcoords, dim=0).cpu().numpy() # 4 x 18

        self.calibrator.commit_bb3d_updates(batch_dict['chosen_tile_coords'], voxel_dists)

        post_bb3d_times = batch_dict['post_bb3d_times']
        rem_time_ms = (batch_dict['abs_deadline_sec'] - time.time()) * 1000
        diffs = post_bb3d_times < rem_time_ms

        m = int(self.sched_algo == SchedAlgo.MirrorRR)
        if not diffs[batch_dict['chosen_tile_coords'].shape[0]-1-m]:
            tiles_idx=1
            while tiles_idx < diffs.shape[0] and diffs[tiles_idx]:
                tiles_idx += 1

            ctc = batch_dict['tiles_queue'][:tiles_idx-m]
            if self.sched_algo == SchedAlgo.MirrorRR:
                batch_dict['chosen_tile_coords'] = np.concatenate((self.mtiles, ctc))
            elif self.sched_algo == SchedAlgo.RoundRobin or self.sched_algo == SchedAlgo.AdaptiveRR:
                batch_dict['chosen_tile_coords'] = ctc
                self.last_tile_coord = ctc[-1].item()

        if self.sched_algo == SchedAlgo.MirrorRR and \
                batch_dict['chosen_tile_coords'].shape[0] > self.mtiles.shape[0]:
            self.last_tile_coord = batch_dict['chosen_tile_coords'][-1].item()
        ctc = batch_dict['chosen_tile_coords'].tolist()
        self.add_dict['chosen_tiles_2'].append(ctc)

        if self.sched_algo == SchedAlgo.AdaptiveRR:
            self.processed_area_perc += len(ctc) / self.cur_netc_num_tiles

        return batch_dict

    # This method adds the latests detections to the queue
    def add_past_proj_to_queue(self):
        batch_dict = self.latest_batch_dict
        if batch_dict is None or self.cur_pose is None:
            return batch_dict

        pred_dict = batch_dict['final_box_dicts'][0]

        # Before appending the dets, extract the projected ones
        proj_mask = pred_dict['pred_scores'] >= self.score_thresh
        for k in ('pred_boxes', 'pred_labels', 'pred_scores'):
            pred_dict[k] = pred_dict[k][proj_mask]

        new_dets_dict = {}
        score_inds = torch.argsort(pred_dict['pred_scores'])
        for k in ('pred_boxes', 'pred_labels'):
            new_dets_dict[k] = pred_dict[k][score_inds]

        # update num dets per tile
        W, W_start = self.pc_range[3] - self.pc_range[0], self.pc_range[0]
        div = W / self.tcount
        tile_inds = torch.div((new_dets_dict['pred_boxes'][:, 0] - W_start), div, \
                rounding_mode='trunc').short()
        tile_bins = torch.bincount(tile_inds, minlength=self.tcount)
        ctc = torch.from_numpy(batch_dict['chosen_tile_coords']).long()
        self.num_dets_per_tile[ctc] = tile_bins[ctc]

        # NOTE The cur_pose and cur_ts here actually belongs to previous sample
        self.past_poses = torch.cat((self.past_poses, self.cur_pose.unsqueeze(0)))
        self.past_ts = torch.cat((self.past_ts, self.cur_ts))
        # Append the pose idx for the detection that will be added
        num_dets = new_dets_dict['pred_boxes'].size(0)
        past_poi = self.past_detections['pose_idx']
        poi = torch.full((num_dets,), self.past_poses.size(0)-1, dtype=past_poi.dtype)
        self.past_detections['pose_idx'] = torch.cat((past_poi, poi))

        for k in ('pred_boxes', 'pred_labels'):
            self.past_detections[k] = torch.cat((self.past_detections[k], new_dets_dict[k]))

        return batch_dict

    def schedule3(self, batch_dict):
        batch_dict['projections_nms'] = None
        if self.enable_projection:
            # Add the detection results of previous sample
            self.add_past_proj_to_queue()
            latest_token = batch_dict['metadata'][0]['token']
            scene_token = self.token_to_scene[latest_token]
            self.cur_pose = self.token_to_pose[latest_token]
            self.cur_ts = self.token_to_ts[latest_token]

            # Remove detections which are no more needed
            active_num_dets = torch.sum(self.num_dets_per_tile)
            #NOTE we need to tune the coeff here , 2.0 is giving good results!
            coeff = 2.0
            max_num_proj = int(active_num_dets * coeff)
            if self.past_detections['pred_boxes'].size(0) > max_num_proj:
                # Remove oldest dets
                for k in ['pose_idx', 'pred_boxes', 'pred_labels']:
                    self.past_detections[k] = self.past_detections[k][-max_num_proj:]

            # Weed out using the pose_idx of first det
            if self.past_detections['pose_idx'].size(0) > 0:
                pose_idx_0 = self.past_detections['pose_idx'][0]
                self.past_poses = self.past_poses[pose_idx_0:]
                self.past_ts = self.past_ts[pose_idx_0:]
                self.past_detections['pose_idx'] = self.past_detections['pose_idx'] - pose_idx_0

            # Do projection in the GPU
            if self.past_detections['pred_boxes'].size(0) > 0:
                proj_dict = {}
                proj_dict['pred_scores'] = self.score_thresh - \
                        (self.score_thresh / (self.past_detections['pose_idx'] + 2))
                proj_dict['pred_labels'] = (self.past_detections['pred_labels'] - 1)
                with torch.cuda.stream(self.projection_stream):
                    proj_dict['pred_boxes'] = cuda_projection.project_past_detections(
                            self.past_detections['pred_boxes'].cuda(),
                            self.past_detections['pose_idx'].cuda(),
                            self.past_poses.cuda(),
                            self.cur_pose.cuda(),
                            self.past_ts.cuda(),
                            self.cur_ts.item())
                    self.projection_stream.synchronize() # maybe not needed?
                    proj_dict['pred_boxes'] = proj_dict['pred_boxes'].cpu()

                # clear up detections which fall under the chosen tiles and also the overtimed ones
                projb = proj_dict['pred_boxes']
                box_x, box_y = projb[:,0], projb[:,1]
                range_mask = box_x >= self.pc_range[0]
                range_mask = torch.logical_and(range_mask, box_x <= self.pc_range[3])
                range_mask = torch.logical_and(range_mask, box_y >= self.pc_range[1])
                range_mask = torch.logical_and(range_mask, box_y <= self.pc_range[4])

                # This op can make nms faster
                with torch.cuda.stream(self.projection_stream):
                    batch_dict['projections_nms'] = cuda_projection.split_projections(
			    proj_dict['pred_boxes'][range_mask],
			    proj_dict['pred_scores'][range_mask],
			    proj_dict['pred_labels'][range_mask],
			    self.cls_id_to_det_head_idx_map,
			    self.num_det_heads,
			    True) # moves results to gpu if true

                for k in ('pred_boxes', 'pred_labels', 'pose_idx'):
                    self.past_detections[k] = self.past_detections[k][range_mask]

                self.projection_stream.synchronize()

        return batch_dict

    def schedule4(self, batch_dict):
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
                box_preds=pred_boxes.cuda(),
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict

    def sched_reset(self):
        self.processed_area_perc = 0.
        self.num_blacklisted_tiles = 0
        self.reset_ts = None

    def projection_reset(self):
        # Poses include [cst(3) csr(4) ept(3) epr(4)]
        self.cur_pose, self.cur_ts = None, None
        if self.sched_algo == SchedAlgo.ProjectionOnly:
            for k in ('pred_boxes', 'pred_labels', 'pose_idx'):
                self.past_detections[k] = []
            self.past_poses, self.past_ts = [], []
        else:
            self.past_detections = self.get_empty_det_dict()
            self.past_detections['pose_idx'] = torch.zeros([0], dtype=torch.long)
#                device=self.past_detections["pred_labels"].device)
            self.past_poses = torch.zeros([0, 14], dtype=torch.float)
            self.past_ts = torch.zeros([0], dtype=torch.long)
            self.num_dets_per_tile = torch.zeros([self.tcount], dtype=torch.long)
        self.last_tile_coord = self.init_tile_coord

    def calibrate(self):

        self.calibrator = AnytimeCalibrator(self)

        collect_data = False
        try:
            self.calibrator.read_calib_data(f"calib_data_{self.sched_algo}.json")
        except OSError:
            collect_data = True

        score_threshold = self.dense_head.model_cfg.POST_PROCESSING.SCORE_THRESH
        # this temporary threshold will allow us to do calibrate cudnn benchmarking
        # of all detection heads, preventing to skip any of them
        self.dense_head.model_cfg.POST_PROCESSING.SCORE_THRESH = 0.0001
        super().calibrate(1)
        self.dense_head.model_cfg.POST_PROCESSING.SCORE_THRESH = score_threshold

        self.enable_projection = (not self.keep_projection_disabled)
        self.projection_reset()
        self.sched_reset()
        if self.training:
            return None

        if collect_data:
            self.calibrator.collect_data(self.sched_algo, f"calib_data_{self.sched_algo}.json")

        return None

    def post_eval(self):
        # remove first ones due to calibration
        self.add_dict['bb3d_preds'] = self.add_dict['bb3d_preds'][1:]
        self.add_dict['nonempty_tiles'] = self.add_dict['nonempty_tiles'][1:]
        self.add_dict['chosen_tiles_1'] = self.add_dict['chosen_tiles_1'][1:]
        self.add_dict['chosen_tiles_2'] = self.add_dict['chosen_tiles_2'][1:]

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
                    torch.zeros((num_dets,), dtype=torch.long)) #, device='cuda'))
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


