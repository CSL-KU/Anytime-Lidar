from .detector3d_template import Detector3DTemplate
from .anytime_calibrator import AnytimeCalibrator
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

from ..model_utils import model_nms_utils
from ...ops.cuda_projection import cuda_projection
from ...ops.cuda_point_tile_mask import cuda_point_tile_mask
from .. import load_data_to_gpu

class AnytimeTemplateV2(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        if 'BACKBONE_2D' in self.model_cfg:
            self.model_cfg.BACKBONE_2D.TILE_COUNT = self.model_cfg.TILE_COUNT
            self.model_cfg.BACKBONE_2D.METHOD = self.model_cfg.METHOD
        if 'DENSE_HEAD' in self.model_cfg:
            self.model_cfg.DENSE_HEAD.TILE_COUNT = self.model_cfg.TILE_COUNT
            self.model_cfg.DENSE_HEAD.METHOD = self.model_cfg.METHOD
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

        # This number will be determined by the scheduling algorithm initially for each input
        self.projection_stream = torch.cuda.Stream()

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

        #self.calibrating_now = False
        self.add_dict = self._eval_dict['additional']
        self.add_dict['bb3d_preds'] = []
        self.add_dict['nonempty_tiles'] = []
        self.add_dict['chosen_tiles_1'] = []
        self.add_dict['chosen_tiles_2'] = []
        #for k in ('voxel_counts', 'num_tiles', 'PostSched'):
        #    self.add_dict[k] = []

        self.proj_time_limit_musec = 1.0 * 1000000

        self.sched_algo = self.model_cfg.METHOD

        if self.sched_algo == SchedAlgo.RoundRobin:
            self.init_tile_coord = -1
        elif self.sched_algo == SchedAlgo.MirrorRR:
            self.init_tile_coord = 0
            m2 = self.tcount//2
            m1 = m2 - 1
            self.mtiles = np.array([m1, m2], dtype=np.int32)
        self.last_tile_coord = self.init_tile_coord

        self.past_detections = {'num_dets': []}
        self.prev_scene_token = ''
        if self.sched_algo == SchedAlgo.ProjectionOnly:
            self.past_poses = []
            self.past_ts = []
        else:
            # Poses include [cst(3) csr(4) ept(3) epr(4)]
            self.past_poses = torch.zeros([0, 14], dtype=torch.float)
            self.past_ts = torch.zeros([0], dtype=torch.long)

        # Needs to be calibrated
        self.score_thresh = self.model_cfg.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH

        total_num_classes = sum([m.size(0) for m in self.dense_head.class_id_mapping_each_head])
        self.cls_id_to_det_head_idx_map = torch.zeros((total_num_classes,), dtype=torch.int)
        self.num_det_heads = len(self.dense_head.class_id_mapping_each_head)
        for i, cls_ids in enumerate(self.dense_head.class_id_mapping_each_head):
            for cls_id in cls_ids:
                self.cls_id_to_det_head_idx_map[cls_id] = i
        self.cls_id_to_det_head_idx_map = self.cls_id_to_det_head_idx_map.cuda()

    # When projecting, set the pred scores to a number below 0.3.
    # After running nms, remove the dets that are projected using their
    # pred score when adding the new dets to the past detections.
    # However, output the entire detections.
    def projection(self, batch_dict):
        batch_dict['projections'] = None
        if not self.enable_projection:
            return batch_dict

        with torch.cuda.stream(self.projection_stream):
            # Do post processing of previous sample here to facilitate calculating
            # remaning time to dealdine
            self.projection_post()

            latest_token = batch_dict['metadata'][0]['token']
            scene_token = self.token_to_scene[latest_token]
            self.cur_pose = self.token_to_pose[latest_token]
            self.cur_ts = self.token_to_ts[latest_token]

            if scene_token != self.prev_scene_token:
                self.projection_reset()
                self.prev_scene_token = scene_token

            if self.sched_algo == SchedAlgo.ProjectionOnly:
                return self.projection_for_test(batch_dict)
            # Clear unuseful dets
            if self.past_ts.size(0) > 0 and self.cur_ts - self.past_ts[0] > self.proj_time_limit_musec:
                self.past_poses = self.past_poses[1:]
                self.past_ts = self.past_ts[1:]
                nd = self.past_detections['num_dets'].pop(0)
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
                proj_dict['pred_labels'] = (self.past_detections['pred_labels'] - 1)
                batch_dict['projections'] = proj_dict

                proj_dicts = cuda_projection.split_projections(
                        proj_dict['pred_boxes'],
                        proj_dict['pred_scores'],
                        proj_dict['pred_labels'],
                        self.cls_id_to_det_head_idx_map,
                        self.num_det_heads)
                batch_dict['projections'] = proj_dicts

        return batch_dict


    def projection_post(self):
        batch_dict = self.latest_batch_dict
        if not self.enable_projection or batch_dict is None or self.cur_pose is None:
            return

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
            if self.sched_algo == SchedAlgo.MirrorRR:
                netc, voxel_counts = fill_tile_gaps(netc, voxel_counts)

            return voxel_tile_coords, netc, voxel_counts

    def schedule1(self, batch_dict):
        voxel_coords = batch_dict['voxel_coords']
        if self.training:
            batch_dict['chosen_tile_coords'] = self.get_nonempty_tiles(voxel_coords)
            return batch_dict
        self.measure_time_start('Sched')
        voxel_tile_coords, netc, netc_vcounts = self.get_nonempty_tiles(voxel_coords)
        batch_dict['nonempty_tile_coords'] = netc

        if self.sched_algo == SchedAlgo.RoundRobin:
            num_tiles, vcounts_all, tiles_queue = round_robin_sched_helper(
                    batch_dict['nonempty_tile_coords'], netc_vcounts,
                    self.last_tile_coord, self.tcount)
        elif self.sched_algo == SchedAlgo.MirrorRR:
            num_tiles, vcounts_all, tiles_queue = mirror_sched_helper(
                    batch_dict['nonempty_tile_coords'], netc_vcounts,
                    self.last_tile_coord, self.tcount)

        if self.sched_algo == SchedAlgo.RoundRobin or self.sched_algo == SchedAlgo.MirrorRR:
            batch_dict['tiles_queue'] = tiles_queue
            self.add_dict['nonempty_tiles'].append(batch_dict['nonempty_tile_coords'].tolist())
            self.projection_stream.synchronize()

            vcounts_all = torch.from_numpy(vcounts_all)
            num_tiles = torch.from_numpy(num_tiles)
            bb3d_times, post_bb3d_times = self.calibrator.pred_req_times_ms(vcounts_all, num_tiles)
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
            else:
                tiles_idx=1
                while tiles_idx < diffs.shape[0] and diffs[tiles_idx]:
                    tiles_idx += 1

                self.add_dict['bb3d_preds'].append(float(bb3d_times[tiles_idx-1]))

                # Voxel filtering is needed
                if self.sched_algo == SchedAlgo.RoundRobin:
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

        return batch_dict

    # Recalculate chosen tiles based on the time spent on bb3d
    def schedule2(self, batch_dict):
        torch.cuda.synchronize()
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
            elif self.sched_algo == SchedAlgo.RoundRobin:
                batch_dict['chosen_tile_coords'] = ctc
                self.last_tile_coord = ctc[-1].item()

        if self.sched_algo == SchedAlgo.MirrorRR and \
                batch_dict['chosen_tile_coords'].shape[0] > self.mtiles.shape[0]:
            self.last_tile_coord = batch_dict['chosen_tile_coords'][-1].item()
        self.add_dict['chosen_tiles_2'].append(batch_dict['chosen_tile_coords'].tolist())

        return batch_dict

    # NOTE this is just a sanity check, not actual scheduling
    def schedule3(self, batch_dict):
        rem_time_ms = (batch_dict['abs_deadline_sec'] - time.time()) * 1000
        req_time_ms = self.calibrator.pred_final_req_time_ms(batch_dict['dethead_indexes'])
        tdiff = rem_time_ms - req_time_ms
        if tdiff < 0:
            print('Remaining and requires time (ms):', rem_time_ms, req_time_ms)
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
        self.cur_pose, self.cur_ts = None, None
        if self.sched_algo == SchedAlgo.ProjectionOnly:
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
        self.enable_projection = True
        self.projection_reset()

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


