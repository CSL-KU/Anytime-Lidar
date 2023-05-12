from .detector3d_template import Detector3DTemplate
import torch
from nuscenes.nuscenes import NuScenes
import time
import sys
import json
import numpy as np
import scipy
import gc

from ...ops.cuda_projection import cuda_projection
from ...ops.cuda_point_tile_mask import cuda_point_tile_mask
from .. import load_data_to_gpu

#os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

class AnytimeTemplate(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        if 'BACKBONE_2D' in self.model_cfg:
            self.model_cfg.BACKBONE_2D.TILE_COUNT = self.model_cfg.TILE_COUNT
        if 'DENSE_HEAD' in self.model_cfg:
            self.model_cfg.DENSE_HEAD.TILE_COUNT = self.model_cfg.TILE_COUNT
        self.module_list = self.build_networks()
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.manual_seed(0)
#        torch.use_deterministic_algorithms(True)

        ################################################################################
        #self.tcount= torch.tensor(self.model_cfg.TILE_COUNT).long().cuda()
        self.tcount = self.model_cfg.TILE_COUNT
        self.tcount_cuda = torch.tensor(self.model_cfg.TILE_COUNT).long().cuda()
        self.total_num_tiles = self.tcount[0] * self.tcount[1]

        # This number will be determined by the scheduling algorithm initially for each input
        self.last_tile_coord = -1
        #self.reduce_mask_stream = torch.cuda.Stream()
        self.tile_size_voxels = torch.from_numpy(\
                self.dataset.grid_size[:2] / self.tcount).cuda().long()

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

        self.past_detections = {'num_dets': []}
        # Poses include [cst(3) csr(4) ept(3) epr(4)]
        self.past_poses = torch.zeros([0, 14], dtype=torch.float)
        self.past_ts = torch.zeros([0], dtype=torch.long)
        self.det_timeout_limit = int(0.5 * 1000000) # in microseconds
        self.prev_scene_token = ''
        ################################################################################

        self.calibrating_now = False
        self.add_dict = self._eval_dict['additional']
        for k in ('voxel_counts', 'num_tiles', 'PostSched'):
            self.add_dict[k] = []

        # these values needs calibration
        self.time_pred_coeffs = []
        self.pred_net_time_stats = []

        self.calib_num_tiles = -1
        self.skip_projection=False

        self.tile_ages = torch.ones(self.total_num_tiles, dtype=torch.long, device='cuda')
        self.RoundRobin = 0
        self.AgingWithDistance = 1

        self.sched_algo = self.RoundRobin
#        self.sched_algo = self.AgingWithDistance

    def projection_init(self, batch_dict):
        latest_token = batch_dict['metadata'][0]['token']
        scene_token = self.token_to_scene[latest_token]
        if scene_token != self.prev_scene_token:
            self.projection_reset()
            self.prev_scene_token = scene_token

        self.cur_pose = self.token_to_pose[latest_token]
        self.cur_ts = self.token_to_ts[latest_token]

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

            # Filter the predictions outside of chosen tiles
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
        if self.past_poses.size(0) > 0 and not self.skip_projection:

            # The mask here allows filtering the projections on chosen tile coords
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

        self.skip_projection=False

        return batch_dict

    def get_nonempty_tiles(self, voxel_coords):
        # Calculate where each voxel resides in which tile
        tile_coords = torch.div(voxel_coords[:, -2:], self.tile_size_voxels, \
                rounding_mode='trunc').long()

        voxel_tile_coords = tile_coords[:, 1] * self.tcount[1] + tile_coords[:, 0]
        nonempty_tile_coords, voxel_counts = torch.unique(voxel_tile_coords, \
                sorted=True, return_counts=True)
        return voxel_tile_coords, nonempty_tile_coords, voxel_counts

    def schedule(self, batch_dict):
        self.measure_time_start('Sched')
        voxel_coords = batch_dict['voxel_coords']
        voxel_tile_coords, netc, netc_vcounts= self.get_nonempty_tiles(voxel_coords)

        if self.training:
            batch_dict['chosen_tile_coords'] = netc
            return

        num_nonempty_tiles = netc.size(0)

        if self.sched_algo == self.RoundRobin:

            # Here I need to run the scheduling algorithm
            # find the index+1 of the last tile that was processed in the previous round
            # if it doesn't exist, find the one that is smaller closest
            tile_begin_idx = \
                    (netc > self.last_tile_coord).type(torch.uint8).argmax()
            tl_end = tile_begin_idx + num_nonempty_tiles
            ntc = netc.expand(2, num_nonempty_tiles).flatten()
            netc = ntc[tile_begin_idx:tl_end].contiguous()

            num_tiles = torch.arange(1, netc.size(0)+1, device=netc_vcounts.device).float()
            cnts = netc_vcounts.expand(2, netc_vcounts.size(0)).flatten()
            cnts = cnts[tile_begin_idx:tl_end].contiguous()
            cnts_cumsum = torch.cumsum(cnts, dim=0).float()

        elif self.sched_algo == self.AgingWithDistance:

            netc_y = torch.div(netc, self.tcount[1], rounding_mode='trunc') - (self.tcount[1]//2)
            netc_x = (netc % self.tcount[1]) - (self.tcount[0]//2)
            s = torch.pow(netc_y, 2) + torch.pow(netc_x, 2)
            s = s.float() / ((self.tcount[0]//2)**2 + (self.tcount[1]//2)**2) # normalize
            dists = 1.0 - s # prioritize closer tiles

            ages = self.tile_ages[netc]
            prios = ages * dists # NOTE, this is an ad-hoc formula

            inds = torch.argsort(prios, descending=True)
            netc = netc[inds]
            netc_vcounts = netc_vcounts[inds]

            num_tiles = torch.arange(1, netc.size(0)+1, device=netc_vcounts.device).float()
            cnts_cumsum = torch.cumsum(netc_vcounts, dim=0).float()

        # Get execution time predictions
        if self.time_pred_coeffs:
            coeffs = self.time_pred_coeffs
            pred_time_stats = self.pred_net_time_stats
        else:
            # do it 2 for worst case
            coeffs = [torch.zeros(6, device='cuda')] * 2
            pred_time_stats = [{'95perc':0.0, '99perc':0.0, 'max': 0.0}] * 2

        tpreds = []
        for C in coeffs:
            #tpreds = C[0]*cnts_cumsum + C[1]*num_tiles + C[2]
            x1, y1 = cnts_cumsum, num_tiles
            XY = torch.stack((torch.ones(x1.size(0), device='cuda'), x1, y1, \
                    x1*y1, x1**2, y1**2), dim=1)
            tpreds.append(torch.matmul(XY, C))

        tpreds = sum(tpreds)
        torch.cuda.synchronize()
        self.psched_start_time = time.time()
        rem_time = batch_dict['abs_deadline_sec'] - self.psched_start_time
        for pt_stats in pred_time_stats:
            rem_time -= pt_stats['99perc']

        diffs = (tpreds < rem_time).cpu()
        idx = torch.sum(diffs).item()

        if self.calibrating_now:
            idx = self.calib_num_tiles
        elif idx < 1: #self.total_num_tiles//20:
            idx = int(num_tiles[0]) #self.total_num_tiles//20 # setting to 1 can cause problems

        if idx == num_nonempty_tiles:
            # Use all tiles
            self.skip_projection=True
            chosen_tile_coords = netc
            self.last_tile_coord = -1
        else:
            # Voxel filtering is needed
            chosen_tile_coords = netc[:idx]
            self.last_tile_coord = chosen_tile_coords[-1]
            tile_filter = cuda_point_tile_mask.point_tile_mask(voxel_tile_coords, \
                    chosen_tile_coords)
            batch_dict['voxel_features'] = batch_dict['voxel_features'][tile_filter].contiguous()
            batch_dict['voxel_coords'] = voxel_coords[tile_filter].contiguous()
            #print('Filtered voxel coords:',batch_dict['voxel_coords'].size())
        batch_dict['chosen_tile_coords'] = chosen_tile_coords

        self.add_dict['voxel_counts'].append(\
                batch_dict['voxel_coords'].size(0))
        self.add_dict['num_tiles'].append(batch_dict['chosen_tile_coords'].size(0))

        self.tile_ages += 1
        self.tile_ages[batch_dict['chosen_tile_coords']] = 1

        self.measure_time_end('Sched')

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
        self.last_tile_coord = -1
        self.tile_ages = torch.ones(self.total_num_tiles, dtype=torch.long, device='cuda')

    def calc_time_pred_coeffs(self, num_voxels, num_tiles, psched_time):
        X = np.array(num_voxels, dtype=np.float32)
        Y = np.array(num_tiles, dtype=np.float32)
        Z = np.array(psched_time, dtype=np.float32)
        x1, y1, z1 = X.flatten(), Y.flatten(), Z.flatten()

        # linear
        #A = np.c_[x1, y1, np.ones(x1.shape[0])]
        #C,_,_,_ = scipy.linalg.lstsq(A, z1)    # coefficients
        #plane_z = C[0]*x1 + C[1]*y1 + C[2]

        # quadratic
        xy = np.stack([x1, y1], axis=1)
        A = np.c_[np.ones(x1.shape[0]), xy, np.prod(xy, axis=1), xy**2]
        C,_,_,_ = scipy.linalg.lstsq(A, z1)
        plane_z = np.dot(np.c_[np.ones(x1.shape), x1, y1, x1*y1, x1**2, y1**2], C)

        self.time_pred_coeffs.append(torch.from_numpy(C).float().cuda())
        print('self.time_pred_coeffs', self.time_pred_coeffs)

        diff = z1 - plane_z
        perc95 = np.percentile(diff, 95, method='lower')
        perc99 = np.percentile(diff, 99, method='lower')
        self.pred_net_time_stats.append({
            'min': float(min(diff)),
            'avrg': float(sum(diff)/len(diff)),
            '95perc': float(perc95),
            '99perc': float(perc99),
            'max': float(max(diff))})

    def calibrate(self, fname='calib_raw_data.json'):
        super().calibrate(1)
        self.enable_projection = True
        self.projection_reset()

        for l in self.add_dict.values():
            l.clear()

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

            psched_time = calib_dict["PostSched"]
            if 'PostSched_mid' in calib_dict:
                psched_until_head = calib_dict['PostSched_mid']
                psched_after_head = np.array(psched_time) - np.array(psched_until_head)
                self.calc_time_pred_coeffs(num_voxels, num_tiles, psched_until_head)
                self.calc_time_pred_coeffs(num_voxels, num_tiles, psched_after_head)
            else:
                self.calc_time_pred_coeffs(num_voxels, num_tiles, psched_time)

        except FileNotFoundError:
            print(f'Calibration file {fname} not found, running calibration')
            self.calibrating_now = True # time calibration!
            self.calibration_procedure(fname)
            sys.exit()

        print('Time prediction stats:')
        print(self.pred_net_time_stats)

        return None #pred_dicts

    def calibration_procedure(self, fname="calib_raw_data.json"):
        gc.disable()
        all_max_num_tiles = []
        for i in range(len(self.dataset)):
            data_dict = self.dataset.getitem_pre(i)
            data_dict = self.dataset.getitem_post(data_dict)
            data_dict = self.dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            _, nonempty_tile_coords, _ = self.get_nonempty_tiles(data_dict['voxel_coords'])
            max_num_tiles = nonempty_tile_coords.size(0)
            all_max_num_tiles.append(max_num_tiles)

        print(min(all_max_num_tiles), max(all_max_num_tiles))
        torch.cuda.empty_cache()
        gc.collect()

        # 10 different tiles should be enough
        for num_tiles in range(1, max(all_max_num_tiles)+1, min(all_max_num_tiles)//10):
            print('Num tiles:', num_tiles)
            for i in range(len(self.dataset)):
                if num_tiles <= all_max_num_tiles[i]:
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
        self.add_dict['exec_times'] = self.get_time_dict()
        self.add_dict['exec_time_stats'] = self.get_time_dict_stats()
        print('Time calibration Complete')
        with open(fname, 'w') as handle:
            json.dump(self.add_dict, handle, indent=4)

    def post_eval(self):
        self.add_dict['tcount'] = self.tcount
        print(f"\nDeadlines missed: {self._eval_dict['deadlines_missed']}\n")
