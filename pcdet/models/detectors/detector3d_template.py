import os
import copy
import time
import json
import datetime
import numpy as np
import socket

import torch
import torch.nn as nn

from nuscenes.utils.splits import all_speed_scenes
from nuscenes.utils.splits import train as train_scenes

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import find_all_spconv_keys
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads, load_data_to_gpu
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils
from ...ops.cuda_projection import cuda_projection

# NOTES
# Post processing is split into two functions, pre and post.
# Pre is called before the time measurement finishes.
# Post is called after the time measurement.
# The child class SHOULD NOT call any postprocessing function
# inside the forward function. It should only override
# the pre and post functions, or just let base class handle post processing.

def pre_forward_hook(module, inp_args):
    dataset_indexes = inp_args[0]
    data_dicts = [module.dataset.getitem_pre(i) for i in dataset_indexes]

    latest_token = data_dicts[0]['metadata']['token']

    if module.deadline_range is not None and not module.is_calibrating():
        use_calib_dataset = os.getenv('CALIBRATION', 0)
        scenes = train_scenes if use_calib_dataset else all_speed_scenes
        scene_name = module.token_to_scene_name[latest_token]
        dl_scale = scenes.index(scene_name) / (len(scenes) - 1)
        dr = module.deadline_range
        deadline_sec = ((dr[1] - dr[0]) * dl_scale + dr[0]) / 1000.0
    else:
        deadline_sec = module._eval_dict['deadline_sec']

    # NOTE The following prevents orin from spending unnecessary 50 ms
    # for tensor initialization, seems like a BUG
    dummy_tensor = torch.zeros(1024*1024, device='cuda')
    torch.cuda.synchronize()

    deadline_sec_override, reset = module.initialize(latest_token)
    if reset:
        module.latest_batch_dict = None
        module.latest_valid_dets = None
        module.dl_miss_streak = 0
    data_dicts = [module.dataset.getitem_post(dd) for dd in data_dicts]
    batch_dict = module.dataset.collate_batch(data_dicts)
    module.measure_time_start('End-to-end')
    module.measure_time_start('PreProcess')
    start_time = time.time()
    load_data_to_gpu(batch_dict)

    # if deadline is set in the init to override, use that, otherwise, use the regular one
    batch_dict['deadline_sec'] = deadline_sec_override if deadline_sec_override != 0. else deadline_sec
    batch_dict['start_time_sec'] = start_time
    batch_dict['abs_deadline_sec'] = start_time + batch_dict['deadline_sec']

    module.measure_time_end('PreProcess')
    return batch_dict

def post_forward_hook(module, inp_args, outp_args):
    assert not isinstance(outp_args, tuple)
    batch_dict = outp_args
    pp_args = module.post_processing_pre(batch_dict) # NMS
    if 'final_box_dicts' in  batch_dict:
        if 'pred_ious' in batch_dict['final_box_dicts'][0]:
            del batch_dict['final_box_dicts'][0]['pred_ious']
        for k,v in batch_dict['final_box_dicts'][0].items():
            batch_dict['final_box_dicts'][0][k] = v.cpu()
    elif 'batch_cls_preds' in  batch_dict:
        for arg_arr in pp_args[:3]:
            arg_arr[0].cpu()
        module.measure_time_end('DetectionHead')
    torch.cuda.synchronize()
    module.finish_time = time.time()
    module.measure_time_end('End-to-end')

    if 'bb3d_layer_time_events' in batch_dict and \
            'bb3d_layer_times' in module.add_dict:
        events = batch_dict['bb3d_layer_time_events']
        times = [events[i].elapsed_time(events[i+1]) for i in range(len(events)-1)]
        module.add_dict['bb3d_layer_times'].append(times)

    pred_dicts, recall_dict = module.post_processing_post(pp_args)

    tdiff = round(module.finish_time - batch_dict['abs_deadline_sec'], 3)
    module._eval_dict['deadline_diffs'].append(tdiff)

    dl_missed = (tdiff > 0)

   # print('post_bb3d time:', (module.finish_time- module.sync_time_ms)*1000.0, 'ms')
    ignore_dl_miss = module.is_calibrating() or (int(os.getenv('IGNORE_DL_MISS', 0)) == 1)
    if not ignore_dl_miss and not module.do_streaming_eval:
        if dl_missed:
            module._eval_dict['deadlines_missed'] += 1
            module.dl_miss_streak += 1
            print('Deadline', round(batch_dict['deadline_sec'] * 1000.0, 1), ' missed,',
                    tdiff * 1000.0, 'ms late. Total missed:',
                    module._eval_dict['deadlines_missed'])

            # Assume the program will abort the process when it misses the deadline
            if module.latest_valid_dets is not None:
                pred_dicts = copy.deepcopy(module.latest_valid_dets)
            elif module._det_dict_copy is not None:
                pred_dicts = [ module.get_dummy_det_dict() for p in pred_dicts ]
            else:
                print('Warning! Using pred_dicts even though the deadline was missed.')
        else:
            module.dl_miss_streak = 0
            module.latest_valid_dets = pred_dicts
        # total 2 seconds
        if module.dl_miss_streak == round(2000 / module.data_period_ms):
            #clear the buffer
            module.latest_valid_dets = None
    else:
        module.latest_valid_dets = pred_dicts

    torch.cuda.synchronize()
    module.calc_elapsed_times()
    module.last_elapsed_time_musec = int(module._time_dict['End-to-end'][-1] * 1000)
    module.latest_batch_dict = batch_dict

    return pred_dicts, recall_dict

class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        
        self._time_dict = {
            'End-to-end': [],
            'PreProcess': [],
        }
        self.update_time_dict(dict())
       
        self._eval_dict = {}
        self._eval_dict['additional'] = {}
        self.add_dict = self._eval_dict['additional']

        if self.model_cfg.get('DEADLINE_SEC', None) is not None:
            self._default_deadline_sec = float(model_cfg.DEADLINE_SEC)
            self._eval_dict['deadline_sec'] = self._default_deadline_sec
        else:
            self._default_deadline_sec = 10.0
            self._eval_dict['deadline_sec'] = 10.0  # loong deadline

        if 'DEADLINE_RANGE_MS' in os.environ:
            drange = os.getenv('DEADLINE_RANGE_MS').split('-')
            self.deadline_range = [float(r) for r in drange]
        else:
            self.deadline_range = None

        self.token_to_scene = {}
        self.token_to_scene_name = {}
        self.token_to_ts = {}
        try:
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
                self.token_to_scene_name[k] = v['scene_name']
        except:
            print("Couldn't find token_to_pos.json, not loading it.")
            pass

        self._eval_dict['deadlines_missed'] = 0
        self._eval_dict['deadline_diffs'] = []

        self.do_streaming_eval = self.model_cfg.get('STREAMING_EVAL', False)
        if self.do_streaming_eval:
            print('Doing streaming evaluation!')

        print('Default deadline is:', self._eval_dict['deadline_sec'])

        self._det_dict_copy = None
        self.pre_hook_handle = self.register_forward_pre_hook(pre_forward_hook)
        self.post_hook_handle = self.register_forward_hook(post_forward_hook)
        self.hooks_binded = True
    
        self.latest_batch_dict = None
        self.latest_valid_dets = None
        self.dl_miss_streak = 0

        self.calibrating = 0
        self.prev_scene_token = ''
        self.data_period_ms = int(os.getenv('DATASET_PERIOD', 500))

        ########################
        ###### PROJECTION ######
        ########################
        self.enable_projection = False
        self.enable_splitting_projections = True
        self.cur_pose, self.cur_ts = None, None
        self.past_detections = {}  #{'num_dets': []}
        # Poses include [cst(3) csr(4) ept(3) epr(4)]
        self.past_poses = torch.zeros([0, 14], dtype=torch.float)
        self.past_ts = torch.zeros([0], dtype=torch.long)

        self.pc_range = torch.tensor(self.dataset.point_cloud_range)
        ##self.pc_range = self.vfe.point_cloud_range.cpu() # what if vfe does not exist?
        self.projection_stream = torch.cuda.Stream()

        self.projection_coeff = float(self.model_cfg.get('PROJECTION_COEFF', 1.0))
        #print('Projection coefficient is', self.projection_coeff)

        # tile count
        self.tcount = self.model_cfg.get('TILE_COUNT', 18)
        if self.tcount is not None:
            self.tcount_cuda = torch.tensor(self.tcount).long().cuda()
        self.num_dets_per_tile = torch.zeros([self.tcount], dtype=torch.long)

        # Now, the problem is, we don't have dense head

        dh_cfg = self.model_cfg.get('DENSE_HEAD', None)
        if dh_cfg is None:
            self.score_thresh = 0.1
        else:
            self.score_thresh = dh_cfg.POST_PROCESSING.SCORE_THRESH


    def initialize(self, latest_token : str) -> (float, bool):
        scene_token = self.token_to_scene[latest_token]
        if scene_token != self.prev_scene_token:
            deadline_sec_override, reset = 0., True # don't do override for now
        else:
            deadline_sec_override, reset = 0., False
        self.prev_scene_token = scene_token

        if reset and self.enable_projection:
            self.projection_reset()

        return deadline_sec_override, reset 

    def projection_reset(self):
        # Poses include [cst(3) csr(4) ept(3) epr(4)]
        self.cur_pose, self.cur_ts = None, None
        self.past_detections = self.get_empty_det_dict()
        self.past_detections['pose_idx'] = torch.zeros([0], dtype=torch.long)
#                device=self.past_detections["pred_labels"].device)
        self.past_poses = torch.zeros([0, 14], dtype=torch.float)
        self.past_ts = torch.zeros([0], dtype=torch.long)
        self.num_dets_per_tile = torch.zeros([self.tcount], dtype=torch.long)

    # This method adds the latests detections to the queue
    def add_past_proj_to_queue(self, proj_all=False):
        batch_dict = self.latest_batch_dict
        if batch_dict is None or self.cur_pose is None:
            return batch_dict

        pred_dict = batch_dict['final_box_dicts'][0]

        # Before appending the dets, extract the projected ones
        if self.score_thresh == 0:
            for k in ('pred_boxes', 'pred_labels', 'pred_scores'):
                pred_dict[k] = pred_dict['orig_'+k]
        else:
            proj_mask = pred_dict['pred_scores'] > self.score_thresh
            for k in ('pred_boxes', 'pred_labels', 'pred_scores'):
                pred_dict[k] = pred_dict[k][proj_mask]

        new_dets_dict = {}
        score_inds = torch.argsort(pred_dict['pred_scores'])
        for k in ('pred_boxes', 'pred_labels'):
            new_dets_dict[k] = pred_dict[k][score_inds]
        if self.score_thresh == 0:
            new_dets_dict['pred_scores'] = pred_dict['pred_scores'][score_inds]

        # update num dets per tile
        W, W_start = self.pc_range[3] - self.pc_range[0], self.pc_range[0]
        div = W / self.tcount
        tile_inds = torch.div((new_dets_dict['pred_boxes'][:, 0] - W_start), div, \
                rounding_mode='trunc').short()
        tile_bins = torch.bincount(tile_inds, minlength=self.tcount)
        if 'chosen_tile_coords' in batch_dict:
            ctc = torch.from_numpy(batch_dict['chosen_tile_coords']).long()
            self.num_dets_per_tile[ctc] = tile_bins[ctc]
        else:
            self.num_dets_per_tile = tile_bins

        # NOTE The cur_pose and cur_ts here actually belongs to previous sample
        self.past_poses = torch.cat((self.past_poses, self.cur_pose.unsqueeze(0)))
        self.past_ts = torch.cat((self.past_ts, self.target_ts if proj_all else self.cur_ts))
        # Append the pose idx for the detection that will be added
        num_dets = new_dets_dict['pred_boxes'].size(0)
        past_poi = self.past_detections['pose_idx']
        poi = torch.full((num_dets,), self.past_poses.size(0)-1, dtype=past_poi.dtype)
        self.past_detections['pose_idx'] = torch.cat((past_poi, poi))

        for k in ('pred_boxes', 'pred_labels'):
            self.past_detections[k] = torch.cat((self.past_detections[k], new_dets_dict[k]))
        if self.score_thresh == 0:
            self.past_detections['pred_scores'] = torch.cat((self.past_detections['pred_scores'],
                    new_dets_dict['pred_scores']))

        return batch_dict

    # Projection
    def do_projection(self, batch_dict):
        if not self.enable_projection or self.training:
            return batch_dict

        # Add the detection results of previous sample
        proj_all = ('final_box_dicts' in batch_dict)
        self.add_past_proj_to_queue(proj_all)
        latest_token = batch_dict['metadata'][0]['token']
        scene_token = self.token_to_scene[latest_token]
        self.cur_pose = self.token_to_pose[latest_token]
        self.cur_ts = self.token_to_ts[latest_token]

        # Remove detections which are no more needed
        active_num_dets = torch.sum(self.num_dets_per_tile)
        max_num_proj = int(active_num_dets * self.projection_coeff)
        if self.past_detections['pred_boxes'].size(0) > max_num_proj:
            # Remove oldest dets
            for k in ['pose_idx', 'pred_boxes', 'pred_labels']:
                self.past_detections[k] = self.past_detections[k][-max_num_proj:]
            if self.score_thresh == 0:
                self.past_detections['pred_scores'] = self.past_detections['pred_scores'][-max_num_proj:]

        # Weed out using the pose_idx of first det
        if self.past_detections['pose_idx'].size(0) > 0:
            pose_idx_0 = self.past_detections['pose_idx'][0]
            self.past_poses = self.past_poses[pose_idx_0:]
            self.past_ts = self.past_ts[pose_idx_0:]
            self.past_detections['pose_idx'] = self.past_detections['pose_idx'] - pose_idx_0

        proj_dict = {}
        if proj_all:
            # NMS will happen right after projection
            # project the available det results as well for streaming eval
            det_pred_dict = batch_dict['final_box_dicts'][0]
            num_boxes = det_pred_dict['pred_boxes'].size(0)

            if self.past_detections['pred_boxes'].size(0) == 0:
                proj_dict['pred_scores'] = det_pred_dict['pred_scores'].cpu()
                proj_dict['pred_labels'] = det_pred_dict['pred_labels'].cpu() - 1
                boxes_to_forecast = det_pred_dict['pred_boxes']
                proj_dict['pose_idx'] = torch.tensor([0] * num_boxes,
                        dtype=self.past_detections['pose_idx'].dtype, device='cuda')
                proj_dict['past_poses'] = self.cur_pose.unsqueeze(0).cuda()
                proj_dict['past_ts'] = self.cur_ts.cuda()
            else:
                # MOVE TO CPU
                proj_pred_scores = self.score_thresh - \
                        (self.score_thresh / (self.past_detections['pose_idx'] + 2))
                proj_dict['pred_scores'] = torch.cat((proj_pred_scores,
                    det_pred_dict['pred_scores'].cpu()), dim=0)

                # MOVE TO CPU
                proj_pred_labels = (self.past_detections['pred_labels'] - 1)
                proj_dict['pred_labels'] = torch.cat((proj_pred_labels,
                    det_pred_dict['pred_labels'].cpu() - 1), dim=0)

                # MOVE TO GPU
                boxes_to_forecast = torch.cat((self.past_detections['pred_boxes'].cuda(),
                    det_pred_dict['pred_boxes']), dim=0)

                # MOVE TO GPU
                det_pose_idx = (self.past_detections['pose_idx'][-1] + 1).repeat(num_boxes)
                proj_dict['pose_idx'] = torch.cat((self.past_detections['pose_idx'],
                    det_pose_idx), dim=0).cuda()

                # MOVE TO GPU
                proj_dict['past_poses'] = torch.cat((self.past_poses,
                    self.cur_pose.unsqueeze(0)), dim=0).cuda()

                # MOVE TO GPU
                proj_dict['past_ts'] = torch.cat((self.past_ts, self.cur_ts), dim=0).cuda()

            # Calculate the exact current timestamp, for now, ignore the timing of
            # projection and nms
            torch.cuda.synchronize()
            elapsed_musec = (time.time() - batch_dict['start_time_sec']) * 1000000
            self.target_ts = self.cur_ts
            while self.target_ts < self.cur_ts + elapsed_musec:
                self.target_ts += 50000 # keep adding 50 ms

            proj_dict['pred_boxes']= cuda_projection.project_past_detections(
                    boxes_to_forecast,
                    proj_dict['pose_idx'],
                    proj_dict['past_poses'],
                    self.cur_pose.cuda(),
                    proj_dict['past_ts'],
                    self.target_ts.item()).cpu()
        else:
            # Do projection in the GPU
            if self.past_detections['pred_boxes'].size(0) > 0:
                if self.score_thresh == 0:
                    proj_dict['pred_scores'] = self.past_detections['pred_scores']
                else:
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

        if 'pred_boxes' in proj_dict:
            # clear up detections which fall under the chosen tiles and also the overtimed ones
            projb = proj_dict['pred_boxes']
            box_x, box_y = projb[:,0], projb[:,1]
            range_mask = box_x >= self.pc_range[0]
            range_mask = torch.logical_and(range_mask, box_x <= self.pc_range[3])
            range_mask = torch.logical_and(range_mask, box_y >= self.pc_range[1])
            range_mask = torch.logical_and(range_mask, box_y <= self.pc_range[4])

            proj_dict['pred_boxes'] = proj_dict['pred_boxes'][range_mask]
            proj_dict['pred_scores'] = proj_dict['pred_scores'][range_mask]
            proj_dict['pred_labels'] = proj_dict['pred_labels'][range_mask]

            # This op can make nms faster
            target_key = 'projections_nms'
            with torch.cuda.stream(self.projection_stream):
                if self.enable_splitting_projections:
                    batch_dict[target_key] = cuda_projection.split_projections(
                            proj_dict['pred_boxes'],
                            proj_dict['pred_scores'],
                            proj_dict['pred_labels'],
                            self.dense_head.cls_id_to_det_head_idx_map,
                            self.dense_head.num_det_heads,
                            True) # moves results to gpu if true
                else:
                    proj_dict['pred_boxes'] = proj_dict['pred_boxes'].cuda()
                    proj_dict['pred_scores'] = proj_dict['pred_scores'].cuda()
                    proj_dict['pred_labels'] = proj_dict['pred_labels'].cuda()
                    batch_dict[target_key] = proj_dict

            # hmm, might be necessary
            if proj_all:
                range_mask = range_mask[:-num_boxes]
            for k in ('pred_boxes', 'pred_labels', 'pose_idx'):
                self.past_detections[k] = self.past_detections[k][range_mask]
            if self.score_thresh == 0:
                self.past_detections['pred_scores'] = self.past_detections['pred_scores'][range_mask]

            self.projection_stream.synchronize()

        return batch_dict



    def train(self, mode=True):
        super().train(mode)
        if self.hooks_binded:
            self.pre_hook_handle.remove()
            self.post_hook_handle.remove()
            self.hooks_binded = False

    def eval(self):
        super().eval()
        if not self.hooks_binded:
            self.pre_hook_handle = self.register_forward_pre_hook(pre_forward_hook)
            self.post_hook_handle = self.register_forward_hook(post_forward_hook)
            self.hooks_binded = True

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels \
            if hasattr(backbone_3d_module, 'backbone_channels') else None
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict.get('num_bev_features', None)
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'] if 'num_bev_features' in model_info_dict else self.model_cfg.DENSE_HEAD.INPUT_FEATURES,
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            backbone_channels=model_info_dict['backbone_channels'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing_pre(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        #recall_dict = {}
        #pred_dicts = []
        final_boxes_arr, final_scores_arr, final_labels_arr, \
                src_box_preds_arr = [], [], [], []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds
            src_box_preds_arr.append(src_box_preds)

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
            final_boxes_arr.append(final_boxes)
            final_scores_arr.append(final_scores)
            final_labels_arr.append(final_labels)
            src_box_preds_arr.append(src_box_preds)
        return (final_boxes_arr, final_scores_arr, final_labels_arr, \
                src_box_preds_arr, batch_dict)

    def post_processing_post(self, args):
        final_boxes_arr, final_scores_arr, final_labels_arr, \
                src_box_preds_arr, batch_dict = args
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            recall_dict = self.generate_recall_record(
                box_preds=final_boxes_arr[index] if 'rois' not in batch_dict \
                        else src_box_preds_arr[index],
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes_arr[index],
                'pred_scores': final_scores_arr[index],
                'pred_labels': final_labels_arr[index]
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch


    def calc_elapsed_times(self):
        for name, events_list in self._cuda_event_dict.items():
            for events in events_list:
                time_elapsed = events[0].elapsed_time(events[1])
                self._time_dict[name].append(round(time_elapsed,3))

        for v in self._cuda_event_dict.values():
            v.clear()

    def clear_stats(self):
        # timers
        for v1, v2 in zip(self._time_dict.values(), self._cuda_event_dict.values()):
            v1.clear()
            v2.clear()

        # deadline
        self._eval_dict['deadlines_missed'] = 0
        self._eval_dict['deadline_diffs'].clear()

    def get_time_dict_stats(self):
        ret={}

        for name, val in self._time_dict.items():
            if len(val) == 0:
                ret[name] = [0, 0, 0, 0, 0, 0]
            else:
                val_max = max(val)
                perc95 = np.percentile(val, 95, interpolation='lower')
                perc99 = np.percentile(val, 99, interpolation='lower')
                val = [et for et in val if et <= perc99]
                std_dev = np.std(val)
                ret[name] = [min(val), # min
                         sum(val) / len(val),  # avrg
                         perc95,
                         perc99, # 99 percentile
                         val_max, # max
                         std_dev]  # std_dev
        return ret

    def get_time_dict(self):
        return self._time_dict

    def print_time_dict(self):
        for name, vals in self._time_dict.items():
            print('1D_LIST', name, vals)

    def print_time_stats(self):
        # Print these for humans
        max_len=0
        for name in self.get_time_dict_stats().keys():
            max_len = max(len(name), max_len)
        #print((" " * max_len),"Min,Avrg,95perc,99perc,Max,Std_dev")
        print(" ,Min,Avrg,95perc,99perc,Max,Std_dev")
        for name, val in self.get_time_dict_stats().items():
            spaces = " " * (max_len - len(name)+1)
            #print(f"{name}{spaces}{val[0]:.2f},{val[1]:.2f}"
            print(f"{name},{val[0]:.2f},{val[1]:.2f}"
                    f",{val[2]:.2f},{val[3]:.2f},{val[4]:.2f},{val[5]:.2f}")
        print('All numbers are in milliseconds')

    # Does not allow same events to be nested
    def measure_time_start(self, event_name_str, cuda_event=True):
        if cuda_event:
            new_events = [
                torch.cuda.Event(enable_timing=True),  # start
                torch.cuda.Event(enable_timing=True)  # end
            ]
            new_events[0].record()
            self._cuda_event_dict[event_name_str].append(new_events)
            torch.cuda.nvtx.range_push(event_name_str)
            return new_events
        else:
            self._time_dict[event_name_str].append(time.time())

    def measure_time_end(self, event_name_str, cuda_event=True):
        if cuda_event:
            self._cuda_event_dict[event_name_str][-1][1].record()
            torch.cuda.nvtx.range_pop()
            return None
        else:
            time_elapsed = round((time.time() - self._time_dict[event_name_str][-1]) * 1000, 3)
            time_elapsed = round(time_elapsed,3)
            self._time_dict[event_name_str][-1] = time_elapsed
            return time_elapsed

    def update_time_dict(self, dict_to_aggragate):
        self._time_dict.update(dict_to_aggragate)
        self._cuda_event_dict = copy.deepcopy(self._time_dict)

    def dump_eval_dict(self, eval_result_dict=None):
        self._eval_dict['exec_times'] = self.get_time_dict()
        self._eval_dict['exec_time_stats'] = self.get_time_dict_stats()
        self._eval_dict['eval_results_dict'] = eval_result_dict
        self._eval_dict['dataset'] = self.dataset.dataset_cfg.DATASET

        if self.model_cfg.get('METHOD', None) is not None:
            self._eval_dict['method'] = self.model_cfg.METHOD

        print('Dumping evaluation dictionary file')
        current_date_time = datetime.datetime.today()
        dt_string = current_date_time.strftime('%d-%m-%y-%I-%M-%p')
        with open(f"eval_dict_{dt_string}.json", 'w') as handle:
            json.dump(self._eval_dict, handle, indent=4)

    def init_empty_det_dict(self, det_dict_example):
        self._det_dict_copy = {
            "pred_boxes": torch.zeros([0, det_dict_example["pred_boxes"].size()[1]],
            dtype=det_dict_example["pred_boxes"].dtype),
            "pred_scores": torch.zeros([0], dtype=det_dict_example["pred_scores"].dtype),
            "pred_labels": torch.zeros([0], dtype=det_dict_example["pred_labels"].dtype),
        }

    def get_empty_det_dict(self):
        det_dict = {}
        for k,v in self._det_dict_copy.items():
            det_dict[k] = v.clone().detach()
        return det_dict

    def get_dummy_det_dict(self):
        det_dict = {}
        for k,v in self._det_dict_copy.items():
            dims = [1]
            for d in v.size()[1:]:
                dims.append(d)
            det_dict[k] = torch.ones(dims, dtype=v.dtype, device=v.device)
        return det_dict


    def print_dict(self, d):
        for k, v in d.items():
            print(k, ':', end=' ')
            if torch.is_tensor(v):
                print(v.size())
            elif isinstance(v, list) and torch.is_tensor(v[0]):
                for e in v:
                    print(e.size(), end=' ')
                print()
            else:
                print(v)

    #def load_data_with_ds_index(self, dataset_index):
    #    data_dict = self.dataset.collate_batch([self.dataset[dataset_index]])
    #    load_data_to_gpu(data_dict)
    #    return data_dict
    def calibration_on(self):
        self.calibrating+=1
    
    def calibration_off(self):
        self.calibrating-=1

    def is_calibrating(self):
        return self.calibrating > 0

    def calibrate(self, batch_size=1):
        #data_dict = self.load_data_with_ds_index(0)
        #print('\ndata_dict:')
        #self.print_dict(data_dict)

        # just do a regular forward first
        #data_dict["abs_deadline_sec"] = time.time () + 10.0
        self.calibration_on()
        training = self.training
        self.eval()

        enable_proj = self.enable_projection
        self.enable_projection = False
        self._eval_dict['deadline_sec'] = 10.0
        pred_dicts, recall_dict = self([i for i in range(batch_size)]) # this calls forward!
        self.enable_projection = enable_proj
        self._eval_dict['deadline_sec'] = self._default_deadline_sec
        self.prev_scene_token = '' 

        #Print full tensor sizes
        #print('\ndata_dict after forward:')
        #self.print_dict(self.latest_batch_dict)

        #print('\nDetections:')
        #for pd in pred_dicts:
        #    self.print_dict(pd)

        #print('\nRecall dict:')
        #self.print_dict(recall_dict)

        if isinstance(pred_dicts, list):
            det = pred_dicts[0]
        else:
            det = pred_dicts
        self.init_empty_det_dict(det)

        self.clear_stats()
        print('Num params:', sum(p.numel() for p in self.parameters()))
        print('Num params trainable:', sum(p.numel() for p in self.parameters() if p.requires_grad))
        if training:
            self.train()
        self.calibration_off()
        print('Detector3D calibration done')
        return None
