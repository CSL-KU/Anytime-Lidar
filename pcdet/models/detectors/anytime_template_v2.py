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
import os

from ...ops.cuda_point_tile_mask import cuda_point_tile_mask
from .. import load_data_to_gpu

from typing import List

@torch.jit.script
def do_inds_calc(vinds : List[torch.Tensor], vcount_area : torch.Tensor, \
        tcount : int, dividers: torch.Tensor):
    # how would be the overhead if I create a stream here?
    outp = []
    outp.append(vcount_area)
    for i, vind in enumerate(vinds):
        voxel_tile_coords = torch.div(vind[:, -1], dividers[i+1], \
                rounding_mode='trunc').int()
        cnts = torch.bincount(voxel_tile_coords, \
                minlength=tcount).unsqueeze(0)
        outp.append(cnts)
    return torch.cat(outp, dim=0).cpu() # num sparse layer groups x num_tiles

# The fork call must be done in torch scripted function for it to be async
@torch.jit.script
def do_inds_calc_wrapper(vinds : List[torch.Tensor], \
        vcount_area : torch.Tensor, \
        tcount : int, dividers: torch.Tensor):
    fut = torch.jit.fork(do_inds_calc, vinds, vcount_area, tcount, dividers)
    return fut

@torch.jit.script
def get_voxel_x_coords_from_points(points_coords : torch.Tensor,
        scale_xy : int, scale_y : int) -> torch.Tensor:
    #merge_coords = points[:, 0].int() * scale_xy + \ ( assume back size is 1 )
    merge_coords = points_coords[:, 0] * scale_y + \
                   points_coords[:, 1]
    unq_coords = torch.unique(merge_coords, sorted=False, dim=0)
    return (unq_coords % scale_xy) // scale_y

@torch.jit.script
def tile_calculations(coords_x : torch.Tensor, tile_size_voxels: float, tcount : int):
    tile_coords = torch.div(coords_x, tile_size_voxels, rounding_mode='trunc').long()
    counts_area = torch.bincount(tile_coords, minlength=tcount).cpu()[:tcount]
    netc = torch.arange(tcount, dtype=torch.long)[counts_area > 0] # shouldn't take noticable time
    return tile_coords, netc, counts_area


@torch.jit.script
def get_netc(coords_x : torch.Tensor, tile_size_points: float, tcount : int):
    tile_coords = torch.div(coords_x, tile_size_points, rounding_mode='trunc').long()
    netc = torch.zeros(tcount, dtype=torch.bool, device=coords_x.device)
    netc[tile_coords] = True
    return torch.nonzero(netc).flatten().cpu()

@torch.jit.script
def tile_calculations_all(points_coords : torch.Tensor, tile_size_voxels : float,
            scale_xy : int, scale_y : int, tcount : int):
    fut = torch.jit.fork(tile_calculations, points_coords[:, 0], tile_size_voxels, tcount)
    voxel_x_coords = get_voxel_x_coords_from_points(points_coords, scale_xy, scale_y)
    _, netc, vcount_area = tile_calculations(voxel_x_coords, tile_size_voxels, tcount)
    point_tile_coords, _, pcount_area = torch.jit.wait(fut)
    return point_tile_coords, netc, pcount_area, vcount_area

class AnytimeTemplateV2(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.model_name = self.model_cfg.NAME + '_' + self.model_cfg.NAME_POSTFIX

        self.sched_disabled = (self.model_cfg.METHOD == SchedAlgo.RoundRobin_NoSchedNoProj)

        self.keep_forecasting_disabled = (self.model_cfg.METHOD == SchedAlgo.RoundRobin_NoProj or \
                self.model_cfg.METHOD == SchedAlgo.RoundRobin_NoSchedNoProj)

        self.use_voxelnext = (self.model_cfg.METHOD == SchedAlgo.RoundRobin_VN or \
                self.model_cfg.METHOD == SchedAlgo.RoundRobin_VN_BLTP)

        self.use_baseline_bb3d_predictor = (self.model_cfg.METHOD == SchedAlgo.RoundRobin_BLTP or \
                self.model_cfg.METHOD == SchedAlgo.RoundRobin_VN_BLTP or \
                self.model_cfg.METHOD == SchedAlgo.RoundRobin_DSVT)

        if self.use_baseline_bb3d_predictor:
            print('***** Using baseline time predictor! *****')

        self.sched_algo = SchedAlgo.RoundRobin
        self.sched_vfe = (self.model_cfg.METHOD != SchedAlgo.RoundRobin_NoVFESched) and \
            ('DynPillarVFE' == self.model_cfg.VFE.NAME or \
            'DynamicPillarVFESimple2D' == self.model_cfg.VFE.NAME)
        self.sched_bb3d = ('BACKBONE_3D' in self.model_cfg)
        print('sched vfe:', self.sched_vfe)
        print('sched bb3d:', self.sched_bb3d)

        self.enable_tile_drop =  (self.model_cfg.METHOD != SchedAlgo.RoundRobin_NoTileDrop) and \
                            (not self.use_voxelnext) and (self.sched_vfe or self.sched_bb3d)
        print('tile dropping:', self.enable_tile_drop)

        if 'BACKBONE_2D' in self.model_cfg:
            self.model_cfg.BACKBONE_2D.TILE_COUNT = self.model_cfg.TILE_COUNT
            self.model_cfg.BACKBONE_2D.METHOD = self.sched_algo
        if 'DENSE_HEAD' in self.model_cfg:
            self.model_cfg.DENSE_HEAD.TILE_COUNT = self.model_cfg.TILE_COUNT
            self.model_cfg.DENSE_HEAD.METHOD = self.sched_algo
        torch.cuda.manual_seed(0)
        self.module_list = self.build_networks()

        ################################################################################
        #self.total_num_tiles = self.tcount

        # divide the tiles in X axis only
        self.tile_size_voxels = float(self.dataset.grid_size[1] / self.tcount)
        pc_range = self.dataset.point_cloud_range
        self.tile_size_points = float((pc_range[3] - pc_range[0]) / self.tcount)
        if self.sched_bb3d:
            self.backbone_3d.tile_size_voxels = self.tile_size_voxels

        ####Projection###

        self.clear_add_dict()
        self.init_tile_coord = -1
        self.last_tile_coord = self.init_tile_coord

        ##########################################
        if self.sched_bb3d:
            ng = self.backbone_3d.num_layer_groups
            ng += 1 if self.use_voxelnext else 0

            self.dividers = self.backbone_3d.get_inds_dividers(self.tile_size_voxels)
            if self.use_voxelnext:
                self.dividers.append(float(self.tile_size_voxels / 8))
            self.dividers.insert(0, 1.) # this will be ignored
            print('dividers', self.dividers, 'ng', ng)
        #self.move_indscalc_to_init = True
        ##########################################

        self.resolution_dividers = self.model_cfg.get('RESOLUTION_DIV', [1.0])
        self.num_res = len(self.resolution_dividers)
        self.res_idx = 0
        self.calibrators = [None] * self.num_res

    def clear_add_dict(self):
        self.add_dict['vfe_layer_times'] = []
        self.add_dict['vfe_point_nums'] = []
        self.add_dict['vfe_preds'] = [] # debug

        self.add_dict['bb3d_layer_times'] = []
        self.add_dict['bb3d_voxel_nums'] = []
        self.add_dict['bb3d_preds'] = [] # debug

        self.add_dict['bb3d_preds_layerwise'] = [] # debug
        self.add_dict['bb3d_voxel_preds'] = [] # debug
        self.add_dict['nonempty_tiles'] = []
        self.add_dict['chosen_tiles_1'] = []
        self.add_dict['chosen_tiles_2'] = []

    def initialize(self, latest_token : str) -> (float, bool):
        deadline_sec_override, reset = super().initialize(latest_token)
        if reset:
            self.last_tile_coord = self.init_tile_coord

        if self.sched_disabled:
            return deadline_sec_override, reset
        elif reset:
            self.sched_reset()

        return deadline_sec_override, reset

    def schedule1(self, batch_dict):
        calibrator = self.calibrators[self.res_idx]
        cur_tile_size_voxels = self.tile_size_voxels 
        if self.training or self.sched_disabled:
            voxel_x_coords = self.get_voxel_x_coords_from_points(batch_dict)
            voxel_tile_coords = torch.div(voxel_x_coords, cur_tile_size_voxels, \
                    rounding_mode='trunc').long()
            # choose all nonempty
            batch_dict['chosen_tile_coords'] = torch.unique(voxel_tile_coords, sorted=True)
            return batch_dict

        if self.latest_batch_dict is not None and \
                'bb3d_intermediary_vinds' in self.latest_batch_dict:
            self.fut = do_inds_calc_wrapper(
                    self.latest_batch_dict['bb3d_intermediary_vinds'],
                    self.latest_batch_dict['vcount_area'],
                    self.tcount, torch.tensor([d for d in self.dividers]))

        pcount_area, vcount_area = None, None
        if self.sched_vfe and self.sched_bb3d:
            batch_dict['points_coords'] = self.vfe.calc_points_coords(batch_dict['points'])
            torch.cuda.synchronize() # prevents tile calculations failure
            point_tile_coords, netc, pcount_area, vcount_area = tile_calculations_all(
                    batch_dict['points_coords'], cur_tile_size_voxels, self.vfe.scale_xy,
                    self.vfe.scale_y, self.tcount)
        elif self.sched_bb3d:
            point_tile_coords, netc, vcount_area = tile_calculations(
                    batch_dict['voxel_coords'][:, -1], cur_tile_size_voxels, self.tcount)
        elif self.sched_vfe:
            points_x_shifted = batch_dict['points'][:, 1] - self.vfe.point_cloud_range[0]
            point_tile_coords, netc, pcount_area = tile_calculations(
                    points_x_shifted, self.tile_size_points, self.tcount)
        else:
            points_x_shifted = batch_dict['points'][:, 1] - self.vfe.point_cloud_range[0]
            netc = get_netc(points_x_shifted, self.tile_size_points, self.tcount)

        netc = np.sort(netc.numpy())
        if pcount_area is not None:
            pcount_area = pcount_area.unsqueeze(0).numpy()
        if vcount_area is not None:
            vcount_area = vcount_area.unsqueeze(0).numpy()

        if self.sched_bb3d:
            batch_dict['vcount_area'] = torch.from_numpy(vcount_area).int().cuda()
        batch_dict['nonempty_tile_coords'] = netc

        num_tiles, tiles_queue = round_robin_sched_helper(
                    netc, self.last_tile_coord, self.tcount)
        batch_dict['tiles_queue'] = tiles_queue
        self.add_dict['nonempty_tiles'].append(netc.tolist())

        if self.latest_batch_dict is not None and \
                'bb3d_intermediary_vinds' in self.latest_batch_dict:
            voxel_dists = torch.jit.wait(self.fut)
            calibrator.commit_bb3d_updates(
                    self.latest_batch_dict['chosen_tile_coords'],
                    voxel_dists.numpy())

        vfe_times, bb3d_times_layerwise, post_bb3d_times, num_voxel_preds = \
                calibrator.pred_req_times_ms(pcount_area, \
                vcount_area, tiles_queue, num_tiles)
        batch_dict['post_bb3d_times'] = copy.deepcopy(post_bb3d_times)
        tpreds = post_bb3d_times
        if bb3d_times_layerwise is not None:
            if not self.use_baseline_bb3d_predictor:
                bb3d_times = np.sum(bb3d_times_layerwise, axis=-1)
            else:
                bb3d_times = bb3d_times_layerwise
            tpreds += bb3d_times
        if vfe_times is not None:
            tpreds += vfe_times


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

        # when reset, process all ignoring deadline
        choose_all = (not self.is_calibrating() and diffs[-1]) or \
                (self.is_calibrating() and len(netc) <= calibrator.get_chosen_tile_num())
        if choose_all:
            # choose all
            chosen_tile_coords = netc
            tiles_idx=0
        else:
            if self.is_calibrating():
                tiles_idx = calibrator.get_chosen_tile_num()
                if tiles_idx >= len(diffs):
                    tiles_idx = len(diffs)
            else:
                tiles_idx=1
                while tiles_idx < diffs.shape[0] and diffs[tiles_idx]:
                    tiles_idx += 1

            # Voxel filtering is needed
            chosen_tile_coords = tiles_queue[:tiles_idx]

            # Filter the points, let the voxel be generated from filtered points
            if self.sched_vfe or self.sched_bb3d:
                # don't deal with filtering if it will be cropped anyway
                tile_filter = cuda_point_tile_mask.point_tile_mask(point_tile_coords, \
                        torch.from_numpy(chosen_tile_coords).cuda())
                filter_list = ['points', 'points_coords'] if self.sched_vfe else \
                        ['voxel_coords', 'voxel_features']
                for k in filter_list:
                    if k in batch_dict:
                        batch_dict[k] = batch_dict[k][tile_filter]
                if not self.sched_vfe:
                    batch_dict['pillar_coords'] = batch_dict['voxel_coords']
                    batch_dict['pillar_features'] = batch_dict['voxel_features']

        self.last_tile_coord = chosen_tile_coords[-1].item()

        tidx = -1 if choose_all else tiles_idx-1
        if self.sched_vfe:
            predicted_vfe_time  = float(vfe_times[tidx])
            self.add_dict['vfe_preds'].append(predicted_vfe_time)
            calibrator.last_pred[1] = predicted_vfe_time

        if self.sched_bb3d:
            predicted_bb3d_time = float(bb3d_times[tidx])
            predicted_bb3d_time_layerwise = bb3d_times_layerwise[tidx]
            predicted_voxels = num_voxel_preds[tidx].astype(int).flatten()
            calibrator.last_pred[2] = predicted_bb3d_time
            self.add_dict['bb3d_preds'].append(predicted_bb3d_time)
            self.add_dict['bb3d_preds_layerwise'].append(predicted_bb3d_time_layerwise.tolist())
            self.add_dict['bb3d_voxel_preds'].append(predicted_voxels.tolist())

        dhp_wcet_ms = calibrator.det_head_post_wcet_ms
        calibrator.last_pred[3] = batch_dict['post_bb3d_times'][tidx] - dhp_wcet_ms
        calibrator.last_pred[4] = dhp_wcet_ms

        batch_dict['chosen_tile_coords'] = chosen_tile_coords
        self.add_dict['chosen_tiles_1'].append(chosen_tile_coords.tolist())

        if self.sched_vfe:
            self.add_dict['vfe_point_nums'].append([batch_dict['points'].size(0)])

        if self.sched_bb3d:
            batch_dict['record_int_vcoords'] = False
            batch_dict['record_int_indices'] = not self.sched_disabled and \
                    not self.use_baseline_bb3d_predictor
            batch_dict['record_time'] = True
            batch_dict['tile_size_voxels'] = cur_tile_size_voxels

        return batch_dict

    def schedule2(self, batch_dict):
        if self.sched_disabled:
            return batch_dict

        # bb3d time predictor commit
        if self.sched_bb3d:
            if not self.use_baseline_bb3d_predictor:
                if self.use_voxelnext:
                    out = batch_dict['encoded_spconv_tensor']
                    batch_dict['bb3d_intermediary_vinds'].append(out.indices)
                num_voxels_actual = np.array([batch_dict['voxel_coords'].size(0)] + \
                        [inds.size(0) for inds in batch_dict['bb3d_intermediary_vinds']], dtype=int)
                self.add_dict['bb3d_voxel_nums'].append(num_voxels_actual.tolist())
            else:
                self.add_dict['bb3d_voxel_nums'].append([batch_dict['voxel_coords'].size(0)])

        # Tile dropping
        if self.enable_tile_drop:
            torch.cuda.synchronize()
            post_bb3d_times = batch_dict['post_bb3d_times']
            rem_time_ms = (batch_dict['abs_deadline_sec'] - time.time()) * 1000
            diffs = post_bb3d_times < rem_time_ms

            if not diffs[batch_dict['chosen_tile_coords'].shape[0]-1]:
                tiles_idx=1
                while tiles_idx < diffs.shape[0] and diffs[tiles_idx]:
                    tiles_idx += 1

                ctc = batch_dict['tiles_queue'][:tiles_idx]
                batch_dict['chosen_tile_coords'] = ctc
                self.last_tile_coord = ctc[-1].item()

        ctc = batch_dict['chosen_tile_coords'].tolist()
        self.add_dict['chosen_tiles_2'].append(ctc)

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

    def calibrate(self, batch_size=1):
        if self.training:
            super().calibrate(1)
            return None

        self.collect_calib_data = [False] * self.num_res
        self.calib_fnames = [""] * self.num_res
        m = self.model_cfg.METHOD
        for res_idx in range(self.num_res):
            self.res_idx = res_idx
            self.calibrators[res_idx] = AnytimeCalibrator(self)
            power_mode = os.getenv('PMODE', 'UNKNOWN_POWER_MODE')
            self.calib_fnames[res_idx] = f"calib_files/{self.model_name}_{power_mode}_m{m}.json"
            try:
                self.calibrators[res_idx].read_calib_data(self.calib_fnames[res_idx])
            except OSError:
                self.collect_calib_data[res_idx] = True

            self.calibration_on()
            self.enable_forecasting = False
            print(f'Calibrating resolution {res_idx}')
            super().calibrate(1)
            self.enable_forecasting = (not self.keep_forecasting_disabled)
            print('Forecasting', 'enabled' if self.enable_forecasting else 'disabled')
            self.last_tile_coord = self.init_tile_coord
            self.sched_reset()

            if self.collect_calib_data[res_idx]:
                self.calibrators[res_idx].collect_data_v2(self.sched_algo, self.calib_fnames[res_idx])
                # After this, the calibration data should be processed with dynamic deadline
            self.clear_stats()
            self.clear_add_dict()
            self.calibration_off()
        if any(self.collect_calib_data):
            sys.exit()

        return None

    def post_eval(self):
        if self.collect_calib_data[self.res_idx]:
            #NOTE it only works for the latest resolution calibrated
            # We need to put bb3d time prediction data in the calibration file
            with open(self.calib_fnames[self.res_idx], 'r') as handle:
                calib_dict = json.load(handle)
                if self.sched_bb3d:
                    calib_dict['bb3d_preds'] = self.add_dict['bb3d_preds']
                if self.sched_vfe:
                    calib_dict['vfe_preds'] = self.add_dict['vfe_preds']
                calib_dict['exec_times'] = self.get_time_dict()
            with open(self.calib_fnames[self.res_idx], 'w') as handle:
                json.dump(calib_dict, handle, indent=4)

        self.add_dict['tcount'] = self.tcount
        self.add_dict['bb3d_pred_shift_ms'] = self.calibrators[self.res_idx].expected_bb3d_err
        print(f"\nDeadlines missed: {self._eval_dict['deadlines_missed']}\n")

        self.plot_post_eval_data()

    def plot_post_eval_data(self):
        import matplotlib.pyplot as plt
        from datetime import datetime
        from pathlib import Path
        root_path = '../../latest_exp_plots/'
        os.makedirs(root_path, exist_ok=True)
        timedata = datetime.now().strftime("%m_%d_%H_%M")

        # plot 3d backbone time pred error
        time_dict = self.get_time_dict()

        if self.sched_bb3d:
            if 'Backbone3D-IL' in  time_dict:
                Backbone3D_times = np.array(time_dict['Backbone3D-IL']) + \
                        np.array(time_dict['Backbone3D-Fwd'])
            else:
                Backbone3D_times = time_dict['Backbone3D']


            if len(Backbone3D_times) == len(self.add_dict['bb3d_preds']):
                bb3d_pred_err = np.array(Backbone3D_times) - np.array(self.add_dict['bb3d_preds'])
                if 'VoxelHead-conv-hm' in time_dict:
                    bb3d_pred_err += np.array(time_dict['VoxelHead-conv-hm'])

                # plot the data
                min_, mean_, perc1_, perc5_, perc95_, perc99_, max_ = get_stats(bb3d_pred_err)
                hist, bin_edges = np.histogram(bb3d_pred_err, bins=100, density=True)
                cdf = np.cumsum(hist * np.diff(bin_edges))
                plt.plot(bin_edges[1:], cdf, linestyle='-')
                plt.grid(True)
                plt.xlim([perc1_, perc99_])
                plt.ylim([0, 1])
                plt.xlabel('Actual - Predicted bb3d execution time (msec)')
                plt.ylabel('CDF')
                plt.savefig(f'{root_path}/{self.model_name}_bb3d_time_pred_err_{timedata}.pdf')
                plt.clf()

            if self.sched_bb3d and not self.use_baseline_bb3d_predictor and not self.sched_disabled:
                # plot 3d backbone time pred error layerwise
                layer_times_actual = np.array(self.add_dict['bb3d_layer_times'])
                layer_times_pred = np.array(self.add_dict['bb3d_preds_layerwise'])
                layer_time_err = layer_times_actual - layer_times_pred
                for i in range(layer_time_err.shape[1]):
                    #min_, mean_, perc1_, perc5_, perc95_, perc99_, max_ = get_stats(bb3d_vpred_err[:, i])
                    #perc1_min = min(perc1_min, perc1_)
                    #perc99_max = max(perc99_max, perc99_)
                    hist, bin_edges = np.histogram(layer_time_err[:, i], bins=100, density=True)
                    cdf = np.cumsum(hist * np.diff(bin_edges))
                    plt.plot(bin_edges[1:], cdf, linestyle='-', label=f"layer {i}")

                #plt.xlim([perc1_min, perc99_max])
                plt.grid(True)
                plt.legend()
                plt.xlabel('Actual - Predicted bb3d layer times')
                plt.ylabel('CDF')
                plt.savefig(f'{root_path}/{self.model_name}_bb3d_layer_time_err_{timedata}.pdf')
                plt.clf()


                # plot 3d backbone voxel pred error layerwise
                vactual = np.array(self.add_dict['bb3d_voxel_nums'])
                vpreds = np.array(self.add_dict['bb3d_voxel_preds'])
                bb3d_vpred_err = vactual - vpreds

                perc1_min, perc99_max = 50000, -50000
                for i in range(bb3d_vpred_err.shape[1]):
                    min_, mean_, perc1_, perc5_, perc95_, perc99_, max_ = get_stats(bb3d_vpred_err[:, i])
                    perc1_min = min(perc1_min, perc1_)
                    perc99_max = max(perc99_max, perc99_)
                    hist, bin_edges = np.histogram(bb3d_vpred_err[:, i], bins=100, density=True)
                    cdf = np.cumsum(hist * np.diff(bin_edges))
                    plt.plot(bin_edges[1:], cdf, linestyle='-', label=f"layer {i}")

                #plt.xlim([perc1_min, perc99_max])
                plt.grid(True)
                plt.legend()
                plt.xlabel('Actual - Predicted bb3d num voxels')
                plt.ylabel('CDF')
                plt.savefig(f'{root_path}/{self.model_name}_bb3d_num_voxel_pred_err_{timedata}.pdf')
                plt.clf()
                print('Num voxel to exec time plot saved.')

                # plot 3d backbone fitted equations
                calibrator = self.calibrators[self.res_idx]
                coeffs_calib, intercepts_calib = calibrator.time_reg_coeffs, \
                        calibrator.time_reg_intercepts
                coeffs_new, intercepts_new = calibrator.fit_voxel_time_data(vactual, \
                        layer_times_actual)
                calib_voxels, calib_times, _, _ = calibrator.get_calib_data_arranged()
                fig, axes = plt.subplots(len(coeffs_calib)//2+1, 2, \
                        figsize=(6, (len(coeffs_calib)-1)*2),
                        sharex=True,
                        constrained_layout=True)
                axes = np.ravel(axes)
                for i in range(len(coeffs_calib)):
                    #vlayer = vactual[:, i]
                    vlayer = calib_voxels[:, i]
                    xlims = [min(vlayer), max(vlayer)]
                    x = np.arange(xlims[0], xlims[1], (xlims[1]-xlims[0])//100)

                    num_voxels_ = np.expand_dims(x, -1) / calibrator.num_voxels_normalizer
                    num_voxels_ = np.concatenate((num_voxels_, np.square(num_voxels_)), axis=-1)
                    bb3d_time_calib = np.sum(num_voxels_ * coeffs_calib[i], axis=-1) + \
                            intercepts_calib[i]
                    bb3d_time_new  = np.sum(num_voxels_ * coeffs_new[i], axis=-1) + \
                            intercepts_new[i]

                    layer_times_ = calib_times[:, i]
                    layer_voxels_ = calib_voxels[:, i]
                    sort_indexes = np.argsort(layer_times_)
                    layer_times_ = layer_times_[sort_indexes] #[0::5]
                    layer_voxels_ = layer_voxels_[sort_indexes] #[0::5]
                    #ax.scatter(layer_voxels_, layer_times_, label="calib")
                    #ax.scatter(vlayer,layer_times_actual[:, i] , label="new")
                    #ax.plot(x, bb3d_time_calib, label="calib")
                    #ax.plot(x, bb3d_time_new, label="new")

                    #ax.grid('True', ls='--')
                    ax = axes[i]
                    ax.scatter(layer_voxels_, layer_times_, label=f"Data")
                    ax.plot(x, bb3d_time_calib, label="Model", color='orange')
                    ax.set_ylim([0, max(layer_times_)*1.1])
                    ax.set_title(f"Block {i+1}", fontsize='medium', loc='left')
                    ax.legend(fontsize='medium')
                    #ax.set_ylabel(f'Layer block {i}\nexecution\ntime (msec)', fontsize='x-large')
                fig.supxlabel('Number of input voxels', fontsize='x-large')
                fig.supylabel('Block execution time (msec)', fontsize='x-large')
                #ax.set_ylabel(f'Layer block {i}\nexecution\ntime (msec)', fontsize='x-large')
                #plt.subplots_adjust(wspace=0, hspace=0)

                plt.savefig(f'{root_path}/{self.model_name}_bb3d_fitted_data_all_{timedata}.pdf')
                #ax.set_xlim([0, 70000])
