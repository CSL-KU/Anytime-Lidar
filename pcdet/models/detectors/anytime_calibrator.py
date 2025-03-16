import torch
import time
import json
import numpy as np
import numba
import gc
import os
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression

if __name__ != "__main__":
    from .sched_helpers import SchedAlgo, get_num_tiles
    from ...ops.cuda_point_tile_mask import cuda_point_tile_mask

def calc_grid_size(pc_range, voxel_size):
    return np.array([ int((pc_range[i+3]-pc_range[i]) / vs)
            for i, vs in enumerate(voxel_size)])

@numba.njit()
def tile_coords_to_id(tile_coords):
    tid = 0
    for tc in tile_coords:
        tid += 2 ** tc
    return int(tid)

def get_stats(np_arr):
    min_, max_, mean_ = np.min(np_arr), np.max(np_arr), np.mean(np_arr)
    perc1_ = np.percentile(np_arr, 1, method='lower')
    perc5_ = np.percentile(np_arr, 5, method='lower')
    perc95_ = np.percentile(np_arr, 95, method='lower')
    perc99_ = np.percentile(np_arr, 99, method='lower')
    print("Min\t1Perc\t5Perc\tMean\t95Perc\t99Perc\tMax")
    print(f'{min_:.2f}\t{perc1_:.2f}\t{perc5_:.2f}\t{mean_:.2f}\t{perc95_:.2f}\t{perc99_:.2f}\t{max_:.2f}')
    return (min_, mean_, perc1_, perc5_, perc95_, perc99_, max_)


class AnytimeCalibrator():
    def __init__(self, model):
        self.model = model
        self.calib_data_dict = None
        if model is None:
            self.dataset = None
            #NOTE modify the following params depending on the config file
            #self.num_det_heads = 8
            self.num_tiles = 18
        else:
            self.dataset = model.dataset
            #self.num_det_heads = len(model.dense_head.class_names_each_head)
            self.num_tiles = model.model_cfg.TILE_COUNT

        self.res_idx = model.res_idx

        # only use baseline predictor for this one
        self.sched_vfe = model.sched_vfe
        self.sched_bb3d = model.sched_bb3d

        self.time_reg_degree = 2
        if self.sched_bb3d:
            self.bb3d_num_l_groups = self.model.backbone_3d.num_layer_groups

            if self.model.use_voxelnext:
                # count the convolutions of the detection head to be
                # a part of 3D backbone
                self.bb3d_num_l_groups += 1 # detection head convolutions

            self.use_baseline_bb3d_predictor = self.model.use_baseline_bb3d_predictor
            #self.move_indscalc_to_init = self.model.move_indscalc_to_init # needed?
            if self.use_baseline_bb3d_predictor:
                self.time_reg_coeffs = np.ones((self.time_reg_degree,), dtype=float)
                self.time_reg_intercepts = np.ones((1,), dtype=float)
            else:
                self.time_reg_coeffs = np.ones((self.bb3d_num_l_groups, self.time_reg_degree), dtype=float)
                self.time_reg_intercepts = np.ones((self.bb3d_num_l_groups,), dtype=float)

                self.scale_num_voxels = False # False appears to be better!
                self.voxel_coeffs_over_layers = np.array([[1.] * self.num_tiles \
                        for _ in range(self.bb3d_num_l_groups)])

        if self.sched_vfe:
            self.vfe_num_l_groups = 1
            self.vfe_time_reg_coeffs = np.ones((self.time_reg_degree,), dtype=float)
            self.vfe_time_reg_intercepts = np.ones((1,), dtype=float)

        # backbone2d and detection head heatmap convolutions
        # first elem unused
        self.det_head_post_wcet_ms = .0
        if not self.model.use_voxelnext:
            self.bb2d_times_ms = np.zeros((self.num_tiles+1,), dtype=float)

        self.expected_bb3d_err = 0.
        self.num_voxels_normalizer = 100000.
        self.num_points_normalizer = 1000000.
        self.chosen_tiles_calib = self.num_tiles
        self.last_pred = np.zeros(5)

    # voxel dists should be [self.bb3d_num_l_groups, num_tiles]
    def commit_bb3d_updates(self, ctc, voxel_dists):
        voxel_dists = voxel_dists[:, ctc]
        if self.scale_num_voxels:
            self.voxel_coeffs_over_layers[:, ctc] = voxel_dists / voxel_dists[0]
        else:
            self.voxel_coeffs_over_layers[:, ctc] = voxel_dists

    def make_pred_baseline(self, count_area, tiles_queue, reg_coeffs,
            reg_intercepts, normalizer):
        counts = count_area.flatten()
        num_points = np.empty((tiles_queue.shape[0]),dtype=float)
        for i in range(len(tiles_queue)):
            num_points[i] = np.sum(counts[tiles_queue[:i+1]])

        num_points_n_ = np.expand_dims(num_points, -1) / normalizer
        num_points_n_ = np.concatenate((num_points_n_, np.square(num_points_n_)), axis=-1)
        time_preds = np.sum(num_points_n_ * reg_coeffs.flatten(), \
                axis=-1) + reg_intercepts
        return time_preds, num_points

    # overhead on jetson-agx: 1 ms
    def pred_req_times_ms(self, pcount_area, vcount_area, tiles_queue, num_tiles): # [num_nonempty_tiles, num_max_tiles]
        vfe_time_preds = None
        if pcount_area is not None:
            vfe_time_preds, num_points = self.make_pred_baseline(pcount_area, tiles_queue,
                  self.vfe_time_reg_coeffs, self.vfe_time_reg_intercepts,
                  self.num_points_normalizer)

        bb3d_time_preds = None
        num_voxels = None
        if vcount_area is not None:
            if self.use_baseline_bb3d_predictor:
                bb3d_time_preds, num_voxels = self.make_pred_baseline(vcount_area, tiles_queue,
                      self.time_reg_coeffs, self.time_reg_intercepts,
                      self.num_voxels_normalizer)
                bb3d_time_preds += self.expected_bb3d_err
            else:
                if self.scale_num_voxels:
                    vcounts = vcount_area * self.voxel_coeffs_over_layers
                else:
                    vcounts = self.voxel_coeffs_over_layers
                    vcounts[0] = vcount_area.flatten()
                num_voxels = np.empty((tiles_queue.shape[0], vcounts.shape[0]), dtype=float)
                for i in range(len(tiles_queue)):
                    num_voxels[i] = np.sum(vcounts[:, tiles_queue[:i+1]], axis=1)
                if self.time_reg_degree == 1:
                    bb3d_time_preds = num_voxels / self.num_voxels_normalizer * \
                            self.time_reg_coeffs.flatten() + \
                            self.time_reg_intercepts
                elif self.time_reg_degree == 2:
                    num_voxels_n_ = np.expand_dims(num_voxels, -1) / self.num_voxels_normalizer
                    num_voxels_n_ = np.concatenate((num_voxels_n_, np.square(num_voxels_n_)), axis=-1)
                    bb3d_time_preds = np.sum(num_voxels_n_ * self.time_reg_coeffs, axis=-1) + \
                            self.time_reg_intercepts
                    # need to divide this cuz adding it to each layer individually
                    bb3d_time_preds[:, 0] += self.expected_bb3d_err

        dense_ops_time_preds = self.det_head_post_wcet_ms
        if not self.model.use_voxelnext:
            dense_ops_time_preds += self.bb2d_times_ms[num_tiles]

        return vfe_time_preds, bb3d_time_preds, dense_ops_time_preds, num_voxels

    def pred_final_req_time_ms(self, dethead_indexes):
        return self.det_head_post_wcet_ms

    # this can be used for both points and voxels
    def fit_voxel_time_data(self, voxel_data, times_data):
        coeffs, intercepts = [], []
        for i in range(self.bb3d_num_l_groups): # should be 4, num bb3d conv blocks
            voxels = voxel_data[:, i:i+1] / self.num_voxels_normalizer
            times = times_data[:, i:i+1]

            if self.time_reg_degree == 2:
                voxels = np.concatenate((voxels, np.square(voxels)), axis=-1)
            reg = LinearRegression().fit(voxels, times)

            coeffs.append(reg.coef_.flatten())
            intercepts.append(reg.intercept_[0])
        return np.array(coeffs), np.array(intercepts)

    def get_calib_data_arranged(self):
        all_voxels = self.calib_data_dict.get('bb3d_voxels', list())
        all_bb3d_times = self.calib_data_dict.get('bb3d_time_ms', list())
        all_points = self.calib_data_dict.get('vfe_points', list())
        all_vfe_times = self.calib_data_dict.get('vfe_time_ms', list())

        if len(all_voxels)>0 and len(all_bb3d_times)>0:
            all_voxels=np.array(all_voxels, dtype=float)
            all_bb3d_times=np.array(all_bb3d_times, dtype=float)
        if len(all_points)>0 and len(all_vfe_times)>0:
            all_points=np.array(all_points, dtype=float)
            all_vfe_times=np.array(all_vfe_times, dtype=float)
        return all_voxels, all_bb3d_times, all_points, all_vfe_times

    def read_calib_data(self, fname='calib_data.json'):
        f = open(fname)
        self.calib_data_dict = json.load(f)
        f.close()

        # Fit the linear model for bb3
        all_voxels, all_bb3d_times, all_points, all_vfe_times = self.get_calib_data_arranged()

        def remove_noise(times, voxels):
            max_times = np.max(times, axis=1)
            perc99 = np.percentile(max_times, 99)
            mask = (times < perc99)
            return np.expand_dims(times[mask], -1), np.expand_dims(voxels[mask], -1)

        #all_bb3d_times, all_voxels = remove_noise(all_bb3d_times, all_voxels) # corrupts data
        if len(all_vfe_times) > 0:
            all_vfe_times, all_points = remove_noise(all_vfe_times, all_points)

        # plot voxel to time graph
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
        if len(all_vfe_times) > 0:
            vfe_times  = np.sum(all_vfe_times, axis=-1, keepdims=True)
            vfe_points = all_points[:, :1]
            axes[0].scatter(vfe_points, vfe_times, label='data') #, label='data')
            axes[0].set_xlabel('Number of input points', fontsize='x-large')
            axes[0].set_ylabel('VFE execution time (msec)', fontsize='x-large')
        if len(all_bb3d_times) > 0:
            bb3d_times  = np.sum(all_bb3d_times, axis=-1, keepdims=True)
            bb3d_voxels = all_voxels[:, :1]
            axes[1].scatter(bb3d_voxels, bb3d_times, label='data') #, label='data')
            axes[1].set_xlabel('Number of input voxels', fontsize='x-large')
            axes[1].set_ylabel('3D backbone\nexecution time (msec)', fontsize='x-large')

        # As a baseline predictor, do linear regression using
        # the number of voxels or points

        if self.sched_vfe and len(all_vfe_times) > 0:
            vfe_points_n = vfe_points / self.num_points_normalizer
            if self.time_reg_degree == 2:
                vfe_points_n = np.concatenate((vfe_points_n,
                    np.square(vfe_points_n)), axis=-1)
            reg = LinearRegression().fit(vfe_points_n, vfe_times)

            self.vfe_time_reg_coeffs = reg.coef_
            self.vfe_time_reg_intercepts = reg.intercept_

            pred_times = np.sum(vfe_points_n * self.vfe_time_reg_coeffs.flatten(), \
                    axis=-1) +  self.vfe_time_reg_intercepts
            axes[0].scatter(vfe_points.flatten(), pred_times.flatten(), label='pred')

        if self.sched_bb3d and self.use_baseline_bb3d_predictor:
            bb3d_voxels_n = bb3d_voxels / self.num_voxels_normalizer
            if self.time_reg_degree == 2:
                bb3d_voxels_n = np.concatenate((bb3d_voxels_n,
                    np.square(bb3d_voxels_n)), axis=-1)
            reg = LinearRegression().fit(bb3d_voxels_n, bb3d_times)

            self.time_reg_coeffs = reg.coef_
            self.time_reg_intercepts = reg.intercept_

            pred_times = np.sum(bb3d_voxels_n * self.time_reg_coeffs.flatten(), \
                    axis=-1) +  self.time_reg_intercepts
            axes[1].scatter(bb3d_voxels.flatten(), pred_times.flatten(), label='pred')
        plt.legend()
        plt.savefig(f'../../latest_exp_plots/{self.model.model_name}_vfe_and_bb3d_time'
                f'_res{self.res_idx}.pdf')
        plt.clf()

        if self.sched_bb3d and not self.use_baseline_bb3d_predictor:
            self.time_reg_coeffs, self.time_reg_intercepts = self.fit_voxel_time_data(all_voxels, all_bb3d_times)

            # the input is voxels: [NUM_CHOSEN_TILES, self.bb3d_num_l_groups],
            # the output is times: [NUM_CHOSEN_TILEs, self.bb3d_num_l_groups]
            all_voxels_n = np.expand_dims(all_voxels, -1) / self.num_voxels_normalizer
            all_voxels_n = np.concatenate((all_voxels_n, np.square(all_voxels_n)), axis=-1)
            all_preds = np.sum(all_voxels_n * self.time_reg_coeffs, axis=-1)
            all_preds += self.time_reg_intercepts
            diffs = all_bb3d_times - all_preds
            print('Excepted time prediction error for each 3D backbone layer\n' \
                    ' assuming the number of voxels are predicted perfectly:')
            for i in range(self.bb3d_num_l_groups):
                get_stats(diffs[:,i])

        dh_post_time_data = self.calib_data_dict['det_head_post_time_ms']
        self.det_head_post_wcet_ms = np.percentile(dh_post_time_data, \
                99, method='lower')
        print('det_head_post_wcet_ms', self.det_head_post_wcet_ms)

        if not self.model.use_voxelnext:
            bb2d_time_data = self.calib_data_dict['bb2d_time_ms']
            self.bb2d_times_ms = np.array([np.percentile(arr if arr else [0], 90, method='lower') \
                    for arr in bb2d_time_data])

            assert self.bb2d_times_ms[-1] != 0.
            for i in range(self.bb2d_times_ms.shape[0]-1, 0, -1):
                if self.bb2d_times_ms[i] == 0.:
                    self.bb2d_times_ms[i] = self.bb2d_times_ms[i+1]

            print('Fused dense convolutions times:')
            print(self.bb2d_times_ms)

        if 'exec_times' in self.calib_data_dict:
            # calculate the 3dbb err cdf
            time_dict = self.calib_data_dict['exec_times']
            if self.sched_bb3d and 'Backbone3D' in time_dict:
                if 'Backbone3D-IL' in time_dict:
                    Backbone3D_times = np.array(time_dict['Backbone3D-IL']) + \
                            np.array(time_dict['Backbone3D-Fwd'])
                else:
                    Backbone3D_times = time_dict['Backbone3D']

                bb3d_pred_err = np.array(Backbone3D_times) - \
                        np.array(self.calib_data_dict['bb3d_preds'])
                if 'VoxelHead-conv-hm' in time_dict:
                    bb3d_pred_err += np.array(time_dict['VoxelHead-conv-hm'])

                print('Overall 3D Backbone time prediction error stats:')
                min_, mean_, perc1_, perc5_, perc95_, perc99_, max_ = get_stats(bb3d_pred_err)
                self.expected_bb3d_err = int(os.getenv('PRED_ERR_MS', 0))
                #print('Expected bb3d error ms:', self.expected_bb3d_err)

            if self.sched_vfe and 'VFE' in time_dict:
                vfe_nn_times = np.array(time_dict['VFE'])
                vfe_pred_err = vfe_nn_times - np.array(self.calib_data_dict['vfe_preds'])

                print('Overall VFE time prediction error stats:')
                min_, mean_, perc1_, perc5_, perc95_, perc99_, max_ = get_stats(vfe_pred_err)
                #self.expected_vfe_err = int(os.getenv('PRED_ERR_MS', 0))
                #print('Expected bb3d error ms:', self.expected_vfe_err)

    def get_chosen_tile_num(self):
        return self.chosen_tiles_calib

    def collect_data_v2(self, sched_algo, fname="calib_data.json"):
        print('Calibration starting...')
        print('POINT_CLOUD_RANGE:', self.model.vfe.point_cloud_range)
        print('VOXEL_SIZE:', self.model.vfe.voxel_size)
        print('GRID SIZE:', self.model.vfe.grid_size)

        num_samples = min(512, len(self.dataset))
        print('Number of samples:', num_samples)

        if not self.model.use_voxelnext:
            bb2d_time_data =  [list() for _ in range(self.bb2d_times_ms.shape[0])]
        dh_post_time_data = []

        gc.disable()
        sample_idx, tile_num = 0, 1
        while sample_idx < num_samples:
            time_begin = time.time()
            print(f'Processing sample {sample_idx}-{sample_idx+5}', end='', flush=True)

            # Enforce a number of tile
            for i in range(5): # supports up to 45 tiles assuming number of samples is 240
                if sample_idx < num_samples:
                    self.chosen_tiles_calib = self.num_tiles if i == 0 else tile_num
                    self.model([sample_idx])
                    lbd = self.model.latest_batch_dict
                    if not self.model.use_voxelnext:
                        e1, e2 = lbd['bb2d_time_events']
                        bb2d_time = e1.elapsed_time(e2)
                        nt = get_num_tiles(lbd['chosen_tile_coords'])
                        bb2d_time_data[nt].append(bb2d_time)
                    e1, e2 = lbd['detheadpost_time_events']
                    dh_post_time_data.append(e1.elapsed_time(e2))
                    sample_idx += 1
                    gc.collect()
            tile_num = (tile_num % self.num_tiles) + 1

            time_end = time.time()
            #print(torch.cuda.memory_allocated() // 1024**2, "MB is being used by tensors.")
            print(f' took {round(time_end-time_begin, 2)} seconds.')
        gc.enable()

        self.calib_data_dict = {
                "version": 2,
                "chosen_tile_coords": self.model.add_dict['chosen_tiles_1'][1:],
                "det_head_post_time_ms": dh_post_time_data,
        }

        if self.sched_vfe:
            self.calib_data_dict.update({
                    "vfe_time_ms": self.model.add_dict['vfe_layer_times'][1:],
                    "vfe_points": self.model.add_dict['vfe_point_nums'][1:]})

        if self.sched_bb3d:
            self.calib_data_dict.update({
                    "bb3d_time_ms": self.model.add_dict['bb3d_layer_times'][1:],
                    "bb3d_voxels": self.model.add_dict['bb3d_voxel_nums'][1:]})

        if not self.model.use_voxelnext:
            self.calib_data_dict["bb2d_time_ms"] = bb2d_time_data

        with open(fname, "w") as outfile:
            json.dump(self.calib_data_dict, outfile, indent=4)

        # Read and parse calib data after dumping
        self.read_calib_data(fname)

