import torch
import time
import json
import numpy as np
import numba
import gc
import os
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from sklearn.linear_model import LinearRegression

def get_stats(np_arr):
    min_, max_, mean_ = np.min(np_arr), np.max(np_arr), np.mean(np_arr)
    perc1_ = np.percentile(np_arr, 1, method='lower')
    perc5_ = np.percentile(np_arr, 5, method='lower')
    perc95_ = np.percentile(np_arr, 95, method='lower')
    perc99_ = np.percentile(np_arr, 99, method='lower')
    print("Min\t1Perc\t5Perc\tMean\t95Perc\t99Perc\tMax")
    print(f'{min_:.2f}\t{perc1_:.2f}\t{perc5_:.2f}\t{mean_:.2f}\t{perc95_:.2f}\t{perc99_:.2f}\t{max_:.2f}')
    return (min_, mean_, perc1_, perc5_, perc95_, perc99_, max_)

def find_index_or_next_largest(arr, element):
    # Find the exact index if the element exists
    index = np.where(arr == element)[0]
    if index.size > 0:
        return index[0]

    # Find the smallest element that is greater than the given element
    larger_elements = np.where(arr > element)[0]
    if larger_elements.size > 0:
        return larger_elements[0]
    
    # If no element is greater than the given element, return -1 or a message
    return -1

def expand_dim_if_one(arr):
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=-1)
    return arr

class ValorCalibrator():
    def __init__(self, model):
        self.model = model
        self.dataset = model.dataset

        self.res_idx = model.res_idx
        self.resdiv = model.resolution_dividers[self.res_idx]

        self.time_reg_degree = 2 # if 2, quadratic func

        self.preprocess_wcet_ms = 0.

        # quadratic predictor is ok for vfe
        self.vfe_num_l_groups = 1
        self.num_points_normalizer = 1000000.
        self.vfe_time_reg_coeffs = np.ones((self.time_reg_degree,), dtype=float)
        self.vfe_time_reg_intercepts = np.ones((1,), dtype=float)

        self.bb3d_exist = ('BACKBONE_3D' in model.model_cfg)
        self.num_voxels_normalizer = 100000.
        if self.bb3d_exist:
            self.treat_bb3d_as_single_l_group = False
            if self.treat_bb3d_as_single_l_group:
                self.bb3d_num_l_groups = 1
                self.time_reg_coeffs = np.ones((self.time_reg_degree,), dtype=float)
                self.time_reg_intercepts = np.ones((1,), dtype=float)
            else:
                self.bb3d_num_l_groups = self.model.backbone_3d.num_layer_groups
                self.time_reg_coeffs = np.ones((self.bb3d_num_l_groups, self.time_reg_degree), dtype=float)
                self.time_reg_intercepts = np.ones((self.bb3d_num_l_groups,), dtype=float)

        self.dense_ops_times_ms = [] # key is wsize, value is ms

        self.postprocess_wcet_ms = .0

        self.calib_data_dict = None

    # NOTE batch size has to be 1 !
    # Call like this:
    # pred_time = module.calibrators[module.res_idx].pred_exec_time_ms(
    #        batch_dict['points'].size(0),
    #        np.array([batch_dict['bb3d_num_voxels']]),
    #        batch_dict['x_lims'][1] - batch_dict['x_lims'][0])
    def pred_exec_time_ms(self, num_points : int, num_voxels : np.ndarray, dense_wsize : int):
        vfe_time_pred = self.quadratic_time_pred(num_points, self.vfe_time_reg_coeffs,
                self.vfe_time_reg_intercepts, self.num_points_normalizer)

        bb3d_time_pred = 0.
        if self.bb3d_exist:
            bb3d_time_pred = self.quadratic_time_pred(num_voxels, self.bb3d_time_reg_coeffs,
                    self.bb3d_time_reg_intercepts, self.num_voxels_normalizer)
            if not self.treat_bb3d_as_single_l_group:
                bb3d_time_pred = bb3d_time_pred.sum()

        idx = find_index_or_next_largest(self.dense_ops_times_ms[:, 0], dense_wsize)
        dense_ops_time_pred = self.dense_ops_times_ms[idx, 1]

        return (self.preprocess_wcet_ms + vfe_time_pred + bb3d_time_pred + \
                dense_ops_time_pred + self.postprocess_wcet_ms).item()

    # fit to quadratic function
    def fit_data(self, input_data, times_data, num_l_groups, normalizer):
        coeffs, intercepts = [], []
        input_data = expand_dim_if_one(input_data)
        times_data = expand_dim_if_one(times_data)

        for i in range(num_l_groups): # should be 4, num bb3d conv blocks
            inputs = input_data[:, i:i+1] / normalizer
            times = times_data[:, i:i+1]

            inputs = np.concatenate((inputs, np.square(inputs)), axis=-1)
            reg = LinearRegression().fit(inputs, times)

            coeffs.append(reg.coef_.flatten())
            intercepts.append(reg.intercept_[0])
        return np.array(coeffs), np.array(intercepts)

    def quadratic_time_pred(self, data_arr, reg_coeffs, reg_intercepts, normalizer):
        data_arr_n_ = np.expand_dims(data_arr, -1) / normalizer
        data_arr_n_ = np.concatenate((data_arr_n_, np.square(data_arr_n_)), axis=-1)
        time_preds = np.sum(data_arr_n_ * reg_coeffs, axis=-1) + reg_intercepts
        return time_preds

    def get_calib_data_arranged(self):
        num_voxels = self.calib_data_dict.get('num_voxels', list())
        bb3d_times = self.calib_data_dict.get('bb3d_times_ms', list())
        num_points = self.calib_data_dict.get('num_points', list())
        vfe_times = self.calib_data_dict.get('vfe_times_ms', list())

        if len(num_voxels)>0 and len(bb3d_times)>0:
            num_voxels=expand_dim_if_one(np.array(num_voxels, dtype=float))
            bb3d_times=expand_dim_if_one(np.array(bb3d_times, dtype=float))
        if len(num_points)>0 and len(vfe_times)>0:
            num_points=np.array(num_points, dtype=float).flatten()
            vfe_times=np.array(vfe_times, dtype=float).flatten()
        return num_voxels, bb3d_times, num_points, vfe_times

    def read_calib_data(self, fname='calib_data.json'):
        f = open(fname)
        self.calib_data_dict = json.load(f)
        f.close()

        self.preprocess_wcet_ms = np.percentile(self.calib_data_dict['preprocess_times_ms'], 99.)

        # Fit the linear model for bb3
        num_voxels, bb3d_times, num_points, vfe_times = self.get_calib_data_arranged()

        #print(vfe_times)
        perc99 = np.percentile(vfe_times, 99)
        mask = (vfe_times < perc99)
        vfe_times, num_points = vfe_times[mask], num_points[mask]

        # Fit vfe data
        self.vfe_time_reg_coeffs, self.vfe_time_reg_intercepts = \
                self.fit_data(num_points, vfe_times, 1, self.num_points_normalizer)

        rootpth = '../../calib_plots/'
        if len(vfe_times) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
            ax.scatter(num_points, vfe_times, label='data')
            ax.set_xlabel('Number of input points', fontsize='x-large')
            ax.set_ylabel('VFE execution time (msec)', fontsize='x-large')
            vfe_time_pred = self.quadratic_time_pred(num_points, self.vfe_time_reg_coeffs,
                    self.vfe_time_reg_intercepts, self.num_points_normalizer)
            ax.scatter(num_points, vfe_time_pred, label='pred')
            plt.legend()
            plt.savefig(rootpth + f'{self.model.model_name}_vfe_res{self.res_idx}.pdf')
            plt.clf()

        if self.bb3d_exist and len(bb3d_times) > 0:
            # Fit bb3d data
            if self.treat_bb3d_as_single_l_group:
                self.bb3d_time_reg_coeffs, self.bb3d_time_reg_intercepts = self.fit_data( \
                        num_voxels[:, :1], bb3d_times.sum(axis=1), \
                        self.bb3d_num_l_groups, self.num_voxels_normalizer)
            else:
                self.bb3d_time_reg_coeffs, self.bb3d_time_reg_intercepts = self.fit_data( \
                        num_voxels, bb3d_times, self.bb3d_num_l_groups,self.num_voxels_normalizer)

            numplots = 1 if self.treat_bb3d_as_single_l_group else self.bb3d_num_l_groups + 1
            fig, axes = plt.subplots(numplots, 1, figsize=(6, 16), constrained_layout=True)

            if self.treat_bb3d_as_single_l_group:
                bb3d_time_pred = self.quadratic_time_pred(num_voxels[:, :1], self.bb3d_time_reg_coeffs,
                        self.bb3d_time_reg_intercepts, self.num_voxels_normalizer)
            else:
                bb3d_time_pred = self.quadratic_time_pred(num_voxels, self.bb3d_time_reg_coeffs,
                        self.bb3d_time_reg_intercepts, self.num_voxels_normalizer)

            axes[0].scatter(num_voxels[:, 0], bb3d_times.sum(axis=1), label='data')
            axes[0].scatter(num_voxels[:, 0], bb3d_time_pred.sum(axis=1), label='pred')
            axes[0].set_xlabel('Number of input voxels', fontsize='x-large')
            axes[0].set_ylabel('3D Backbone\nexecution time (msec)', fontsize='x-large')

            if not self.treat_bb3d_as_single_l_group:
                for i, ax in enumerate(axes[1:]):
                    ax.scatter(num_voxels[:, i], bb3d_times[:, i], label='data')
                    ax.scatter(num_voxels[:, i], bb3d_time_pred[:, i], label='pred')
                    ax.set_xlabel('Number of input voxels', fontsize='x-large')
                    ax.set_ylabel(f'3D Backbone block {i+1}\nexecution time (msec)', fontsize='x-large')

            plt.legend()
            plt.savefig(rootpth + f'{self.model.model_name}_bb3d_res{self.res_idx}.pdf')
            plt.clf()

        self.dense_ops_times_dict = self.calib_data_dict['dense_ops_ms_dict']
        dense_ops_times_tuples = []
        for k,v in self.dense_ops_times_dict.items():
            tmp  = np.percentile(v, 99)
            dense_ops_times_tuples.append((float(k), tmp))
        dense_ops_times_arr = np.array(dense_ops_times_tuples)
        inds = np.argsort(dense_ops_times_arr[:, 0])
        self.dense_ops_times_ms = dense_ops_times_arr[inds]

        self.postprocess_wcet_ms = np.percentile(self.calib_data_dict['postprocess_times_ms'], 99)
        if False:
            print('preprocess_wcet_ms', self.preprocess_wcet_ms)
            print('dense_ops_times_ms')
            print(self.dense_ops_times_ms)
            print('postprocess_wcet_ms', self.postprocess_wcet_ms)

        if 'e2e_times_ms' in self.calib_data_dict:
            print('End to end execution time stats (ms):')
            get_stats(np.array(self.calib_data_dict['e2e_times_ms']))

    def collect_data(self, fname="calib_data.json"):
        print('Calibration starting...')
        pc_range = self.model.dataset.point_cloud_range
        print('POINT_CLOUD_RANGE:', pc_range)
        print('VOXEL_SIZE:', self.model.vfe.voxel_size)
        print('GRID SIZE:', self.model.vfe.grid_size)

        num_samples = min(len(self.dataset), 512)
        print('Number of samples:', num_samples)

        preprocess_ms_arr = np.empty(num_samples, dtype=float)
        
        num_points_arr = np.empty(num_samples, dtype=int)
        vfe_ms_arr = np.empty(num_samples, dtype=float)

        num_voxels_arr = np.empty((num_samples, self.bb3d_num_l_groups), dtype=int)
        bb3d_ms_arr = np.empty((num_samples, self.bb3d_num_l_groups), dtype=float)

        dense_ops_ms_dict = {}

        postprocess_ms_arr = np.empty(num_samples, dtype=float)
        e2e_ms_arr = np.empty(num_samples, dtype=float)

        gc.disable()
        sample_idx, tile_num = 0, 1
        time_begin = time.time()
        pc_xwidth = pc_range[3] - pc_range[0]
        while sample_idx < num_samples:
            if sample_idx % 10 == 0 and sample_idx > 0:
                elapsed_sec = round(time.time() - time_begin, 2)
                print(f'Processing samples {sample_idx-10}-{sample_idx} took {elapsed_sec} seconds.')
                time_begin = time.time()

            # Enforce different point clound ranges to hit different input sizes
            squeeze_amount_meters = sample_idx % int(pc_xwidth*0.5)
            self.model.calib_pc_range[3] = pc_range[3] - squeeze_amount_meters

            self.model([sample_idx])

            lbd = self.model.latest_batch_dict

            pp_ms =  self.model._time_dict['PreProcess'][-1]
            sched_ms =  self.model._time_dict['Sched'][-1]
            preprocess_ms_arr[sample_idx] = pp_ms + sched_ms

            num_points_arr[sample_idx] = lbd['points'].size(0)
            vfe_ms_arr[sample_idx] = self.model._time_dict['VFE'][-1]

            if self.bb3d_exist:
                if 'bb3d_layer_times' in lbd:
                    num_voxels_arr[sample_idx, :] = lbd['bb3d_num_voxels']
                    bb3d_ms_arr[sample_idx, :] = lbd['bb3d_layer_times']
                else:
                    nv = lbd['voxel_coords' if 'voxel_coords' in lbd else 'pillar_coords']
                    num_voxels_arr[sample_idx, 0] = nv.size(0)
                    bb3d_ms_arr[sample_idx, 0] = self.model._time_dict['Backbone3D'][-1]

            dense_ops_ms = float(self.model._time_dict['DenseOps'][-1])
            x_min, x_max = lbd['x_lims']
            tensor_width = int(x_max - x_min)
            if tensor_width in dense_ops_ms_dict:
                dense_ops_ms_dict[tensor_width].append(dense_ops_ms)
            else:
                dense_ops_ms_dict[tensor_width] = [dense_ops_ms]

            genbox_ms =  self.model._time_dict['CenterHead-GenBox'][-1]
            postp_ms =  self.model._time_dict['PostProcess'][-1]
            postprocess_ms_arr[sample_idx] = postp_ms + genbox_ms

            e2e_ms_arr[sample_idx] = self.model._time_dict['End-to-end'][-1]

            sample_idx += 1
            if sample_idx % 100 == 0:
                gc.collect()
        gc.enable()

        self.calib_data_dict = {
                "preprocess_times_ms": preprocess_ms_arr.tolist(),
                "num_points" : num_points_arr.tolist(),
                "vfe_times_ms" : vfe_ms_arr.tolist(),
                "dense_ops_ms_dict" : dense_ops_ms_dict,
                "postprocess_times_ms": postprocess_ms_arr.tolist(),
                "e2e_times_ms": e2e_ms_arr.tolist()
        }

        if self.bb3d_exist:
            self.calib_data_dict.update({
                "num_voxels": num_voxels_arr.tolist(),
                "bb3d_times_ms": bb3d_ms_arr.tolist()
            })

        with open(fname, "w") as outfile:
            json.dump(self.calib_data_dict, outfile, indent=4)

        # Read and parse calib data after dumping
        self.read_calib_data(fname)


