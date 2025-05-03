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

class MURALCalibrator():
    def __init__(self, model, res_idx, num_slices):
        self.model = model
        self.dataset = model.dataset

        self.res_idx = res_idx
        self.resdiv = model.resolution_dividers[self.res_idx]

        self.time_reg_degree = 2 # if 2, quadratic func

        self.preprocess_wcet_ms = 0.

        # quadratic predictor is ok for vfe
        self.vfe_num_l_groups = 1
        self.num_points_normalizer = 1000000.
        self.vfe_time_reg_coeffs = np.ones((1,self.time_reg_degree), dtype=np.float32)
        self.vfe_time_reg_intercepts = np.ones((1,), dtype=np.float32)

        self.bb3d_exist = ('BACKBONE_3D' in model.model_cfg)
        self.num_voxels_normalizer = 100000.
        if self.bb3d_exist:
            self.treat_bb3d_as_single_l_group = False
            if self.treat_bb3d_as_single_l_group:
                self.bb3d_num_l_groups = 1
                self.bb3d_time_reg_coeffs = np.ones((self.bb3d_num_l_groups, self.time_reg_degree,),
                                                    dtype=np.float32)
                self.bb3d_time_reg_intercepts = np.ones((1,), dtype=np.float32)
            else:
                self.bb3d_num_l_groups = self.model.backbone_3d.num_layer_groups
                self.bb3d_time_reg_coeffs = np.ones((self.bb3d_num_l_groups, self.time_reg_degree),
                                                    dtype=np.float32)
                self.bb3d_time_reg_intercepts = np.ones((self.bb3d_num_l_groups,), dtype=np.float32)

        # key is wsize, value is ms
        self.dense_inp_slice_sz = self.model.dense_inp_slice_sz
        self.num_slices = num_slices
        self.dense_ops_times_ms = np.zeros(num_slices).astype(float)
        self.postprocess_wcet_ms = .0
        self.calib_data_dict = None
        self.last_pred = np.zeros(5)
        self.e2e_wcet_ms = 999.0
        self.e2e_times_ms_arr = np.zeros(512)

        self.data_sched_thr = float(os.environ.get('DATA_SCHED_THR', 0.7))
        self.repeat_points = 0

    def get_e2e_wcet_ms(self):
        return self.e2e_wcet_ms

    def get_e2e_min_ms(self):
        return self.e2e_min_ms

    def quadratic_time_pred(self, data_arr, reg_coeffs, reg_intercepts, normalizer):
        data_arr_n = data_arr / normalizer
        data_arr_n_sq = np.square(data_arr_n)
        time_preds = data_arr_n * reg_coeffs[:, 0] + data_arr_n_sq * reg_coeffs[:, 1] + \
                reg_intercepts

        return time_preds

    # NOTE batch size has to be 1 !
    def pred_exec_time_ms(self, num_points : int, pillar_counts: np.ndarray, num_slices: int,
                          consider_prep_time=False) -> float:
        vfe_time_pred = self.quadratic_time_pred(num_points, self.vfe_time_reg_coeffs,
                self.vfe_time_reg_intercepts, self.num_points_normalizer)

        bb3d_time_pred = 0.
        if self.bb3d_exist:
            if len(pillar_counts.shape) > 1:
                pillar_counts = pillar_counts.sum(1)
            bb3d_time_pred = self.quadratic_time_pred(pillar_counts, self.bb3d_time_reg_coeffs,
                    self.bb3d_time_reg_intercepts, self.num_voxels_normalizer)
            if not self.treat_bb3d_as_single_l_group:
                bb3d_time_pred = bb3d_time_pred.sum()

        dense_ops_time_pred = self.dense_ops_times_ms[num_slices - 1]

        self.last_pred = np.array([self.preprocess_wcet_ms, vfe_time_pred[0], bb3d_time_pred,
                                   dense_ops_time_pred, self.postprocess_wcet_ms])
        return np.sum(self.last_pred if consider_prep_time else self.last_pred[1:]).item()

    def find_config_to_meet_dl(self,
                            num_points : int,
                            pillar_counts : np.ndarray,
                            slc_bgn_idx: int,
                            slc_end_idx: int,
                            deadline_ms: float,
                            flip:bool = False): # -> Tuple[float, int, int]:
        vfe_time_pred = self.quadratic_time_pred(num_points, self.vfe_time_reg_coeffs,
                self.vfe_time_reg_intercepts, self.num_points_normalizer)

        if flip:
            pillar_counts = np.fliplr(pillar_counts)
        pillar_counts_cs = np.cumsum(pillar_counts, 1).T
        bb3d_time_preds = self.quadratic_time_pred(pillar_counts_cs, self.bb3d_time_reg_coeffs,
                self.bb3d_time_reg_intercepts, self.num_voxels_normalizer)
        if not self.treat_bb3d_as_single_l_group:
            bb3d_time_preds = bb3d_time_preds.sum(1)
        if flip:
            bb3d_time_preds = np.flip(bb3d_time_preds)

        num_chosen_slices = slc_end_idx - slc_bgn_idx + 1
        time_pred = vfe_time_pred[0] + self.postprocess_wcet_ms
        bb3d_t = bb3d_time_preds[slc_bgn_idx if flip else slc_end_idx]
        dense_t = self.dense_ops_times_ms[num_chosen_slices-1]
        total_time_pred = time_pred + bb3d_t + dense_t

        num_slc_lower_bound = int(num_chosen_slices * self.data_sched_thr)
        while total_time_pred > deadline_ms and num_chosen_slices > num_slc_lower_bound:
            if flip:
                slc_bgn_idx += 1
                bb3d_t = bb3d_time_preds[slc_bgn_idx]
            else:
                slc_end_idx -= 1
                bb3d_t = bb3d_time_preds[slc_end_idx]
            num_chosen_slices -= 1
            dense_t = self.dense_ops_times_ms[num_chosen_slices]
            total_time_pred = time_pred + bb3d_t + dense_t

        if total_time_pred < deadline_ms:
            self.last_pred = np.array([self.preprocess_wcet_ms, vfe_time_pred[0], bb3d_t,
                                   dense_t, self.postprocess_wcet_ms])

        return (total_time_pred, slc_bgn_idx, slc_end_idx)

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
        return np.array(coeffs).astype(np.float32), np.array(intercepts).astype(np.float32)

    def get_calib_data_arranged(self):
        num_voxels = self.calib_data_dict.get('num_voxels', list())
        bb3d_times = self.calib_data_dict.get('bb3d_times_ms', list())
        num_points = self.calib_data_dict.get('num_points', list())
        vfe_times = self.calib_data_dict.get('vfe_times_ms', list())

        if len(num_voxels)>0 and len(bb3d_times)>0:
            num_voxels=expand_dim_if_one(np.array(num_voxels, dtype=np.float32))
            bb3d_times=expand_dim_if_one(np.array(bb3d_times, dtype=np.float32))
        if len(num_points)>0 and len(vfe_times)>0:
            num_points=np.array(num_points, dtype=np.float32).flatten()
            vfe_times=np.array(vfe_times, dtype=np.float32).flatten()
        return num_voxels, bb3d_times, num_points, vfe_times

    def read_calib_data(self, fname='calib_data.json'):
        f = open(fname)
        self.calib_data_dict = json.load(f)
        f.close()

        self.preprocess_wcet_ms = np.percentile(self.calib_data_dict['preprocess_times_ms'], 50.)

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
        for sz, latency in self.dense_ops_times_dict.items():
            latency99perc = np.percentile(latency, 99)
            self.dense_ops_times_ms[int(sz)//self.dense_inp_slice_sz - 1] = latency99perc

        self.postprocess_wcet_ms = np.percentile(self.calib_data_dict['postprocess_times_ms'], 50)
        if False:
            print('preprocess_wcet_ms', self.preprocess_wcet_ms)
            print('dense_ops_times_ms')
            print(self.dense_ops_times_ms)
            print('postprocess_wcet_ms', self.postprocess_wcet_ms)

        if 'e2e_times_ms' in self.calib_data_dict:
            self.e2e_times_ms_arr = np.array(self.calib_data_dict['e2e_times_ms'])
            print('End to end execution time stats (ms):')
            min_, mean_, perc1_, perc5_, perc95_, perc99_, max_ = get_stats(self.e2e_times_ms_arr)
            self.e2e_wcet_ms = perc99_
            self.e2e_min_ms = min_

    def collect_data(self, fname="calib_data.json"):
        print('Calibration starting...')
        pc_range = self.model.filter_pc_range.cpu().numpy()
        print('POINT CLOUD FILTER RANGE:', pc_range)
        print('VOXEL_SIZE:', self.model.vfe.voxel_size)
        print('GRID SIZE:', self.model.vfe.grid_size)

        num_samples = min(len(self.dataset), 512)
        print('Number of samples:', num_samples)

        preprocess_ms_arr = np.empty(num_samples, dtype=np.float32)
        
        num_points_arr = np.empty(num_samples, dtype=np.int32)
        vfe_ms_arr = np.empty(num_samples, dtype=np.float32)

        if self.bb3d_exist:
            num_voxels_arr = np.empty((num_samples, self.bb3d_num_l_groups), dtype=np.int32)
            bb3d_ms_arr = np.empty((num_samples, self.bb3d_num_l_groups), dtype=np.float32)

        dense_ops_ms_dict = {}

        postprocess_ms_arr = np.empty(num_samples, dtype=np.float32)
        e2e_ms_arr = np.empty(num_samples, dtype=np.float32)

        gc.disable()
        deadline_backup = self.model._default_deadline_sec
        self.model._default_deadline_sec = 100.0
        sample_idx, tile_num = 0, 1
        time_begin = time.time()
        pc_xwidth = pc_range[3] - pc_range[0]
        while sample_idx < num_samples:
            if sample_idx % 10 == 0 and sample_idx > 0:
                elapsed_sec = round(time.time() - time_begin, 2)
                print(f'Processing samples {sample_idx-10}-{sample_idx} took {elapsed_sec} seconds.')
                time_begin = time.time()

            self.repeat_points = sample_idx % 4

            # Enforce different point clound ranges to hit different input sizes
            squeeze_amount_meters = sample_idx % int(pc_xwidth*0.5)
            self.model.calib_pc_range[3] = pc_range[3] - squeeze_amount_meters

            self.model([sample_idx])

            lbd = self.model.latest_batch_dict

            preprocess_ms_arr[sample_idx] =  self.model._time_dict['PreProcess'][-1]
            if 'Sched' in self.model._time_dict:
                preprocess_ms_arr[sample_idx] += self.model._time_dict['Sched'][-1]

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
            x_min, x_max = lbd['tensor_slice_inds']
            tensor_width = str(int(x_max - x_min + 1) * self.dense_inp_slice_sz)
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


        if self.model.dense_conv_opt_on:
            # Calculate DenseOps times that were not calculated
            dense_ops_inp_sz = self.model.inp_tensor_sizes[self.res_idx]
            dummy_inp = torch.rand(dense_ops_inp_sz).cuda()
            max_width = dense_ops_inp_sz[3]
            #print(sorted(list(dense_ops_ms_dict.keys())))
            slc_sz = self.dense_inp_slice_sz
            min_inp_width = slc_sz * 2
            for target_width in range(max_width, min_inp_width-1, -slc_sz):
                if str(target_width) not in dense_ops_ms_dict:
                    dense_ops_ms_dict[str(target_width)] = []
                    print('Calibrating dense ops for missing slice width:', target_width)
                    torch.cuda.synchronize()
                    for i in range(10):
                        cevents = [torch.cuda.Event(enable_timing=True) for e in range(2)]
                        cevents[0].record()
                        dummy_inp_slice = dummy_inp[..., :target_width].contiguous()
                        self.model.forward_eval_dense(dummy_inp_slice)
                        cevents[1].record()
                        torch.cuda.synchronize()
                        dense_ops_ms_dict[str(target_width)].append(cevents[0].elapsed_time(cevents[1]))

        self.model._default_deadline_sec = deadline_backup
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


