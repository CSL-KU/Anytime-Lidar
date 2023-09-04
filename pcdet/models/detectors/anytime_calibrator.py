import torch
import time
import json
import numpy as np
import numba
import gc
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
    perc5_ = np.percentile(np_arr, 5, method='lower')
    perc95_ = np.percentile(np_arr, 95, method='lower')
    print("Min\t5Perc\tMean\t95Perc\tMax")
    print(f'{min_:.2f}\t{perc5_:.2f}\t{mean_:.2f}\t{perc95_:.2f}\t{max_:.2f}')
    return min_, mean_, perc5_, perc95_, max_


class AnytimeCalibrator():
    def __init__(self, model):
        self.model = model
        self.calib_data_dict = None
        if model is None:
            self.dataset = None
            self.num_det_heads = 6
            self.num_tiles = 18
        else:
            self.dataset = model.dataset
            self.num_det_heads = len(model.dense_head.class_names_each_head)
            self.num_tiles = model.model_cfg.TILE_COUNT

        # backbone2d and detection head heatmap convolutions
        # first elem unused
        self.bb2d_times_ms = np.zeros((self.num_tiles+1,), dtype=float)
        self.det_head_post_times_ms = np.zeros((2**self.num_det_heads,), dtype=float)

        self.time_reg_degree = 2
        self.bb3d_num_l_groups  =self.model.backbone_3d.num_layer_groups
        self.time_reg_coeffs = np.ones((self.bb3d_num_l_groups, self.time_reg_degree), dtype=float)
        self.time_reg_intercepts = np.ones((self.bb3d_num_l_groups,), dtype=float)

        self.voxel_coeffs_over_layers = np.array([[1.] * self.num_tiles \
                for _ in range(self.bb3d_num_l_groups)])

    # voxel dists should be [self.bb3d_num_l_groups, num_tiles]
    def commit_bb3d_updates(self, ctc, voxel_dists):
        voxel_dists = voxel_dists[:, ctc]
        self.voxel_coeffs_over_layers[:, ctc] = voxel_dists / voxel_dists[0]

    # overhead on jetson-agx: 1 ms
    def pred_req_times_ms(self, vcount_area, tiles_queue, num_tiles): # [num_nonempty_tiles, num_max_tiles]
        vcounts = vcount_area * self.voxel_coeffs_over_layers
        num_voxels = np.empty((tiles_queue.shape[0], vcounts.shape[0]), dtype=vcounts.dtype)
        for i in range(len(tiles_queue)):
            num_voxels[i] = np.sum(vcounts[:, tiles_queue[:i+1]], axis=1)
        if self.time_reg_degree == 1:
            bb3d_time_preds = num_voxels * self.time_reg_coeffs.flatten() + \
                    self.time_reg_intercepts
            bb3d_time_preds = np.sum(bb3d_time_preds, axis=-1)
        elif self.time_reg_degree == 2:
            num_voxels = np.expand_dims(num_voxels, -1)
            num_voxels = np.concatenate((num_voxels, np.square(num_voxels)), axis=-1)
            bb3d_time_preds = np.sum(num_voxels * self.time_reg_coeffs, axis=-1) + \
                    self.time_reg_intercepts
            bb3d_time_preds = np.sum(bb3d_time_preds, axis=-1)

        return bb3d_time_preds, self.bb2d_times_ms[num_tiles] + \
                self.det_head_post_times_ms[-1]

    def pred_final_req_time_ms(self, dethead_indexes):
        hid = tile_coords_to_id(dethead_indexes)
        return self.det_head_post_times_ms[hid]

    def read_calib_data(self, fname='calib_data.json'):
        f = open(fname)
        self.calib_data_dict = json.load(f)
        f.close()

        # Fit the linear model for bb3
        #vcounts_samples = self.calib_data_dict['voxel_counts']
        bb3d_voxels_samples = self.calib_data_dict['bb3d_voxels']
        exec_times_ms_samples = self.calib_data_dict['bb3d_time_ms']

        all_times, all_voxels = [], []
        for bb3d_voxels_s, exec_times_ms_s in zip(bb3d_voxels_samples, exec_times_ms_samples):
            all_voxels.extend(bb3d_voxels_s)
            all_times.extend(exec_times_ms_s)

        all_times=np.array(all_times, dtype=float)
        all_voxels=np.array(all_voxels, dtype=float)

        # 2 bec second order: v, v_2
        self.time_reg_coeffs = np.empty((self.bb3d_num_l_groups, self.time_reg_degree), dtype=float)
        self.time_reg_intercepts = np.empty((self.bb3d_num_l_groups,), dtype=float)
        for i in range(self.bb3d_num_l_groups): # should be 4, num bb3d conv blocks
            voxels = all_voxels[:, i:i+1]
            times = all_times[:, i:i+1]

            if self.time_reg_degree == 2:
                voxels = np.concatenate((voxels, np.square(voxels)), axis=-1)
            reg = LinearRegression().fit(voxels, times)

            self.time_reg_coeffs[i] = reg.coef_
            self.time_reg_intercepts[i] = reg.intercept_

        # the input is voxels: [NUM_CHOSEN_TILES, self.bb3d_num_l_groups],
        # the output is times: [NUM_CHOSEN_TILEs, self.bb3d_num_l_groups]
        all_voxels = np.expand_dims(all_voxels, -1)
        all_voxels = np.concatenate((all_voxels, np.square(all_voxels)), axis=-1)
        all_preds = np.sum(all_voxels * self.time_reg_coeffs, axis=-1) + self.time_reg_intercepts
        diffs = all_preds - all_times
        for i in range(self.bb3d_num_l_groups):
            get_stats(diffs[:,i])

        # Predicting how voxel counts will change over layers is hard
        # Use history to approximate
        ctc_samples = self.calib_data_dict['chosen_tile_coords']
        voxels_dataset = [[] for _ in range(self.num_tiles)]
        for ctc_s, bb3d_v_s in zip(ctc_samples, bb3d_voxels_samples):
            for ctc, bb3d_v in zip(ctc_s, bb3d_v_s):
                if len(ctc) == 1:
                    # Normalize and append
                    bb3d_v = np.array(bb3d_v, dtype=float)
                    voxels_dataset[ctc[0]].append(bb3d_v / bb3d_v[0])
                else:
                    break # skip to next sample

        self.voxel_coeffs_over_layers = [sum(vd) / len(vd) for vd in voxels_dataset]
        self.voxel_coeffs_over_layers = np.array(self.voxel_coeffs_over_layers).transpose()
        print('coefficient that guess the changes in number of voxels in bb3d:')
        print(self.voxel_coeffs_over_layers.T)

        bb2d_time_data = self.calib_data_dict['bb2d_time_ms']
        self.bb2d_times_ms = np.array([np.percentile(arr if arr else [0], 99, method='lower') \
                for arr in bb2d_time_data])
        print('bb2d_times_ms')
        print(self.bb2d_times_ms)
        dh_post_time_data = self.calib_data_dict['det_head_post_time_ms']
        self.det_head_post_times_ms = np.array([np.percentile(arr if arr else [0], \
                99, method='lower') for arr in dh_post_time_data])

        print('det_head_post_times_ms')
        print(self.det_head_post_times_ms)

    def get_points(self, index):
        batch_dict = self.dataset.collate_batch([self.dataset[index]])
        batch_dict['points'] = torch.from_numpy(batch_dict['points']).cuda()
        assert 'batch_size' in batch_dict
        return batch_dict

    def process(self, batch_dict, record=True, noprint=False):
        # I need to use cuda events to measure the time of each section
        bb3d_times_ms, bb3d_voxels, bb2d_time_ms, det_head_post_time_ms, hid = [], [], 0., 0., 0
        with torch.no_grad():
            cuda_events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            voxel_tile_coords = batch_dict['voxel_tile_coords']
            chosen_tile_coords = batch_dict['chosen_tile_coords']
            torch.cuda.synchronize()
            if record:
                #cuda_events[0].record()
                batch_dict['record_time'] = True
                batch_dict['record_int_vcounts'] = True
                batch_dict['record_int_vcoords'] = True
                batch_dict['tile_size_voxels'] = self.tile_size_voxels
                batch_dict['num_tiles'] = self.num_tiles

            tile_filter = cuda_point_tile_mask.point_tile_mask(voxel_tile_coords, \
                        torch.from_numpy(chosen_tile_coords).cuda())
            for k in ('voxel_features', 'voxel_coords'):
                batch_dict[k] = batch_dict[k][tile_filter].contiguous()

            # it will sync itself when recording
            batch_dict = self.model.backbone_3d(batch_dict)
            #torch.cuda.synchronize()

            if record:
                bb3d_times_ms = batch_dict['bb3d_layer_times_ms']
                bb3d_voxels = batch_dict['bb3d_num_voxels']
                cuda_events[0].record()

            batch_dict = self.model.map_to_bev(batch_dict)
            batch_dict = self.model.backbone_2d(batch_dict)
            batch_dict = self.model.dense_head.forward_eval_pre(batch_dict)
            ## synchronized here

            if record:
                cuda_events[1].record()

            batch_dict = self.model.dense_head.forward_eval_post(batch_dict)

            if record:
                cuda_events[2].record()

            torch.cuda.synchronize()

            if record:
                # timing doesn't change much
                #bb3d_time_ms = cuda_events[0].elapsed_time(cuda_events[1]) # return
                bb2d_time_ms = cuda_events[0].elapsed_time(cuda_events[1])

                # all possibilities are touched from what I see in the calib data
                det_head_post_time_ms = cuda_events[1].elapsed_time(cuda_events[2])
                hid = tile_coords_to_id(batch_dict['dethead_indexes'])

                self.model.dense_head.calc_skip_times()

        if record and not noprint:
            print(f'Elapsed times: {bb3d_times_ms}, {bb2d_time_ms}'
                    ', {det_head_post_time_ms}')

        return bb3d_times_ms, bb3d_voxels, bb2d_time_ms, det_head_post_time_ms, hid

    def collect_data(self, sched_algo, fname="calib_data.json"):
        print('Calibration starting...')
        print('NUM_POINT_FEATURES:', self.model.vfe.num_point_features)
        print('POINT_CLOUD_RANGE:', self.model.vfe.point_cloud_range)
        print('VOXEL_SIZE:', self.model.vfe.voxel_size)
        print('GRID SIZE:', self.model.vfe.grid_size)

        self.tile_size_voxels = torch.tensor(\
                self.model.vfe.grid_size[1] / self.num_tiles).cuda().int()

        # This inital processing is code to warmup the cache
        batch_dict = self.get_points(1)
        batch_dict = self.model.initialize(batch_dict)
        batch_dict = self.model.vfe(batch_dict)
        batch_dict['voxel_tile_coords'], ctc, _ = \
                self.model.get_nonempty_tiles(batch_dict['voxel_coords'])
        batch_dict['chosen_tile_coords'] = ctc
        self.process(batch_dict, record=False, noprint=True)

        num_samples = len(self.dataset)
        print('Number of samples:', num_samples)

        # Let's try X scan!
        scene_tokens = [None for _ in range(num_samples)]
        voxel_counts_series = [list() for _ in range(num_samples)]
        chosen_tc_series =  [list() for _ in range(num_samples)]
        bb3d_time_series = [list() for _ in range(num_samples)]
        bb3d_voxel_series = [list() for _ in range(num_samples)]

        bb2d_time_data =  [list() for _ in range(self.bb2d_times_ms.shape[0])]
        dh_post_time_data =  [list() for _ in range(self.det_head_post_times_ms.shape[0])]

        gc.disable()
        for sample_idx in range(num_samples):
            print(f'Processing sample {sample_idx}', end='', flush=True)
            time_begin = time.time()

            batch_dict = self.get_points(sample_idx)
            batch_dict = self.model.initialize(batch_dict)
            scene_tokens[sample_idx] = self.model.token_to_scene[\
                    batch_dict['metadata'][0]['token']]
            batch_dict = self.model.schedule0(batch_dict)
            batch_dict = self.model.vfe(batch_dict)

            voxel_coords = batch_dict['voxel_coords']
            voxel_features = batch_dict['voxel_features']

            voxel_tile_coords, nonempty_tile_coords, voxel_counts = \
                    self.model.get_nonempty_tiles(voxel_coords)
            batch_dict['voxel_tile_coords'] = voxel_tile_coords

            if sched_algo == SchedAlgo.MirrorRR:
                m2 = self.num_tiles//2
                m1 = m2 - 1
                mandatory_tiles = np.array([m1, m2], dtype=int)
                rtiles = np.arange(m2+1, nonempty_tile_coords[-1]+1)
                ltiles = np.arange(m1-1, nonempty_tile_coords[0]-1, -1)
                rltiles = np.concatenate((rtiles, ltiles, rtiles, ltiles))
                v = np.zeros((self.num_tiles,), dtype=voxel_counts.dtype)
                v[nonempty_tile_coords] = voxel_counts
                voxel_counts = v

                # Initially, process only the mandatory tiles
                chosen_tc_series[sample_idx].append(mandatory_tiles)
                batch_dict['voxel_coords'] = voxel_coords
                batch_dict['voxel_features'] = voxel_features
                batch_dict['chosen_tile_coords'] = mandatory_tiles
                bb3d_time, bb3d_voxels, bb2d_time, dh_post_time, hid = self.process(batch_dict,
                        record=True, noprint=True)

                bb3d_time_series[sample_idx].append(bb3d_time)
                bb3d_voxel_series[sample_idx].append(bb3d_voxels)
                nt = mandatory_tiles.shape[0]
                bb2d_time_data[nt].append(bb2d_time)
                dh_post_time_data[hid].append(dh_post_time)
                vcounts = np.zeros((self.num_tiles,), dtype=voxel_counts.dtype)
                vcounts[mandatory_tiles] = voxel_counts[mandatory_tiles]
                voxel_counts_series[sample_idx].append(vcounts)
                tiles_range = rltiles.shape[0]//2
            elif sched_algo == SchedAlgo.RoundRobin:
                all_tiles = np.concatenate((nonempty_tile_coords, nonempty_tile_coords))
                all_voxel_counts= np.concatenate((voxel_counts, voxel_counts))
                ntc_sz = nonempty_tile_coords.shape[0]
                tiles_range = ntc_sz

            # Process rest of the possibilities expect the final one
            for tiles in range(1, tiles_range):
                for start_idx in range(tiles_range):
                    if sched_algo == SchedAlgo.RoundRobin:
                        chosen_tile_coords = all_tiles[start_idx:(start_idx+tiles)]
                        chosen_tc_series[sample_idx].append(chosen_tile_coords)
                        chosen_voxel_counts = all_voxel_counts[start_idx:(start_idx+tiles)]
                        nt = get_num_tiles(chosen_tile_coords)
                    elif sched_algo == SchedAlgo.MirrorRR:
                        chosen_tile_coords = np.concatenate((mandatory_tiles, \
                                rltiles[start_idx:(start_idx+tiles)]))
                        chosen_tc_series[sample_idx].append(chosen_tile_coords)
                        chosen_voxel_counts = voxel_counts[chosen_tile_coords]
                        nt = chosen_tile_coords.shape[0]

                    batch_dict['voxel_coords'] = voxel_coords
                    batch_dict['voxel_features'] = voxel_features
                    batch_dict['chosen_tile_coords'] = chosen_tile_coords
                    bb3d_time, bb3d_voxels, bb2d_time, dh_post_time, hid = self.process(batch_dict,
                            record=True, noprint=True)
                    bb3d_time_series[sample_idx].append(bb3d_time)
                    bb3d_voxel_series[sample_idx].append(bb3d_voxels)
                    bb2d_time_data[nt].append(bb2d_time)
                    dh_post_time_data[hid].append(dh_post_time)

                    vcounts = np.zeros((self.num_tiles,), dtype=chosen_voxel_counts.dtype)
                    vcounts[chosen_tile_coords] = chosen_voxel_counts
                    voxel_counts_series[sample_idx].append(vcounts)

            # Finally, process the entire point cloud without filtering
            chosen_tc_series[sample_idx].append(nonempty_tile_coords)

            batch_dict['voxel_coords'] = voxel_coords
            batch_dict['voxel_features'] = voxel_features
            batch_dict['chosen_tile_coords'] = nonempty_tile_coords
            bb3d_time, bb3d_voxels, bb2d_time, dh_post_time, hid = self.process(batch_dict,
                    record=True, noprint=True)
            bb3d_time_series[sample_idx].append(bb3d_time)
            bb3d_voxel_series[sample_idx].append(bb3d_voxels)
            if sched_algo == SchedAlgo.RoundRobin:
                nt = get_num_tiles(nonempty_tile_coords)
                bb2d_time_data[nt].append(bb2d_time)
                dh_post_time_data[hid].append(dh_post_time)
                vcounts = np.zeros((self.num_tiles,), dtype=voxel_counts.dtype)
                vcounts[nonempty_tile_coords] = voxel_counts
                voxel_counts_series[sample_idx].append(vcounts)
            elif sched_algo == SchedAlgo.MirrorRR:
                nt = nonempty_tile_coords.shape[0]
                bb2d_time_data[nt].append(bb2d_time)
                dh_post_time_data[hid].append(dh_post_time)
                voxel_counts_series[sample_idx].append(voxel_counts)

            gc.collect()
            time_end = time.time()
            #print(torch.cuda.memory_allocated() // 1024**2, "MB is being used by tensors.")
            print(f' took {round(time_end-time_begin, 2)} seconds.')
        gc.enable()

        for i, vc_l in enumerate(voxel_counts_series):
            for j, vc in enumerate(vc_l):
                voxel_counts_series[i][j] = vc.tolist()
                chosen_tc_series[i][j] = chosen_tc_series[i][j].tolist()

        self.calib_data_dict = {
                "voxel_counts": voxel_counts_series,
                "bb3d_time_ms": bb3d_time_series,
                "bb3d_voxels": bb3d_voxel_series,
                "scene_tokens": scene_tokens,
                "chosen_tile_coords": chosen_tc_series,
                "bb2d_time_ms": bb2d_time_data,
                "det_head_post_time_ms": dh_post_time_data,
                "det_head_attr_skip_gains": self.model.dense_head.get_attr_skip_gains(),
                "num_tiles": self.num_tiles,
                "num_det_heads" : self.num_det_heads,
        }

        with open(fname, "w") as outfile:
            json.dump(self.calib_data_dict, outfile, indent=4)


    def plot_data(self):
        vcounts_samples = self.calib_data_dict['voxel_counts']
        exec_times_ms_samples = self.calib_data_dict['bb3d_time_ms']
        bb3d_voxels_samples = self.calib_data_dict['bb3d_voxels']

#        n = len(vcounts_samples)
#        ets_max = np.empty((n,1), dtype=int)
#        vcounts_all = np.empty((n, len(vcounts_samples[0][0])), dtype=float)
#        min_et = 9999.9
#        for i in range(n):
#            vcounts_all[i] = vcounts_samples[i][-1]
#            ets_max[i,0] = exec_times_ms_samples[i][-1]
#            min_et = min(min(exec_times_ms_samples[i]), min_et)
#        vcounts_all = np.array(vcounts_all)
        #reg = LinearRegression().fit(vcounts_all, ets_max)
        #ets_predicted = reg.predict(vcounts_all)
        #diff = ets_max - ets_predicted
        #get_stats(diff)

#        reg = self.exec_time_model
#
#        colors='rgbcmyk'
#        for sample_idx in range(len(vcounts_samples)):
#            vcounts = np.array(vcounts_samples[sample_idx])
#            num_voxels = vcounts.sum(1, keepdims=True)
#
#            pred_exec_times = reg.predict(vcounts)
#            pred_exec_times = np.squeeze(np.sum(pred_exec_times, axis=1))
#            #ets_predicted = reg.predict(vcounts[-1, None])
#            #A = (ets_predicted - min_et) / num_voxels[-1]
#            #pred_exec_times = A * num_voxels + min_et
#            #pred_exec_times = np.squeeze(pred_exec_times)
#            gt_exec_times = np.sum(np.array(exec_times_ms_samples[sample_idx]), axis=1)
#            gt_exec_times = np.squeeze(gt_exec_times)
#            diff = pred_exec_times - gt_exec_times
#            get_stats(diff)
#            plt.scatter(num_voxels, gt_exec_times, label="Actual")
#            plt.scatter(num_voxels, pred_exec_times, label="Predicted")
#            plt.xlim([0, 100000])
#            plt.ylim([0, 200])
#            plt.xlabel('Number of voxels')
#            plt.ylabel('Execution time (ms)')
#            plt.legend()
#            plt.savefig(f'/root/Anytime-Lidar/tools/plots/data{sample_idx}.png')
#            plt.clf()

if __name__ == "__main__":
    calibrator = AnytimeCalibrator(None)
    calibrator.read_calib_data('/root/Anytime-Lidar/tools/calib_data_1.json')
    calibrator.plot_data()

