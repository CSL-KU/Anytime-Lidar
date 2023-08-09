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
    from ...ops.cuda_point_tile_mask import cuda_point_tile_mask
    #from ...ops.cuda_voxel_nb_count import cuda_voxel_nb_count

def calc_grid_size(pc_range, voxel_size):
    return np.array([ int((pc_range[i+3]-pc_range[i]) / vs)
            for i, vs in enumerate(voxel_size)])

@numba.njit()
def tile_coords_to_id(tile_coords):
    tid = 0
    for tc in tile_coords:
        tid += 2 ** tc
    return int(tid)

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
        self.filtering_times_ms = []
        self.filtering_wcet_ms = 1.0
        self.bb2d_times_ms = np.zeros((2**self.num_tiles,), dtype=float)
        self.det_head_pre_times_ms = np.zeros((2**self.num_tiles,), dtype=float)
        self.det_head_post_times_ms = np.zeros((2**self.num_det_heads,), dtype=float)
        self.combined_bb2d_dhpre_times_ms = self.bb2d_times_ms + self.det_head_pre_times_ms

        self.min_bb3d_time_ms = 9999.9
        self.max_exec_time_model = LinearRegression().fit(
                [np.arange(self.num_tiles)], [1])

    def pred_req_times_ms(self, vcounts, tile_coords):
        num_voxels = vcounts.sum(1, keepdims=True)

        ets_predicted = self.max_exec_time_model.predict(vcounts[-1, None])
        # Ax + b = exec_time
        A = (ets_predicted - self.min_bb3d_time_ms) / num_voxels[-1]
        bb3d_times = (A * num_voxels + self.min_bb3d_time_ms).flatten()
        bb3d_times += self.filtering_wcet_ms

        post_bb3d_times = np.empty((tile_coords.shape[0],), dtype=float)
        for i in range(tile_coords.shape[0]):
            tid = tile_coords_to_id(tile_coords[:i+1])
            post_bb3d_times[i] = self.combined_bb2d_dhpre_times_ms[tid]

        return bb3d_times, post_bb3d_times + self.det_head_post_times_ms[-1] # wcet

    def pred_final_req_time_ms(self, dethead_indexes):
        hid = tile_coords_to_id(dethead_indexes)
        return self.det_head_post_times_ms[hid]

    def read_calib_data(self, fname='calib_data.json'):
        f = open(fname)
        self.calib_data_dict = json.load(f)
        f.close()

        # Fit the linear model for bb3
        vcounts_samples = self.calib_data_dict['voxel_counts']
        exec_times_ms_samples = self.calib_data_dict['bb3d_time_ms']
        n = len(vcounts_samples)
        max_ets = np.empty((n,1), dtype=int)
        vcounts_all = np.empty((n, len(vcounts_samples[0][0])), dtype=float)
        glob_min_et = 9999.9
        for i in range(n):
            vcounts_all[i] = vcounts_samples[i][-1]
            max_ets[i,0] = exec_times_ms_samples[i][-1]
            glob_min_et = min(min(exec_times_ms_samples[i]), glob_min_et)

        self.min_bb3d_time_ms = glob_min_et
        self.max_exec_time_model = LinearRegression().fit(vcounts_all, max_ets)

        self.filtering_times_ms = self.calib_data_dict['filtering_times_ms']
        self.filtering_wcet_ms = np.percentile(self.filtering_times_ms, 99, method='lower')

        self.bb2d_times_ms = np.array(self.calib_data_dict['bb2d_times_ms'])
        self.det_head_pre_times_ms = np.array(self.calib_data_dict['det_head_pre_times_ms'])
        self.combined_bb2d_dhpre_times_ms = self.bb2d_times_ms + self.det_head_pre_times_ms

        self.det_head_post_times_ms = np.array(self.calib_data_dict['det_head_post_times_ms'])


    def get_points(self, index):
        batch_dict = self.dataset.collate_batch([self.dataset[index]])
        batch_dict['points'] = torch.from_numpy(batch_dict['points']).cuda()
        assert 'batch_size' in batch_dict
        return batch_dict

    def process(self, batch_dict, record=True, noprint=False):
        # I need to use cuda events to measure the time of each section
        with torch.no_grad():
            cuda_events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            voxel_tile_coords = batch_dict['voxel_tile_coords']
            chosen_tile_coords = batch_dict['chosen_tile_coords'].cpu().numpy()
            torch.cuda.synchronize()
            if record:
                cuda_events[0].record()
            tile_filter = cuda_point_tile_mask.point_tile_mask(voxel_tile_coords, \
                        torch.from_numpy(chosen_tile_coords).cuda())
            for k in ('voxel_features', 'voxel_coords'):
                batch_dict[k] = batch_dict[k][tile_filter].contiguous()

            if record:
                cuda_events[1].record()

            batch_dict = self.model.backbone_3d(batch_dict)
            torch.cuda.synchronize()

            if record:
                cuda_events[2].record()

            batch_dict = self.model.map_to_bev(batch_dict)
            batch_dict = self.model.backbone_2d(batch_dict)

            if record:
                cuda_events[3].record()

            batch_dict = self.model.dense_head.forward_eval_pre(batch_dict)
            ## synchronized here

            if record:
                cuda_events[4].record()
                batch_dict['record'] = True

            batch_dict = self.model.dense_head.forward_eval_post(batch_dict)

            if record:
                cuda_events[5].record()

            torch.cuda.synchronize()

            if record:
                # timing doesn't change much
                filter_time_ms = cuda_events[0].elapsed_time(cuda_events[1])
                self.filtering_times_ms.append(filter_time_ms) # take 99perc later

                # use neural network
                bb3d_time_ms = cuda_events[1].elapsed_time(cuda_events[2]) # return

                # all possibilities are touched
                bb2d_time_ms = cuda_events[2].elapsed_time(cuda_events[3])
                tid = tile_coords_to_id(chosen_tile_coords)
                self.bb2d_times_ms[tid] = bb2d_time_ms

                # all possibilities are touched
                det_head_pre_time_ms  = cuda_events[3].elapsed_time(cuda_events[4])
                self.det_head_pre_times_ms[tid] = det_head_pre_time_ms

                # all possibilities are touched from what I see in the calib data
                det_head_post_time_ms = cuda_events[4].elapsed_time(cuda_events[5])
                hid = tile_coords_to_id(batch_dict['dethead_indexes'])
                self.det_head_post_times_ms[hid] = det_head_post_time_ms
                self.model.dense_head.calc_skip_times()

        if record and not noprint:
            print(f'Elapsed times: {filter_time_ms}, {bb3d_time_ms}, {bb2d_time_ms}'
                    ', {det_head_pre_time_ms}, {det_head_post_time_ms}')

        return (bb3d_time_ms if record else 0.)

    def collect_data(self, fname="calib_data.json"):
        print('Calibration starting...')
        print('NUM_POINT_FEATURES:', self.model.vfe.num_point_features)
        print('POINT_CLOUD_RANGE:', self.model.vfe.point_cloud_range)
        print('VOXEL_SIZE:', self.model.vfe.voxel_size)
        print('GRID SIZE:', self.model.vfe.grid_size)

        # This inital processing is code to warmup the cache
        batch_dict = self.get_points(1)
        batch_dict = self.model.projection(batch_dict)
        batch_dict = self.model.vfe(batch_dict)
        batch_dict['voxel_tile_coords'], batch_dict['chosen_tile_coords'], _ = \
                self.model.get_nonempty_tiles(batch_dict['voxel_coords'])
        self.process(batch_dict, record=False, noprint=True)

        # Let's try X scan!
        voxel_counts_series = []
        chosen_tc_series = []
        bb3d_time_series = []
        scene_tokens = []

        gc.disable()
        print('Number of samples:', len(self.dataset))
        for sample_idx in range(len(self.dataset)):
            print(f'Processing sample {sample_idx}', end='')
            time_begin = time.time()

            batch_dict = self.get_points(sample_idx)
            batch_dict = self.model.projection(batch_dict)
            scene_tokens.append(self.model.token_to_scene[batch_dict['metadata'][0]['token']])
            batch_dict = self.model.vfe(batch_dict)

            voxel_coords = batch_dict['voxel_coords']
            voxel_features = batch_dict['voxel_features']

            voxel_tile_coords, nonempty_tile_coords, voxel_counts = \
                    self.model.get_nonempty_tiles(voxel_coords)
            batch_dict['voxel_tile_coords'] = voxel_tile_coords

            all_tiles = torch.cat((nonempty_tile_coords, nonempty_tile_coords))
            all_voxel_counts= torch.cat((voxel_counts, voxel_counts)).contiguous()

            ntc_sz = nonempty_tile_coords.shape[0]

            bb3d_time_series.append([])
            voxel_counts_series.append([])
            chosen_tc_series.append([])
            for tiles in range(1, ntc_sz):
                for start_idx in range(ntc_sz):
                    chosen_tile_coords = all_tiles[start_idx:(start_idx+tiles)]
                    chosen_tc_series[-1].append(chosen_tile_coords)
                    chosen_voxel_counts = all_voxel_counts[start_idx:(start_idx+tiles)]

                    batch_dict['voxel_coords'] = voxel_coords
                    batch_dict['voxel_features'] = voxel_features
                    batch_dict['chosen_tile_coords'] = chosen_tile_coords
                    bb3d_time = self.process(batch_dict, record=True, noprint=True)
                    bb3d_time_series[-1].append(bb3d_time)
                    vcounts = torch.zeros((self.num_tiles,), dtype=torch.long, device='cuda')
                    vcounts[chosen_tile_coords] = chosen_voxel_counts
                    voxel_counts_series[-1].append(vcounts)

            # Finally, process the entire point cloud without filtering
            chosen_tc_series[-1].append(nonempty_tile_coords)

            batch_dict['voxel_coords'] = voxel_coords
            batch_dict['voxel_features'] = voxel_features
            batch_dict['chosen_tile_coords'] = nonempty_tile_coords
            bb3d_time = self.process(batch_dict, record=True, noprint=True)

            bb3d_time_series[-1].append(bb3d_time)
            vcounts = torch.zeros((self.num_tiles,), dtype=torch.long, device='cuda')
            vcounts[nonempty_tile_coords] = voxel_counts
            voxel_counts_series[-1].append(vcounts)

            time_end = time.time()
            print(f' took {round(time_end-time_begin, 2)} seconds.')
            gc.collect()
        gc.enable()

        for i, vc_l in enumerate(voxel_counts_series):
            for j, vc in enumerate(vc_l):
                voxel_counts_series[i][j] = vc.cpu().tolist()
                chosen_tc_series[i][j] = chosen_tc_series[i][j].cpu().tolist()

        self.calib_data_dict = {
                "voxel_counts": voxel_counts_series,
                "bb3d_time_ms": bb3d_time_series,
                "scene_tokens": scene_tokens,
                "chosen_tile_coords": chosen_tc_series,
                "filtering_times_ms": self.filtering_times_ms,
                "bb2d_times_ms": self.bb2d_times_ms.tolist(),
                "det_head_pre_times_ms": self.det_head_pre_times_ms.tolist(),
                "det_head_post_times_ms": self.det_head_post_times_ms.tolist(),
                "det_head_attr_skip_gains": self.model.dense_head.get_attr_skip_gains(),
                "num_tiles": self.num_tiles,
                "num_det_heads" : self.num_det_heads,
        }

        with open(fname, "w") as outfile:
            json.dump(self.calib_data_dict, outfile, indent=4)


    def plot_data(self):
        vcounts_samples = self.calib_data_dict['voxel_counts']
        exec_times_ms_samples = self.calib_data_dict['bb3d_time_ms']

        # First, let's see if we can predict the max execution time
        n = len(vcounts_samples)
        ets_max = np.empty((n,1), dtype=int)
        vcounts_all = np.empty((n, len(vcounts_samples[0][0])), dtype=float)
        min_et = 9999.9
        for i in range(n):
            vcounts_all[i] = vcounts_samples[i][-1]
            ets_max[i,0] = exec_times_ms_samples[i][-1]
            min_et = min(min(exec_times_ms_samples[i]), min_et)
        vcounts_all = np.array(vcounts_all)

        def get_stats(np_arr):
            min_, max_, mean_ = np.min(np_arr), np.max(np_arr), np.mean(np_arr)
            perc95_ = np.percentile(np_arr, 95, method='lower')
            perc99_ = np.percentile(np_arr, 99, method='lower')
            print("Min\tMean\t95Perc\t99Perc\tMax")
            print(f'{min_:.2f}\t{mean_:.2f}\t{perc95_:.2f}\t{perc99_:.2f}\t{max_:.2f}')
            return min_, mean_, perc95_, perc99_, max_

        reg = LinearRegression().fit(vcounts_all, ets_max)
        ets_predicted = reg.predict(vcounts_all)
        diff = ets_max - ets_predicted
        get_stats(diff)

        colors='rgbcmyk'
        for sample_idx in range(len(vcounts_samples)):
            vcounts = np.array(vcounts_samples[sample_idx])
            num_voxels = vcounts.sum(1, keepdims=True)

            ets_predicted = reg.predict(vcounts[-1, None])
            A = (ets_predicted - min_et) / num_voxels[-1]

            pred_exec_times = A * num_voxels + min_et
            pred_exec_times = np.squeeze(pred_exec_times)
            gt_exec_times = np.array(exec_times_ms_samples[sample_idx])
            diff = pred_exec_times - gt_exec_times
            get_stats(diff)
            plt.scatter(num_voxels, gt_exec_times, label="Actual")
            plt.scatter(num_voxels, pred_exec_times, label="Predicted")
            plt.xlim([0, 100000])
            plt.ylim([0, 200])
            plt.xlabel('Number of voxels')
            plt.ylabel('Execution time (ms)')
            plt.legend()
            plt.savefig(f'/root/Anytime-Lidar/tools/plots/data{sample_idx}.png')
            plt.clf()

if __name__ == "__main__":
    calibrator = AnytimeCalibrator(None)
    calibrator.read_calib_data('/root/Anytime-Lidar/tools/calib_data.json')
    calibrator.plot_data()

