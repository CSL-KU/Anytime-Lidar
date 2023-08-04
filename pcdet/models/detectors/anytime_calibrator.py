import torch
import time
import json
import numpy as np
#import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from ...ops.cuda_point_tile_mask import cuda_point_tile_mask
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from torch.utils.data import TensorDataset, DataLoader

def calc_grid_size(pc_range, voxel_size):
    return np.array([ int((pc_range[i+3]-pc_range[i]) / vs)
            for i, vs in enumerate(voxel_size)])

def tile_coords_to_id(tile_coords):
    tid = 0
    for tc in tile_coords:
        tid += 2 ** tc
    return tid

class AnytimeCalibrator():
    def __init__(self, model):
        self.model = model
        self.dataset = model.dataset
        self.num_det_heads = len(model.dense_head.class_names_each_head)

        self.num_tiles = model.model_cfg.TILE_COUNT
        self.filtering_times_ms = []
        self.bb2d_times_ms = np.zeros((2**self.num_tiles,), dtype=np.float)
        self.det_head_pre_times_ms = np.zeros((2**self.num_tiles,), dtype=np.float)
        self.det_head_post_times_ms = np.zeros((2**self.num_det_heads,), dtype=np.float)

    def get_points(self, index):
        batch_dict = self.dataset.collate_batch([self.dataset[index]])
        batch_dict['points'] = torch.from_numpy(batch_dict['points']).cuda()
        assert 'batch_size' in batch_dict
        return batch_dict

    def process(self, batch_dict, record=True, noprint=False):
        # I need to use cuda events to measure the time of each section
        with torch.no_grad():
            cuda_events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            torch.cuda.synchronize()
            if record:
                cuda_events[0].record()
            voxel_tile_coords = batch_dict['voxel_tile_coords']
            chosen_tile_coords = batch_dict['chosen_tile_coords']
            tile_filter = cuda_point_tile_mask.point_tile_mask(voxel_tile_coords, \
                        chosen_tile_coords)
            batch_dict['mask'] = tile_filter
            for k in ('voxel_features', 'voxel_coords'):
                batch_dict[k] = batch_dict[k][tile_filter].contiguous()

            if record:
                cuda_events[1].record()

            batch_dict = self.model.backbone_3d(batch_dict)

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
                tid = tile_coords_to_id(chosen_tile_coords.cpu().numpy())
                self.bb2d_times_ms[tid] = bb2d_time_ms

                # all possibilities are touched
                det_head_pre_time_ms  = cuda_events[3].elapsed_time(cuda_events[4])
                self.det_head_pre_times_ms[tid] = det_head_pre_time_ms

                # when makin prediction, use number of heads to determine wcet since we might no
                # be able to hit all the possible detection head configurations
                det_head_post_time_ms = cuda_events[4].elapsed_time(cuda_events[5])
                hid = tile_coords_to_id(batch_dict['dethead_indexes'])
                self.det_head_post_times_ms[hid] = det_head_post_time_ms
                self.model.dense_head.calc_skip_times()

        if record and not noprint:
            print(f'Elapsed times: {filter_time_ms}, {bb3d_time_ms}, {bb2d_time_ms}'
                    ', {det_head_pre_time_ms}, {det_head_post_time_ms}')

        return (bb3d_time_ms if record else 0.)

    def collect_data(self):
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
        model_coeffs = []
        scene_tokens = []

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

            all_tiles = torch.cat((nonempty_tile_coords, nonempty_tile_coords)).contiguous()
            all_voxel_counts= torch.cat((voxel_counts, voxel_counts)).contiguous()

            ntc_sz = nonempty_tile_coords.size(0)

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
            batch_dict['chosen_tile_coords'] = chosen_tile_coords
            bb3d_time = self.process(batch_dict, record=True, noprint=True)

            bb3d_time_series[-1].append(bb3d_time)
            vcounts = torch.zeros((self.num_tiles,), dtype=torch.long, device='cuda')
            vcounts[nonempty_tile_coords] = voxel_counts
            voxel_counts_series[-1].append(vcounts)
            
            time_end = time.time()
            print(f' took {round(time_end-time_begin, 2)} seconds.')

        for i, vc_l in enumerate(voxel_counts_series):
            for j, vc in enumerate(vc_l):
                voxel_counts_series[i][j] = vc.cpu().tolist()
                chosen_tc_series[i][j] = chosen_tc_series[i][j].cpu().tolist()


        dataset = { "voxel_counts": voxel_counts_series,
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

        with open("calib_dataset.json", "w") as outfile:
            json.dump(dataset, outfile, indent=4)


def define_pred_model(dataset_dict):
    inp_size = dataset_dict['self.num_tiles']
    outp_size = 1 # Execution time
    #outp_size = inp_size * (inp_size - 1) + 1
    #outp_size = inp_size * (inp_size + 1) // 2
    # Model 1
    possibilities = inp_size * (inp_size +1) // 2
    model = torch.nn.Sequential(
        torch.nn.Linear(inp_size, possibilities),
        torch.nn.ReLU(),
        torch.nn.Linear(possibilities, outp_size)).cuda()

    # Model 2
    #model = torch.nn.Sequential(
    #    torch.nn.Linear(inp_size, 64),
    #    torch.nn.ReLU(),
    #    torch.nn.Linear(64, 128),
    #    torch.nn.ReLU(),
    #    torch.nn.Linear(128, outp_size)).cuda()
    return model


def remove_bad_data(dataset_dict):
    scene_tokens = dataset_dict["scene_tokens"]

    prev_st = ''
    mask = np.ones((len(scene_tokens),), dtype=bool)
    for i, st in enumerate(scene_tokens):
        if prev_st != st:
            mask[i] = False
        prev_st = st

    for key in ('voxel_counts', 'bb3d_time_ms', 'chosen_tile_coords', 'scene_tokens'):
        arr = dataset_dict[key]
        dataset_dict[key] = [a for i, a in enumerate(arr) if mask[i]]

    return dataset_dict

def plot_data(dataset_dict, num_samples_to_compare=2):
    vcounts_samples = dataset_dict['voxel_counts']
    exec_times_ms_samples = dataset_dict['bb3d_time_ms']

    colors='rgbcmyk'
    for i in range(len(vcounts_samples) - num_samples_to_compare + 1):
        sample_ids = ''
        for j in range(num_samples_to_compare):
            sample_idx = i + j
            vcounts = np.array(vcounts_samples[sample_idx])
            num_voxels = vcounts.sum(1)
            exec_times = np.array(exec_times_ms_samples[sample_idx])

            #model = LinearRegression()
            #x = num_voxels.reshape((-1, 1))
            #xvals = np.hstack((x,x**2))
            #model.fit(xvals, pct)
            #pct_pred = model.predict(xvals)
            #if i == 0:
            #    print('Model params:', model.coef_, model.intercept_)
            #    model_coeffs.append((model.coef_, model.intercept_))
            #p = plt.plot(nv, pct_pred, label=f"Sample {num}", c=colors[num%len(colors)])
            plt.scatter(num_voxels, exec_times, label=f"Sample {sample_idx}",
                    c=colors[sample_idx%len(colors)])
            sample_ids += '_' + str(sample_idx)
        plt.xlim([0, 100000])
        plt.ylim([0, 200])
        plt.xlabel('Number of voxels')
        plt.ylabel('Execution time (ms)')
        plt.legend()
        plt.savefig(f'./plots/data{sample_ids}.png')
        plt.clf()

def fit_transform(x):
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s

    return x, m, s

def transform(x, m, s):
    x -= m
    x /= s

    return x

def train_pred_model_v1(dataset_dict, model):
    # First, remove the samples which has less number of sweeps than max
    vcounts_samples = dataset_dict['voxel_counts']
    exec_times_ms_samples = dataset_dict['bb3d_time_ms']
    chosen_tc_samples = dataset_dict["chosen_tile_coords"]
    scene_tokens = dataset_dict["scene_tokens"]
    self.num_tiles = dataset_dict["self.num_tiles"]

    # Second, extend the data to be able to turn it into a tensor
    vcounts, exec_times_ms, chosen_tc = [], [], []
    for v, e, c in zip(vcounts_samples, exec_times_ms_samples, chosen_tc_samples):
        vcounts.extend(v)
        exec_times_ms.extend(e)
        chosen_tc.extend(c)

    d = 'cuda'
    vcounts = torch.tensor(vcounts, dtype=torch.float, device=d)
    exec_times_ms = torch.tensor(exec_times_ms, dtype=torch.float,
            device=d).unsqueeze(-1)

    dataset = TensorDataset(vcounts, exec_times_ms)

    train_dset, test_dset = torch.utils.data.random_split(dataset, [0.7, 0.3])
    dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Create the loss function and optimizer.
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the network.
    model.train()
    print('Training start')
    for epoch in range(500):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data

            optimizer.zero_grad()

            inputs, i_m, i_s = fit_transform(inputs)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % 5 == 0:
            l =  running_loss/len(dataloader)
            print(f'Epoch {epoch} loss {l}')
            

    torch.save(model.state_dict(), './voxel_time_model.pt')


def test_pred_model_v1(dataset_dict, model):
    model.load_state_dict(torch.load('./voxel_time_model.pt'))
    model.cuda()
    model.eval()

    vcounts_samples = dataset_dict['voxel_counts']
    exec_times_ms_samples = dataset_dict['bb3d_time_ms']
    chosen_tc_samples = dataset_dict["chosen_tile_coords"]

    vcounts = []
    for v in vcounts_samples:
        vcounts.extend(v)
    vcounts = torch.tensor(vcounts, dtype=torch.float, device='cuda')
    vcounts, m, s = fit_transform(vcounts)

    for i, (vc, et_ms) in enumerate(zip(vcounts_samples, exec_times_ms_samples)):
        vc = torch.tensor(vc)
        vc_sum = vc.sum(dim=1).numpy()
        vc = vc.cuda().float()
        vc = transform(vc, m, s)
        with torch.no_grad():
            #torch.cuda.synchronize()
            #tstart = time.time()
            et_ms_predicted = model(vc)
            #torch.cuda.synchronize()
            #tdiff_msec = round((time.time() - tstart)*1000, 2)
            #print(f'Size:{vc.size()}, Time:{tdiff_msec} ms')

        plt.scatter(vc_sum, et_ms, label="actual")
        et_ms_predicted = et_ms_predicted.cpu()
        plt.scatter(vc_sum, et_ms_predicted.numpy(), label="predicted")
        plt.legend()
        plt.savefig(f'./plots/sample{i}.png')
        plt.clf()
        diff = (et_ms_predicted - torch.tensor(et_ms)).flatten()
        print(f'Sample {i} exec time diff mean', torch.mean(diff).item(),
                'min', torch.min(diff).item(), 'max', torch.max(diff).item())

#def main():
#    #collect_data()
#
#    f = open('dataset.json')
#    dataset_dict = json.load(f)
#    f.close()
#    dataset_dict = remove_bad_data(dataset_dict)
#
#    #plot_data(dataset_dict)
#    model = define_pred_model(dataset_dict)
#    #train_pred_model_v1(dataset_dict, model)
#    test_pred_model_v1(dataset_dict, model)
#
#def remove_flipping_data(dataset_dict):
#    chosen_tc_samples = dataset_dict["chosen_tile_coords"]
#    for i, ctcs_of_sample in enumerate(chosen_tc_samples):
#        #Using chosen tile coords, determine the indexes of exec times in the outp tensor
#        mask = np.ones((len(ctcs_of_sample),), dtype=bool)
#        for j, chosen_tile_coords in enumerate(ctcs_of_sample):
#            s = chosen_tile_coords[0]
#            e = chosen_tile_coords[-1]
#            mask[j] = (e >= s)
#        for key in ('voxel_counts', 'bb3d_time_ms', 'chosen_tile_coords'):
#            arr = dataset_dict[key][i]
#            dataset_dict[key][i] = [a for j, a in enumerate(arr) if mask[j]]
#
#    return dataset_dict
#
#
