import copy
import pickle
import torch
from pathlib import Path

import numpy as np
from tqdm import tqdm
import os

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate

#Patchwork++
#remove_ground = False
#try:
#    import sys
#    patchwork_module_path = "/root/patchwork-plusplus/build/python_wrapper"
#    sys.path.insert(0, patchwork_module_path)
#    import pypatchworkpp
#    params = pypatchworkpp.Parameters()
#    params.sensor_height = 1.84
#    params.max_range = 54.0
#    params.enable_RNR = True
#    params.enable_TGR = True
#    params.verbose = False
#    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)
#    # NOTE uncomment the following line to enable ground removal
##    remove_ground = True
#except ImportError:
#    print("Cannot find pypatchworkpp, won't remove ground.")

class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []

        # Force using train data for calibration
        do_calib = (int(os.getenv('CALIBRATION', '0')) > 0)
        if do_calib:
            print('Using CALIBRATION split')
        self.include_nuscenes_data("train" if do_calib else self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)

        self.infos.extend(nuscenes_infos)
        self.logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = []
        sweep_times_list = []

        #for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
        for k in range(max_sweeps - 2, -1, -1):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])

            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        sweep_points_list.append(points)
        sweep_times_list.append(np.zeros((points.shape[0], 1)))

        # REMOVE GROUND HERE
        #global remove_ground
        #if remove_ground:
        #    global PatchworkPLUSPLUS
        #    for i, sweep in enumerate(sweep_points_list):
        #        PatchworkPLUSPLUS.estimateGround(sweep)
        #        sweep_points_list[i] = PatchworkPLUSPLUS.getNonground()
        #        sweep_times_list[i] = sweep_times_list[i][:sweep_points_list[i].shape[0]]

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def get_metadata_dict(self, index):
        info = copy.deepcopy(self.infos[index])

        input_dict = {
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        return input_dict

    def getitem_pre(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })

        return self.prepare_data_pre(data_dict=input_dict)

    def get_gt_as_pred_dict(self, index):
        info = self.infos[index]
        gt_dict = {}
        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            gt_dict.update({
                'pred_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'pred_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })

            if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
                gt_boxes = gt_dict['pred_boxes']
                gt_boxes[np.isnan(gt_boxes)] = 0
                gt_dict['pred_boxes'] = gt_boxes

            if not self.dataset_cfg.PRED_VELOCITY:
                gt_dict['pred_boxes'] = gt_dict['pred_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

            selected = common_utils.keep_arrays_by_name(gt_dict['pred_names'], self.class_names)
            gt_dict['pred_boxes'] = torch.from_numpy(gt_dict['pred_boxes'][selected]).float()
            gt_dict['pred_names'] = gt_dict['pred_names'][selected]
            gt_dict['pred_labels'] = torch.tensor([self.class_names.index(n) + 1 \
                    for n in gt_dict['pred_names']], dtype=torch.long)
            gt_dict['pred_scores'] = torch.ones_like(gt_dict['pred_labels'], dtype=torch.float)

        return gt_dict


    def getitem_post(self, data_dict):
        self.prepare_data_post(data_dict=data_dict)
        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        return data_dict

    def __getitem__(self, index):
        return self.getitem_post(self.getitem_pre(index))

    def evaluation(self, det_annos, class_names, **kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import nuscenes_utils
        if 'loaded_nusc' in kwargs:
            nusc = kwargs['loaded_nusc']
        else:
            nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path),
                    verbose=True)
        if 'boxes_in_global_coords' in kwargs:
            boxes_in_global_coords = kwargs['boxes_in_global_coords']
        else:
            boxes_in_global_coords = False
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos( \
                det_annos, nusc, boxes_in_global_coords=boxes_in_global_coords)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }
        if 'nusc_annos_outp' in kwargs:
            kwargs['nusc_annos_outp'].update(nusc_annos)

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)

        if not hasattr(self, 'logger'):
            self.logger=common_utils.create_logger()

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        dt = kwargs['det_elapsed_musec'] if 'det_elapsed_musec' in kwargs else None
        do_calib = (int(os.getenv('CALIBRATION', '0')) > 0)
        print('Do calibration flag is', do_calib)
        es = "train" if do_calib else eval_set_map[self.dataset_cfg.VERSION]
        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path='', #res_path,
            eval_set=es,
            output_dir=str(output_path),
            verbose=True,
            det_elapsed_musec=dt,
            data_dict=nusc_annos
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    def tracking_evaluation(self, output_path, res_path, **kwargs):
        from nuscenes.eval.tracking.evaluate import TrackingEval
        from nuscenes.eval.common.config import config_factory as track_configs

        cfg = track_configs("tracking_nips_2019")

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }

        do_calib = (int(os.getenv('CALIBRATION', '0')) > 0)
        es = "train" if do_calib else eval_set_map[self.dataset_cfg.VERSION]
        nusc_eval = TrackingEval(
            config=cfg,
            result_path=res_path,
            eval_set=es,
            output_dir=str(output_path),
            verbose=True,
            nusc_version=self.dataset_cfg.VERSION,
            nusc_dataroot=str(self.root_path),
        )
        metrics_summary = nusc_eval.main(render_curves=False)


    def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        import torch

        database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo'
        db_info_save_path = self.root_path / f'nuscenes_dbinfos_{max_sweeps}sweeps_withvelo.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_nuscenes_info(version, data_path, save_path, max_sweeps=10):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from . import nuscenes_utils
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = nuscenes_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps
    )

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    args = parser.parse_args()

    if args.func == 'create_nuscenes_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        create_nuscenes_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'nuscenes',
            save_path=ROOT_DIR / 'data' / 'nuscenes',
            max_sweeps=dataset_cfg.MAX_SWEEPS,
        )

#        nuscenes_dataset = NuScenesDataset(
#            dataset_cfg=dataset_cfg, class_names=None,
#            root_path=ROOT_DIR / 'data' / 'nuscenes',
#            logger=common_utils.create_logger(), training=True
#        )
#        nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
