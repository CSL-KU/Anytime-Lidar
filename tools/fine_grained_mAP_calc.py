import os, re
import glob
import sys
import copy
import json
import pickle
import numpy as np
from tqdm import tqdm
from itertools import product
import concurrent.futures

#from collections import OrderedDict
#import matplotlib
#from matplotlib import pyplot as plt
#matplotlib.use('Agg')
#from itertools import cycle
#import numba

#"fields": [
#        "scene",
#        "time_segment",
#        "dist_th",
#        "class",
#        "resolution",
#        "seg_sample_stats"
#    ],

def get_dl_data(datas, dl):
    for d in datas:
        if d[0] == dl:
            return d
    return datas[0]

def gen_features(bboxes, scores, labels, use_raw_data=True):
    #bboxes = det_annos['boxes_lidar']

    if use_raw_data:
        feature_coords = bboxes[:, :2]
        features = np.empty((bboxes.shape[0], 8), dtype=float)
        features[:, :3] = bboxes[:, 3:6]
        features[:, 3] = bboxes[:, 6]
        features[:, 4:6] = bboxes[:, 7:9]
        features[:, 6] = scores
        features[:, 7] = labels
    else:
        feature_coords = (bboxes[:, :2] + 57.6)
         # sizes(3), heading(1), vel(2), score(1), label(1)
        features = np.empty((bboxes.shape[0], 8), dtype=float)
        features[:, :3] = bboxes[:, 3:6] / np.array([40., 10., 15.]) # max sizes x y z
        features[:, 3] = bboxes[:, 6] / 3.14
        # assuming max vel is 15 meters per second
        features[:, 4:6] = bboxes[:, 7:9] / 15.0
        features[:, 6] = scores
        features[:, 7] = (labels-1) / 10.

    return feature_coords, features

#def create_bev_tensor(feature_coords, features):
#    bev_tensor = np.zeros((1, 8, 64, 64), dtype=float)
#    coords = feature_coords[:, :2].astype(int)//2
#    bev_tensor[0, :, coords[:,1].ravel(), coords[:,0].ravel()] = features.T
#    return bev_tensor

def calc_ap(all_tp, all_scr, all_num_gt, nelem=101) -> float:
    """ Calculated average precision. """
    sort_inds = np.argsort(-all_scr) # descending
    tp = all_tp[sort_inds]
    #scr =  all_scr[sort_inds] # this is confidence score
    fp = np.logical_not(tp)
    #print('class:', cls_nm, 'dist thr:', dist_th, 'num dets:', tp.shape)

    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)

    prec = tp / (fp + tp)
    rec = tp / float(all_num_gt)
    #nelem = 101
    rec_interp = np.linspace(0, 1, nelem)
    precision = np.interp(rec_interp, rec, prec, right=0)
    #conf = np.interp(rec_interp, rec, scr, right=0)
    #rec = rec_interp

    min_recall = 0.1
    min_precision = 0.1

    prec = np.copy(precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


class SegInfo:
    def __init__(self, inp_dir):
        self.class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
                      'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.dist_thresholds = ['0.5', '1.0', '2.0', '4.0']

        self.dataset_dict = {}

        eval_path_list = glob.glob(inp_dir + "/sampled_dets_*.pkl")
        for path in eval_path_list:
            print('Loading', path)
            with open(path, 'rb') as handle:
                d = pickle.load(handle)
                resolution = int(d['resolution'])
                print('resolution index is', resolution)
                coords_glob = d['annos_in_glob_coords']
                assert not coords_glob

                det_annos_all = d['det_annos']
                print('Num samples:', len(det_annos_all))
                for det_annos in det_annos_all:
                    scores = det_annos['score']
                    if len(scores) > 0:
                        feature_coords, features = gen_features(det_annos['boxes_lidar'], scores, 
                                det_annos['pred_labels'])

                        sample_token = det_annos['metadata']['token']
                        tpl = [feature_coords, features, resolution] # in in out
                        if sample_token in self.dataset_dict:
                            self.dataset_dict[sample_token].append(tpl)
                        else:
                            self.dataset_dict[sample_token] = [tpl]

        self.tuple_fields = ('resolution_idx', 'class', 'dist_th', 'scene', 'start_time',
                'sample_tokens', 'num_gt_arr', 'tp_arr', 'scr_arr')

        self.seg_info_tuples = []
        seg_info_path_list = glob.glob(inp_dir + "/segment_*.json")
        for path in seg_info_path_list:
            print('Loading', path)

            with open(path, 'r') as handle:
                seg_info = json.load(handle)
                res_idx = seg_info['resolution_idx']
                tuples_dict = seg_info['tuples']
                for cls_name in seg_info['class_names']:
                    for dist_th in seg_info['distance_thresholds']:
                        cur_tuples = tuples_dict[cls_name][str(dist_th)]
                        for scene_idx, msec, sample_tokens, num_gt, pred_data in cur_tuples:
                            pred_data = [np.array(pd) for pd in pred_data]
                            tp_arr = [(pd > 0.).astype(int) for pd in pred_data]
                            scr_arr = [np.abs(pd).astype(float) for pd in pred_data]
                            self.seg_info_tuples.append((res_idx, cls_name, dist_th, \
                                    scene_idx, msec, sample_tokens, num_gt, \
                                    tp_arr, scr_arr))

        print(f'All files loaded.')

        self.resolution_idx = self.tuple_fields.index('resolution_idx')
        self.class_idx = self.tuple_fields.index('class')
        self.dist_th_idx = self.tuple_fields.index('dist_th')
        self.scene_idx = self.tuple_fields.index('scene')
        self.time_segment_idx = self.tuple_fields.index('start_time')
        self.sample_tokens_idx = self.tuple_fields.index('sample_tokens')
        self.num_gt_arr_idx = self.tuple_fields.index('num_gt_arr')
        self.tp_arr_idx = self.tuple_fields.index('tp_arr')
        self.scr_arr_idx = self.tuple_fields.index('scr_arr')

        # turn time segs into ints
        seg_len = seg_info['segment_time_length_ms']
        print(f'Time segment length: {seg_len}')

        # each value of this dict holds eval data of same scene, time, dist_th and class 
        # but different resolutions
        seg_eval_data_dict = {}
        # this one on the other hand merges the classes and dist thresholds
        seg_prec_dict = {}
        # this one will be used to build the dataset
        sample_token_to_seg_dict = {}
        #sample_token_to_egovel = {}
        all_resolutions = set()
        all_segments = set()
        for i, tpl in enumerate(self.seg_info_tuples):
            # these segments were inserted considering cls scores, from high to low
            num_gt_seg = np.sum(tpl[self.num_gt_arr_idx])
            tp_arr = np.concatenate(tpl[self.tp_arr_idx])
            scr_arr = np.concatenate(tpl[self.scr_arr_idx])

            resolution = int(tpl[self.resolution_idx])
            all_resolutions.add(resolution)
            data = [resolution, tp_arr, scr_arr, num_gt_seg]

            key = self.get_key(i, use_cls=True, use_dist_th=True, use_dl=False)
            if key not in seg_eval_data_dict:
                seg_eval_data_dict[key] = []
            seg_eval_data_dict[key].append(data)

            key = self.get_key(i, use_cls=False, use_dist_th=False, use_dl=False)
            tokens = tpl[self.sample_tokens_idx]
            for tkn in tokens:
                if tkn in sample_token_to_seg_dict:
                    assert sample_token_to_seg_dict[tkn] == key
                else:
                    sample_token_to_seg_dict[tkn] = key
            all_segments.add(key)

        self.seg_eval_data_dict = seg_eval_data_dict
        self.sample_token_to_seg_dict = sample_token_to_seg_dict
        #self.sample_token_to_egovel = sample_token_to_egovel
        self.all_segments = list(all_segments)
        self.all_resolutions = list(all_resolutions)
        self.global_worst_dl = 0.

        if len(self.dataset_dict) > 0:
            to_del = [k for k,v in self.dataset_dict.items() if len(v) != len(self.all_resolutions)]
            for k in to_del:
                del self.dataset_dict[k]
            print('Dataset dict len after pruning:', len(self.dataset_dict))

    def calc_max_mAP(self, lims):
        if lims[0] == 0:
            progress_bar = tqdm(total=lims[1]-lims[0], leave=True, dynamic_ncols=True)
        else:
            progress_bar = None

        max_mAP =.0
        for it, perm in enumerate(product(self.all_resolutions, repeat=len(self.all_segments))):
            if it >= lims[0] and it < lims[1]:
                cur_seg_dls = {seg:dl for dl, seg in zip(perm, self.all_segments)}
                seg_eval_dict = {}
                for key, datas in self.seg_eval_data_dict.items():
                    seg_i = '='.join(key.split('=')[:2])
                    seg_eval_dict[key] = get_dl_data(datas, cur_seg_dls[seg_i])
                mAP = self.calc_mAP(seg_eval_dict, False)
                if mAP > max_mAP:
                    max_mAP = mAP
                    best_perm = perm
                    best_seg_eval_dict = seg_eval_dict
                    #progress_bar.set_postfix({'mAP':max_mAP})
                if progress_bar is not None:
                    progress_bar.update()

        if progress_bar is not None:
            progress_bar.close()

        return (max_mAP, best_perm, best_seg_eval_dict)

    def do_eval(self, resolution=None, upper_bound_calc_method='heuristic'):
        #if resolution is none, it will pick the resolution that gives the best result

        if resolution is None: # calculate upper bound
            max_mAP = 0.
            best_seg_eval_dict = None

            if upper_bound_calc_method == 'exhaustive':
                num_iters = len(self.all_resolutions)**len(self.all_segments)
                num_procs = 12
                step = num_iters // num_procs
                lims_all = [[i*step, (i+1)*step] for i in range(num_procs)]
                lims_all[-1][-1] = num_iters
                #print(num_iters,  lims)
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
                    futures =[executor.submit(self.calc_max_mAP, lims) for lims in lims_all]
                    for fut in futures:
                        mAP, perm_ret, seg_eval_dict_ret = fut.result()
                        if mAP > max_mAP:
                            max_mAP = mAP
                            best_perm = perm_ret
                            best_seg_eval_dict = seg_eval_dict_ret
            elif upper_bound_calc_method == 'heuristic':
                # for each segment, try all resolutions and find the best resolution
                # that gives the most boost to the mAP.
                num_iters = len(self.all_resolutions)*len(self.all_segments)
                progress_bar = tqdm(total=num_iters, leave=True, dynamic_ncols=True)
                init_dl = self.global_worst_dl if self.global_worst_dl != 0. else self.all_resolutions[0]
                cur_seg_dls = {seg:init_dl for seg in self.all_segments}
                print('Heuristic started by initializing all resolutions to', init_dl)

                with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                    for i, seg in enumerate(self.all_segments):
                        for new_dl in self.all_resolutions:
                            old_dl = cur_seg_dls[seg]
                            cur_seg_dls[seg] = new_dl
                            seg_eval_dict = {}
                            for key, datas in self.seg_eval_data_dict.items():
                                seg_i = '='.join(key.split('=')[:2])
                                seg_eval_dict[key] = get_dl_data(datas, cur_seg_dls[seg_i])
                            mAP = self.calc_mAP(seg_eval_dict, False, executor)
                            if mAP > max_mAP:
                                max_mAP = mAP
                                best_seg_eval_dict = seg_eval_dict
                                progress_bar.set_postfix({'mAP':max_mAP})
                            else:
                                cur_seg_dls[seg] = old_dl
                            progress_bar.update()
                best_perm = list(cur_seg_dls.values())
            elif upper_bound_calc_method == 'ap_based':
                num_iters = len(self.all_segments)
                progress_bar = tqdm(total=num_iters, leave=True, dynamic_ncols=True)

                cur_seg_dls = {} #seg:[] for seg in self.all_segments}

                seg_dls, best_seg_eval_dict, best_perm = {}, {}, []
                for j, seg in enumerate(self.all_segments):
                    datas_all = {dist_th:[] for dist_th in self.dist_thresholds}
                    keys_all = []
                    for key, datas in self.seg_eval_data_dict.items():
                        fields = key.split('=')
                        seg_i = '='.join(fields[:2])
                        if seg == seg_i:
                            dist_th = fields[3]
                            datas_all[dist_th].extend(datas)
                            keys_all.append(key)
                    best_dl, max_ap = self.all_resolutions[0], 0.
                    seg_ap_scores = [0.] * len(self.all_resolutions)
                    for i, dl in enumerate(self.all_resolutions):
                        ap_list = []
                        for dist_th in self.dist_thresholds:
                            datas_all_dist = datas_all[dist_th]
                            dl_datas = [d for d in datas_all_dist if d[0] == dl]
                            assert len(dl_datas) > 0
                            ap_list.append(calc_ap(
                                np.concatenate([d[1] for d in dl_datas]), # tp
                                np.concatenate([d[2] for d in dl_datas]), # scr
                                np.sum([d[3] for d in dl_datas]).item() # gt
                            ))

                        new_ap = np.mean(ap_list)
                        seg_ap_scores[i] = new_ap

                        if new_ap > max_ap:
                            best_dl = dl
                            max_ap = new_ap

                    for key in keys_all:
                        datas = get_dl_data(self.seg_eval_data_dict[key], best_dl)
                        best_seg_eval_dict[key] = datas
                    best_perm.append(best_dl)
                    cur_seg_dls[seg] = seg_ap_scores

                    progress_bar.update()
                max_mAP = self.calc_mAP(best_seg_eval_dict, False)
            else:
                print('Unkown upper bound calculation method', upper_bound_calc_method)
                return

            progress_bar.close()
            print('Resolutions of each segment:')
            print(best_perm)
            print('Upper bound mAP:', max_mAP)
            print('Resolution stats:')
            resolutions, occurances = np.unique(np.array(best_perm), return_counts=True)
            print(resolutions)
            print(occurances)
            print(np.round(occurances / np.sum(occurances), 2))

            if len(self.dataset_dict) > 0 and (upper_bound_calc_method == 'heuristic' or \
                    upper_bound_calc_method == 'ap_based'):
                dataset_tuples=[]

                for sample_tkn, inout_list in self.dataset_dict.items():
                    segkey = self.sample_token_to_seg_dict[sample_tkn]
                    if upper_bound_calc_method == 'heuristic':
                        dl = cur_seg_dls[segkey]
                        idx = [l[-1] for l in inout_list].index(dl)
                        dataset_tuples.append(inout_list[idx] + [sample_tkn])
#                            [self.sample_token_to_egovel[sample_tkn]]
                    else:
                        # Use all resolution data for train
                        seg_ap_scores = cur_seg_dls[segkey]
                        for l in inout_list:
                            dataset_tuples.append((l[0], l[1], seg_ap_scores, sample_tkn))

                print('Duplicates are not removed')
                # dump the tuples
                print(f'Dumping {len(dataset_tuples)} samples as dataset')
                with open('resolution_dataset.pkl', 'wb') as f:
                    pickle.dump({
                        'fields': ('coords', 'features', 'resolution', 'sample_tkn'),
                        'upper_bound_calc_method': upper_bound_calc_method,
                        'data':dataset_tuples}, f)
            return max_mAP
        else:
            seg_eval_dict = {}
            for key, datas in self.seg_eval_data_dict.items():
                seg_eval_dict[key] = get_dl_data(datas, resolution)
            mAP = self.calc_mAP(seg_eval_dict)
            print('mAP:', mAP)
            return mAP

    def calc_mAP(self, seg_eval_dict, print_detailed=True, parallel_executor=None):
        average_precs = []
        ap_targets = {(cls_nm, dist_th):[[],[],0] for cls_nm in self.class_names \
                for dist_th in self.dist_thresholds}

        for key, data in seg_eval_dict.items():
            tokens = key.split('=')
            stats = ap_targets[(tokens[2], tokens[3])]
            resolution, tp_arr, scr_arr, num_gt_seg = data
            stats[0].append(tp_arr)
            stats[1].append(scr_arr)
            stats[2] += num_gt_seg

        for key, stats in ap_targets.items():
            all_tp, all_scr, all_num_gt = stats
            if all_num_gt == 0:
                ap = 0.
            else:
                all_tp = np.concatenate(all_tp)
                all_scr = np.concatenate(all_scr)

                if parallel_executor is not None:
                    ap = parallel_executor.submit(calc_ap, all_tp, all_scr, all_num_gt)
                else:
                    ap = calc_ap(all_tp, all_scr, all_num_gt)

            average_precs.append([key[0], key[1], ap])

        if parallel_executor is not None:
            for ap in average_precs:
                ap[-1] = ap[-1].result()

        def print_fixed_width(s, width):
            # Format the string to a fixed width, with spaces padded to the right
            formatted_str = f"{s:<{width}}"
            print(formatted_str, end='')

        all_aps = []
        for ap in average_precs:
            if print_detailed:
                print_fixed_width(ap[0], 30)
                print_fixed_width(ap[1], 5)
                print(round(ap[2], 3))
            all_aps.append(ap[2])
        mAP = np.mean(all_aps)
        return mAP

    def get_seg_prec_recall(self, eval_data_tpl):
        resolution, tp_arr, scr_arr, num_gt_seg = eval_data_tpl

        if num_gt_seg == 0:
            return 0., 0.

        tp = np.sum(tp_arr)
        fp = len(tp_arr) - tp

        if fp == 0. and tp == 0.:
            return 0., 0.
        else:
            return (tp / (fp + tp)).item(), (tp / float(num_gt_seg)).item()

    #def __getitem__(self, idx_field_tuple):
    #    return self.seg_info_tuples[idx_field_tuple[0]][self.field_inds[idx_field_tuple[1]]]

    def __len__(self):
        return len(self.seg_info_tuples)

    def get_key(self, tpl_idx, use_cls=True, use_dist_th=True, use_dl=True):
        tpl = self.seg_info_tuples[tpl_idx]
        key = str(tpl[self.scene_idx]) + '=' + \
                str(tpl[self.time_segment_idx]) + '='
        if use_cls:
            key += str(tpl[self.class_idx]) + '='
        if use_dist_th:
            key += str(tpl[self.dist_th_idx]) +  '='
        if use_dl:
            key += str(tpl[self.resolution_idx]) + '='
        return key[:-1]

    def do_eval_all_resolutions(self):
        #VALO
        mAPs = []
        global_worst_mAP = 1.
        resolutions_sorted = sorted(seg_info.all_resolutions)
        for dl in resolutions_sorted:
            print('Resolution', dl)
            mAPs.append(seg_info.do_eval(dl))
            if mAPs[-1] < global_worst_mAP:
                self.global_worst_dl = dl
                global_worst_mAP = mAPs[-1]

        print('Resolutions mAP')
        for mAP, dl in zip(mAPs, resolutions_sorted):
            print(f"{dl}\t{mAP}")

if __name__ == '__main__':  
    inp_dir = sys.argv[1]
    seg_info = SegInfo(inp_dir)

    seg_info.do_eval_all_resolutions()
    seg_info.do_eval(None, 'ap_based')
    #seg_info.do_eval(None, 'heuristic')
    #seg_info.do_eval(None, 'exhaustive')
