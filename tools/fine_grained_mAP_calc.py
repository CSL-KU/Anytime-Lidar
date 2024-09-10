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
#        "deadline_ms",
#        "seg_sample_stats"
#    ],

def get_dl_data(datas, dl):
    for d in datas:
        if d[0] == dl:
            return d
    return datas[0]

def gen_features(bboxes, scores, labels, coords_2d=False):
    #bboxes = det_annos['boxes_lidar']
    #TODO, consider duplicated feature coords
    if coords_2d:
       feature_coords = (bboxes[:, :2] + 57.6).astype(int) # since pc range is -57.6 +57+6
    else:
       feature_coords = (bboxes[:, :3] + 57.6)

     # sizes(3), heading(1), vel(2), score(1), label(1)
    features = np.empty((bboxes.shape[0], 8), dtype=float)
    features[:, :3] = bboxes[:, 3:6] / np.array([40., 10., 15.]) # max sizes x y z
    features[:, 3] = bboxes[:, 6] / 3.14
    # assuming max vel is 15 meters per second 
    features[:, 4:6] = bboxes[:, 7:9] / 15.0
    features[:, 6] = scores
    features[:, 7] = (labels-1) / 10.

    return feature_coords, features

def create_bev_tensor(feature_coords, features):
    bev_tensor = np.zeros((1, 8, 64, 64), dtype=float)
    coords = feature_coords[:, :2].astype(int)//2
    bev_tensor[0, :, coords[:,1].ravel(), coords[:,0].ravel()] = features.T
    return bev_tensor

def calc_ap(all_tp, all_scr, all_num_gt) -> float:
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
    nelem = 101
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

        eval_path_list = glob.glob(inp_dir + "/*.pkl")
        for path in eval_path_list:
            print('Loading', path)
            with open(path, 'rb') as handle:
                d = pickle.load(handle)
                deadline_ms = d['calib_deadline_ms']
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
                        tpl = [feature_coords, features, deadline_ms] # in in out
                        if sample_token in self.dataset_dict:
                            self.dataset_dict[sample_token].append(tpl)
                        else:
                            self.dataset_dict[sample_token] = [tpl]

        self.seg_info_tuples = []
        seg_info_path_list = glob.glob(inp_dir + "/*.json")
        for path in seg_info_path_list:
            print('Loading', path)
            with open(path, 'r') as handle:
                seg_info = json.load(handle)
                seg_info_fields = seg_info['fields']
                self.seg_info_tuples += seg_info['tuples']

        print(f'All files loaded.')

        for i, f in enumerate(seg_info_fields):
            self.__setattr__(f+'_idx', i)

#        if dist_th is not None:
#        self.seg_info_tuples = [t for t in self.seg_info_tuples \
#                 if t[self.dist_th_idx] == 2.0]

#        if class_name is not None:
#        self.seg_info_tuples = [t for t in self.seg_info_tuples \
#                 if t[self.class_idx] == 'car']
        
        self.scene_to_idx={}
        self.scene_to_idx_counter=0

        # turn time segs into ints
        seg = self.seg_info_tuples[0][self.time_segment_idx]
        seg_len = seg[1] - seg[0] # assume all segments have the same length
        print(f'Time segment length: {seg_len}')
        for t in self.seg_info_tuples:
            t[self.time_segment_idx] = int(t[self.time_segment_idx][0] / seg_len)

        # each value of this dict holds eval data of same scene, time, dist_th and class 
        # but different deadlines
        seg_eval_data_dict = {}
        # this one on the other hand merges the classes and dist thresholds
        seg_prec_dict = {}
        # this one will be used to build the dataset
        sample_token_to_seg_dict = {}
        sample_token_to_egovel = {}
        all_deadlines = set()
        all_segments = set()
        for i, tpl in enumerate(self.seg_info_tuples):
            seg_sample_stats = tpl[self.seg_sample_stats_idx] # list of dicts
            num_gt_seg, tp_arr, scr_arr = 0, [], []
            # these segments were inserted considering cls scores, from high to low
            for sample in seg_sample_stats: # all samples in the segment
                num_gt_seg += sample['num_gt']
                #sample['egopose_translation_xy']
                sample_token_to_egovel[sample['sample_token']] = sample['egovel_xy']
                predictions = sample['pred_data']
                if len(predictions) > 0:
                    if isinstance(predictions[0], dict):
                        tp_arr += [p['is_true_pos'] for p in predictions]
                        scr_arr += [p['detection_score'] for p in predictions]
                    else:
                        tp_arr += [p[0] for p in predictions]
                        scr_arr += [p[1] for p in predictions]

            tp_arr = np.array(tp_arr, dtype=int)
            scr_arr = np.array(scr_arr, dtype=float)

            deadline = int(float(tpl[self.deadline_ms_idx]))
            all_deadlines.add(deadline)
            data = [deadline, tp_arr, scr_arr, num_gt_seg]

            key = self.get_key(i, use_cls=True, use_dist_th=True, use_dl=False)
            if key not in seg_eval_data_dict:
                seg_eval_data_dict[key] = []
            seg_eval_data_dict[key].append(data)

            key = self.get_key(i, use_cls=False, use_dist_th=False, use_dl=False)
            tokens = [s['sample_token'] for s in seg_sample_stats]
            for tkn in tokens:
                if tkn in sample_token_to_seg_dict:
                    assert sample_token_to_seg_dict[tkn] == key
                else:
                    sample_token_to_seg_dict[tkn] = key
            all_segments.add(key)

        self.seg_eval_data_dict = seg_eval_data_dict
        self.sample_token_to_seg_dict = sample_token_to_seg_dict
        self.sample_token_to_egovel = sample_token_to_egovel
        self.all_segments = list(all_segments)
        self.all_deadlines = list(all_deadlines)
        self.global_worst_dl = 0.

        if len(self.dataset_dict) > 0:
            to_del = [k for k,v in self.dataset_dict.items() if len(v) != len(self.all_deadlines)]
            for k in to_del:
                del self.dataset_dict[k]
            print('Dataset dict len after pruning:', len(self.dataset_dict))

    def calc_max_mAP(self, lims):
        if lims[0] == 0:
            progress_bar = tqdm(total=lims[1]-lims[0], leave=True, dynamic_ncols=True)
        else:
            progress_bar = None

        max_mAP =.0
        for it, perm in enumerate(product(self.all_deadlines, repeat=len(self.all_segments))):
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

    def do_eval(self, deadline_ms=None, upper_bound_calc_method='heuristic'):
        #if deadline is none, it will pick the deadline that gives the best result

        if deadline_ms is None: # calculate upper bound
            max_mAP = 0.
            best_seg_eval_dict = None

            if upper_bound_calc_method == 'exhaustive':
                num_iters = len(self.all_deadlines)**len(self.all_segments)
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
                # for each segment, try all deadlines and find the best deadline
                # that gives the most boost to the mAP.
                num_iters = len(self.all_deadlines)*len(self.all_segments)
                progress_bar = tqdm(total=num_iters, leave=True, dynamic_ncols=True)
                init_dl = self.global_worst_dl if self.global_worst_dl != 0. else self.all_deadlines[0]
                cur_seg_dls = {seg:init_dl for seg in self.all_segments}
                print('Heuristic started by initializing all deadlines to', init_dl)

                with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                    for i, seg in enumerate(self.all_segments):
                        for new_dl in self.all_deadlines:
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
            elif upper_bound_calc_method == 'precision_based':
                num_iters = len(self.all_segments)
                progress_bar = tqdm(total=num_iters, leave=True, dynamic_ncols=True)
                seg_dls, best_seg_eval_dict, best_perm = {}, {}, []
                for j, seg in enumerate(self.all_segments):
                    datas_all = []
                    keys_all = []
                    for key, datas in self.seg_eval_data_dict.items():
                        seg_i = '='.join(key.split('=')[:2])
                        if seg == seg_i:
                            datas_all.extend(datas)
                            keys_all.append(key)
                    #determine the best deadline for this segment
                    deadlines, cnt = np.unique([d[0] for d in datas_all], return_counts=True)
                    cnt = -(cnt - np.max(cnt)) # num zeros to add
                    best_dl, max_prec = datas_all[0][0], 0.
                    for i, dl in enumerate(deadlines):
                        dl_datas = [d for d in datas_all if d[0] == dl]

                        class_agnostic = False
                        if class_agnostic:
                            merged_datas = (
                                dl_datas[0],
                                np.concatenate([d[1] for d in dl_datas]),
                                np.concatenate([d[2] for d in dl_datas]),
                                np.sum([d[3] for d in dl_datas]).item()
                            )
                            mean_prec, mean_rec = self.get_seg_prec_recall(merged_datas)
                        else:
                            precs_and_recs  = [self.get_seg_prec_recall(d) for d in dl_datas]
                            precs_and_recs += [0., 0.] * cnt[i]
                            precs_and_recs = np.array(precs_and_recs)
                            mean_prec = np.mean(precs_and_recs[:, 0])
                            mean_rec = np.mean(precs_and_recs[:, 1])
                            #mean_prec += mean_rec

                        if mean_prec > max_prec:
                            best_dl = dl
                            max_prec = mean_prec

                    for key in keys_all:
                        datas = get_dl_data(self.seg_eval_data_dict[key], best_dl)
                        best_seg_eval_dict[key] = datas
                    best_perm.append(best_dl)

                    progress_bar.update()
                max_mAP = self.calc_mAP(best_seg_eval_dict, False)
            else:
                print('Unkown upper bound calculation method', upper_bound_calc_method)
                return

            progress_bar.close()
            print('Deadlines of each segment:')
            print(best_perm)
            print('Upper bound mAP:', max_mAP)
            print('Deadline stats:')
            deadlines, occurances = np.unique(np.array(best_perm), return_counts=True)
            print(deadlines)
            print(occurances)
            print(np.round(occurances / np.sum(occurances), 2))

            if len(self.dataset_dict) > 0 and upper_bound_calc_method == 'heuristic':
                dataset_tuples=[]

                for sample_tkn, inout_list in self.dataset_dict.items():
                    segkey = self.sample_token_to_seg_dict[sample_tkn]
                    dl = cur_seg_dls[segkey]
                    idx = [l[-1] for l in inout_list].index(dl)
                    dataset_tuples.append(inout_list[idx] + \
                            [self.sample_token_to_egovel[sample_tkn]])

                print('Duplicates are not removed')
                # dump the tuples
                print(f'Dumping {len(dataset_tuples)} samples as dataset')
                with open('deadline_dataset.pkl', 'wb') as f:
                    pickle.dump({
                        'fields': ('coords', 'features', 'deadline', 'egovel_xy'),
                        'data':dataset_tuples}, f)
            return max_mAP
        else:
            seg_eval_dict = {}
            for key, datas in self.seg_eval_data_dict.items():
                seg_eval_dict[key] = get_dl_data(datas, deadline_ms)
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
            deadline_ms, tp_arr, scr_arr, num_gt_seg = data
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
        deadline_ms, tp_arr, scr_arr, num_gt_seg = eval_data_tpl

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
            key += str(tpl[self.deadline_ms_idx]) + '='
        return key[:-1]

    def get_keyidx(self, tpl_idx):
        t = self.seg_info_tuples[tpl_idx]

        scene = t[self.scene_idx]
        if scene not in self.scene_to_idx:
            self.scene_to_idx[scene] = self.scene_to_idx_counter
            self.scene_to_idx_counter += 1

        num_time_seg_in_scene = 20
        return self.scene_to_idx[scene] * num_time_seg_in_scene + t[fi['time_segment']]

    def do_eval_all_deadlines(self):
        #VALO
        mAPs = []
        global_worst_mAP = 1.
        deadlines_sorted = sorted(seg_info.all_deadlines)
        for dl in deadlines_sorted:
            print('Deadline', dl)
            mAPs.append(seg_info.do_eval(dl))
            if mAPs[-1] < global_worst_mAP:
                self.global_worst_dl = dl
                global_worst_mAP = mAPs[-1]

        print('Deadline(ms) mAP')
        for mAP, dl in zip(mAPs, deadlines_sorted):
            print(f"{dl}\t{mAP}")

if __name__ == '__main__':  
    inp_dir = sys.argv[1]
    seg_info = SegInfo(inp_dir)

    seg_info.do_eval_all_deadlines()
    seg_info.do_eval(None, 'heuristic')
    #seg_info.do_eval(None, 'precision_based')
    #seg_info.do_eval(None, 'exhaustive')
