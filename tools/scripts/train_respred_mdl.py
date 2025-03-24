#!/usr/bin/python3
import glob
import pickle
import json
import numpy as np
from alive_progress import alive_bar
import concurrent.futures
import sys
import os
import time
import copy
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
#from visual_utils.bev_visualizer import visualize_bev_detections

#NUMRES = 3
INP_DATA_RES=2
num_train_scenes = 75
num_test_scenes = 75
gen_dataset=True
calc_detscores=True
#NUM_BINS=10

NUM_CLASSES = 10
TIME_SLICE_PER_SCENE = 10 # ~2 seconds
DISTANCE_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]
class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
class_range = {
    "car": 50,
    "truck": 50,
    "bus": 50,
    "trailer": 50,
    "construction_vehicle": 50,
    "pedestrian": 40,
    "motorcycle": 40,
    "bicycle": 40,
    "traffic_cone": 30,
    "barrier": 30
}

class_id_range = [0] + [class_range[name] for name in class_names]

def parse_detresult(result_str):
    result_str = result_str.split('\n')

    cls_AP_scores = []
    for i, line in enumerate(result_str):
        if line[:3] == "***":
            cls = line.split(' ')[0][3:]
            mean_cls_AP = float(result_str[i+1].split(' ')[-1])
            cls_AP_scores.append(mean_cls_AP)

    mAP = float(result_str[-3:][0].split(' ')[-1])
    NDS = float(result_str[-3:][1].split(' ')[-1])
    return mAP, NDS, cls_AP_scores

def calc_AP(boxes_l, scores_l, gt_boxes_l, dist_th, num_all_dets, num_all_gt):
    gt_boxes_l_cpy = copy.deepcopy(gt_boxes_l)

    merged_scores = np.empty(num_all_dets, dtype=float)
    merged_frame_ids = np.empty(num_all_dets, dtype=int)
    merged_boxes = np.empty((num_all_dets, 2), dtype=float)

    i = 0
    for frame_id, scores in enumerate(scores_l):
        endi = len(scores) + i
        merged_scores[i:endi] = scores
        merged_frame_ids[i:endi] = frame_id
        merged_boxes[i:endi] = boxes_l[frame_id][:, :2]
        i = endi

    scores_sort_inds = np.argsort(merged_scores)[::-1] # reverse it
    tp = np.zeros(num_all_dets, dtype=bool)
    for i, score_i in enumerate(scores_sort_inds):
        frame_id = merged_frame_ids[score_i]
        gt_boxes = gt_boxes_l_cpy[frame_id]
        if gt_boxes.shape[0] == 0:
            continue

        score = merged_scores[score_i]
        box = merged_boxes[score_i]
        box_xy = np.expand_dims(box, 0)

        dists = distance.cdist(gt_boxes[:, :2], box_xy, 'euclidean').flatten()
        min_idx = np.argmin(dists)

        if dists[min_idx] < dist_th:
            tp[i] = True
            gt_boxes[min_idx, :2] = 9999. # make sure it won't match again

    simple_precision = np.sum(tp) / len(tp)

    fp = np.logical_not(tp)
    tp = np.cumsum(tp.astype(float))
    fp = np.cumsum(fp.astype(float))

    prec = tp / (fp + tp)
    rec = tp / float(num_all_gt)
    nelem = 101
    rec_interp = np.linspace(0, 1, nelem)
    precision = np.interp(rec_interp, rec, prec, right=0)
    #conf = np.interp(rec_interp, rec, scr, right=0)
    rec = rec_interp

    min_recall =  0.1
    min_precision = 0.1

    prec = np.copy(precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision), simple_precision

def calc_detscore(eval_data_arr):
    detscore = []
    for class_idx, eval_data_dict in enumerate(eval_data_arr): #for each class
        boxes_l, scores_l, gt_boxes_l = eval_data_dict['boxes'], \
                eval_data_dict['scores'], eval_data_dict['gt_boxes']

        num_gt_per_frame = np.array([gtb.shape[0] for gtb in gt_boxes_l])
        num_gt = np.sum(num_gt_per_frame)
        if num_gt == 0:
            continue # skip this class

        num_all_dets = sum([b.shape[0] for b in boxes_l])
        if num_all_dets != 0:
            for dist_th in DISTANCE_THRESHOLDS:
                #Append frame id of each score
                ap_score, simple_precision = calc_AP(boxes_l, scores_l, gt_boxes_l, dist_th,
                                   num_all_dets, num_gt)
                detscore.append(ap_score) #* num_gt

    return sum(detscore) / (len(DISTANCE_THRESHOLDS) * NUM_CLASSES)

def calc_resolution_detscores(sampled_dets, all_tslc_inds, all_gt_boxes, ridx):
    print('Calculating detection scores for resolution', ridx)
    num_timeslices = len(all_tslc_inds)
    tslc_scores = np.empty(num_timeslices)
    for tidx, (si, ei) in enumerate(all_tslc_inds): # for each timeslice
        num_tslc_elems = ei - si
        eval_data_arr = [{
            'boxes': [None] * num_tslc_elems, # list of frames
            'scores': [None] * num_tslc_elems,
            'gt_boxes': [None] * num_tslc_elems,
        } for c in range(NUM_CLASSES)]
        for cls_id in range(1, NUM_CLASSES+1): #1 to 10
            slc_eval_data = eval_data_arr[cls_id-1]
            boxes_l, scores_l, gt_boxes_l = slc_eval_data['boxes'], \
                    slc_eval_data['scores'], slc_eval_data['gt_boxes']
            cls_range = class_id_range[cls_id]
            for i in range(si, ei):
                j = i - si
                gt_labels = all_gt_boxes[i][:, -1].astype(int)
                cls_mask = (gt_labels == cls_id)
                cur_gt_boxes = all_gt_boxes[i][cls_mask, :-1]
                range_mask = (cur_gt_boxes[:, :2] <= cls_range).all(1)
                gt_boxes_l[j] = cur_gt_boxes[range_mask].astype(float)

                pred_dicts = sampled_dets[i]
                if pred_dicts is None:
                    boxes_l[j] = np.zeros((0, 9), dtype=float)
                    scores_l[j] = np.zeros((0,), dtype=float)
                else:
                    # ind is 0 since batch size assumed to be 1
                    pd = pred_dicts[0]
                    labels = pd['pred_labels']
                    boxes = pd['pred_boxes']
                    scores = pd['pred_scores']
                    cls_mask = (labels == cls_id)
                    boxes = boxes[cls_mask]
                    scores = scores[cls_mask]
                    range_mask = (boxes[:, :2] <= cls_range).all(1)
                    boxes_l[j] = boxes[range_mask]
                    scores_l[j] = scores[range_mask]
        tslc_scores[tidx] = calc_detscore(eval_data_arr)
        if tidx % (int(num_timeslices / 10)+1) == 0:
            print(f"Resolution {ridx} detscore calc progress: %{int(tidx/num_timeslices*100)}")

    return tslc_scores


def read_data(pth, typ):
    end = ('1' if typ == 'train' else '0') + '.pkl'
    gt_database_path = glob.glob(pth + "/*gt_database*" + end)
    print('Loading', gt_database_path[0])
    with open(gt_database_path[0], 'rb') as f:
        gt_tuples = pickle.load(f) # list of (sample_token, gt_boxes, time_preds)

    sample_tokens, all_gt_boxes, time_preds = [], [], []
    for st, gtb, tp in gt_tuples:
        sample_tokens.append(st)
        all_gt_boxes.append(gtb)
        time_preds.append(tp)

    eval_dict_paths = glob.glob(pth + "/*calib*" + end)
    num_res = len(eval_dict_paths)
    eval_dicts = [None] * num_res
    for evald_pth in eval_dict_paths:
        print('Loading', evald_pth)
        with open(evald_pth, 'rb') as f:
            eval_d = pickle.load(f)
        eval_dicts[eval_d['resolution']] = eval_d
        mAP, NDS, class_AP_scrs = parse_detresult(eval_d['result_str'])
        print(f'Resolution {eval_d["resolution"]} mAP: {mAP} NDS: {NDS}')

    print('Calculating time slice indices')
    eval_d = eval_dicts[0] # any of them works to get timeslice
    all_time_slice_inds = []
    scene_begin_inds = eval_d['scene_begin_inds']
    sampled_dets_0 = eval_d['objects']
    begin_inds = np.array([0] + scene_begin_inds)
    end_inds = np.array(scene_begin_inds + [len(sampled_dets_0)])
    num_samples_in_scene = end_inds - begin_inds
    for num_smpl, bi, ei in zip(num_samples_in_scene, begin_inds, end_inds):
        slice_inds = np.linspace(bi, ei, TIME_SLICE_PER_SCENE, dtype=int)
        for i in range(len(slice_inds)-1):
            all_time_slice_inds.append(slice_inds[i:i+2])
    all_tslc_inds = np.array(all_time_slice_inds)
    num_slices = len(all_tslc_inds)

    # I am converting the objs here cuz pickling torch tensors is troublesome
    for eval_d in eval_dicts:
        for pred_dicts in eval_d['objects']:
            if pred_dicts is not None:
                pd = pred_dicts[0] # batch size assumes to be 1
                boxes = pd['pred_boxes'] 
                if not isinstance(boxes, np.ndarray):
                    scores, labels = pd['pred_scores'], pd['pred_labels']
                    pd['pred_boxes'] = boxes.numpy().astype(float)
                    pd['pred_scores'] = scores.numpy().astype(float)
                    pd['pred_labels'] = labels.numpy().astype(int)

    # For each resolution, calculate detection score of each time slice
    detscore_path = os.path.join(pth, 'detscore.json')
    if not calc_detscores:
        with open(detscore_path, 'r') as f:
            all_tslc_scores = json.load(f)
    else:
        all_tslc_scores = np.zeros((len(all_tslc_inds), num_res))
        dataset_mAPs = np.zeros(num_res)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_res * 2) as executor:
            futs = []
            noslc_futs = []
            for eval_d in eval_dicts:
                ridx = eval_d['resolution']
                fut = executor.submit(calc_resolution_detscores, eval_d['objects'], \
                        all_tslc_inds, all_gt_boxes, ridx)
                futs.append((ridx, fut))

                noslc_fut = executor.submit(calc_resolution_detscores, eval_d['objects'], \
                        [[0, len(all_gt_boxes)]], all_gt_boxes, ridx)
                noslc_futs.append((ridx, noslc_fut))

            for ridx, fut in futs:
                detscores = fut.result()
                all_tslc_scores[:, ridx] = detscores

            for ridx, fut in noslc_futs:
                detscores = fut.result()
                dataset_mAPs[ridx] = detscores[0]

        print('Manually calculated dataset mAPs:')
        print(dataset_mAPs)

        with open(detscore_path, 'w') as f:
            json.dump(all_tslc_scores.tolist(), f, indent=4)

    # now work on what would be the input of the prediction model
    # use global best data
    eval_d = eval_dicts[INP_DATA_RES]
    egovels = eval_d['egovels']
    exec_times_ms = eval_d['exec_times_ms'] # use for it?
    sampled_dets = eval_d['objects']
    num_dets = len(sampled_dets)
    mask = np.zeros(num_dets, dtype=bool)
    egovel_inds = [i for i, ev in enumerate(egovels) if ev is not None]
    data_tuples = []
    for tidx, (si, ei) in enumerate(all_tslc_inds): # for each timeslice
        scene_egovel_inds = [i for i in egovel_inds if i >= si and i < ei]
        for idx in range(1, len(scene_egovel_inds)-1):
            time_tpl = exec_times_ms[scene_egovel_inds[idx]]
            if time_tpl is None:
                print(idx, scene_egovel_inds[idx], 'No time tuple.')
                continue
            #etime, sim_time_ms = time_tpl
            #if sim_time_ms < 800:
            #    continue
            ev = egovels[scene_egovel_inds[idx]]
            if np.isnan(ev).any():
                print(idx, scene_egovel_inds[idx], 'Egovel is bad.')
                continue
            pred_dict = sampled_dets[scene_egovel_inds[idx+1]] #[0]
            if pred_dict is None:
                print(idx, scene_egovel_inds[idx], 'Pred dict is none.')
                continue
            pred_dict = pred_dict[0]
            boxes = pred_dict['pred_boxes']
            if boxes.shape[0] == 0:
                print(idx, scene_egovel_inds[idx], 'No boxes.')
                continue

            # Velocity helps a lot!
            obj_velos = boxes[:, 7:9]
            obj_velos[np.isnan(obj_velos).any(1)] = 0. #Assume these to be stationary
            rel_velos = obj_velos - ev # using ev makes it worse?
            bj_velos = np.linalg.norm(obj_velos, axis=1)
            rel_velos = np.linalg.norm(rel_velos, axis=1)
            relvel_mean = np.mean(rel_velos)
            relvel_perc5, relvel_perc95 = np.percentile(rel_velos, (5, 95))

            #NOTE this part did not improve either
            prev_pred_dicts = sampled_dets[scene_egovel_inds[idx]]
            if prev_pred_dicts is None:
                print(idx, scene_egovel_inds[idx], 'No prev pred dict.')
                continue
            prev_pd = prev_pred_dicts[0]
            prev_boxes = prev_pd['pred_boxes']
            if prev_boxes.shape[0] == 0:
                print(idx, scene_egovel_inds[idx], 'No prev boxes.')
                continue
#            # assume all objects to be of same class
            #precs = np.empty(4)
            #prev_box_locs = prev_boxes[:, :2] + np.repeat(prev_pd['pred_labels'] * 100, 2).reshape(-1, 2)
            #box_locs = boxes[:, :2] + np.repeat(pred_dict['pred_labels'] * 100, 2).reshape(-1, 2)
            #boxes_l, scores_l, gt_boxes_l = [prev_box_locs], [prev_pd['pred_scores']], [box_locs]
            #num_all_dets, num_all_gt = len(prev_boxes), len(boxes)
            #for j, dist_th in enumerate(DISTANCE_THRESHOLDS):
            #   _, precision = calc_AP(boxes_l, scores_l, gt_boxes_l, dist_th, num_all_dets, num_all_gt)
            #   precs[j] = precision * (1 / dist_th)
            #mean_prec = precs.mean()

            # This helps!
            objpos = boxes[:, :2]
            objpos = np.linalg.norm(objpos, axis=1)
            objpos_mean = np.mean(objpos)
            objpos_perc5, objpos_perc95 = np.percentile(objpos, (5, 95))

            #labels = pred_dict['pred_labels']
            #num_obj = labels.shape[0]
            #label_dist = np.bincount(labels-1, minlength=NUM_CLASSES) / labels.shape[0]

            data_tuples.append((objpos_perc5, objpos_mean, objpos_perc95, #))
                np.linalg.norm(ev),
                relvel_perc5, relvel_perc95, relvel_mean,
            #    mean_prec,
            ))
            mask[scene_egovel_inds[idx]] = True

    data_tuples = np.array(data_tuples)
    tpreds = np.array(time_preds)[mask]
    X = np.concatenate((tpreds, data_tuples), axis=1)

    # num time_preds is as same as num sampled_dets
    y = np.empty((len(time_preds), num_res))
    oracle_dict = {}
    for tidx, (si, ei) in enumerate(all_tslc_inds): # for each timeslice
        scores = all_tslc_scores[tidx]
        for st in sample_tokens[si:ei]:
            oracle_dict[st] = np.argmax(scores).item()
        num_elem = ei - si
        scores = np.tile(scores, num_elem).reshape(-1, len(scores))
        y[si:ei] = scores
    masked_y = y[mask]

    return X, masked_y, oracle_dict

def get_bad_data_mask(X, y):
    mask = np.logical_or(np.isnan(X).any(1), np.isnan(y).any(1))
    mask = np.logical_or(mask, (y == 0).all(1)) # if all scores are 0, skip
    return np.logical_not(mask)

if __name__ == '__main__':
    inp_dir = sys.argv[1]
    if gen_dataset:
        X_train, y_train, oracle_train = read_data(inp_dir, 'train')
        X_test, y_test, oracle_test = read_data(inp_dir, 'test')
        print('Dumping generated dataset and oracle')
        with open(os.path.join(inp_dir, 'generated_dataset.pkl'), 'wb') as f:
            pickle.dump((X_train, y_train, X_test, y_test), f)
        with open('oracle_test.json', 'w') as f:
            json.dump(oracle_test, f, indent=4)
    else:
        print('Loading generated dataset')
        with open(os.path.join(inp_dir, 'generated_dataset.pkl'), 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)

    #Filter nans and zeros
    mask = get_bad_data_mask(X_train, y_train)
    X_train, y_train = X_train[mask], y_train[mask]
    mask = get_bad_data_mask(X_test, y_test)
    X_test, y_test = X_test[mask], y_test[mask]

    y_train_labels = np.argmax(y_train, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    #train_max_scores = np.take_along_axis(y_train, np.expand_dims(y_train_labels, -1), axis=1)
    #train_max_score = np.sum(train_max_scores)

    test_max_scores = np.take_along_axis(y_test, np.expand_dims(y_test_labels, -1), axis=1)
    test_max_score = np.sum(test_max_scores)

    num_res = y_test.shape[1]
    fixed_res_scores = np.empty(num_res)
    for ridx in range(num_res):
        fixed_res_scores[ridx] = np.sum(y_test[:, ridx]) / test_max_score * 100

    print('***** Fixed resolution scores:', np.round(fixed_res_scores, 2))

    print('Train data shapes and labels distribution:')
    print(X_train.shape, y_train.shape, np.bincount(y_train_labels))
    print('Test data shapes and labels distribution:')
    print(X_test.shape, y_test.shape, np.bincount(y_test_labels))

    # Create and train the Random Forest classifier
    use_grid_search = False
    if use_grid_search:
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            'n_estimators': [64, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=40),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train_labels)
        print("Best parameters:", grid_search.best_params_)
        rf_classifier = grid_search
    else:
        rf_classifier = RandomForestClassifier(
            n_estimators=100,  # number of trees
            max_depth=10,    # maximum depth of trees
            max_features='sqrt',
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=40
        )

    # Train the model
    rf_classifier.fit(X_train, y_train_labels)
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_classifier, f)

    # Make predictions on test set
    y_pred = rf_classifier.predict(X_test)
    pred_scores = np.take_along_axis(y_test, np.expand_dims(y_pred, -1), axis=1)
    pred_score = np.sum(pred_scores)
    print('***** Predictor score:', round(pred_score / test_max_score * 100, 2))

    #speed test
    t1 = time.monotonic()
    for i in range(50):
        rf_classifier.predict(X_test[i:i+1])
    t2 = time.monotonic()
    tdiff = (t2 - t1) * 1000 / 50
    print('Inference time:', tdiff, 'ms')

    # Calculate accuracy
    accuracy = accuracy_score(y_test_labels, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_labels, y_pred))

    #from sklearn.model_selection import cross_val_score
    #
    ## Perform 5-fold cross-validation
    #cv_scores = cross_val_score(rf_classifier, X, y, cv=5)
    #print("\nCross-validation scores:", cv_scores)
    #print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    feature_importance = pd.DataFrame({
        'feature': range(X_train.shape[1]),
        'importance': rf_classifier.feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))
