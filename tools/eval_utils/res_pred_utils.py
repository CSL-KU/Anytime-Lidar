from pyquaternion import Quaternion
import numpy as np
import torch

def get_2d_egovel(prev_ts, prev_egopose, cur_ts, cur_egopose):
    tdiff_sec = (cur_ts - prev_ts) * 1e-6 # musec to sec

    cur_transl = cur_egopose[7:10]
    prev_transl = prev_egopose[7:10]

    egovel = (cur_transl - prev_transl) / tdiff_sec

    cur_rot = Quaternion(cur_egopose[10:14].numpy())
    egovel = cur_rot.inverse.rotate(egovel.numpy())

    return egovel[[1,0]] # return x y vel

# NOTE This is noisy!
def get_egopose_and_egovel(nusc, sample_tkn, norm=False, global_coords=False):
    sample = nusc.get('sample', sample_tkn)
    sd_tkn = sample['data']['LIDAR_TOP']
    sample_data = nusc.get('sample_data', sd_tkn)
    ep = nusc.get('ego_pose', sample_data['ego_pose_token'])
    # timestamps are in microseconds
    ts = sample_data['timestamp']
    if sample_data['prev'] == '':
        #No prev data, calc speed w.r.t next
        next_sample_data = nusc.get('sample_data', sample_data['next'])
        next_ep = nusc.get('ego_pose', next_sample_data['ego_pose_token'])
        next_ts = next_sample_data['timestamp']
        trnsl = np.array(ep['translation'])
        next_trnsl = np.array(next_ep['translation'])
        egovel = (next_trnsl - trnsl) / ((next_ts - ts) * 1e-6)
    else:
        prev_sample_data = nusc.get('sample_data', sample_data['prev'])
        prev_ep = nusc.get('ego_pose', prev_sample_data['ego_pose_token'])
        prev_ts = prev_sample_data['timestamp']
        trnsl = np.array(ep['translation'])
        prev_trnsl = np.array(prev_ep['translation'])
        egovel = (trnsl - prev_trnsl) / ((ts - prev_ts) * 1e-6)

    if not global_coords:
        rotation = Quaternion(ep['rotation'])
        # Convert the global velocity to ego frame
        egovel = rotation.inverse.rotate(egovel)
    
    if norm:
        egovel = np.linalg.norm(egovel)

    return ep, egovel[[1,0,2]]

def get_smooth_egovel(nusc, sample_tkn, target_time_diff_ms=250, global_coords=False):
    sample = nusc.get('sample', sample_tkn)
    sd_tkn = sample['data']['LIDAR_TOP']
    sample_data = nusc.get('sample_data', sd_tkn)
    ts = sample_data['timestamp'] # microseconds

    past_sample_data = sample_data
    past_ts = ts
    while past_sample_data['prev'] != '' and (ts-past_ts) < target_time_diff_ms*1000:
        past_sample_data = nusc.get('sample_data', past_sample_data['prev'])
        past_ts = past_sample_data['timestamp']

    ep = nusc.get('ego_pose', sample_data['ego_pose_token'])
    past_ep = nusc.get('ego_pose', past_sample_data['ego_pose_token'])
    trnsl = np.array(ep['translation'])
    past_trnsl = np.array(past_ep['translation'])
    egovel = (trnsl - past_trnsl) / ((ts - past_ts) * 1e-6)

    if not global_coords:
        rotation = Quaternion(ep['rotation'])
        # Convert the global velocity to ego frame
        egovel = rotation.inverse.rotate(egovel)

    return ep, egovel

def calc_falsepos_when_shifted(time_diff_sec, coords, rel_velos, labels,
                               dist_thresholds=[0.5, 1.0, 2.0, 4.0],
                               scores=None,
                               class_ids=None):
    if scores is None:
        scores = np.ones(len(coords))

    if class_ids is None:
        class_ids = np.unique(labels)

    # assert not np.isnan(rel_velos).any()
    # assert not np.isnan(coords).any()

    false_pos = 0
    future_coords = coords + rel_velos * time_diff_sec
    for cls_id in class_ids:
        c_mask = (labels == cls_id)
        c_scores = scores[c_mask]
        c_coords = coords[c_mask]
        c_fut_coords = future_coords[c_mask]

        scr_inds = np.argsort(-c_scores) # descending
        for dist_th in dist_thresholds:
            c_fut_coords_ = c_fut_coords.copy()
            for idx in scr_inds:
                pred_coord = c_coords[idx]
                dist_diffs = np.linalg.norm(c_fut_coords_ - pred_coord, axis=1)
                mindx = np.argmin(dist_diffs)
                if dist_diffs[mindx] <= dist_th:
                    c_fut_coords_[mindx] = 9999. # can't be matched to anything now
                else:
                    false_pos += 1
    return false_pos

# res_exec_times should be sorted descending
# rightmost correspond to lowest resolution
def pick_best_resolution(res_exec_times_sec, egovel, pred_dict, score_thr=.5):
    dist_thresholds = [0.5, 1.0, 2.0, 4.0]

    scores = pred_dict['pred_scores'].numpy()
    bboxes = pred_dict['pred_boxes'].numpy()
    labels = pred_dict['pred_labels'].numpy()

    scores_mask = scores > score_thr
    scores = scores[scores_mask]
    bboxes = bboxes[scores_mask]
    labels = labels[scores_mask]

    coords = bboxes[:, :2]
    velos = bboxes[:, 7:9]
    rel_velos = velos - egovel
    
    class_ids = np.unique(labels)

    false_pos = np.empty(len(res_exec_times_sec))
    for et_idx, et in enumerate(res_exec_times_sec):
        false_pos[et_idx] = calc_falsepos_when_shifted(et, coords, rel_velos, labels,
                                                       dist_thresholds, scores, class_ids)

    # increase the resolution if it is not going to yield more false positives
    chosen_res = len(res_exec_times_sec)-1
    while chosen_res > 0 and false_pos[chosen_res-1] <= false_pos[chosen_res]:
        chosen_res -= 1

    return chosen_res



@torch.jit.script
def get_tp_fp_gt(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_class : int):
    inds = torch.argsort(pred_scores, descending=True)
    pred_labels = pred_labels[inds]
    pred_xyz = pred_boxes[inds, :3]
    gt_xyz = gt_boxes[:, :3]

    distance_thresholds = [0.5, 1.0, 2.0, 4.0]
    tp_fp_gt = torch.zeros((num_class, len(distance_thresholds), 3))
    for cls in range(1, num_class+1):
        pred_mask = (pred_labels == cls)
        pred_xyz_l = pred_xyz[pred_mask]
        gt_xyz_l = gt_xyz[gt_labels == cls]

        cls_i = cls-1
        tp_fp_gt[cls_i, :, 2] = len(gt_xyz_l)
        if len(pred_xyz_l) == 0:
            continue

        if len(gt_xyz_l) == 0:
            tp_fp_gt[cls_i, :, 1] += len(pred_xyz_l)
            continue

        dists = torch.cdist(pred_xyz_l, gt_xyz_l)
        for d, dist_th in enumerate(distance_thresholds):
            for row in dists:
                min_dist, gt_idx = torch.min(row, dim=0)
                if (min_dist <= dist_th):
                    tp_fp_gt[cls_i, d, 0] += 1
                    dists[:, gt_idx] = 5.0 # can't match with anything now
                else:
                    tp_fp_gt[cls_i, d, 1] += 1
    return tp_fp_gt