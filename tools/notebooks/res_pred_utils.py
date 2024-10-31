from pyquaternion import Quaternion
import numpy as np

def get_2d_egovel(prev_ts, prev_egopose, cur_ts, cur_egopose):
    tdiff_sec = (cur_ts - prev_ts) / 1000000. # musec to sec

    cur_transl = cur_egopose[7:10]
    prev_transl = prev_egopose[7:10]

    egovel = (cur_transl - prev_transl) / tdiff_sec

    cur_rot = Quaternion(cur_egopose[10:14].numpy())
    egovel = cur_rot.inverse.rotate(egovel.numpy())

    return egovel[[1,0]] # return x y vel

def get_egopose_and_egovel(nusc, sample_tkn, norm=False):
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
        egovel = (next_trnsl - trnsl) / ((next_ts - ts) / 1000000.)
    else:
        prev_sample_data = nusc.get('sample_data', sample_data['prev'])
        prev_ep = nusc.get('ego_pose', prev_sample_data['ego_pose_token'])
        prev_ts = prev_sample_data['timestamp']
        trnsl = np.array(ep['translation'])
        prev_trnsl = np.array(prev_ep['translation'])
        egovel = (trnsl - prev_trnsl) / ((ts - prev_ts) / 1000000.)

    rotation = Quaternion(ep['rotation'])

    # Convert the global velocity to ego frame
    egovel = rotation.inverse.rotate(egovel)
    
    if norm:
        egovel = np.linalg.norm(egovel)

    return ep, egovel[[1,0,2]]

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