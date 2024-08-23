import torch
from ...ops.iou3d_nms import iou3d_nms_utils
from ...ops.ioubev_nms import ioubev_nms_utils
from typing import Optional, List

def class_agnostic_nms(box_scores : torch.Tensor, box_preds : torch.Tensor, nms_type : str , nms_thresh : float,
        nms_post_maxsize : int, nms_pre_maxsize : int, score_thresh : Optional[float] = None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]
    else:
        scores_mask = torch.ones(box_scores.size(0), dtype=torch.long, device=box_scores.device)

    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_pre_maxsize, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        if nms_type == 'nms_gpu':
            keep_idx = iou3d_nms_utils.nms_gpu(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_thresh, nms_pre_maxsize
            )
        else: #nms_type == 'nms_normal_gpu':
            keep_idx = iou3d_nms_utils.nms_normal_gpu(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_thresh
            )

        selected = indices[keep_idx[:nms_post_maxsize]]
    else:
        selected = torch.zeros(0, dtype=torch.long, device=box_scores.device)

    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]

def multi_classes_nms(cls_scores : torch.Tensor, box_preds : torch.Tensor, nms_type : str , nms_thresh : float,
        nms_post_maxsize : int, nms_pre_maxsize : int, score_thresh : Optional[float] = None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]
            cur_box_preds = box_preds

        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_pre_maxsize, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            if nms_type == 'nms_gpu':
                keep_idx = torch.ops.iou3d_nms_utils.nms_gpu(
                        boxes_for_nms[:, 0:7], box_scores_nms, nms_thresh, nms_pre_maxsize
                )
            else: #nms_type == 'nms_normal_gpu':
                keep_idx = iou3d_nms_utils.nms_normal_gpu(
                        boxes_for_nms[:, 0:7], box_scores_nms, nms_thresh
                )


            selected = indices[keep_idx[:nms_post_maxsize]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(selected.size(0)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes

def multi_classes_nms_mmdet(box_scores : torch.Tensor, box_preds : torch.Tensor, box_labels : torch.Tensor,
        nms_thresh_l : List[float], nms_post_maxsize_l : List[int], nms_pre_maxsize_l : List[int],
        score_thresh : Optional[float]):
    """
    Args:
        cls_scores: (N,)
        box_preds: (N, 7 + C)
        box_labels: (N,)

    """
    selected : List[torch.Tensor] = []
    for k in range(len(nms_thresh_l)):
        curr_mask = box_labels == k
        if score_thresh is not None: # and isinstance(score_thresh, float):
            curr_mask *= (box_scores > score_thresh)
        #if score_thresh_l is not None:  #and isinstance(score_thresh, list):
        #    curr_mask *= (box_scores > score_thresh_l[k])
        curr_idx = torch.nonzero(curr_mask)[:, 0]
        curr_box_scores = box_scores[curr_mask]
        cur_box_preds = box_preds[curr_mask]

        curr_box_preds_bev = cur_box_preds[:, [0,1,3,4,6]]
        # xywhr2xyxyr
        curr_boxes_bev_for_nms = torch.zeros_like(curr_box_preds_bev)
        half_w = curr_box_preds_bev[:, 2] / 2
        half_h = curr_box_preds_bev[:, 3] / 2
        curr_boxes_bev_for_nms[:, 0] = curr_box_preds_bev[:, 0] - half_w
        curr_boxes_bev_for_nms[:, 1] = curr_box_preds_bev[:, 1] - half_h
        curr_boxes_bev_for_nms[:, 2] = curr_box_preds_bev[:, 0] + half_w
        curr_boxes_bev_for_nms[:, 3] = curr_box_preds_bev[:, 1] + half_h
        curr_boxes_bev_for_nms[:, 4] = curr_box_preds_bev[:, 4]

        if curr_box_scores.shape[0] > 0:
            # box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            curr_box_scores_nms = curr_box_scores
            curr_boxes_for_nms = curr_boxes_bev_for_nms
            # transformer to old mmdet3d coordinate
            """
            .. code-block:: none

                                up z    x front (yaw=-0.5*pi)
                                ^   ^
                                |  /
                                | /
            (yaw=-pi) left y <------ 0 -------- (yaw=0)
            """
            pi = 3.141592
            curr_boxes_bev_for_nms[:, 4] = (-curr_boxes_bev_for_nms[:, 4] + pi / 2 * 1)
            curr_boxes_bev_for_nms[:, 4] = (curr_boxes_bev_for_nms[:, 4] + pi) % (2*pi) - pi

            keep_idx = ioubev_nms_utils.nms_gpu_bev(
                curr_boxes_for_nms[:, 0:5],
                curr_box_scores_nms,
                nms_thresh_l[k],
                nms_pre_maxsize_l[k],
                nms_post_maxsize_l[k]
            )
            curr_selected = curr_idx[keep_idx]
            selected.append(curr_selected)

    sel_all = torch.cat(selected)
    return sel_all, box_scores[sel_all]
