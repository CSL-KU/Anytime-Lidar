import torch
from torch.onnx import register_custom_op_symbolic

from ...ops.iou3d_nms.iou3d_nms_cuda import boxes_iou_bev_cpu
from typing import Dict, Final, List, Tuple

import pcdet.ops.utils as pcdet_utils
pcdet_utils.load_torch_op_shr_lib("pcdet/ops/forecasting")

def forecast_past_dets(g, pred_boxes, past_pose_indexes, past_poses, cur_pose, \
        past_timestamps, target_timestamp):
    return g.op("kucsl::forecast_past_dets", pred_boxes, past_pose_indexes, past_poses, cur_pose, \
            past_timestamps, target_timestamp)
register_custom_op_symbolic("kucsl::forecast_past_dets", forecast_past_dets, 17)


def move_to_world_coords(pred_boxes, poses, pose_idx):
    return torch.ops.kucsl.move_to_world_coords(pred_boxes, poses, pose_idx)

# Needed for multihead detection heads
@torch.jit.script
def split_dets(cls_id_to_det_head_idx_map: torch.Tensor,
               num_det_heads : int,
               pred_boxes : torch.Tensor,
               pred_scores : torch.Tensor,
               pred_labels: torch.Tensor,
               move_to_gpu : bool) -> List[Dict[str,torch.Tensor]]:
    det_head_mappings = cls_id_to_det_head_idx_map[pred_labels]

    forc_dicts : List[Dict[str,torch.Tensor]] = []
    pred_merged = torch.cat((pred_boxes, pred_scores.unsqueeze(-1),
            pred_labels.float().unsqueeze(-1)), -1)
    for i in range(num_det_heads):
        pred_masked = pred_merged[det_head_mappings == i]
        pred_masked = pred_masked.cuda() if move_to_gpu else pred_masked
        forc_dicts.append({
                'pred_boxes': pred_masked[:, :-2],
                'pred_scores': pred_masked[:, -2],
                'pred_labels': pred_masked[:, -1].long()
        })

    return forc_dicts

class Forecaster(torch.nn.Module):
    pc_range : Final[Tuple[float]]
    tcount : Final[int]
    score_thresh : Final[float]
    forecasting_coeff : Final[float]
    num_det_heads : Final[int]
    remove_considering_time : Final[bool]

    cls_id_to_det_head_idx_map : torch.Tensor
    num_dets_per_tile : torch.Tensor
    past_poses : torch.Tensor
    past_ts : torch.Tensor
    past_detections : Dict[str,torch.Tensor]

    def __init__(self,
            pc_range : Tuple[float],
            tcount : int,
            score_thresh : float,
            forecasting_coeff : float = 1.0,
            num_det_heads : int = 1,
            cls_id_to_det_head_idx_map : torch.Tensor = torch.tensor([0])):
        super().__init__()

        self.pc_range = pc_range
        self.tcount = tcount
        self.score_thresh = score_thresh
        self.forecasting_coeff = forecasting_coeff
        self.num_det_heads = num_det_heads
        self.cls_id_to_det_head_idx_map = cls_id_to_det_head_idx_map
        self.remove_considering_time = False

        self.reset_vars()

    def reset_vars(self):
        # Poses include [cst(3) csr(4) ept(3) epr(4)]
        self.num_dets_per_tile = torch.zeros([self.tcount], dtype=torch.long)
        self.past_poses = torch.zeros([0, 14], dtype=torch.float)
        self.past_ts = torch.zeros([0], dtype=torch.long)
        # The forecasting queue
        self.past_detections = {
                'pred_boxes': torch.zeros([0, 9], dtype=torch.float),
                'pred_labels': torch.zeros([0], dtype=torch.long),
                'pred_scores': torch.zeros([0], dtype=torch.float),
                'pose_idx': torch.zeros([0], dtype=torch.long)
        }

    # This method adds the latests detections to the queue
    def add_past_forc_to_queue(self, 
            last_pred_dict : Dict[str,torch.Tensor],
            last_ctc : torch.Tensor, # should be long
            last_pose : torch.Tensor,
            last_ts : int,
        ):
        # Before appending the dets, extract the forcected ones
        forc_mask = last_pred_dict['pred_scores'] > self.score_thresh
        for k in ('pred_boxes', 'pred_labels', 'pred_scores'):
            last_pred_dict[k] = last_pred_dict[k][forc_mask]

        # Now, apply nms and remove duplicates
        if self.remove_considering_time:
            #NOTE seems like not working as expected
            moved_past_pboxes = torch.ops.kucsl.move_to_world_coords(
                    self.past_detections['pred_boxes'],
                    self.past_poses,
                    self.past_detections['pose_idx'])

            moved_cur_pboxes = torch.ops.kucsl.move_to_world_coords(
                    last_pred_dict['pred_boxes'],
                    last_pose.unsqueeze(0),
                    torch.zeros(last_pred_dict['pred_boxes'].size(0), dtype=torch.long))

            nms_mask = torch.ops.kucsl.forecasting_nms(
                    moved_cur_pboxes,
                    last_pred_dict['pred_labels'],
                    moved_past_pboxes,
                    self.past_detections['pred_labels'],
                    0.001)

            for k in ('pred_boxes', 'pred_labels', 'pred_scores', 'pose_idx'):
                self.past_detections[k] = self.past_detections[k][nms_mask]

        new_dets_dict : Dict[str,torch.Tensor] = {}
        score_inds = torch.argsort(last_pred_dict['pred_scores'])
        for k in ('pred_boxes', 'pred_labels', 'pred_scores'):
            new_dets_dict[k] = last_pred_dict[k][score_inds]

        if not self.remove_considering_time:
            # update num dets per tile
            W, W_start = self.pc_range[3] - self.pc_range[0], self.pc_range[0]
            div = W / self.tcount
            tile_inds = torch.div((new_dets_dict['pred_boxes'][:, 0] - W_start), div, \
                    rounding_mode='trunc').short()
            tile_bins = torch.bincount(tile_inds, minlength=self.tcount)
            self.num_dets_per_tile[last_ctc] = tile_bins[last_ctc]

        self.past_poses = torch.cat((self.past_poses, last_pose.unsqueeze(0)))
        self.past_ts = torch.cat((self.past_ts, torch.tensor([last_ts], dtype=torch.long)))
        # Append the pose idx for the detection that will be added
        num_dets = new_dets_dict['pred_boxes'].size(0)
        past_poi = self.past_detections['pose_idx']
        poi = torch.full((num_dets,), self.past_poses.size(0)-1, dtype=past_poi.dtype)
        self.past_detections['pose_idx'] = torch.cat((past_poi, poi))

        for k in ('pred_boxes', 'pred_labels', 'pred_scores'):
            self.past_detections[k] = torch.cat((self.past_detections[k], new_dets_dict[k]))

    @torch.jit.export
    def fork_forward(self, last_pred_dict : Dict[str,torch.Tensor], \
            last_ctc : torch.Tensor, last_pose : torch.Tensor, \
            last_ts : int, cur_pose : torch.Tensor, cur_ts : int, reset : bool) \
            -> torch.jit.Future[List[Dict[str,torch.Tensor]]]:
        return torch.jit.fork(self.forward, last_pred_dict, last_ctc, \
                last_pose, last_ts, cur_pose, cur_ts, reset)

    # Projection
    def forward(self, last_pred_dict : Dict[str,torch.Tensor],
            last_ctc : torch.Tensor, last_pose : torch.Tensor,
            last_ts : int, cur_pose : torch.Tensor, cur_ts : int, reset : bool):

        if reset:
            self.reset_vars()

        self.add_past_forc_to_queue(last_pred_dict, last_ctc, last_pose, last_ts)

        # Remove detections which are no more needed
        if self.remove_considering_time:
            # Remove based on time past time and score
            # time limit = score * 10 * 110ms, assuming score is betw 1 and 0

            past_ts_all = self.past_ts[self.past_detections['pose_idx']]
            tdiff = cur_ts - past_ts_all
            tlims = self.past_detections['pred_scores'] * 10 * 110000
            keep_mask_dets = tdiff < tlims

            for k in ('pose_idx', 'pred_boxes', 'pred_labels', 'pred_scores'):
                self.past_detections[k] = self.past_detections[k][keep_mask_dets]
        else:
            active_num_dets = torch.sum(self.num_dets_per_tile)
            max_num_forc = int(active_num_dets * self.forecasting_coeff)
            if self.past_detections['pred_boxes'].size(0) > max_num_forc:
                # Remove oldest dets
                for k in ('pose_idx', 'pred_boxes', 'pred_labels', 'pred_scores'):
                    self.past_detections[k] = self.past_detections[k][-max_num_forc:]

        # Weed out using the pose_idx of first det
        if self.past_detections['pose_idx'].size(0) > 0:
            pose_idx_0 = self.past_detections['pose_idx'][0]
            self.past_poses = self.past_poses[pose_idx_0:]
            self.past_ts = self.past_ts[pose_idx_0:]
            self.past_detections['pose_idx'] = self.past_detections['pose_idx'] - pose_idx_0

        # Do forecasting in the GPU
        if self.past_detections['pred_boxes'].size(0) == 0:
            return [{'pred_boxes': torch.empty([0, 9], dtype=torch.float),
                'pred_labels': torch.empty([0], dtype=torch.long),
                'pred_scores': torch.empty([0], dtype=torch.float)}]

        forc_dict : Dict[str,torch.Tensor] = {}
        if self.score_thresh == 0:
            forc_dict['pred_scores'] = self.past_detections['pred_scores']
        else:
            forc_dict['pred_scores'] = self.score_thresh - \
                    (self.score_thresh / (self.past_detections['pose_idx'] + 2))
        forc_dict['pred_labels'] = (self.past_detections['pred_labels'] - 1)

        #print('past dets', self.past_detections['pred_boxes'].size(0))
        forc_dict['pred_boxes'] = torch.ops.kucsl.forecast_past_dets(
                self.past_detections['pred_boxes'],
                self.past_detections['pose_idx'],
                self.past_poses,
                cur_pose,
                self.past_ts,
                cur_ts)

        # remove those going out of range
        forcb = forc_dict['pred_boxes']
        box_x, box_y = forcb[:,0], forcb[:,1]
        range_mask = box_x >= self.pc_range[0]
        range_mask = torch.logical_and(range_mask, box_x <= self.pc_range[3])
        range_mask = torch.logical_and(range_mask, box_y >= self.pc_range[1])
        mask = torch.logical_and(range_mask, box_y <= self.pc_range[4])

        forc_dict['pred_boxes'] = forc_dict['pred_boxes'][mask]
        forc_dict['pred_scores'] = forc_dict['pred_scores'][mask]
        forc_dict['pred_labels'] = forc_dict['pred_labels'][mask]

        # This op can make nms faster
        if self.num_det_heads > 1:
            forecasted_dets = split_dets(
                    self.cls_id_to_det_head_idx_map,
                    self.num_det_heads,
                    forc_dict['pred_boxes'],
                    forc_dict['pred_scores'],
                    forc_dict['pred_labels'],
                    False) # moves results to gpu if true
        else:
            forecasted_dets = [forc_dict]

        for k in ('pred_boxes', 'pred_labels', 'pose_idx', 'pred_scores'):
            self.past_detections[k] = self.past_detections[k][mask]

        return forecasted_dets

    @torch.jit.export
    def fork_forward_1tile(self,
            last_pred_dict : Dict[str,torch.Tensor],
            last_pose : torch.Tensor,
            last_ts : int,
            cur_pose : torch.Tensor,
            cur_ts : int) -> torch.jit.Future[List[Dict[str,torch.Tensor]]]:
        return torch.jit.fork(self.forward_1tile, last_pred_dict, \
                last_pose, last_ts, cur_pose, cur_ts)

    @torch.jit.export
    def forward_1tile(self,
            last_pred_dict : Dict[str,torch.Tensor],
            last_pose : torch.Tensor,
            last_ts : int,
            cur_pose : torch.Tensor,
            cur_ts : int) -> List[Dict[str,torch.Tensor]]:

        boxes = last_pred_dict['pred_boxes']
        if boxes.size(0) == 0:
            return [{'pred_boxes': torch.empty([0, 9], dtype=torch.float),
                'pred_labels': torch.empty([0], dtype=torch.long),
                'pred_scores': torch.empty([0], dtype=torch.float)}]

        # mask
        scores = last_pred_dict['pred_scores']
        mask = (scores >= self.score_thresh)
        boxes = boxes[mask]
        labels = last_pred_dict['pred_labels'][mask] - 1

        if self.score_thresh > 0:
            scores = torch.full([boxes.size(0)], self.score_thresh * 0.9, dtype=scores.dtype)
        else:
            scores = scores[mask]

        pose_idx = torch.zeros(boxes.size(0), dtype=torch.long)
        forcb = torch.ops.kucsl.forecast_past_dets(
                boxes,
                pose_idx,
                last_pose.unsqueeze(0),
                cur_pose,
                torch.tensor([last_ts]).long(),
                cur_ts)

        # remove those going out of range, which might not be necessary since eval doesn't care
        box_x, box_y = forcb[:,0], forcb[:,1]
        range_mask = box_x >= self.pc_range[0]
        range_mask = torch.logical_and(range_mask, box_x <= self.pc_range[3])
        range_mask = torch.logical_and(range_mask, box_y >= self.pc_range[1])
        mask = torch.logical_and(range_mask, box_y <= self.pc_range[4])

        forc_dict : Dict[str,torch.Tensor] = {}
        forc_dict['pred_boxes'] = forcb[mask]
        forc_dict['pred_scores'] = scores[mask]
        forc_dict['pred_labels'] = labels[mask]

        # This op can make nms faster
        if self.num_det_heads > 1:
            forecasted_dets = split_dets(
                    self.cls_id_to_det_head_idx_map,
                    self.num_det_heads,
                    forc_dict['pred_boxes'],
                    forc_dict['pred_scores'],
                    forc_dict['pred_labels'],
                    False) # moves results to gpu if true
        else:
            forecasted_dets = [forc_dict]

        return forecasted_dets
