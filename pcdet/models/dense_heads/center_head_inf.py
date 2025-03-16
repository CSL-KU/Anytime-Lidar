import copy
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from ...ops.cuda_slicer import cuda_slicer_utils
from typing import Dict, List, Tuple, Optional, Final
from functools import partial
from pcdet.ops.norm_funcs.res_aware_bnorm import ResAwareBatchNorm2d
from pcdet.ops.norm_funcs.fn_instance_norm import FnInstanceNorm

class SeparateHead(nn.Module):
    vel_conv_available : Final[bool]
    iou_conv_available : Final[bool]

    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False, norm_func=None, enable_normalization=True, optimize_attr_convs=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict
        self.conv_names = tuple(sep_head_dict.keys())
        self.refs_to_bns = []

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []

            for k in range(num_conv - 1):
                p = 0 if optimize_attr_convs and cur_name != 'hm' else 1
                inner_fc_list = [nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=p, bias=use_bias)]
                if enable_normalization: #TODO I havent made an exception for hm, but its ok
                    inner_fc_list.append(nn.BatchNorm2d(input_channels) if norm_func is None else norm_func(input_channels))
                    if optimize_attr_convs and cur_name != 'hm':
                        self.refs_to_bns.append(inner_fc_list[-1])
                inner_fc_list.append(nn.ReLU())
                fc_list.append(nn.Sequential(*inner_fc_list))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=p, bias=True))
            fc = nn.Sequential(*fc_list)

            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

        self.vel_conv_available = ('vel' in self.sep_head_dict)
        self.iou_conv_available = ('iou' in self.sep_head_dict)

    def instancenorm_mode(self):
        for bn in self.refs_to_bns:
            if isinstance(bn, ResAwareBatchNorm2d):
                for i in range(len(bn.layers)):
                    bn.layers[i].momentum = 0.
                    bn.layers[i].track_running_stats = False
            else:
                bn.momentum = 0.
                bn.track_running_stats = False

    def pd_to_list(self, pd : Dict[str,torch.Tensor]) -> List[torch.Tensor]:
        lst = []
        # TODO, get this order from CenterHeadInf
        for nm in ['hm', 'center', 'center_z', 'dim', 'rot', 'vel', 'iou']:
            if nm in pd:
                lst.append(pd[nm])

        return lst

    def forward_hm(self, x) -> torch.Tensor:
        return self.hm(x).sigmoid()

    def forward_attr(self, x) -> Dict[str, torch.Tensor]:
        ret_dict = {
            'center': self.center(x),
            'center_z': self.center_z(x),
            'dim': self.dim(x),
            'rot': self.rot(x),
            }
        if self.vel_conv_available:
            ret_dict['vel'] = self.vel(x)
        if self.iou_conv_available:
            ret_dict['iou'] = self.iou(x)

        return ret_dict


    def forward(self, x : torch.Tensor) -> Dict[str, torch.Tensor]:
        ret_dict = {'hm': self.forward_hm(x)}
        ret_dict.update(self.forward_attr(x))
        return ret_dict

# Inference only, torchscript compatible
class CenterHeadInf(nn.Module): 
    feature_map_stride : Final[int]
    nms_type : Final[str]
    nms_thresh : Final[List[float]]
    nms_pre_maxsize : Final[List[int]]
    nms_post_maxsize : Final[List[int]]
    score_thresh : Final[float]
    use_iou_to_rectify_score : Final[bool]
    head_order : Final[List[str]]
    max_obj_per_sample : Final[int]
    num_det_heads : Final[int]
    tcount : Final[int]
    optimize_attr_convs : Final[bool]
    initial_voxel_size : Final[List[float]]
    grid_size : Final[List[int]]
    res_divs: Final[List[float]]

    point_cloud_range : List[float]
    voxel_size : List[float]
    class_id_mapping_each_head : List[torch.Tensor]
    det_dict_copy : Dict[str,torch.Tensor]
    iou_rectifier : torch.Tensor
    post_center_limit_range : torch.Tensor

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=False):
        super().__init__()
        assert not predict_boxes_when_training
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size.tolist()
        self.voxel_size = voxel_size
        self.initial_voxel_size = voxel_size.copy()
        self.point_cloud_range = point_cloud_range.tolist()

        self.all_pc_ranges = self.model_cfg.get('ALL_PC_RANGES', [self.point_cloud_range])

        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', 1)

        self.class_names = class_names
        self.class_names_each_head = []
        class_id_mapping_each_head : List[torch.Tensor] = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.tensor(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )
            class_id_mapping_each_head.append(cur_class_id_mapping)
        self.class_id_mapping_each_head = class_id_mapping_each_head

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.cls_id_to_det_head_idx_map = torch.zeros((total_classes,), dtype=torch.int)
        self.num_det_heads = len(self.class_id_mapping_each_head)
        for i, cls_ids in enumerate(self.class_id_mapping_each_head):
            for cls_id in cls_ids:
                self.cls_id_to_det_head_idx_map[cls_id] = i

        self.res_divs = model_cfg.get('RESOLUTION_DIV', [1.0])
        resdiv_mask = self.model_cfg.get('RESDIV_MASK', [True] * len(self.res_divs))
        norm_method = self.model_cfg.get('NORM_METHOD', 'Batch')
        if norm_method == 'Batch':
            norm_func = partial(nn.BatchNorm2d, eps=self.model_cfg.get('BN_EPS', 1e-5), momentum=self.model_cfg.get('BN_MOM', 0.1))
        elif norm_method == 'ResAwareBatch':
            norm_func = partial(ResAwareBatchNorm2d, res_divs=self.res_divs, \
                    resdiv_mask=resdiv_mask, eps=1e-3, momentum=0.01)
        elif norm_method == 'Instance':
            norm_func = partial(FnInstanceNorm, eps=self.model_cfg.get('BN_EPS', 1e-5), momentum=self.model_cfg.get('BN_MOM', 0.1))
        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            norm_func(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.optimize_attr_convs = self.model_cfg.get('OPTIMIZE_ATTR_CONVS', True)

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                    norm_func=norm_func,
                    enable_normalization=self.model_cfg.get('ENABLE_NORM_IN_ATTR_LAYERS', True),
                    optimize_attr_convs=self.optimize_attr_convs
                )
            )
        self.det_dict_copy = {
            "pred_boxes": torch.zeros([0, 9], dtype=torch.float, device='cuda'),
            "pred_scores": torch.zeros([0], dtype=torch.float,device='cuda'),
            "pred_labels": torch.zeros([0], dtype=torch.int, device='cuda'),
        }

        post_process_cfg = self.model_cfg.POST_PROCESSING
        self.post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).float()
        self.nms_type = post_process_cfg.NMS_CONFIG.NMS_TYPE

        nms_thresh = post_process_cfg.NMS_CONFIG.NMS_THRESH
        nms_pre_maxsize = post_process_cfg.NMS_CONFIG.NMS_PRE_MAXSIZE
        nms_post_maxsize = post_process_cfg.NMS_CONFIG.NMS_POST_MAXSIZE
        self.nms_thresh = nms_thresh if isinstance(nms_thresh, list) else [nms_thresh]
        self.nms_pre_maxsize = nms_pre_maxsize if isinstance(nms_pre_maxsize, list) else [nms_pre_maxsize]
        self.nms_post_maxsize = nms_post_maxsize if isinstance(nms_post_maxsize, list) else [nms_post_maxsize]

        self.score_thresh = post_process_cfg.SCORE_THRESH
        self.use_iou_to_rectify_score = post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False)
        self.iou_rectifier = torch.tensor(post_process_cfg.IOU_RECTIFIER if self.use_iou_to_rectify_score else [0], dtype=torch.float, device='cuda')
        self.head_order = self.separate_head_cfg.HEAD_ORDER
        self.max_obj_per_sample = post_process_cfg.MAX_OBJ_PER_SAMPLE

        self.tcount = self.model_cfg.get('TILE_COUNT', 1)

    def instancenorm_mode(self):
        for dh in self.heads_list:
            dh.instancenorm_mode()

    def generate_predicted_boxes_single_head(self, cls_mapping : torch.Tensor, \
            pred_dict: Dict[str,torch.Tensor], topk_output : List[torch.Tensor], \
            forecasted_dets : Optional[Dict[str,torch.Tensor]]) \
            -> Dict[str,torch.Tensor]:

        pred_dict = {k:v.cpu() for k,v in pred_dict.items()}
        topk_output = [t.cpu() for t in topk_output]

        if self.optimize_attr_convs:
            # Remove the HW dimentions by flattenning
            center = pred_dict['center'].flatten(-3)
            center_z = pred_dict['center_z'].flatten(-3)
            dim = pred_dict['dim'].flatten(-3).exp()
            rot = pred_dict['rot'].flatten(-3)
            rot_cos = rot[:, :1] # (num_obj, 1)
            rot_sin = rot[:, 1:2]
            vel = pred_dict['vel'].flatten(-3) if 'vel' in self.head_order else None
            iou = (pred_dict['iou'].flatten(-3) + 1) * 0.5 if 'iou' in pred_dict else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap_sliced(
                rot_cos, rot_sin,
                center, center_z, dim,
                self.point_cloud_range, self.voxel_size,
                self.feature_map_stride, self.max_obj_per_sample,
                self.post_center_limit_range, topk_output,
                vel, iou, self.score_thresh
            )
        else:
            batch_hm = pred_dict['hm'] #.sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.head_order else None
            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                batch_hm, batch_rot_cos, batch_rot_sin,
                batch_center, batch_center_z, batch_dim,
                self.point_cloud_range, self.voxel_size,
                self.feature_map_stride, self.max_obj_per_sample,
                self.post_center_limit_range, topk_output,
                batch_vel, batch_iou, self.score_thresh
            )

        #for k, final_dict in enumerate(final_pred_dicts): # for all batches
        final_dict = final_pred_dicts[0]
        final_dict['pred_labels'] = cls_mapping[final_dict['pred_labels'].long()]

        if self.use_iou_to_rectify_score and 'pred_iou' in final_dict:
            pred_iou = torch.clamp(final_dict['pred_iou'], min=0, max=1.0)
            ps = final_dict['pred_scores']
            final_dict['pred_scores'] = torch.pow(final_dict['pred_scores'], 1 - self.iou_rectifier[final_dict['pred_labels']]) * \
                    torch.pow(pred_iou, self.iou_rectifier[final_dict['pred_labels']])

        if self.nms_type not in ['circle_nms', 'multi_class_nms']:
            if forecasted_dets is not None:
                for j in forecasted_dets.keys():
                    final_dict[j] = torch.cat((final_dict[j], forecasted_dets[j]), dim=0)

            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                final_dict['pred_scores'], final_dict['pred_boxes'], self.nms_type, self.nms_thresh[0],
                self.nms_post_maxsize[0], self.nms_pre_maxsize[0])
                #score_thresh=None
        elif self.nms_type == 'multi_class_nms':
            if forecasted_dets is not None:
                for j in forecasted_dets.keys():
                    final_dict[j] = torch.cat((final_dict[j], forecasted_dets[j]), dim=0)

            selected, selected_scores = model_nms_utils.multi_classes_nms_mmdet(
                final_dict['pred_scores'], final_dict['pred_boxes'], final_dict['pred_labels'],
                self.nms_thresh, self.nms_post_maxsize, self.nms_pre_maxsize, None
            )
        else:
            selected = torch.ones(final_dict['pred_boxes'].size(0), dtype=torch.long, device=final_dict['pred_boxes'].device)
            selected_scores = final_dict['pred_scores']

        final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
        final_dict['pred_scores'] = selected_scores
        final_dict['pred_labels'] = final_dict['pred_labels'][selected]

        return final_dict

    def generate_predicted_boxes(self, batch_size: int, pred_dicts: List[Dict[str,torch.Tensor]],\
            topk_outputs : List[List[torch.Tensor]], forecasted_dets : Optional[List[Dict[str,torch.Tensor]]]) \
            -> List[Dict[str,torch.Tensor]]:

        assert batch_size == 1

        ret_dict : List[Dict[str,List[torch.Tensor]]] = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]

        futs : List[torch.jit.Future[Dict[str,torch.Tensor]]] = []
        torch.cuda.synchronize()
        for idx, pred_dict in enumerate(pred_dicts):  # num det heads
            cls_mapping = self.class_id_mapping_each_head[idx]
            final_dict_fut = torch.jit.fork(self.generate_predicted_boxes_single_head,
                    cls_mapping, pred_dict, topk_outputs[idx],
                    forecasted_dets[idx] if forecasted_dets is not None else None)
            futs.append(final_dict_fut)

        for final_dict_fut in futs:
            final_dict = torch.jit.wait(final_dict_fut)
            ret_dict[0]['pred_boxes'].append(final_dict['pred_boxes'])
            ret_dict[0]['pred_scores'].append(final_dict['pred_scores'])
            ret_dict[0]['pred_labels'].append(final_dict['pred_labels'])

        final_ret_dict : List[Dict[str,torch.Tensor]] = []
        for k in range(batch_size):
            if not ret_dict[k]['pred_boxes']:
                final_ret_dict.append(self.get_empty_det_dict())
            else:
                final_ret_dict.append({
                    'pred_boxes': torch.cat(ret_dict[k]['pred_boxes'], dim=0),
                    'pred_scores' : torch.cat(ret_dict[k]['pred_scores'], dim=0),
                    'pred_labels' : torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1})

        return final_ret_dict

    def ordered_outp_names(self, include_hm=True):
        names =  (['hm'] if include_hm else []) + list(self.separate_head_cfg.HEAD_ORDER)
        if 'iou' in self.separate_head_cfg.HEAD_DICT:
            names += ['iou']
        return names

    # This func is for the baseline, not for valo
    def convert_out_to_pred_dicts(self, out_tensors):
        head_order = self.ordered_outp_names()
        num_convs_per_head = len(out_tensors) // self.num_det_heads
        pred_dicts = []
        for i in range(self.num_det_heads):
            ot = out_tensors[i*num_convs_per_head:(i+1)*num_convs_per_head]
            pred_dicts.append({name : t for name, t in zip(head_order, ot)})
        return pred_dicts

    def forward_up_to_topk(self, spatial_features_2d : torch.Tensor) -> List[torch.Tensor]:
        x = self.shared_conv(spatial_features_2d)
        pred_dicts = [h.forward(x) for h in self.heads_list]
        conv_order = self.ordered_outp_names()
        out_tensors_ordered = [pd[conv_name] for pd in pred_dicts for conv_name in conv_order]
        return out_tensors_ordered

    def forward(self, spatial_features_2d : torch.Tensor,
            forecasted_dets : Optional[List[Dict[str,torch.Tensor]]]):
        assert not self.training

        tensors = self.forward_pre(spatial_features_2d)
        x = tensors[0]
        pred_dicts = [{'hm':t} for t in tensors[1:]]
        pred_dicts = self.forward_post(x, pred_dicts)
        topk_outputs = self.forward_topk(pred_dicts)
        return self.forward_genbox(x.size(0), pred_dicts, topk_outputs, forecasted_dets)

    def forward_pre(self, spatial_features_2d) -> List[torch.Tensor]:
        x = self.shared_conv(spatial_features_2d)
        return [x] + [head.forward_hm(x) for head in self.heads_list]

    def forward_post(self, x : torch.Tensor, pred_dicts : List[Dict[str,torch.Tensor]]) -> List[Dict[str,torch.Tensor]]:
        for i, head in enumerate(self.heads_list):
            pred_dicts[i].update(head.forward_attr(x))
        return pred_dicts

    @torch.jit.export
    def slice_shr_conv_outp(self, shr_conv_outp : torch.Tensor,
            ys_all : List[torch.Tensor], xs_all : List[torch.Tensor]):
            #topk_outputs : List[List[torch.Tensor]]):

        slice_size = 5 # two convs, each ksize=3
        shr_conv_outp_nhwc = shr_conv_outp.permute(0,2,3,1).contiguous()
        p = slice_size // 2
        padded_x = torch.nn.functional.pad(shr_conv_outp_nhwc, (0,0,p,p,p,p))

        y_inds = torch.cat(ys_all).int().flatten()
        x_inds = torch.cat(xs_all).int().flatten()

        mops = self.max_obj_per_sample
        b_id = torch.zeros(self.num_det_heads * mops,
                dtype=torch.int, device=shr_conv_outp.device) # since batch size is 1
        indices = torch.stack((b_id, y_inds, x_inds), dim=1)
        all_slices = cuda_slicer_utils.slice_and_batch_nhwc(padded_x, indices) #, slice_size)

        return [all_slices[i*mops:(i+1)*mops] for i in range(self.num_det_heads)]

    @torch.jit.export
    def forward_sliced_inp(self, slices_per_head : List[torch.Tensor],
            pred_dicts : List[Dict[str,torch.Tensor]]) -> List[Dict[str,torch.Tensor]]:
        for i, head in enumerate(self.heads_list):
            pred_dicts[i].update(head.forward_attr(slices_per_head[i]))
        return pred_dicts

    def forward_sliced_inp_trt(self, slices_per_head : List[torch.Tensor]) \
            -> List[torch.Tensor]:
        out_tensors = []
        for i, head in enumerate(self.heads_list):
            pd = head.forward_attr(slices_per_head[i])
            out_tensors += head.pd_to_list(pd)
        return out_tensors

    def get_sliced_forward_outp_names(self):
        names = list(self.separate_head_cfg.HEAD_ORDER)
        if 'iou' in self.separate_head_cfg.HEAD_DICT:
            names += ['iou']
        return names

    @torch.jit.export
    def forward_topk(self, pred_dicts : List[Dict[str,torch.Tensor]]) -> List[List[torch.Tensor]]:
        return [centernet_utils._topk(pd['hm'], K=self.max_obj_per_sample) for pd in pred_dicts]

    @torch.jit.export
    def forward_topk_trt(self, heatmaps: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        return [centernet_utils.topk_trt(hm, K=self.max_obj_per_sample) for hm in heatmaps]

    @torch.jit.export
    def forward_genbox(self, batch_size: int, pred_dicts: List[Dict[str,torch.Tensor]],\
            topk_outputs : List[List[torch.Tensor]], forecasted_dets : Optional[List[Dict[str,torch.Tensor]]]) \
            -> List[Dict[str,torch.Tensor]]:
        return self.generate_predicted_boxes(batch_size, pred_dicts, topk_outputs, forecasted_dets)

    def get_empty_final_box_dicts(self):
        return [self.get_empty_det_dict() for i in range(self.num_det_heads)]

    def get_empty_det_dict(self):
        det_dict = {}
        for k,v in self.det_dict_copy.items():
            det_dict[k] = v.clone().detach()
        return det_dict

    @torch.jit.export
    def adjust_voxel_size_wrt_resolution(self, res_idx : int):
        self.point_cloud_range = self.all_pc_ranges[res_idx]
        resdiv = self.res_divs[res_idx]
        new_grid_size_x = int(self.grid_size[0] / resdiv)
        new_grid_size_y = int(self.grid_size[1] / resdiv)
        Xlen = self.point_cloud_range[3] - self.point_cloud_range[0]
        Ylen = self.point_cloud_range[4] - self.point_cloud_range[1]
        self.voxel_size[0] = Xlen / new_grid_size_x
        self.voxel_size[1] = Ylen / new_grid_size_y

@torch.jit.script
def scatter_sliced_tensors(chosen_tile_coords : List[int],
        sliced_tensors : List[torch.Tensor],
        tcount : int) -> List[torch.Tensor]:
    ctc = chosen_tile_coords
    if len(ctc) == tcount:
        return sliced_tensors # no need to scatter

    #Based on chosen_tile_coords, we need to scatter the output
    ctc_s, ctc_e = ctc[0], ctc[-1]

    chunk_r, chunk_l = [0, 0], [0, 0]
    if ctc_s <= ctc_e:
        # contiguous
        num_tiles = ctc_e - ctc_s + 1
    else:
        # Two chunks, find the point of switching
        i = 0
        while ctc[i] < ctc[i+1]:
            i += 1
        chunk_r[0], chunk_r[1] = ctc_s, ctc[i]
        chunk_l[0], chunk_l[1] = ctc[i+1], ctc_e
        num_tiles = (chunk_r[1] - chunk_r[0] + 1) + (chunk_l[1] - chunk_l[0] + 1)

    scattered_tensors = []
    for tensor in sliced_tensors:
        tile_sz = tensor.size(-1) // num_tiles
        full_sz = tile_sz * tcount
        scat_tensor = torch.zeros((tensor.size(0), tensor.size(1), tensor.size(2), full_sz),
                device=tensor.device, dtype=tensor.dtype)

        if ctc_s <= ctc_e:
            # contiguous
            scat_tensor[..., (ctc_s * tile_sz):((ctc_e + 1) * tile_sz)] = tensor
        else:
            # Two chunks, find the point of switching
            c_sz_l = (chunk_l[1] - chunk_l[0] + 1) * tile_sz
            c_sz_r = (chunk_r[1] - chunk_r[0] + 1) * tile_sz
            #Example: 7 8 2 3 4  -> . . 2 3 4 . . 7 8
            scat_tensor[..., (chunk_r[0]*tile_sz):((chunk_r[1]+1)*tile_sz)] = \
                    tensor[..., :c_sz_r]
            scat_tensor[..., (chunk_l[0]*tile_sz):((chunk_l[1]+1)*tile_sz)] = \
                    tensor[..., -c_sz_l:]
        scattered_tensors.append(scat_tensor)
    return scattered_tensors
