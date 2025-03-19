import torch
from pcdet.ops.norm_funcs.res_aware_bnorm import ResAwareBatchNorm1d, ResAwareBatchNorm2d
from pcdet.models.backbones_3d.spconv_backbone_2d import PillarRes18BackBone8x_pillar_calc, PillarRes18BackBone8x_pillar_calc_v2

from typing import Dict, List, Tuple, Optional, Final

def set_bn_resolution(resawarebns, res_idx):
    for rabn in resawarebns:
        rabn.setResIndex(res_idx)

def interpolate_batch_norms(resawarebns, max_grid_l):
    for rabn in resawarebns:
        rabn.interpolate(max_grid_l)

def get_all_resawarebn(model):
    resaware1dbns, resaware2dbns = [], []
    for module in model.modules():
        if isinstance(module, ResAwareBatchNorm1d):
            resaware1dbns.append(module)
        elif isinstance(module, ResAwareBatchNorm2d):
            resaware2dbns.append(module)
    return resaware1dbns, resaware2dbns

@torch.jit.script
def get_slice_range(down_scale_factor : int, x_min: int, x_max: int, maxsz: int) \
        -> Tuple[int, int]:
    dsf = down_scale_factor
    x_min, x_max = x_min // dsf, x_max // dsf + 1
    denom = 4 # denom is dependent on strides within the dense covs
    minsz = 16

    rng = (x_max - x_min)
    if rng < minsz:
        diff = minsz - rng
        if x_max + diff <= maxsz:
            x_max += diff
        elif x_min - diff >= 0:
            x_min -= diff
        #else: # very unlikely
        #    pass
        rng = (x_max - x_min)

    pad = denom - (rng % denom)
    if pad == denom:
        pass
    elif x_min >= pad: # pad from left
        x_min -= pad
    elif (maxsz - x_max) >= pad: # pad from right
        x_max += pad
    else: # don't slice
        x_min, x_max = 0 , maxsz
    return x_min, x_max

@torch.jit.script
def slice_tensor(down_scale_factor : int, x_min: int, x_max: int, inp : torch.Tensor) \
        -> Tuple[torch.Tensor, int, int]:
    x_min, x_max = get_slice_range(down_scale_factor, x_min, x_max, inp.size(3))
    return inp[..., x_min:x_max].contiguous(), x_min, x_max

# This will be used to generate the onnx
class DenseConvsPipeline(torch.nn.Module):
    def __init__(self, backbone_3d, backbone_2d, dense_head):
        super().__init__()
        self.backbone_3d = backbone_3d
        self.backbone_2d = backbone_2d
        self.dense_head = dense_head
        self.optimize_attr_convs = dense_head.model_cfg.OPTIMIZE_ATTR_CONVS

    def forward(self, x_conv4 : torch.Tensor) -> List[torch.Tensor]:
        x_conv5 = self.backbone_3d.forward_dense(x_conv4)
        data_dict = self.backbone_2d({"multi_scale_2d_features" : 
            {"x_conv4": x_conv4, "x_conv5": x_conv5}})

        if self.optimize_attr_convs:
            outputs = self.dense_head.forward_pre(data_dict['spatial_features_2d'])
            shr_conv_outp = outputs[0]
            heatmaps = outputs[1:]

            topk_outputs = self.dense_head.forward_topk_trt(heatmaps)

            ys_all = [topk_outp[2] for topk_outp in topk_outputs]
            xs_all = [topk_outp[3] for topk_outp in topk_outputs]

            sliced_inp = self.dense_head.slice_shr_conv_outp(shr_conv_outp, ys_all, xs_all)
            outputs = self.dense_head.forward_sliced_inp_trt(sliced_inp)
            for topk_output in topk_outputs:
                outputs += topk_output

            return outputs
        else:
            return self.dense_head.forward_up_to_topk(data_dict['spatial_features_2d'])

class MultiPillarCounter(torch.nn.Module):
    # Pass the args in cpu , pillar sizes should be [N,2], pc_range should be [6]
    grid_sizes: Final[List[List[int]]]
    num_slices: Final[List[int]]
    num_res : Final[int]
    pillar_sizes : torch.Tensor
    pc_range_min: torch.Tensor
    pc_range_cpu: torch.Tensor

    def __init__(self, pillar_sizes : torch.Tensor, pc_range : torch.Tensor,
                 slice_sz: int = 32):
        super().__init__()
        xy_length = pc_range[[3,4]] - pc_range[[0,1]]
        grid_sizes = torch.empty((pillar_sizes.size(0), 2), dtype=torch.int) # cpu
        self.num_slices = [0] * pillar_sizes.size(0)
        for i, ps in enumerate(pillar_sizes):
            grid_sizes[i] = torch.round(xy_length / ps)
            self.num_slices[i] = (grid_sizes[i, 0] // slice_sz).item()
        self.grid_sizes = grid_sizes.tolist()

        self.pillar_sizes = pillar_sizes.cuda()
        self.pc_range_cpu = pc_range
        self.pc_range_min = pc_range[[0,1]].cuda()
        self.num_res = len(self.grid_sizes)

        print('num_slices', self.num_slices)
        print('grid_sizes', self.grid_sizes)
        print('pillar_sizes', self.pillar_sizes)


    @torch.jit.export
    def forward_new(self, points_xy : torch.Tensor, get_xminmax : bool) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        points_xy_s = points_xy - self.pc_range_min

        grid_sz = self.grid_sizes[0] # get biggest grid
        batch_grid = torch.zeros([1, self.num_res, grid_sz[0], grid_sz[1]],
                                      device=points_xy.device, dtype=torch.float32)

        expanded_pts = points_xy_s.unsqueeze(1).expand(-1, self.num_res, -1)
        batch_point_coords = (expanded_pts / self.pillar_sizes).int()

        inds = torch.arange(self.num_res, device=points_xy.device).unsqueeze(0)
        inds = inds.expand(batch_point_coords.size(0), -1).flatten()
        batch_grid[:, inds, batch_point_coords[:, :, 1].flatten(),
                   batch_point_coords[:, :, 0].flatten()] = 1.0

        pc0, pillar_counts = PillarRes18BackBone8x_pillar_calc_v2(batch_grid,
                                                               self.num_slices[0])
        x_minmax = torch.empty((self.num_res, 2), dtype=torch.int)
        if get_xminmax:
            mask = (pc0 > 0).cpu()
            for i, row in enumerate(mask): # for each resolution
                inds = torch.where(row)[0]
                x_minmax[i,0], x_minmax[i,1] = inds[0], inds[-1]
            x_minmax = x_minmax
        pillar_counts = pillar_counts.T.cpu().int()

        return pillar_counts, x_minmax

    def forward(self, points_xy : torch.Tensor) -> List[torch.Tensor]:
        points_xy_s = points_xy - self.pc_range_min
        grid_sz = self.grid_sizes[0] # get biggest grid
        batch_grid = torch.zeros([1, self.num_res, grid_sz[0], grid_sz[1]],
                                      device=points_xy.device, dtype=torch.float32)

        expanded_pts = points_xy_s.unsqueeze(1).expand(-1, self.num_res, -1)
        batch_point_coords = (expanded_pts / self.pillar_sizes).int()

        inds = torch.arange(self.num_res, device=points_xy.device).unsqueeze(0)
        inds = inds.expand(batch_point_coords.size(0), -1).flatten()
        batch_grid[:, inds, batch_point_coords[:, :, 1].flatten(),
                   batch_point_coords[:, :, 0].flatten()] = 1.0

        pillar_counts = PillarRes18BackBone8x_pillar_calc(batch_grid, self.num_slices[0], True)
        pillar_counts = pillar_counts.int().permute(1,0,2).cpu()

        pcl = []
        for i in range(self.num_res):
            ns, pc = self.num_slices[i], pillar_counts[i]
            pcl.append(pc[:, :ns])

        return pcl

    @torch.jit.export
    def get_minmax_inds(self, points_x : torch.Tensor, res_idx : int) -> torch.Tensor:
        xmin, xmax = torch.aminmax(points_x)
        minmax = torch.cat((xmin.unsqueeze(-1), xmax.unsqueeze(-1))) - self.pc_range_min[0]
        minmax = (minmax / self.pillar_sizes[res_idx, 0]).cpu().int()
        slice_sz = self.grid_sizes[res_idx][1] / self.num_slices[res_idx]
        return (minmax / slice_sz).int()

    @torch.jit.export
    def forward_one_res(self, points_xy_s : torch.Tensor, res_idx : int) -> torch.Tensor:
        grid_sz = self.grid_sizes[res_idx]
        point_coords = torch.floor(points_xy_s / self.pillar_sizes[res_idx]).long()
        grid = torch.zeros([1, 1, grid_sz[0], grid_sz[1]], device=points_xy_s.device)
        grid[:, :, point_coords[:, 1], point_coords[:, 0]] = 1.
        pillar_counts = PillarRes18BackBone8x_pillar_calc(grid, self.num_slices[res_idx], False)

        #return the nonzero slice inds
        return pillar_counts

    @torch.jit.export
    def fork_forward(self, points : torch.Tensor) -> torch.jit.Future[List[torch.Tensor]]:
        points_xy = points[:, 1:3]
        fut = torch.jit.fork(self.forward_old, points_xy, split_counts=True)
        return fut

    def forward_old(self, points_xy : torch.Tensor, split_counts : bool = False) -> List[torch.Tensor]:
        points_xy_s = points_xy - self.pc_range_min
        counts = [self.forward_one_res(points_xy_s, res_idx) \
                for res_idx in range(len(self.grid_sizes))]
        all_pillar_counts = [torch.cat(counts, dim=1)]
        if split_counts:
            all_pillar_counts = self.split_pillar_counts(all_pillar_counts[0].int().cpu())
        return all_pillar_counts

    @torch.jit.export
    def split_pillar_counts(self, all_pillar_counts : torch.Tensor) -> List[torch.Tensor]:
        chunks, bgn = [], 0
        for num_slice in self.num_slices:
            chunks.append(all_pillar_counts[:, bgn:bgn+num_slice])
            bgn+=num_slice
        return chunks

    @torch.jit.export
    def slice_inds_to_point_cloud_range(self, res_idx : int, minx : torch.Tensor, maxx : torch.Tensor):
        ns = self.num_slices[res_idx]
        minx_perc = minx / ns
        maxx_perc = (maxx+1) / ns
        rng = (self.pc_range_cpu[3] - self.pc_range_cpu[0])
        minmax = torch.empty(2)
        minmax[0] = (minx_perc * rng) + self.pc_range_cpu[0]
        minmax[1] = (maxx_perc * rng) + self.pc_range_cpu[0]
        return minmax.cuda()

