import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from pcdet.ops.norm_funcs.res_aware_bnorm import ResAwareBatchNorm1d

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate

class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False,
                 res_divs=[1.0],
                 resdiv_mask=[True],
                 norm_method='Batch'):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            if norm_method == 'Batch':
                self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
            elif norm_method == 'ResAwareBatch':
                self.norm = ResAwareBatchNorm1d(out_channels, \
                        res_divs=res_divs, \
                        resdiv_mask=resdiv_mask,
                        eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.relu = nn.ReLU()

    def forward(self, inputs : torch.Tensor, unq_inv : torch.Tensor, \
            num_out_inds : int) -> torch.Tensor:
        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)

        x_max = torch.zeros((num_out_inds, x.size(1)), dtype=x.dtype, device=x.device)
        torch_scatter.scatter_max(x, unq_inv, dim=0, out=x_max)

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated

class DynamicPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        res_divs = model_cfg.get('RESOLUTION_DIV', [1.0])
        norm_method = self.model_cfg.get('NORM_METHOD', 'Batch')
        all_pc_ranges = self.model_cfg.get('ALL_PC_RANGES', None)
        resdiv_mask = self.model_cfg.get('RESDIV_MASK', [True] * len(res_divs))

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)
        self.num_point_features = num_point_features

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2),
                res_divs=res_divs, resdiv_mask=resdiv_mask, norm_method=norm_method)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_params = []
        for i, resdiv in enumerate(res_divs):
            if all_pc_ranges is not None:
                point_cloud_range = all_pc_ranges[i]

            grid_size_tmp = [int(gs / resdiv) for gs in grid_size[:2]]
            Xlen = point_cloud_range[3] - point_cloud_range[0]
            Ylen = point_cloud_range[4] - point_cloud_range[1]
            voxel_size_tmp = [Xlen/grid_size_tmp[0], Ylen/grid_size_tmp[1]]
            self.voxel_params.append((
                    voxel_size_tmp[0], #voxel_x
                    voxel_size_tmp[1], #voxel_y
                    voxel_size[2], #voxel_z
                    voxel_size_tmp[0] / 2 + point_cloud_range[0], #x_offset
                    voxel_size_tmp[1] / 2 + point_cloud_range[1], #y_offset
                    voxel_size[2] / 2 + point_cloud_range[2], #z_offset
                    grid_size_tmp[0] * grid_size_tmp[1], #scale_xy
                    grid_size_tmp[1], #scale_y
                    torch.tensor(grid_size_tmp).cuda(), # grid_size
                    torch.tensor(voxel_size_tmp + [voxel_size[2]]).cuda(),
                    torch.tensor(point_cloud_range).cuda()
            ))

        self.set_params(0)

    # Allows switching between different pillar sizes
    def set_params(self, idx):
        self.voxel_x, self.voxel_y, self.voxel_z, \
                self.x_offset, self.y_offset, self.z_offset,  \
                self.scale_xy, self.scale_y, \
                self.grid_size, self.voxel_size, \
                self.point_cloud_range = self.voxel_params[idx]

    def adjust_voxel_size_wrt_resolution(self, res_idx):
        self.set_params(res_idx)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @torch.no_grad()
    def calc_points_coords(self, points):
        points_coords = torch.floor((points[:, [1,2]] - self.point_cloud_range[[0,1]])
                / self.voxel_size[[0,1]]).int()
        return points_coords

    def forward_gen_pillars(self, points : torch.Tensor, points_coords : Optional[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if points_coords is None:
            points_coords = self.calc_points_coords(points)
        points_xyz = points[:, [1, 2, 3]].contiguous()
        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]
        
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, dim=0)
        num_out_inds = torch.max(unq_inv) + 1
        points_mean = torch.zeros((num_out_inds, points_xyz.size(1)),
                dtype=points_xyz.dtype, device=points_xyz.device)

        torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0, out=points_mean)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]
        
        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        z_ = torch.zeros(unq_coords.shape[0], device=unq_coords.device, dtype=torch.int)
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                   z_,
                                   unq_coords % self.scale_y,
                                   (unq_coords % self.scale_xy) // self.scale_y), dim=1)

        return voxel_coords, features, unq_inv, num_out_inds

    def forward_nn(self, features : torch.Tensor, unq_inv : torch.Tensor, num_out_inds : int) -> torch.Tensor:

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv, num_out_inds)

        return features

    def forward(self, points : torch.Tensor, points_coords : Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        voxel_coords, features, unq_inv, num_out_inds = self.forward_gen_pillars(points, points_coords)
        features = self.forward_nn(features, unq_inv, num_out_inds)
        return voxel_coords, features


class DynamicPillarVFESimple2D(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        # self.use_cluster_xyz = self.model_cfg.get('USE_CLUSTER_XYZ', True)
        if self.use_absolute_xyz:
            num_point_features += 3
        # if self.use_cluster_xyz:
        #     num_point_features += 3
        if self.with_distance:
            num_point_features += 1

        res_divs = model_cfg.get('RESOLUTION_DIV', [1.0])
        norm_method = self.model_cfg.get('NORM_METHOD', 'Batch')
        all_pc_ranges = self.model_cfg.get('ALL_PC_RANGES', None)
        resdiv_mask = self.model_cfg.get('RESDIV_MASK', [True] * len(res_divs))

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2),
                    res_divs=res_divs, resdiv_mask=resdiv_mask, norm_method=norm_method)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_params = []
        for i, resdiv in enumerate(res_divs):
            if all_pc_ranges is not None:
                point_cloud_range = all_pc_ranges[i]

            grid_size_tmp = [int(gs / resdiv) for gs in grid_size[:2]]
            Xlen = point_cloud_range[3] - point_cloud_range[0]
            Ylen = point_cloud_range[4] - point_cloud_range[1]
            voxel_size_tmp = [Xlen/grid_size_tmp[0], Ylen/grid_size_tmp[1]]
            self.voxel_params.append((
                    voxel_size_tmp[0], #voxel_x
                    voxel_size_tmp[1], #voxel_y
                    voxel_size[2], #voxel_z
                    voxel_size_tmp[0] / 2 + point_cloud_range[0], #x_offset
                    voxel_size_tmp[1] / 2 + point_cloud_range[1], #y_offset
                    voxel_size[2] / 2 + point_cloud_range[2], #z_offset
                    grid_size_tmp[0] * grid_size_tmp[1], #scale_xy
                    grid_size_tmp[1], #scale_y
                    torch.tensor(grid_size_tmp).cuda(), # grid_size
                    torch.tensor(voxel_size_tmp + [voxel_size[2]]).cuda(),
                    torch.tensor(point_cloud_range).cuda()
            ))

        self.set_params(0)

    # Allows switching between different pillar sizes
    def set_params(self, idx):
        self.voxel_x, self.voxel_y, self.voxel_z, \
                self.x_offset, self.y_offset, self.z_offset,  \
                self.scale_xy, self.scale_y, \
                self.grid_size, self.voxel_size, \
                self.point_cloud_range = self.voxel_params[idx]

    def adjust_voxel_size_wrt_resolution(self, res_idx):
        self.set_params(res_idx)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @torch.no_grad()
    def calc_points_coords(self, points):
        points_coords = torch.floor((points[:, [1,2]] - self.point_cloud_range[[0,1]])
                / self.voxel_size[[0,1]]).int()
        return points_coords

    @torch.no_grad()
    def forward_gen_pillars(self, points : torch.Tensor, points_coords : Optional[torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if points_coords is None:
            points_coords = self.calc_points_coords(points)
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        features = [f_center]
        if self.use_absolute_xyz:
            features.append(points[:, 1:])
        else:
            features.append(points[:, 4:])

        # if self.use_cluster_xyz:
        #     points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        #     f_cluster = points_xyz - points_mean[unq_inv, :]
        #     features.append(f_cluster)

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        pillar_coords = torch.stack((unq_coords // self.scale_xy,
                                     (unq_coords % self.scale_xy) // self.scale_y,
                                     unq_coords % self.scale_y,
                                     ), dim=1)
        pillar_coords = pillar_coords[:, [0, 2, 1]]

        return pillar_coords, features, unq_inv, torch.max(unq_inv) + 1

    def forward_nn(self, features : torch.Tensor, unq_inv : torch.Tensor, num_out_inds : int) -> torch.Tensor:
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv, num_out_inds)

        return features

    def forward(self, points : torch.Tensor, points_coords : Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        pillar_coords, features, unq_inv, num_out_inds = self.forward_gen_pillars(points, points_coords)
        features = self.forward_nn(features, unq_inv, num_out_inds)
        return pillar_coords, features
