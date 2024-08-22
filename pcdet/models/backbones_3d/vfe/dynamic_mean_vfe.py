import torch

from .vfe_template import VFETemplate

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from typing import Tuple

class DynamicMeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

    def get_output_feature_dim(self):
        return self.num_point_features

    @torch.no_grad()
    def range_filter(self, batch_dict, filter_z=True):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        if filter_z:
            points_z = points[:, 3]
            mask = torch.logical_and(points_z > self.point_cloud_range[2], points_z < self.point_cloud_range[5])
            points = points[mask]

        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        mask = ((point_coords >= 0) & (point_coords < self.grid_size)).all(dim=1)
        batch_dict['points'] = points[mask]
        batch_dict['points_coords'] = point_coords[mask]
        return batch_dict

    @torch.no_grad()
    def forward_gen_voxels(self, points : torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                        point_coords[:, 0] * self.scale_yz + \
                        point_coords[:, 1] * self.scale_z + \
                        point_coords[:, 2]
        points_data = points[:, 1:].contiguous()
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        return voxel_coords.contiguous(), points_mean.contiguous()

    def forward(self, points : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        voxel_coords, features = self.forward_gen_voxels(points)
        return voxel_coords, features
