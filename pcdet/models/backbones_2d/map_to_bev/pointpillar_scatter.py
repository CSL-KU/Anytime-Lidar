import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Final

class PointPillarScatter(nn.Module):
    num_bev_features : Final[int]
    channels_first : Final[bool]
    grid_sizes : Final[List[Tuple[int,int]]]
    nz : Final[int]
    ny : int
    nx : int

    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size.tolist()
        assert self.nz == 1

        res_divs = model_cfg.get('RESOLUTION_DIV', [1.0])
        self.grid_sizes = []
        for resdiv in res_divs:
            self.grid_sizes.append((int(self.nx / resdiv), int(self.ny / resdiv)))

        self.channels_first = kwargs.get('channels_first', True)

    @torch.jit.export
    def adjust_grid_size_wrt_resolution(self, res_idx : int):
        self.nx, self.ny = self.grid_sizes[res_idx]

    def forward(self, pillar_features : torch.Tensor, coords : torch.Tensor, 
            batch_size : int = 1) -> torch.Tensor:
        batch_spatial_features = []
        #batch_size = coords[:, 0].max().int().item() + 1
        #channels_first = 'chosen_tile_coords' not in batch_dict
        dim1 = self.num_bev_features if self.channels_first else self.nz * self.nx * self.ny
        dim2 = self.nz * self.nx * self.ny if self.channels_first else self.num_bev_features
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(dim1, dim2, dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            if self.channels_first:
                pillars = pillars.t()
                spatial_feature[:, indices] = pillars
            else:
                spatial_feature[indices, :] = pillars

            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        if self.channels_first:
            batch_spatial_features = batch_spatial_features.view(\
                    batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        else:
            batch_spatial_features = batch_spatial_features.view(\
                    batch_size, self.ny, self.nx, self.num_bev_features * self.nz)
        return batch_spatial_features.contiguous()

class PointPillarScatter3d(nn.Module):
    num_bev_features_before_compression: Final[int]
    nz : Final[int]
    ny : Final[int]
    nx : Final[int]

    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        
        self.model_cfg = model_cfg
        self.nx, self.ny, self.nz = self.model_cfg.INPUT_SHAPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features_before_compression = self.model_cfg.NUM_BEV_FEATURES // self.nz

    def forward_training(self, batch_size, pillar_features, coords):
        batch_spatial_features = []
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features_before_compression,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.to(dtype=torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        return batch_spatial_features.contiguous()

    def forward(self, pillar_features : torch.Tensor, coords : torch.Tensor) -> torch.Tensor:
        spatial_feature = torch.zeros(
            self.num_bev_features_before_compression,
            self.nz * self.nx * self.ny,
            dtype=pillar_features.dtype,
            device=pillar_features.device)

        this_coords = coords
        indices = this_coords[:, 1] * self.ny * self.nx + this_coords[:, 2] * self.nx + this_coords[:, 3]
        indices = indices.to(dtype=torch.long)
        pillars = pillar_features
        pillars = pillars.t()
        spatial_feature[:, indices] = pillars

        spatial_feature = spatial_feature.view(1, self.num_bev_features_before_compression * self.nz, self.ny, self.nx)
        return spatial_feature.contiguous()
