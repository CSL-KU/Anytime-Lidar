from functools import partial

import torch.nn as nn
import torch

from ...utils.spconv_utils import replace_feature, spconv
from pcdet.ops.norm_funcs.res_aware_bnorm import ResAwareBatchNorm1d, ResAwareBatchNorm2d
from pcdet.ops.norm_funcs.fn_instance_norm import FnInstanceNorm
from typing import Tuple

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_fn=None):
    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out
    
    
class PillarBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        
        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            dense_block(256, 256, 3, norm_fn=norm_fn, padding=1),
            dense_block(256, 256, 3, norm_fn=norm_fn, padding=1),
        )
        
        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }


    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })

        return batch_dict

#static method
def PillarRes18BackBone8x_pillar_calc(bev_img : torch.Tensor, num_slices : int) \
        -> Tuple[torch.Tensor,torch.Tensor]:
    bi_sz = bev_img.shape
    p0_ = bev_img.view(bi_sz[0], bi_sz[1], bi_sz[2], num_slices, \
            bi_sz[3]//num_slices).sum(dim=[0, 2, 4])

    x1 = torch.nn.functional.max_pool2d(bev_img, kernel_size=3, stride=2, padding=1)
    x2 = torch.nn.functional.max_pool2d(x1, kernel_size=3, stride=2, padding=1)
    x3 = torch.nn.functional.max_pool2d(x2, kernel_size=3, stride=2, padding=1)
    dims = [0, 2, 3] # leave C (num res)
    p1 = x1.sum(dim=dims)
    p2 = x2.sum(dim=dims)
    p3 = x3.sum(dim=dims)
    p0 = p0_.sum(dim=1)
    return p0_, torch.stack((p0, p1, p2, p3)) # return (num_res, num_slc) (num_layer, numres)

class PillarRes18BackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.res_divs = model_cfg.get('RESOLUTION_DIV', [1.0])
        self.resdiv_mask = self.model_cfg.get('RESDIV_MASK', [True] * len(self.res_divs))
        norm_method = self.model_cfg.get('NORM_METHOD', 'Batch')
        if norm_method == 'ResAwareBatch':
            norm_fn = partial(ResAwareBatchNorm1d, res_divs=self.res_divs, \
                    resdiv_mask=self.resdiv_mask, eps=1e-3, momentum=0.01)
        else: #norm_method == 'Batch':
            norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        #elif norm_method == 'Instance':

        self.sparse_shape = grid_size[[1, 0]]

        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )
        
        if norm_method == 'Batch':
            norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        elif norm_method == 'ResAwareBatch':
            norm_fn = partial(ResAwareBatchNorm2d, res_divs=self.res_divs, \
                    resdiv_mask=self.resdiv_mask, eps=1e-3, momentum=0.01)
        elif norm_method == 'Instance':
            norm_fn = partial(FnInstanceNorm, eps=1e-3, momentum=0.01)

        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

        # Grouped with respect to having same input size
        self.num_layer_groups = 4

    def get_inds_dividers(self, tile_size_voxels):
        # numbers here are determined with respect to strides
        return [tile_size_voxels / float(s) for s in (2,4,8)]

    def sparse_outp_downscale_factor(self):
        return 8 # 3 convs with stride of 2

    def forward_up_to_dense(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']

        record_time = batch_dict.get('record_time', False)
        record_vcounts = batch_dict.get('record_int_vcounts', False)
        # int: intermediary
        record_int_vcoords = batch_dict.get('record_int_vcoords', False)
        record_int_indices = batch_dict.get('record_int_indices', False)

        if record_time:
            events=[torch.cuda.Event(enable_timing=True)]
            events[-1].record()
        if record_vcounts:
            num_voxels=[pillar_coords.size(0)]
        if record_int_vcoords:
            vcoords = []
            tile_size_voxels = batch_dict['tile_size_voxels']
            num_tiles = batch_dict['num_tiles']
        if record_int_indices:
            vinds = []

        resdiv = batch_dict.get('resolution_divider', 1)
        sparse_shape = [int(s/resdiv) for s in self.sparse_shape]
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=sparse_shape,
            batch_size=batch_size
        )

        #x_conv1 = self.conv1(input_sp_tensor)
        #x_conv2 = self.conv2(x_conv1)
        #x_conv3 = self.conv3(x_conv2)
        #x_conv4 = self.conv4(x_conv3)

        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2_0 = self.conv2[0][0](x_conv1)

        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if record_vcounts:
            num_voxels.append(x_conv2_0.indices.size(0))
        if record_int_indices:
            vinds.append(x_conv2_0.indices)
        if record_int_vcoords:
            voxel_tile_coords = torch.div(x_conv2_0.indices[:, -1], tile_size_voxels // 2, \
                    rounding_mode='trunc')
            voxel_dist = torch.bincount(voxel_tile_coords, minlength=num_tiles)[:num_tiles]
            vcoords.append(voxel_dist.unsqueeze(0))

        x_conv2_0 = x_conv2_0.replace_feature(self.conv2[0][1](x_conv2_0.features))
        x_conv2_0 = x_conv2_0.replace_feature(self.conv2[0][2](x_conv2_0.features))
        x_conv2_1 = self.conv2[1](x_conv2_0)
        x_conv2 = self.conv2[2](x_conv2_1)
        x_conv3_0 = self.conv3[0][0](x_conv2)

        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if record_vcounts:
            num_voxels.append(x_conv3_0.indices.size(0))
        if record_int_indices:
            vinds.append(x_conv3_0.indices)
        if record_int_vcoords:
            voxel_tile_coords = torch.div(x_conv3_0.indices[:, -1], tile_size_voxels // 4, \
                    rounding_mode='trunc')
            voxel_dist = torch.bincount(voxel_tile_coords, minlength=num_tiles)[:num_tiles]
            vcoords.append(voxel_dist.unsqueeze(0))

        x_conv3_0 = x_conv3_0.replace_feature(self.conv3[0][1](x_conv3_0.features))
        x_conv3_0 = x_conv3_0.replace_feature(self.conv3[0][2](x_conv3_0.features))
        x_conv3_1 = self.conv3[1](x_conv3_0)
        x_conv3 = self.conv3[2](x_conv3_1)
        x_conv4_0 = self.conv4[0][0](x_conv3)

        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
        if record_vcounts:
            num_voxels.append(x_conv4_0.indices.size(0))
        if record_int_indices:
            vinds.append(x_conv4_0.indices)
        if record_int_vcoords:
            voxel_tile_coords = torch.div(x_conv4_0.indices[:, -1], tile_size_voxels // 8, \
                    rounding_mode='trunc')
            voxel_dist = torch.bincount(voxel_tile_coords, minlength=num_tiles)[:num_tiles]
            vcoords.append(voxel_dist.unsqueeze(0))

        x_conv4_0 = x_conv4_0.replace_feature(self.conv4[0][1](x_conv4_0.features))
        x_conv4_0 = x_conv4_0.replace_feature(self.conv4[0][2](x_conv4_0.features))
        x_conv4_1 = self.conv4[1](x_conv4_0)
        x_conv4 = self.conv4[2](x_conv4_1)

        batch_dict['x_conv4_out'] = x_conv4.dense()

        if record_time:
            events.append(torch.cuda.Event(enable_timing=True))
            events[-1].record()
            batch_dict['bb3d_layer_time_events'] = events
        if record_vcounts:
            batch_dict['bb3d_num_voxels'] = num_voxels
        if record_int_indices:
            batch_dict['bb3d_intermediary_vinds'] = vinds
        if record_int_vcoords:
            batch_dict['bb3d_intermediary_vcoords'] = vcoords

        return batch_dict

    # this function emulates the sparse convolutions and calculates the
    # number of voxels for each stage
    def sparse_convs_num_pillars_calc(self, bev_img : torch.Tensor) -> torch.Tensor:
        num_pillars = torch.empty(self.num_layer_groups, device=bev_img.device)
        x1 = torch.nn.functional.max_pool2d(bev_img, kernel_size=3, stride=2, padding=1)
        x2 = torch.nn.functional.max_pool2d(x1, kernel_size=3, stride=2, padding=1)
        x3 = torch.nn.functional.max_pool2d(x2, kernel_size=3, stride=2, padding=1)
        num_pillars[0] = bev_img.sum()
        num_pillars[1] = x1.sum()
        num_pillars[2] = x2.sum()
        num_pillars[3] = x3.sum()
        return num_pillars

    def forward_dense(self, x_conv4):
        return self.conv5(x_conv4)

    def forward(self, batch_dict):
        batch_dict = self.forward_up_to_dense(batch_dict)
        x_conv4 = batch_dict['x_conv4_out']
        x_conv5 = self.forward_dense(x_conv4)

        # batch_dict.update({
        #     'encoded_spconv_tensor': out,
        #     'encoded_spconv_tensor_stride': 8
        # })
        batch_dict.update({
            'multi_scale_2d_features': {
                #'x_conv1': x_conv1,
                #'x_conv2': x_conv2,
                #'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        del batch_dict['x_conv4_out']
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })
        
        return batch_dict
