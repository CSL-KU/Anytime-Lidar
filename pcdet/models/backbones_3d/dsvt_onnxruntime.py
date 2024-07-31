import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .dsvt_input_layer import DSVTInputLayer

from torch.onnx import symbolic_helper as sym_help
from torch.onnx import  _type_utils
from typing import List

def is_nested(g, self):
    t = torch.tensor(False, dtype=torch.bool)
    const_node = g.op("Constant", value_t=t)
    return const_node

def is_autocast_enabled(g):
    t = torch.tensor(False, dtype=torch.bool)
    const_node = g.op("Constant", value_t=t)
    return const_node

torch.onnx.register_custom_op_symbolic("prim::is_nested", is_nested, 9)
torch.onnx.register_custom_op_symbolic("aten::is_autocast_enabled", is_autocast_enabled, 9)

class DSVT_ort(nn.Module):
    '''Dynamic Sparse Voxel Transformer Backbone.
    Args:
        INPUT_LAYER: Config of input layer, which converts the output of vfe to dsvt input.
        block_name (list[string]): Name of blocks for each stage. Length: stage_num.
        set_info (list[list[int, int]]): A list of set config for each stage. Eelement i contains 
            [set_size, block_num], where set_size is the number of voxel in a set and block_num is the
            number of blocks for stage i. Length: stage_num.
        d_model (list[int]): Number of input channels for each stage. Length: stage_num.
        nhead (list[int]): Number of attention heads for each stage. Length: stage_num.
        dim_feedforward (list[int]): Dimensions of the feedforward network in set attention for each stage. 
            Length: stage num.
        dropout (float): Drop rate of set attention. 
        activation (string): Name of activation layer in set attention.
        reduction_type (string): Pooling method between stages. One of: "attention", "maxpool", "linear".
        output_shape (tuple[int, int]): Shape of output bev feature.
        conv_out_channel (int): Number of output channels.

    '''
    def __init__(self, model_cfg, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.input_layer = DSVTInputLayer(self.model_cfg.INPUT_LAYER)
        block_name = self.model_cfg.block_name
        set_info = self.model_cfg.set_info
        d_model = self.model_cfg.d_model
        nhead = self.model_cfg.nhead
        dim_feedforward = self.model_cfg.dim_feedforward
        dropout = self.model_cfg.dropout
        activation = self.model_cfg.activation
        self.reduction_type = self.model_cfg.get('reduction_type', 'attention')
        # save GPU memory
        self.use_torch_ckpt = self.model_cfg.get('ues_checkpoint', False)
 
        # Sparse Regional Attention Blocks
        stage_num = len(block_name)
        for stage_id in range(stage_num):
            num_blocks_this_stage = set_info[stage_id][-1]
            dmodel_this_stage = d_model[stage_id]
            dfeed_this_stage = dim_feedforward[stage_id]
            num_head_this_stage = nhead[stage_id]
            block_name_this_stage = block_name[stage_id]
            block_module = _get_block_module(block_name_this_stage)
            block_list=[]
            norm_list=[]
            for i in range(num_blocks_this_stage):
                block_list.append(
                    block_module(dmodel_this_stage, num_head_this_stage, dfeed_this_stage,
                                 dropout, activation, batch_first=True)
                )
                norm_list.append(nn.LayerNorm(dmodel_this_stage))
            self.__setattr__(f'stage_{stage_id}', nn.ModuleList(block_list))
            self.__setattr__(f'residual_norm_stage_{stage_id}', nn.ModuleList(norm_list))

            # apply pooling except the last stage
            if stage_id < stage_num-1:
                downsample_window = self.model_cfg.INPUT_LAYER.downsample_stride[stage_id]
                dmodel_next_stage = d_model[stage_id+1]
                pool_volume = torch.IntTensor(downsample_window).prod().item()
                if self.reduction_type == 'linear':
                    cat_feat_dim = dmodel_this_stage * torch.IntTensor(downsample_window).prod().item()
                    self.__setattr__(f'stage_{stage_id}_reduction', Stage_Reduction_Block(cat_feat_dim, dmodel_next_stage))
                elif self.reduction_type == 'maxpool':
                    self.__setattr__(f'stage_{stage_id}_reduction', torch.nn.MaxPool1d(pool_volume))
                elif self.reduction_type == 'attention':
                    self.__setattr__(f'stage_{stage_id}_reduction', Stage_ReductionAtt_Block(dmodel_this_stage, pool_volume))
                else:
                    raise NotImplementedError

        self.num_shifts = [2] * stage_num
        self.output_shape = self.model_cfg.output_shape
        self.stage_num = stage_num
        self.set_info = set_info
        self.num_point_features = self.model_cfg.conv_out_channel

        self.num_layer_groups = 1
        self.input_layer_scripted = None
        #self._reset_parameters()

    def get_inds_dividers(self, tile_size_voxels):
        # numbers here are determined with respect to strides
        return []


    #@torch.jit.ignore    
    def get_voxel_info(self, voxel_feats, voxel_coors): #: batch_dict):
        '''
        Args:
            bacth_dict (dict): 
                The dict contains the following keys
                - voxel_features (Tensor[float]): Voxel features after VFE. Shape of (N, d_model[0]), 
                    where N is the number of input voxels.
                - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                    Each row is (batch_id, z, y, x). 
                - ...
        
        Returns:
            bacth_dict (dict):
                The dict contains the following keys
                - pillar_features (Tensor[float]):
                - voxel_coords (Tensor[int]):
                - ...
        '''

        self.dsvtblocks_list = self.stage_0
        self.layer_norms_list = self.residual_norm_stage_0

        if self.input_layer_scripted == None:
            self.input_layer_scripted = torch.jit.script(self.input_layer)
            # Do a few invocations to invoke JIT optimization
            num_voxels = voxel_feats.size(0)
            while True:
                num_voxels -= 2000
                if num_voxels <= 0:
                    break
                vf, vc = voxel_feats[:num_voxels], voxel_coors[:num_voxels]
                self.input_layer_scripted(vf, vc)

        voxel_info = self.input_layer_scripted(voxel_feats, voxel_coors) #batch_dict)

        voxel_feat = voxel_info['voxel_feats_stage0']
        set_voxel_inds_list = [[voxel_info[f'set_voxel_inds_stage{s}_shift{i}'] for i in range(self.num_shifts[s])] for s in range(self.stage_num)]
        set_voxel_masks_list = [[voxel_info[f'set_voxel_mask_stage{s}_shift{i}'] for i in range(self.num_shifts[s])] for s in range(self.stage_num)]
        pos_embed_list = [[[voxel_info[f'pos_embed_stage{s}_block{b}_shift{i}'] for i in range(self.num_shifts[s])] for b in range(self.set_info[s][1])] for s in range(self.stage_num)]

        pooling_mapping_index = [voxel_info[f'pooling_mapping_index_stage{s+1}'] for s in range(self.stage_num-1)]
        pooling_index_in_pool = [voxel_info[f'pooling_index_in_pool_stage{s+1}'] for s in range(self.stage_num-1)]
        pooling_preholder_feats = [voxel_info[f'pooling_preholder_feats_stage{s+1}'] for s in range(self.stage_num-1)]

        assert len(set_voxel_inds_list) == 1 and len(set_voxel_inds_list[0]) == 2
        assert len(set_voxel_masks_list) == 1 and len(set_voxel_masks_list[0]) == 2

        assert len(pooling_mapping_index) == 0
        assert len(pooling_index_in_pool) == 0
        assert len(pooling_preholder_feats) == 0

        set_voxel_inds_tensor_shift_0 = set_voxel_inds_list[0][0].contiguous()
        set_voxel_inds_tensor_shift_1 = set_voxel_inds_list[0][1].contiguous()
        set_voxel_masks_tensor_shift_0 = set_voxel_masks_list[0][0].contiguous()
        set_voxel_masks_tensor_shift_1 = set_voxel_masks_list[0][1].contiguous()

        pos_embed_tensor=torch.stack([torch.stack(v, dim=0) \
                for v in pos_embed_list[0]], dim=0).contiguous()

        return (voxel_feat, set_voxel_inds_tensor_shift_0, set_voxel_inds_tensor_shift_1,
                set_voxel_masks_tensor_shift_0, set_voxel_masks_tensor_shift_1, pos_embed_tensor, 
                voxel_info[f'voxel_coors_stage{self.stage_num - 1}']) # Dont forward last one (coors)
        #assert tuple(pos_embed_list.shape[:3]) == (1, 4, 2)

    def forward(
        self,
        voxel_feat : torch.Tensor, 
        set_voxel_inds_tensor_shift_0 : torch.Tensor,
        set_voxel_inds_tensor_shift_1 : torch.Tensor,
        set_voxel_masks_tensor_shift_0 : torch.Tensor, 
        set_voxel_masks_tensor_shift_1: torch.Tensor,
        pos_embed_tensor : torch.Tensor,
    ):
        outputs = voxel_feat

        residual = outputs
        blc_id = 0
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed)
        outputs = self.dsvtblocks_list[0].encoder_list[0](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed)
        outputs = self.dsvtblocks_list[0].encoder_list[1](*inputs)
        
        outputs = self.layer_norms_list[0](residual + outputs)

        residual = outputs
        blc_id = 1
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed)
        outputs = self.dsvtblocks_list[1].encoder_list[0](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed)
        outputs = self.dsvtblocks_list[1].encoder_list[1](*inputs)
        
        outputs = self.layer_norms_list[1](residual + outputs)

        residual = outputs
        blc_id = 2
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed)
        outputs = self.dsvtblocks_list[2].encoder_list[0](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_0[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_0[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed)
        outputs = self.dsvtblocks_list[2].encoder_list[1](*inputs)
        
        outputs = self.layer_norms_list[2](residual + outputs)

        residual = outputs
        blc_id = 3
        set_id = 0
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed)
        outputs = self.dsvtblocks_list[3].encoder_list[0](*inputs)
        set_id = 1
        set_voxel_inds = set_voxel_inds_tensor_shift_1[set_id:set_id+1].squeeze(0)
        set_voxel_masks = set_voxel_masks_tensor_shift_1[set_id:set_id+1].squeeze(0)
        pos_embed = pos_embed_tensor[blc_id:blc_id+1, set_id:set_id+1].squeeze(0).squeeze(0)
        inputs = (outputs, set_voxel_inds, set_voxel_masks, pos_embed)
        outputs = self.dsvtblocks_list[3].encoder_list[1](*inputs)
        
        outputs = self.layer_norms_list[3](residual + outputs)

        return outputs

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)


class DSVTBlock(nn.Module):
    ''' Consist of two encoder layer, shift and shift back.
    '''
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True):
        super().__init__()

        encoder_1 = DSVT_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                        activation, batch_first)
        encoder_2 = DSVT_EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                        activation, batch_first)
        self.encoder_list = nn.ModuleList([encoder_1, encoder_2])

    @torch.jit.ignore    
    def forward(
            self,
            src : torch.Tensor,
            set_voxel_inds_list,
            set_voxel_masks_list,
            pos_embed_list,
            block_id,
    ):
        num_shifts = 2
        output = src
        # TODO: bug to be fixed, mismatch of pos_embed
        i = 0
        set_id = i
        shift_id = block_id % 2
        pos_embed_id = i
        set_voxel_inds = set_voxel_inds_list[shift_id][set_id]
        set_voxel_masks = set_voxel_masks_list[shift_id][set_id]
        pos_embed = pos_embed_list[pos_embed_id]
        layer = self.encoder_list[0]
        output = layer(output, set_voxel_inds, set_voxel_masks, pos_embed)

        i = 1
        set_id = i
        shift_id = block_id % 2
        pos_embed_id = i
        set_voxel_inds = set_voxel_inds_list[shift_id][set_id]
        set_voxel_masks = set_voxel_masks_list[shift_id][set_id]
        pos_embed = pos_embed_list[pos_embed_id]
        layer = self.encoder_list[1]
        output = layer(output, set_voxel_inds, set_voxel_masks, pos_embed)

        return output


class DSVT_EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True, mlp_dropout=0.):
        super().__init__()
        self.win_attn = SetAttention(d_model, nhead, dropout, dim_feedforward, activation, batch_first, mlp_dropout)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self,src,set_voxel_inds,set_voxel_masks,pos=None) -> torch.Tensor:
        identity = src 
        src = self.win_attn(src, pos, set_voxel_masks, set_voxel_inds)
        src = src + identity
        src = self.norm(src)

        return src

class SetAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout, dim_feedforward=2048, activation="relu", batch_first=True, mlp_dropout=0.):
        super().__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos, key_padding_mask, voxel_inds) -> torch.Tensor:
        '''
        Args:
            src (Tensor[float]): Voxel features with shape (N, C), where N is the number of voxels.
            pos (Tensor[float]): Position embedding vectors with shape (N, C).
            key_padding_mask (Tensor[bool]): Mask for redundant voxels within set. Shape of (set_num, set_size).
            voxel_inds (Tensor[int]): Voxel indexs for each set. Shape of (set_num, set_size).
        Returns:
            src (Tensor[float]): Voxel features.
        '''

        set_features = src[voxel_inds]
        set_pos = pos[voxel_inds]
        query = set_features + set_pos
        key = set_features + set_pos
        value = set_features

        src2 = self.self_attn(query, key, value, key_padding_mask)[0]
        flatten_inds = voxel_inds.reshape(-1).contiguous()
        src2_placeholder = torch.zeros(src.size(), device=src2.device, dtype=src2.dtype)
        src2_flattened = src2.flatten(0, 1).contiguous()
        src2_placeholder[flatten_inds] = src2_flattened
        src2 = src2_placeholder

        # FFN layer
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class Stage_Reduction_Block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.linear1 = nn.Linear(input_channel, output_channel, bias=False)
        self.norm = nn.LayerNorm(output_channel)

    def forward(self, x):
        src = x
        src = self.norm(self.linear1(x))
        return src


class Stage_ReductionAtt_Block(nn.Module):
    def __init__(self, input_channel, pool_volume):
        super().__init__()
        self.pool_volume = pool_volume
        self.query_func = torch.nn.MaxPool1d(pool_volume)
        self.norm = nn.LayerNorm(input_channel)
        self.self_attn = nn.MultiheadAttention(input_channel, 8, batch_first=True)
        self.pos_embedding = nn.Parameter(torch.randn(pool_volume, input_channel))
        nn.init.normal_(self.pos_embedding, std=.01)

    def forward(self, x, key_padding_mask) -> torch.Tensor:
        # x: [voxel_num, c_dim, pool_volume]
        src = self.query_func(x).permute(0, 2, 1)  # voxel_num, 1, c_dim
        key = value = x.permute(0, 2, 1)
        key = key + self.pos_embedding.unsqueeze(0).repeat(src.shape[0], 1, 1)
        query = src.clone()
        output = self.self_attn(query, key, value, key_padding_mask)[0]
        src = self.norm(output + src).squeeze(1)
        return src


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_block_module(name):
    """Return an block module given a string"""
    if name == "DSVTBlock":
        return DSVTBlock
    raise RuntimeError(F"This Block not exist.")
