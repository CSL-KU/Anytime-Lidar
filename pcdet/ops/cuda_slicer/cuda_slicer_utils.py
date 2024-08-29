import torch
import os
from torch.onnx import register_custom_op_symbolic

script_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
for file_name in os.listdir(script_path):
    if file_name.endswith('.so'):
        torch.ops.load_library(script_path + os.sep + file_name)
        break

def slice_and_batch_nhwc(padded_x : torch.Tensor, indices : torch.Tensor): #, slice_size : int):
    return torch.ops.cuda_slicer.slice_and_batch_nhwc(padded_x, indices) #, slice_size)

def slice_and_batch_nhwc_op(g, padded_x : torch.Tensor, indices : torch.Tensor): #, slice_size : int):
    return g.op("cuda_slicer::slice_and_batch_nhwc", padded_x, indices) #, slice_size)
register_custom_op_symbolic("cuda_slicer::slice_and_batch_nhwc", slice_and_batch_nhwc_op, 17)
