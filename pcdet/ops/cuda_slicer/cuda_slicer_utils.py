import torch
import os

script_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
for file_name in os.listdir(script_path):
    if file_name.endswith('.so'):
        torch.ops.load_library(script_path + os.sep + file_name)
        break

def slice_and_batch_nhwc(padded_x : torch.Tensor, indices : torch.Tensor, slice_size : int):
    return torch.ops.cuda_slicer.slice_and_batch_nhwc(padded_x, indices, slice_size)
