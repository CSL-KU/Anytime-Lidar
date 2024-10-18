import torch
import os

def load_torch_op_shr_lib(path):
    for file_name in os.listdir(path):
        if file_name.endswith('.so'):
            torch.ops.load_library(os.path.join(path, file_name))
            break
