import torch
import os

def load_torch_op_shr_lib(path_rel_to_pcdet):
    pcdet_path = os.environ["PCDET_PATH"]
    path = os.path.join(pcdet_path, path_rel_to_pcdet)
    for file_name in os.listdir(path):
        if file_name.endswith('.so'):
            torch.ops.load_library(os.path.join(path, file_name))
            break
