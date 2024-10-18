import torch

from . import ioubev_nms_cuda
from typing import Optional, List
from os import listdir
# to be released
# try:
#     import weighted_nms_ext
# except Exception as e:
#     print(f'Error {e} when import weighted_nms.')

for file_name in listdir("../pcdet/ops/ioubev_nms"):
    if file_name.endswith('.so'):
        torch.ops.load_library("../pcdet/ops/ioubev_nms/" + file_name)
        break



def nms_gpu_bev(boxes : torch.Tensor, scores : torch.Tensor, thresh : float, pre_maxsize : Optional[int], post_max_size : Optional[int]):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (float): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]

    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = torch.ops.ioubev_nms_cuda.nms_gpu(boxes, keep, thresh)
    keep = order[keep[:num_out].cuda()].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep