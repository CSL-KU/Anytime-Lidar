#ifndef IOU3D_NMS_H
#define IOU3D_NMS_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int64_t boxes_aligned_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap);
int64_t boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap);
int64_t boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou);
int64_t nms_gpu(at::Tensor boxes, at::Tensor keep, double nms_overlap_thresh);
int64_t nms_normal_gpu(at::Tensor boxes, at::Tensor keep, double nms_overlap_thresh);

#endif
