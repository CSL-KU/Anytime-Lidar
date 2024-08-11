#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <utility>
#include <string>
#include <vector>
#include <map>

namespace py = pybind11;
using namespace torch::indexing;

torch::Tensor forecast_past_dets(
        const torch::Tensor pred_boxes, // [num_objects, 9], fp_type
        const torch::Tensor past_pose_indexes, // [num_objects], long
        const torch::Tensor past_poses, // [num_past_poses, 14], fp_type
        const torch::Tensor cur_pose, // [14], fp_type
        const torch::Tensor past_timestamps, // [num_past_poses], long
        const long target_timestamp // [1]
);

torch::Tensor move_to_world_coords(
        const torch::Tensor pred_boxes, // [num_objects, 9], fp_type
        const torch::Tensor poses, // [num_objects, 9], fp_type
        const torch::Tensor pose_idx // [14], fp_type
);

// Assumes all boxes are in the same coordinate frame
torch::Tensor forecasting_nms(
        const torch::Tensor pred_boxes_hp,
        const torch::Tensor pred_labels_hp,
        const torch::Tensor pred_boxes_lp,
        const torch::Tensor pred_labels_lp,
        const double iou_threshold
);

// This is only needed if using multihead architecture
std::vector<std::map<std::string, torch::Tensor>> split_dets(
        const torch::Tensor pred_boxes, // [num_objects, 9], fp_type
        const torch::Tensor pred_scores,
        const torch::Tensor pred_labels,
        const torch::Tensor cls_id_to_det_head_idx_map,
        const long num_det_heads,
        const bool move_to_gpu);

TORCH_LIBRARY_FRAGMENT(kucsl, m) { //kucsl here is the domain name
    m.def("forecast_past_dets", &forecast_past_dets);
    m.def("move_to_world_coords", &move_to_world_coords);
    m.def("forecasting_nms", &forecasting_nms);
    //m.def("split_dets", &split_dets); // causes compile error, use python impl instead
}

