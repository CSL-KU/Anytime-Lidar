#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <utility>

using namespace torch::indexing;
namespace py = pybind11;

//torch::Tensor slice_and_batch(
//        torch::Tensor inp,
//        torch::Tensor slice_indices,
//		const int64_t slice_size);

torch::Tensor slice_and_batch_nhwc(
        torch::Tensor inp,
        torch::Tensor slice_indices);
//        const int64_t slice_size);

TORCH_LIBRARY(cuda_slicer, m) {
    //m.def("slice_and_batch", &slice_and_batch, "Slice and Batch CUDA");
    m.def("slice_and_batch_nhwc", &slice_and_batch_nhwc);
}
