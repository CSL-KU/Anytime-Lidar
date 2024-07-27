#include <assert.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

//using namespace torch::indexing;

//inputs:
//<class 'torch.Tensor'> <class 'list'> <class 'list'> <class 'bool'> <class 'list'> <class 'bool'>
//torch.Size([18795, 4]) [360, 360, 1] [30, 30, 1] True [15, 15, 0] False
std::vector<torch::Tensor> get_window_coors(
		const at::Tensor coors,
		const std::vector<long> &sparse_shape,
		const std::vector<long> &window_shape,
		const bool do_shift,
		const std::vector<long> &shift_list,
		const bool return_win_coors);

//std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_pooling_index(
//		at::Tensor coors,
//		std::vector<int>& sparse_shape,
//		std::vector<int>& window_shape);

TORCH_LIBRARY(dsvt_ops, m) { //kucsl here is the domain name
    m.def("dsvt_ops::get_window_coors", &get_window_coors);
//    m.def("kucsl::get_pooling_index", &get_pooling_index);
}

