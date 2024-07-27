#include <assert.h>
#include <vector>
#include <cassert>
#include <stdio.h>
#include <cmath>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CALL(call)                                   \
  do                                                    \
{                                                     \
  const cudaError_t error_code = call;              \
  if (error_code != cudaSuccess)                    \
  {                                                 \
    printf("CUDA Error:\n");                      \
    printf("    File:       %s\n", __FILE__);     \
    printf("    Line:       %d\n", __LINE__);     \
    printf("    Error code: %d\n", error_code);   \
    printf("    Error text: %s\n",                \
        cudaGetErrorString(error_code));          \
    exit(1);                                      \
  }                                                 \
} while (0)

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// #define DEBUG
// #define ASSERTION

template <typename T>
using one_dim_pa32 = torch::PackedTensorAccessor32<T,1,torch::RestrictPtrTraits>;

template <typename T>
using two_dim_pa32 = torch::PackedTensorAccessor32<T,2,torch::RestrictPtrTraits>;

template <typename scalar_t>
__global__ void get_window_coors_kernel(
        const two_dim_pa32<scalar_t> coors, // [N, 4]
        const long shift_x,
        const long shift_y,
        const long shift_z,
        const long win_shape_x,
        const long win_shape_y,
        const long win_shape_z,
        const long max_num_win_x,
        const long max_num_win_y,
        const long max_num_win_z,
        one_dim_pa32<scalar_t> batch_win_inds, // [N]
        two_dim_pa32<scalar_t> coors_in_win // [N, 3]
){

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < coors.size(0)){
    auto coor_x = coors[idx][3];
    auto coor_y = coors[idx][2];
    auto coor_z = coors[idx][1];

    auto shifted_coor_x = coor_x + shift_x;
    auto shifted_coor_y = coor_y + shift_y;
    auto shifted_coor_z = coor_z + shift_z;

    long win_coor_x = shifted_coor_x / win_shape_x;
    long win_coor_y = shifted_coor_y / win_shape_y;
    long win_coor_z = shifted_coor_z / win_shape_z;

    auto max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z;
    batch_win_inds[idx] = coors[idx][0] * max_num_win_per_sample +
         win_coor_x * max_num_win_y * max_num_win_z +
         win_coor_y * max_num_win_z +
         win_coor_z;

    coors_in_win[idx][2] = shifted_coor_x % win_shape_x;
    coors_in_win[idx][1] = shifted_coor_y % win_shape_y;
    coors_in_win[idx][0] = shifted_coor_z % win_shape_z;
  }
}

std::vector<torch::Tensor> get_window_coors(
    const at::Tensor coors,
    const std::vector<long> &sparse_shape,
    const std::vector<long> &window_shape,
    const bool do_shift,	
    const std::vector<long> &shift_list, // make default value [0]
    const bool return_win_coors)
{
  assert(!return_win_coors); // not implemented

  auto win_shape_x = window_shape[0];
  auto win_shape_y = window_shape[1];
  auto win_shape_z = ((window_shape.size() == 2) ? sparse_shape[2] : window_shape[2]);

  auto sparse_shape_x = sparse_shape[0];
  auto sparse_shape_y = sparse_shape[1];
  auto sparse_shape_z = sparse_shape[2];

  auto max_num_win_x = static_cast<long>(std::ceil(
        static_cast<float>(sparse_shape_x) / win_shape_x) + 1);
  auto max_num_win_y = static_cast<long>(std::ceil(
        static_cast<float>(sparse_shape_y) / win_shape_y) + 1);
  auto max_num_win_z = static_cast<long>(std::ceil(
        static_cast<float>(sparse_shape_z) / win_shape_z) + 1);

  long shift_x, shift_y, shift_z;
  bool all_elems_zero = true; // This will be equal to passing None in python version
  for(long e : shift_list){
    if(e != 0){
      all_elems_zero=false;
      break;
    }
  }
  if(!all_elems_zero){
    shift_x = shift_list[0]; shift_y = shift_list[1]; shift_z = shift_list[2];
  }
  else if (do_shift){
    shift_x = win_shape_x/2; shift_y = win_shape_y/2; shift_z = win_shape_z/2; 
  }
  else{
    shift_x = win_shape_x; shift_y = win_shape_y; shift_z = win_shape_z; 
  }

  if (sparse_shape_z == win_shape_z)
    shift_z = 0;

  auto tensor_options = torch::TensorOptions()
      .layout(torch::kStrided)
      .dtype(torch::kInt64)
      .device(coors.device().type())
      .requires_grad(false);

  auto num_inds = coors.size(0);
  const long threads_per_block = 256;
  const long num_blocks = std::ceil(static_cast<float>(num_inds) / threads_per_block);

  torch::Tensor batch_win_inds = torch::empty({num_inds}, tensor_options);
  torch::Tensor coor_win_inds = torch::empty({num_inds, 3}, tensor_options);

  const auto stream = at::cuda::getCurrentCUDAStream().stream();

//  std::cerr << "+  " << shift_x << " " << shift_y << " " << shift_z << " " << win_shape_x <<
//   " " << win_shape_y << " " << win_shape_z << " " <<
//        max_num_win_x << " " << max_num_win_y << " " << max_num_win_z << " " << std::endl;

  AT_DISPATCH_INTEGRAL_TYPES(coors.type(), "get_window_coors_cuda", ([&] {
    get_window_coors_kernel<scalar_t><<<num_blocks, threads_per_block, 0, stream>>>(
        coors.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        shift_x, shift_y, shift_z, win_shape_x, win_shape_y, win_shape_z,
        max_num_win_x, max_num_win_y, max_num_win_z,
        batch_win_inds.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
        coor_win_inds.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
  }));

  std::vector<torch::Tensor> ret = {batch_win_inds, coor_win_inds};
  return ret;

}

//std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get_pooling_index(
//    at::Tensor coors,
//    std::vector<long>& sparse_shape,
//    std::vector<long>& window_shape){
//
//  }
//
