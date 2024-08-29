#include "sliceAndBatchKernel.h"
#include <stdint.h>
#include <cassert>
#include <stdio.h>

// inds is two dimensionals, (num_slices, 3). 3 is batch_id,x,y
__global__ void sliceAndBatchNHWCKernel(
        const float* inp, // NHWC
        const int H_In, const int W_In, const int C_In,
        const int32_t* inds, // [num_inds, 3]
        const int num_inds,
        float* outp, // NCHW
        const int C, const int H, const int W){
  // blockDim.x is the number of threads in a block

//  const int H_In = inp_size[1], W_In = inp_size[2], C_In = inp_size[3];
  const auto slice_idx = blockIdx.x;
  const auto C_per_thread = C_In / blockDim.z;

  const auto slice_h = threadIdx.x;
  const auto slice_w = threadIdx.y;
  const auto c_offset = threadIdx.z * C_per_thread;

  const auto ind_idx = slice_idx * 3;
  const auto n = inds[ind_idx];
  const auto h = inds[ind_idx + 1] + slice_h;
  const auto w = inds[ind_idx + 2] + slice_w;
  const auto src_base_idx = (n * H_In * W_In * C_In) +
          (h * W_In * C_In) +
          (w * C_In) +
          c_offset;

//  const int C = outp_size[1], H = outp_size[2], W = outp_size[3];
  const auto HW = H * W;
  const auto dest_base_idx = (slice_idx * C * HW) +
          (slice_h * W) +
          slice_w;

  for(auto c=0; c < C_per_thread; ++c){
    outp[dest_base_idx + (c_offset + c) * HW] = inp[src_base_idx + c];
  }

}

void sliceAndBatchImpl(
        const float* inp, // NHWC
        const int inp_size[4],
        const int32_t* inds, // [num_inds, 3]
        const int num_inds,
        float* outp, // NCHW, should be allocated before
        const int outp_size[4],
        const int slice_size,
        cudaStream_t stream)
{
    const auto max_threads_per_block = 256;
    const auto C = inp_size[3];
    auto num_active_threads = slice_size * slice_size * C;
    uint32_t C_per_thread = 1;
    while(num_active_threads > max_threads_per_block){
        num_active_threads /= 2;
        C_per_thread *= 2;
    }
    // If conditions doesn't hold, error!
    assert(C % C_per_thread == 0);

    const dim3 grid_dims(num_inds);
    dim3 block_dims(slice_size, slice_size, C / C_per_thread);

    sliceAndBatchNHWCKernel<<<grid_dims, block_dims, 0, stream>>>(
            inp, inp_size[1], inp_size[2], inp_size[3], inds, num_inds,
            outp, outp_size[1], outp_size[2], outp_size[3]);

}


