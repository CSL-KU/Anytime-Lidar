/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SLICE_AND_BATCH_KERNEL_H
#define SLICE_AND_BATCH_KERNEL_H

#include <cstdint>
#include <cuda_runtime.h>

void sliceAndBatchImpl(
        const float* inp, // NHWC
        const int inp_size[4],
        const int32_t* inds, // [num_inds, 3]
        const int num_inds,
        float* outp, // NCHW, should be allocated before
        const int outp_size[4],
        const int slice_size,
        cudaStream_t stream);

#endif // SLICE_AND_BATCH_KERNEL_Hs
