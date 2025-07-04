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
// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "sliceAndBatchKernel.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

//namespace
//{
struct SliceAndBatchParams : public samplesCommon::SampleParams
{
    int32_t slice_size = 5;
    nvinfer1::Dims inp_dims = {
        .nbDims = 4,
        .d = {1, 7, 7, 1, 0, 0, 0, 0}
    };
    nvinfer1::Dims inds_dims = {
        .nbDims = 2,
        .d = {2, 3, 0, 0, 0, 0, 0, 0}
    };
};
//} // namespace

enum IOpos{IN_INP, IN_INDS, OUT_SLICES};

//! \brief  The SampleSliceAndBatchPlugin class implements a SliceAndBatch plugin
//!
//! \details The plugin is able to output the non-zero indices in row major or column major order
//!
class SampleSliceAndBatchPlugin
{
public:
    SampleSliceAndBatchPlugin(SliceAndBatchParams const& params);

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    SliceAndBatchParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims[2];  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    uint32_t mSeed{};

    //!
    //! \brief Creates a TensorRT network and inserts a SliceAndBatch plugin
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config);

    //!
    //! \brief Reads the input and stores the result in a managed buffer
    //!
    bool processInput(samplesCommon::BufferManager const& buffers);

    //!
    //! \brief Verifies the result
    //!
    bool verifyOutput(samplesCommon::BufferManager const& buffers);
};

class SliceAndBatchPluginCreator : public nvinfer1::IPluginCreator
{
public:
    SliceAndBatchPluginCreator();

    AsciiChar const* getPluginName() const noexcept override;

    AsciiChar const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) noexcept override;

    IPluginV2* deserializePlugin(AsciiChar const* name, void const* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;


private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    char mNamespace[64];
};
