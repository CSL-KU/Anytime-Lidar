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
#include <cuda_runtime_api.h>
#include "sliceAndBatchPlugin.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

//using half = __half;

class SliceAndBatchPlugin : public IPluginV2DynamicExt
{
public:
    SliceAndBatchPlugin(SliceAndBatchPlugin const& p) = default;

    SliceAndBatchPlugin(int sliceSize)
        : mSliceSize(sliceSize)
    {
        initFieldsToSerialize();
        mNamespace[0] = '\0';
    }

    ~SliceAndBatchPlugin()
    {
        terminate();
    }

    void initFieldsToSerialize()
    {
        mDataToSerialize.clear();
        mDataToSerialize.emplace_back(PluginField("slice_size", &mSliceSize, PluginFieldType::kINT32, 5));
        mFCToSerialize.nbFields = mDataToSerialize.size();
        mFCToSerialize.fields = mDataToSerialize.data();
    }

    IPluginV2DynamicExt* clone() const noexcept override
    {
        auto clone = std::make_unique<SliceAndBatchPlugin>(*this);
        clone->initFieldsToSerialize();
        return clone.release();
    }

    nvinfer1::DataType getOutputDataType(
            int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        return DataType::kFLOAT;
    }

    AsciiChar const* getPluginType() const noexcept override
    {
        return "slice_and_batch_nhwc";
    }

    AsciiChar const* getPluginVersion() const noexcept override
    {
        return "1";
    }

    void setPluginNamespace(char const* libNamespace) noexcept override
    {
        std::strncpy(mNamespace, libNamespace, 32);
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace;
    }

    // IPluginV3OneBuild methods
    int32_t getNbOutputs() const noexcept override
    {
        return 1;
    }

    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        return;
    }

    void configureWithFormat(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
            DataType type, PluginFormat format, int32_t maxBatchSize) noexcept override
    {
        //pass
    }

    int32_t initialize() noexcept override
    {
        return 0;
    }

    void terminate() noexcept override
    {

    }

    void destroy() noexcept override
    {
        delete this;
    }

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override
    {
        // NOTE accessing other than pos index caused error
        bool typeOk{false};
        if (pos == IOpos::IN_INP || pos == IOpos::OUT_SLICES)
        {
            typeOk = (inOut[pos].type == DataType::kFLOAT);
        }
        else if(pos == IOpos::IN_INDS)
        {
            typeOk = inOut[pos].type == DataType::kINT32;
        }

        typeOk = typeOk && (inOut[pos].format == PluginFormat::kLINEAR);
        return typeOk;
    }

    DimsExprs getOutputDimensions(int outputIndex,
        const DimsExprs* inputs, int nbInputs,
        IExprBuilder& exprBuilder)  noexcept override
    {
        if (inputs[IOpos::IN_INP].nbDims != 4 || inputs[IOpos::IN_INDS].nbDims != 2)
        {
            sample::gLogError << "Input dims are wrong!" << std::endl;
//            return -1;
        }

        DimsExprs output; //(inputs[0]);

        output.nbDims = 4;
        output.d[0] = inputs[IOpos::IN_INDS].d[0];
        output.d[1] = inputs[IOpos::IN_INP].d[3];
        output.d[2] = exprBuilder.constant(mSliceSize);
        output.d[3] = exprBuilder.constant(mSliceSize);

        return output;

    }

    // IPluginV3OneRuntime methods
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        int inp_size[4], outp_size[4];
        for(auto i=0; i<4; ++i){
            inp_size[i] = inputDesc[IOpos::IN_INP].dims.d[i];
            outp_size[i] = outputDesc[0].dims.d[i];
        }

        sliceAndBatchImpl(
                static_cast<const float*>(inputs[IOpos::IN_INP]),
                inp_size,
                static_cast<const int32_t*>(inputs[IOpos::IN_INDS]),
                inputDesc[IOpos::IN_INDS].dims.d[0],
                static_cast<float*>(outputs[0]),
                outp_size,
                mSliceSize,
                stream);

        return 0;
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept
    {
        return &mFCToSerialize;
    }

    size_t getSerializationSize() const noexcept override
    {
        return sizeof(int); // mSliceSize
    }

    void serialize(void* buffer) const noexcept override
    {
        *(int*)buffer = mSliceSize;
    }


    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
            int32_t nbOutputs) const noexcept override
    {
        return 0;
    }



private:
    int mSliceSize{5};
    char mNamespace[64];
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};


SliceAndBatchPluginCreator::SliceAndBatchPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("slice_size", nullptr, PluginFieldType::kINT32, 5));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
    mNamespace[0] = '\0';
}

AsciiChar const* SliceAndBatchPluginCreator::getPluginName() const noexcept
{
    return "slice_and_batch_nhwc";
}

AsciiChar const* SliceAndBatchPluginCreator::getPluginVersion() const noexcept
{
    return "1";
}

PluginFieldCollection const* SliceAndBatchPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SliceAndBatchPluginCreator::createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        int32_t slice_size{5};
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            auto const fieldName(fc->fields[i].name);
            if (std::strcmp(fieldName, "slice_size") == 0)
            {
                slice_size = *static_cast<int32_t const*>(fc->fields[i].data);
            }
        }
        return new SliceAndBatchPlugin(slice_size);
    }
    catch (std::exception const& e)
    {
        sample::gLogError << e.what() << std::endl;
    }
    return nullptr;
}

IPluginV2* SliceAndBatchPluginCreator::deserializePlugin(AsciiChar const* name, void const* serialData, size_t serialLength) noexcept
{
    int slice_size = *(int*)serialData;
    SliceAndBatchPlugin* plugin = new SliceAndBatchPlugin(slice_size);
    plugin->setPluginNamespace(mNamespace);

    return plugin;
}

char const* SliceAndBatchPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace;
}

void SliceAndBatchPluginCreator::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept
{
    std::strncpy(mNamespace, pluginNamespace, 32);
}

SampleSliceAndBatchPlugin::SampleSliceAndBatchPlugin(SliceAndBatchParams const& params)
    : mParams(params)
    , mRuntime(nullptr)
    , mEngine(nullptr)
{
    mSeed = static_cast<uint32_t>(time(nullptr));
}

bool SampleSliceAndBatchPlugin::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
        sample::gLogError << "Builder failed." << std::endl;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
            1U << static_cast<uint32_t>(
            NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    if (!network)
    {
        sample::gLogError << "Network creation failed." << std::endl;
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        sample::gLogError << "Create builder config failed." << std::endl;
        return false;
    }

    auto pluginCreator = std::make_unique<SliceAndBatchPluginCreator>();
    getPluginRegistry()->registerCreator(*pluginCreator.get(), "");

    auto constructed = constructNetwork(builder, network, config);
    if (!constructed)
    {
        sample::gLogError << "Construct network failed." << std::endl;
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        sample::gLogError << "Couldn't create plan." << std::endl;
        return false;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 2);
    mInputDims[IOpos::IN_INP] = network->getInput(IOpos::IN_INP)->getDimensions();
    ASSERT(mInputDims[IOpos::IN_INP].nbDims == 4);
    mInputDims[IOpos::IN_INDS] = network->getInput(IOpos::IN_INDS)->getDimensions();
    ASSERT(mInputDims[IOpos::IN_INDS].nbDims == 2);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 4);

    return true;
}

//!
//! \brief Creates a network with a single custom layer containing the NonZero plugin and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the NonZero plugin
//!
//! \param builder Pointer to the engine builder
//!
bool SampleSliceAndBatchPlugin::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config)
{
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }


    auto* inp = network->addInput("inp", DataType::kFLOAT, mParams.inp_dims);
    auto* inds = network->addInput("inds", DataType::kINT32, mParams.inds_dims);

    sample::gLogInfo << "Added inputs to network" << std::endl;

    ASSERT(inp != nullptr && inds != nullptr);

    std::vector<PluginField> const vecPF{{"slice_size", &mParams.slice_size, PluginFieldType::kINT32, 5}};
    PluginFieldCollection pfc{static_cast<int32_t>(vecPF.size()), vecPF.data()};

    auto creator = getPluginRegistry()->getPluginCreator("slice_and_batch_nhwc", "1");
    IPluginV2 *pluginObj = creator->createPlugin("slice_and_batch_nhwc", &pfc);

    sample::gLogInfo << "Plugin got created" << std::endl;

    std::vector<ITensor*> inputsVec{inp, inds};
    auto layer = network->addPluginV2(inputsVec.data(), 2, *pluginObj);
    ASSERT(layer != nullptr);
    ASSERT(layer->getInput(0) != nullptr);
    ASSERT(layer->getInput(1) != nullptr);
    ASSERT(layer->getOutput(0) != nullptr);

    layer->getOutput(0)->setName("slices");

    network->markOutput(*(layer->getOutput(0)));

    // Destroy the plugin object
    pluginObj->destroy();

    sample::gLogInfo << "Plugin added to network" << std::endl;

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleSliceAndBatchPlugin::infer()
{

    // Since the data dependent output size cannot be inferred from the engine denote a sufficient size for the
    // corresponding output buffer (along with the rest of the I/O tensors)
    int64_t inp_size = mInputDims[0].d[0] * mInputDims[0].d[1] * mInputDims[0].d[2] * mInputDims[0].d[3];
    int64_t inds_size = mInputDims[1].d[0] * mInputDims[1].d[1];
    int64_t outp_size = mInputDims[1].d[0] * mInputDims[0].d[3] * mParams.slice_size * mParams.slice_size;
    std::vector<int64_t> ioVolumes = {inp_size, inds_size, outp_size};

    sample::gLogInfo << "Buffers:" << ioVolumes[0] << " " << ioVolumes[1] << " " << ioVolumes[2] << std::endl;


    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; ++i)
    {
        auto const name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 2);
    if (!processInput(buffers))
    {
        return false;
    }

    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    bool status = context->enqueueV3(stream);
    if (!status)
    {
        return false;
    }

    // Asynchronously copy data from device output buffers to host output buffers.
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete.
    CHECK(cudaStreamSynchronize(stream));

    // Release stream.
    CHECK(cudaStreamDestroy(stream));

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleSliceAndBatchPlugin::processInput(samplesCommon::BufferManager const& buffers)
{
    std::default_random_engine generator(mSeed);
    std::uniform_int_distribution<int64_t> distr(0, 100);

    sample::gLogInfo << mParams.inputTensorNames[0] << ":" << std::endl;
    float* inpBuf = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    auto& d0 = mParams.inp_dims.d;
    for (int32_t n = 0; n < d0[0]; ++n)
    {
        for (int32_t h = 0; h < d0[1]; ++h)
        {
            for (int32_t w = 0; w < d0[2]; ++w)
            {
                for (int32_t c = 0; c < d0[3]; ++c){
                    auto idx = n*d0[1]*d0[2]*d0[3] + h*d0[2]*d0[3] + w*d0[3] + c;
                    inpBuf[idx] = distr(generator);
                    sample::gLogInfo << inpBuf[idx] << ", ";
                }
            }
            sample::gLogInfo << std::endl;
        }
    }

    int32_t* indsBuf = static_cast<int32_t*>(buffers.getHostBuffer(mParams.inputTensorNames[1]));
    ASSERT(indsBuf != nullptr);
    sample::gLogInfo << mParams.inputTensorNames[1] << ":" << std::endl;
    indsBuf[0] = 0; indsBuf[1] = 0; indsBuf[2] = 0;
    indsBuf[3] = 0; indsBuf[4] = 2; indsBuf[5] = 2;

    return true;
}

//!
//! \brief Verify result
//!
bool SampleSliceAndBatchPlugin::verifyOutput(samplesCommon::BufferManager const& buffers)
{
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    int64_t od[4] = { mInputDims[1].d[0], mInputDims[0].d[3], mParams.slice_size, mParams.slice_size};
    sample::gLogInfo << "Output:" << std::endl;
    for (int32_t n = 0; n < od[0]; ++n){
        for (int32_t c = 0; c < od[1]; ++c){
            for (int32_t h = 0; h < od[2]; ++h){
                for (int32_t w = 0; w < od[3]; ++w){
                    auto idx = n * od[1] * od[2] * od[3] + c * od[2] * od[3] + h * od[3] + w;
                    sample::gLogInfo << output[idx] << ", ";
                }
                sample::gLogInfo << std::endl;
            }
        }
    }

    return true;
}

REGISTER_TENSORRT_PLUGIN(SliceAndBatchPluginCreator);
