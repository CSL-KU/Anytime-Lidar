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
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

std::string const kSAMPLE_NAME = "TensorRT.slice_and_batch_nhwc";

//REGISTER_TENSORRT_PLUGIN(SliceAndBatchPluginCreator);

//!
//! \brief Initializes members of the params struct using the command line args
//!
SliceAndBatchParams initializeSampleParams(samplesCommon::Args const& args)
{
    SliceAndBatchParams params;

    params.inputTensorNames.push_back("inp");
    params.inputTensorNames.push_back("inds");
    params.outputTensorNames.push_back("slices");
    params.fp16 = args.runInFp16;
//    params.dummy = args.rowOrder;

//    std::default_random_engine generator(static_cast<uint32_t>(time(nullptr)));
//    std::uniform_int_distribution<int64_t> distr(30000, 60000);
//    params.src_numv = distr(generator);
//    std::uniform_int_distribution<int64_t> distr2(10000, params.src_numv);
//    params.dst_numv = distr2(generator);
//    params.C_dim = 128;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_non_zero_plugin [-h or --help] [-d or --datadir=<path to data directory>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    sample::gLogger.setReportableSeverity(ILogger::Severity::kVERBOSE);

    auto sampleTest = sample::gLogger.defineTest(kSAMPLE_NAME, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    auto params = initializeSampleParams(args);
    SampleSliceAndBatchPlugin sample(params);

    sample::gLogInfo << "Building and running a GPU inference engine for SliceAndBatch plugin" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

//    std::cout << std::endl << "Src numv:" << params.src_numv << " Dst numv:"
//                << params.dst_numv << " C:" << params.C_dim << std::endl;

    return sample::gLogger.reportPass(sampleTest);
}
