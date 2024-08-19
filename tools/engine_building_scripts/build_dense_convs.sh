#!/bin/bash

inp="spatial_features"

inp_onnx_path=$(realpath $1)
outp_engine_path=$(echo $inp_onnx_path | sed 's/\.onnx$/.engine/')

pushd ../deploy_files
#pushd deploy_files
TRT_PATH="/home/humble/shared/libraries/TensorRT-10.1.0.27"
LD_LIBRARY_PATH=$TRT_PATH/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH  \
	$TRT_PATH/bin/trtexec --onnx=$inp_onnx_path  --saveEngine=$outp_engine_path \
    --noTF32 --stronglyTyped --consistency
popd
