#!/bin/bash

inp="spatial_features"
#1x128x360x24
#1x128x360x312
#1x128x360x360

inp_onnx_path=$(realpath $1)
outp_engine_path=$(echo $inp_onnx_path | sed 's/\.onnx$/.engine/')

pushd ../deploy_files
#pushd deploy_files
TRT_PATH="/home/humble/shared/libraries/TensorRT-10.1.0.27"
LD_LIBRARY_PATH=$TRT_PATH/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH  \
	$TRT_PATH/bin/trtexec --onnx=$inp_onnx_path  --saveEngine=$outp_engine_path \
    --noTF32 --stronglyTyped --consistency --minShapes=${inp}:1x128x360x24 \
    --optShapes=${inp}:1x128x360x312 --maxShapes=${inp}:1x128x360x360
popd
