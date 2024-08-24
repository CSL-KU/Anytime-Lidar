#!/bin/bash

inp="spatial_features"
#1x128x360x24
#1x128x360x312
#1x128x360x360

inp_onnx_path=$(realpath $1)
outp_engine_path=$(echo $inp_onnx_path | sed 's/\.onnx$/.engine/')

# DSVT backbone
#MIN_SHAPE=1x128x360x24
#OPT_SHAPE=1x128x360x312
#MAX_SHAPE=1x128x360x360

# CenterPoint
MIN_SHAPE=1x256x180x10
OPT_SHAPE=1x256x180x144
MAX_SHAPE=1x256x180x180

pushd ../deploy_files
#pushd deploy_files
TRT_PATH="/home/humble/shared/libraries/TensorRT-10.1.0.27"
LD_LIBRARY_PATH=$TRT_PATH/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH  \
	$TRT_PATH/bin/trtexec --onnx=$inp_onnx_path  --saveEngine=$outp_engine_path \
    --noTF32 --stronglyTyped --consistency --minShapes=${inp}:$MIN_SHAPE \
    --optShapes=${inp}:$OPT_SHAPE --maxShapes=${inp}:$MAX_SHAPE
popd
