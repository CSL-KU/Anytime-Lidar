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

# CenterPoint 0.075
MIN_SHAPE=1x256x192x8
OPT_SHAPE=1x256x192x168
MAX_SHAPE=1x256x192x192

# CenterPoint 0.1
#MIN_SHAPE=1x256x144x8
#OPT_SHAPE=1x256x144x128
#MAX_SHAPE=1x256x144x144

# CenterPoint pp
#MIN_SHAPE=1x64x576x32
#OPT_SHAPE=1x64x576x512
#MAX_SHAPE=1x64x576x576

pushd ../deploy_files
#pushd deploy_files
TRT_PATH="/home/humble/shared/libraries/TensorRT-10.1.0.27"
LD_LIBRARY_PATH=$TRT_PATH/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH  \
	$TRT_PATH/bin/trtexec --onnx=$inp_onnx_path  --saveEngine=$outp_engine_path \
    --noTF32 --stronglyTyped --consistency --minShapes=${inp}:$MIN_SHAPE \
    --optShapes=${inp}:$OPT_SHAPE --maxShapes=${inp}:$MAX_SHAPE \
	--staticPlugins=../../pcdet/trt_plugins/slice_and_batch_nhwc/build/libslice_and_batch_lib.so
popd
