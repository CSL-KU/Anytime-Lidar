#!/bin/bash

inp_onnx_path=$(realpath $1)
fname=$(echo $inp_onnx_path | awk -F'/' '{print $NF}')
fname_prefix=$(echo $fname | awk -F'.' '{print $1}')
outp_engine_path="./trt_engines/${PMODE}/${fname_prefix}.engine"
mkdir -p "./trt_engines/${PMODE}"

# DSVT backbone
#MIN_SHAPE=1x128x360x24
#OPT_SHAPE=1x128x360x312
#MAX_SHAPE=1x128x360x360

# CenterPoint PillarNet 0.075
#MIN_SHAPE=1x256x192x8
#OPT_SHAPE=1x256x192x168
#MAX_SHAPE=1x256x192x192

# CenterPoint PillarNet 0.1
#MIN_SHAPE=1x256x144x8
#OPT_SHAPE=1x256x144x128
#MAX_SHAPE=1x256x144x144

# PillarNet 0.15
#MIN_SHAPE=1x256x96x6
#OPT_SHAPE=1x256x96x84
#MAX_SHAPE=1x256x96x96

# PillarNet 0.15 also 0.2 ?
#MIN_SHAPE=1x256x72x4
#OPT_SHAPE=1x256x72x64
#MAX_SHAPE=1x256x72x72

# PillarNet 0.2
MIN_SHAPE=1x256x72x4
OPT_SHAPE=1x256x72x64
MAX_SHAPE=1x256x72x72

# PillarNet 0.3
#MIN_SHAPE=1x256x48x6
#OPT_SHAPE=1x256x48x36
#MAX_SHAPE=1x256x48x48

# CenterPoint pp
#MIN_SHAPE=1x64x576x32
#OPT_SHAPE=1x64x576x512
#MAX_SHAPE=1x64x576x576

# CenterPoint inp
#inp="spatial_features"

#PillarNet inp
inp="x_conv4"

TRT_PATH="/home/humble/shared/libraries/TensorRT-10.1.0.27"
LD_LIBRARY_PATH=$TRT_PATH/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH  \
	$TRT_PATH/bin/trtexec --onnx=$inp_onnx_path  --saveEngine=$outp_engine_path \
    --noTF32 --minShapes=${inp}:$MIN_SHAPE \
    --optShapes=${inp}:$OPT_SHAPE --maxShapes=${inp}:$MAX_SHAPE \
	--staticPlugins=../../pcdet/trt_plugins/slice_and_batch_nhwc/build/libslice_and_batch_lib.so
