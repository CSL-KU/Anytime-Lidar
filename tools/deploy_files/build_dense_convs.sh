#!/bin/bash


inp_onnx_path=$(realpath $1)
fname=$(echo $inp_onnx_path | awk -F'/' '{print $NF}')
fname_prefix=$(echo $fname | awk -F'.' '{print $1}')
outp_engine_path="./trt_engines/${PMODE}/${fname_prefix}.engine"
mkdir -p "./trt_engines/${PMODE}"

#inp="spatial_features"
TRT_PATH="/home/humble/shared/libraries/TensorRT-10.1.0.27"
LD_LIBRARY_PATH=$TRT_PATH/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH  \
	$TRT_PATH/bin/trtexec --onnx=$inp_onnx_path  --saveEngine=$outp_engine_path \
    --noTF32
