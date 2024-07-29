#!/bin/bash

inp="spatial_features"
#1x128x360x24
#1x128x360x312
#1x128x360x360

pushd deploy_files_valo
TRT_PATH="/home/humble/shared/libraries/TensorRT-10.1.0.27"
LD_LIBRARY_PATH=$TRT_PATH/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH  \
	$TRT_PATH/bin/trtexec --onnx=./dsvt2.onnx  --saveEngine=./dsvt2.engine \
    --minShapes=${inp}:1x128x360x24 --optShapes=${inp}:1x128x360x312  \
    --maxShapes=${inp}:1x128x360x360 --stronglyTyped --consistency
#	--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT,-EDGE_MASK_CONVOLUTIONS,-JIT_CONVOLUTIONS \
#	--memPoolSize=tacticSharedMem:8G
popd
