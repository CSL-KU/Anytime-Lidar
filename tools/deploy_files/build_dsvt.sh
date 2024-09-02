#!/bin/bash

#polygraphy surgeon sanitize --fold-constants deploy_files/dsvt.onnx -o deploy_files/dsvt_folded.onnx

# 10 sweeps:
#max: [24987, 343, 355, 343, 355, 24987]
#mid: [14903, 232, 235, 232, 235, 14903]
#min: [4243, 92, 90, 92, 90, 4243]

# 5 sweeps:
#max: [18514, 274, 277, 274, 277, 18514]
#mid: [11788, 190, 205, 190, 205, 11788]
#min: [3883, 88, 85, 88, 85, 3883]

# nuscenes model:
m1=1
m2=1

o1=15217
o2=235
o3=246

e1=40000
e2=500

#[torch.Size([13613, 128]), 
#torch.Size([2, 217, 90]), 
#torch.Size([2, 222, 90]), 
#torch.Size([2, 217, 90]), 
#torch.Size([2, 222, 90]), 
#torch.Size([4, 2, 13613, 128])]

inp1="voxel_feat"
inp2="set_voxel_inds_tensor_shift_0"
inp3="set_voxel_inds_tensor_shift_1"
inp4="set_voxel_masks_tensor_shift_0"
inp5="set_voxel_masks_tensor_shift_1"
inp6="pos_embed_tensor"
#inp7="voxel_coords"

inp_onnx_path=$(realpath $1)
fname=$(echo $inp_onnx_path | awk -F'/' '{print $NF}')
fname_prefix=$(echo $fname | awk -F'.' '{print $1}')
outp_engine_path="./trt_engines/${PMODE}/${fname_prefix}.engine"
mkdir -p "./trt_engines/${PMODE}"

TRT_PATH="/home/humble/shared/libraries/TensorRT-10.1.0.27"
LD_LIBRARY_PATH=$TRT_PATH/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH  \
	$TRT_PATH/bin/trtexec --onnx=$inp_onnx_path  --saveEngine=$outp_engine_path --verbose \
	--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT,-EDGE_MASK_CONVOLUTIONS,-JIT_CONVOLUTIONS \
	--minShapes=${inp1}:${m1}x128,${inp2}:2x${m2}x90,${inp3}:2x${m2}x90,${inp4}:2x${m2}x90,${inp5}:2x${m2}x90,${inp6}:4x2x${m1}x128 \
	--optShapes=${inp1}:${o1}x128,${inp2}:2x${o2}x90,${inp3}:2x${o3}x90,${inp4}:2x${o2}x90,${inp5}:2x${o3}x90,${inp6}:4x2x${o1}x128 \
	--maxShapes=${inp1}:${e1}x128,${inp2}:2x${e2}x90,${inp3}:2x${e2}x90,${inp4}:2x${e2}x90,${inp5}:2x${e2}x90,${inp6}:4x2x${e1}x128 \
	--loadInputs=${inp1}:${inp1}.bin,${inp2}:${inp2}.bin,${inp3}:${inp3}.bin,${inp4}:${inp4}.bin,${inp5}:${inp5}.bin,${inp6}:${inp6}.bin \
	--noTF32 --stronglyTyped --consistency
#	--memPoolSize=tacticSharedMem:8G

# Using this plugin doesnt solve the problem
#	--staticPlugins=../../../libraries/IndexPutDeterministicTRT/build/libindex_put_lib.so \
