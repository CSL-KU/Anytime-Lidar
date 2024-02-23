#!/bin/bash
. nusc_sh_utils.sh

## CALIBRATION
export CALIBRATION=1
unset DEADLINE_RANGE_MS
./nusc_dataset_prep.sh
link_data 500
#export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml"
#export CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth"
#./run_tests.sh calibm 2

export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel01_res3d_centerpoint_anytime_16.yaml"
export CKPT_FILE="../models/cbgs_voxel01_res3d_centerpoint_anytime_16.pth"
./run_tests.sh calibm 10

#export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_voxelnext_anytime.yaml"
#export CKPT_FILE="../models/voxelnext_nuscenes_kernel1.pth"
#./run_tests.sh calibm 9

#export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_v1.yaml"
#export CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_v1.pth"
#export TASKSET="taskset 0x3f"
#export OMP_NUM_THREADS=2
#export USE_ALV1=1
#./run_tests.sh calibm 10
unset OMP_NUM_THREADS USE_ALV1 CFG_FILE CKPT_FILE DATASET_RANGE

export CALIBRATION=0
export DATASET_RANGE="0-150"
export DEADLINE_RANGE_MS="50-200"
#export DEADLINE_RANGE_MS="100-350"
./nusc_dataset_prep.sh
link_data 500
#./run_tests.sh methods 0.350 -0.050 0.100
./run_tests.sh methods_dyn
python eval_from_files.py ./exp_data_nsc_methods_dyn


## TEST
#AGX Orin
#start_period_ms=300
#end_period_ms=50
#AGX Xavier
#start_period_ms=350
#end_period_ms=100

#step_ms=50 # better not change it
#step_s=$(echo "$step_ms / 1000.0" | bc -l)
##deadline == period
#for period_ms in $(seq $start_period_ms -$step_ms $end_period_ms); do
#	link_data $period_ms
#	period_s=$(echo "$period_ms / 1000.0" | bc -l)
#	./run_tests.sh methods $(printf %.3f $period_s) -0.050 \
#		$(printf %.3f $(echo "$period_s - $step_s + 0.001" | bc -l))
#	python eval_from_files.py ./exp_data_nsc
#done

#./run_tests.sh calibm 9
#./run_tests.sh calibm 2
#for period_ms in $(seq $start_period_ms -$step_ms $end_period_ms); do
#	period_s=$(echo "$period_ms / 1000.0" | bc -l)
#	./run_tests.sh methods $(printf %.3f $period_s) -0.050 \
#		$(printf %.3f $(echo "$period_s - $step_s + 0.001" | bc -l))
#done
#python eval_from_files.py ./exp_data_nsc

#. streaming_test.sh streaming_eval_ss_cp100
#. streaming_test.sh streaming_eval_ss_cp075
