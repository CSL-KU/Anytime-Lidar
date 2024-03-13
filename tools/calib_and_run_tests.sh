#!/bin/bash
. nusc_sh_utils.sh
#
### CALIBRATION
#export CALIBRATION=1
##unset DEADLINE_RANGE_MS
#export DEADLINE_RANGE_MS="70-200"
#export DEADLINE_RANGE_MS="90-350"
#export DATASET_PERIOD="500"
##./nusc_dataset_prep.sh
#link_data 500
#export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml"
#export CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth"
#./run_tests.sh singlem 4 100.0
#./run_tests.sh singlem 5 100.0
#ln -s calib_data_m4_c18.json calib_data_m8_c18.json
#ln -s calib_data_m4_c18.json calib_data_m9_c18.json
#
#export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_voxelnext_anytime.yaml"
#export CKPT_FILE="../models/voxelnext_nuscenes_kernel1.pth"
#./run_tests.sh singlem 6 100.0
#./run_tests.sh singlem 7 100.0
#
#export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel01_res3d_centerpoint_anytime_16.yaml"
#export CKPT_FILE="../models/cbgs_voxel01_res3d_centerpoint_anytime_16.pth"
#./run_tests.sh singlem 11 100.0
#
#export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_v1.yaml"
#export CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_v1.pth"
#export TASKSET="taskset 0x3f"
#export OMP_NUM_THREADS=2
#export USE_ALV1=1
#./run_tests.sh calibm 10
unset OMP_NUM_THREADS USE_ALV1 TASKSET CFG_FILE CKPT_FILE DEADLINE_RANGE_MS DATASET_RANGE

export CALIBRATION=0
#export DATASET_RANGE="0-150"
export DEADLINE_RANGE_MS="70-200"
#export DEADLINE_RANGE_MS="90-350"
#unset DEADLINE_RANGE_MS
export DATASET_PERIOD="200"
#./nusc_dataset_prep.sh
link_data 200
#link_data 350 
./run_tests.sh methods_dyn
#./run_tests.sh methods 0.070 0.0325 0.200
#./run_tests.sh methods 0.090 0.065 0.350
python eval_from_files.py ./exp_data_nsc_methods_dyn
#python eval_from_files.py ./exp_data_nsc_methods

