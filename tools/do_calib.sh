#!/bin/bash
. nusc_sh_utils.sh

unset OMP_NUM_THREADS USE_ALV1 TASKSET CFG_FILE CKPT_FILE DEADLINE_RANGE_MS DATASET_RANGE
export CALIBRATION=1
#export DEADLINE_RANGE_MS="70-200"
#export DEADLINE_RANGE_MS="90-350"
export DATASET_PERIOD="500"
#./nusc_dataset_prep.sh
link_data 500
NUM_TILES=18

export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml"
export CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth"
if ! [ -f calib_data_m4_c"$NUM_TILES".json ]; then
	./run_tests.sh singlemt 4 100.0 $NUM_TILES
	rm -f calib_data_m8_c"$NUM_TILES".json calib_data_m9_c"$NUM_TILES".json
	ln -s calib_data_m4_c"$NUM_TILES".json calib_data_m8_c"$NUM_TILES".json
	ln -s calib_data_m4_c"$NUM_TILES".json calib_data_m9_c"$NUM_TILES".json
fi

if ! [ -f calib_data_m5_c"$NUM_TILES".json ]; then
	./run_tests.sh singlemt 5 100.0 $NUM_TILES
fi

export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_voxelnext_anytime.yaml"
export CKPT_FILE="../models/voxelnext_nuscenes_kernel1.pth"
for m in 6 7; do
	if ! [ -f calib_data_m"$m"_c"$NUM_TILES".json ]; then
		./run_tests.sh singlemt $m 100.0 $NUM_TILES
	fi
done

export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_v1.yaml"
export CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_v1.pth"
if ! [ -f calib_dict_NuScenesDataset_m10.json ]; then
	TASKSET="taskset 0x3f" OMP_NUM_THREADS=2 USE_ALV1=1 ./run_tests.sh singlem 10
fi

# this one doesn't support 18 tiles but 16
export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel01_res3d_centerpoint_anytime_16.yaml"
export CKPT_FILE="../models/cbgs_voxel01_res3d_centerpoint_anytime_16.pth"
if ! [ -f calib_data_m11_c16.json ]; then
	./run_tests.sh singlemt 11 100.0 16
fi

export CFG_FILE="./cfgs/nuscenes_models/dsvt_anytime.yaml"
export CKPT_FILE="../models/DSVT_Nuscenes_val.pth"
if ! [ -f calib_data_m13_c"$NUM_TILES".json ]; then
	./run_tests.sh singlemt 13 100.0 $NUM_TILES
fi

unset CFG_FILE CKPT_FILE NUM_TILES
printf "Calibration done.\n"
