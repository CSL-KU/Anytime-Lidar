#!/bin/bash
. nusc_sh_utils.sh

unset OMP_NUM_THREADS USE_ALV1 TASKSET CFG_FILE CKPT_FILE DEADLINE_RANGE_MS DATASET_RANGE
export CALIBRATION=1
export DATASET_PERIOD="250"
#./nusc_dataset_prep.sh
link_data 250

############CALIB START################
export CALIBRATION=1
CFG_FILE="./cfgs/nuscenes_models/valo_pillarnet_0100.yaml"
CKPT_FILE="../models/pillarnet0100_e20.pth"
for m in 5 13
do
	CFG_FILE=$CFG_FILE CKPT_FILE=$CKPT_FILE ./run_tests.sh singlem $m 10.0
done

CFG_FILE="./cfgs/nuscenes_models/mural_pillarnet_0100_0128_0200.yaml"
CKPT_FILE="../models/mural_pillarnet_0100_0128_0200_e20.pth"
for m in $(seq 6 12)
do
	CFG_FILE=$CFG_FILE CKPT_FILE=$CKPT_FILE ./run_tests.sh singlem $m 10.0
done

CFG_FILE="./cfgs/nuscenes_models/valo_pointpillars_cp_0200.yaml"
CKPT_FILE="../models/PointPillarsCP0200_e20.pth"
for m in 5 13
do
	CFG_FILE=$CFG_FILE CKPT_FILE=$CKPT_FILE ./run_tests.sh singlem $m 10.0
done

CFG_FILE="./cfgs/nuscenes_models/mural_pp_centerpoint_0200_0256_0400.yaml"
CKPT_FILE="../models/mural_pp_centerpoint_0200_0256_0400_e20.pth"
for m in $(seq 6 12)
do
	CFG_FILE=$CFG_FILE CKPT_FILE=$CKPT_FILE ./run_tests.sh singlem $m 10.0
done
############CALIB END################a

printf "Calibration done.\n"
