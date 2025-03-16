#!/bin/bash
. nusc_sh_utils.sh
unset OMP_NUM_THREADS USE_ALV1 TASKSET CFG_FILE CKPT_FILE DEADLINE_RANGE_MS DATASET_RANGE
unset FIXED_RES_IDX

export IGNORE_DL_MISS=0
export DATASET_PERIOD=250
export OMP_NUM_THREADS=4

############CALIB START################
#export CALIBRATION=1
#CFG_FILE="./cfgs/nuscenes_models/valo_pillarnet_0100.yaml"
#CKPT_FILE="../models/pillarnet0100_e20.pth"
#CFG_FILE=$CFG_FILE CKPT_FILE=$CKPT_FILE ./run_tests.sh singlem 5 10.0

#CFG_FILE="./cfgs/nuscenes_models/mural_pillarnet_0100_4res.yaml"
#CKPT_FILE="../models/mural_pillarnet_0100_4res_e20.pth"
#CFG_FILE=$CFG_FILE CKPT_FILE=$CKPT_FILE ./run_tests.sh singlem 6 10.0
#CFG_FILE=$CFG_FILE CKPT_FILE=$CKPT_FILE ./run_tests.sh singlem 7 10.0
#CFG_FILE=$CFG_FILE CKPT_FILE=$CKPT_FILE ./run_tests.sh singlem 8 10.0
#CFG_FILE=$CFG_FILE CKPT_FILE=$CKPT_FILE ./run_tests.sh singlem 9 10.0
############CALIB END################

export CALIBRATION=0
./run_tests.sh methods 0.050 0.050 0.250
python eval_from_files.py ./exp_data_nsc_methods
