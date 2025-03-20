#!/bin/bash
. nusc_sh_utils.sh
unset OMP_NUM_THREADS USE_ALV1 TASKSET CFG_FILE CKPT_FILE DEADLINE_RANGE_MS DATASET_RANGE
unset FIXED_RES_IDX

export IGNORE_DL_MISS=0
export DATASET_PERIOD=250
export OMP_NUM_THREADS=4

############CALIB START################
export CALIBRATION=1
#CFG_FILE="./cfgs/nuscenes_models/valo_pillarnet_0100.yaml"
#CKPT_FILE="../models/pillarnet0100_e20.pth"
#CFG_FILE=$CFG_FILE CKPT_FILE=$CKPT_FILE ./run_tests.sh singlem 5 10.0
CFG_FILE="./cfgs/nuscenes_models/mural_pillarnet_0100_0128_0200.yaml"
CKPT_FILE="../models/mural_pillarnet_0100_0128_0200_e20.pth"
for m in $(seq 6 11)
do
  CFG_FILE=$CFG_FILE CKPT_FILE=$CKPT_FILE ./run_tests.sh singlem $m 10.0
done
############CALIB END################a

export CALIBRATION=0
./run_tests.sh methods 0.060 0.020 0.140
python eval_from_files.py ./exp_data_nsc_methods
