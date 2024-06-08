#!/bin/bash
. nusc_sh_utils.sh
unset OMP_NUM_THREADS USE_ALV1 TASKSET CFG_FILE CKPT_FILE DEADLINE_RANGE_MS DATASET_RANGE

export CALIBRATION=0
#export DATASET_RANGE="0-150"
#export DEADLINE_RANGE_MS="70-200"
#export DEADLINE_RANGE_MS="90-350"
export DATASET_PERIOD="350"
#export DATASET_PERIOD="200"
#./nusc_dataset_prep.sh
#link_data 200
link_data 350 

#./run_tests.sh methods 0.070 0.0325 0.200
./run_tests.sh methods 0.090 0.065 0.350
python eval_from_files.py ./exp_data_nsc_methods

#./run_tests.sh methods_dyn
#python eval_from_files.py ./exp_data_nsc_methods_dyn
