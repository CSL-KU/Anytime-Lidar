#!/bin/bash
. nusc_sh_utils.sh
unset OMP_NUM_THREADS USE_ALV1 TASKSET CFG_FILE CKPT_FILE DEADLINE_RANGE_MS DATASET_RANGE
unset FIXED_RES_IDX

export IGNORE_DL_MISS=0
export DATASET_PERIOD=250
export OMP_NUM_THREADS=2

link_data 250

export CALIBRATION=0
# The three numbers following "methods" define the deadline range BGN STEP END
./run_tests.sh methods 0.050 0.050 0.250 Pillarnet
python eval_from_files.py ./exp_data_nsc_methods_Pillarnet
python log_plotter.py exp_data_nsc_methods_Pillarnet/ 0 Pillarnet
python log_plotter.py exp_data_nsc_methods_Pillarnet/ 1 Pillarnet

./run_tests.sh methods 0.040 0.020 0.120  PointpillarsCP
python eval_from_files.py ./exp_data_nsc_methods_PointpillarsCP
python log_plotter.py exp_data_nsc_methods_PointpillarsCP/ 0 PointpillarsCP
python log_plotter.py exp_data_nsc_methods_PointpillarsCP/ 1 PointpillarsCP
