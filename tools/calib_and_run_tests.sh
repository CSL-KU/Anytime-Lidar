#!/bin/bash
. nusc_sh_utils.sh

## CALIBRATION
#export CALIBRATION=1
#link_data 250
#./run_tests.sh calibm 2

## TEST
export CALIBRATION=0
export DO_EVAL=0
#link_data 350
#./run_tests.sh methods 0.350 -0.050 0.301
#link_data 300
#./run_tests.sh methods 0.300 -0.050 0.251
#link_data 250
#./run_tests.sh methods 0.250 -0.050 0.201
#link_data 200
#./run_tests.sh methods 0.200 -0.050 0.151
#link_data 150
#./run_tests.sh methods 0.150 -0.050 0.101
#link_data 100
#./run_tests.sh methods 0.100 -0.050 0.051
link_data 50
. streaming_test.sh streaming_eval_ALv2_19scenes
