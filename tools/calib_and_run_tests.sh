#!/bin/bash
. nusc_sh_utils.sh

## CALIBRATION
#link_data 500
#./run_tests.sh calibm 2

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

#period is always 350
#./run_tests.sh calibm 9
#./run_tests.sh calibm 2
#link_data 350
#for period_ms in $(seq $start_period_ms -$step_ms $end_period_ms); do
#	period_s=$(echo "$period_ms / 1000.0" | bc -l)
#	./run_tests.sh methods $(printf %.3f $period_s) -0.050 \
#		$(printf %.3f $(echo "$period_s - $step_s + 0.001" | bc -l))
#done
#python eval_from_files.py ./exp_data_nsc

#. streaming_test.sh streaming_eval_ss_cp100
. streaming_test.sh streaming_eval_ss_cp075
