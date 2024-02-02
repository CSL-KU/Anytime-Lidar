#!/bin/bash
. nusc_sh_utils.sh
link_data 50
mkdir -p $1
# Baseline dyn sched
export DO_DYN_SCHED="1"
export E2E_REL_DEADLINE_S="0.5"
fpath=$1/$2_eval_dict_m0_d10.0000.json
if [ -f $fpath ]; then
	printf "Skipping $fpath test.\n"
else
	./run_tests.sh singlems 0 10.0000
	mv -f eval_dict_*.json $fpath
	fpath=$(echo $fpath | sed 's/json/pkl/g')
	mv -f 'eval.pkl' $fpath
fi

## Baseline no dyn sched
#export DO_DYN_SCHED="0"
#for E2E_REL_DEADLINE_S in $(seq 0.100 0.100 0.400)
#do
#	export E2E_REL_DEADLINE_S=$E2E_REL_DEADLINE_S
#	# no budget
#	fpath=$1/s${E2E_REL_DEADLINE_S}_eval_dict_m0_d10.0000.json
#	if [ -f $fpath ]; then
#		printf "Skipping $fpath test.\n"
#	else
#		./run_tests.sh singlems 0 10.0000
#		mv -f eval_dict_*.json $fpath
#		fpath=$(echo $fpath | sed 's/json/pkl/g')
#		mv -f 'eval.pkl' $fpath
#	fi
#done

## ALv2-ARR
#. nusc_sh_utils.sh
#link_data 50
#export CALIBRATION=0
#export DO_EVAL=0
#export PROJECTION_COEFF=2.0
#MAX_BUDGET=0.250
#for method in 2 3
#do
#	TARGET_DIR=$1"_method"$method
#	mkdir -p $TARGET_DIR
#	for E2E_REL_DEADLINE_S in $(seq 0.200 0.100 0.500)
#	do
#		export E2E_REL_DEADLINE_S=$E2E_REL_DEADLINE_S
#		for BUDGET in $(seq 0.150 0.050 $MAX_BUDGET)
#		do
#			if (( $(echo "$E2E_REL_DEADLINE_S < $BUDGET" | bc -l) )); then
#				break
#			fi
#			fpath=$TARGET_DIR/s${E2E_REL_DEADLINE_S}_eval_dict_m${method}_d${BUDGET}0.json
#			if [ -f $fpath ]; then
#				printf "Skipping $fpath test.\n"
#			else
#				printf "Doing $fpath test.\n"
#				./run_tests.sh singlemsp $method ${BUDGET}0 $PROJECTION_COEFF
#				mv -f eval_dict_*.json $fpath
#				fpath=$(echo $fpath | sed 's/json/pkl/g')
#				mv -f 'eval.pkl' $fpath
#			fi
#		done
#	done
#done
