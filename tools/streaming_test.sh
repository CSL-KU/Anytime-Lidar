#!/bin/bash
#test_prep 50
mkdir -p $1

# Baseline
#for E2E_REL_DEADLINE_S in $(seq 0.200 0.100 0.500)
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
. nusc_sh_utils.sh
link_data 50
export CALIBRATION=0
export DO_EVAL=0
export PROJECTION_COEFF=1.0
MAX_BUDGET=0.250
for method in 3
do
#for PROJ_COEFF in $(seq 1.0 0.5 2.0)
#do
#	TARGET_DIR=$1_$PROJ_COEFF
	TARGET_DIR=$1
	for E2E_REL_DEADLINE_S in $(seq 0.200 0.100 0.500)
	do
		export E2E_REL_DEADLINE_S=$E2E_REL_DEADLINE_S
		for BUDGET in $(seq 0.150 0.050 $MAX_BUDGET)
		do
			if (( $(echo "$E2E_REL_DEADLINE_S < $BUDGET" | bc -l) )); then
				break
			fi
			fpath=$TARGET_DIR/s${E2E_REL_DEADLINE_S}_eval_dict_m${method}_d${BUDGET}0.json
			if [ -f $fpath ]; then
				printf "Skipping $fpath test.\n"
			else
				printf "Doing $fpath test.\n"
				./run_tests.sh singlemsp $method ${BUDGET}0 $PROJECTION_COEFF
				mv -f eval_dict_*.json $fpath
				fpath=$(echo $fpath | sed 's/json/pkl/g')
				mv -f 'eval.pkl' $fpath
			fi
		done
	done
#done
done
