#!/bin/bash
#test_prep 50
mkdir -p $1

# Baseline
for E2E_REL_DEADLINE_S in $(seq 0.200 0.050 0.500)
do
	export E2E_REL_DEADLINE_S=$E2E_REL_DEADLINE_S
	# no budget
	fpath=$1/s${E2E_REL_DEADLINE_S}_eval_dict_m0_d10.0000.json
	if [ -f $fpath ]; then
		printf "Skipping $fpath test.\n"
	else
		./run_tests.sh singlems 0 10.0000
		mv -f eval_dict_*.json $fpath
	fi
done

# ALv2-ARR
#MAX_BUDGET=0.240
#for E2E_REL_DEADLINE_S in $(seq 0.200 0.050 0.500)
#do
#	export E2E_REL_DEADLINE_S=$E2E_REL_DEADLINE_S
#	for BUDGET in $(seq 0.180 0.020 $MAX_BUDGET)
#	do
#		if (( $(echo "$E2E_REL_DEADLINE_S < $BUDGET" |bc -l) )); then
#			break
#		fi
#		fpath=$1/s${E2E_REL_DEADLINE_S}_eval_dict_m4_d${BUDGET}0.json
#		if [ -f $fpath ]; then
#			printf "Skipping $fpath test.\n"
#		else
#			printf "Doing $fpath test.\n"
#			./run_tests.sh singlems 4 ${BUDGET}0
#			mv -f eval_dict_*.json $fpath
#		fi
#	done
#done
