#!/bin/bash
. nusc_sh_utils.sh

export DO_DYN_SCHED="1"
export DO_EVAL="1"
deadline=10.000
mkdir -p streaming_eval_res
rm -f eval_dict_*.json "eval.pkl"

for DATASET_SEL in $(seq 0 1); do
	export DATASET_SEL=$DATASET_SEL
	. nusc_dataset_prep.sh
	link_data 50

	CFG_FILES=( \
		"./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint.yaml" \
		"./cfgs/nuscenes_models/cbgs_dyn_voxel01_res3d_centerpoint.yaml" \
		"./cfgs/nuscenes_models/cbgs_dyn_voxel02_res3d_centerpoint.yaml")
	CKPT_FILES=( \
		"../models/cbgs_voxel0075_centerpoint_5swipes.pth" \
		"../models/cbgs_voxel01_centerpoint_5swipes.pth" \
		"../models/cbgs_voxel02_centerpoint_5swipes.pth")

	for m in ${!CFG_FILES[@]}; do
		export CFG_FILE=${CFG_FILES[$m]}
		export CKPT_FILE=${CKPT_FILES[$m]}
		prefix=$(echo $CFG_FILE | cut -d '_' -f 4)
		fpath="streaming_eval_res/d${DATASET_SEL}_${prefix}_eval_dict.json"
		if [ -f $fpath ]; then
			printf "Skipping $fpath test.\n"
		else
			./run_tests.sh singlems 2 $deadline
			mv -f eval_dict_*.json $fpath
			#fpath=$(echo $fpath | sed 's/json/pkl/g')
			#mv -f 'eval.pkl' $fpath
		fi
	done
	#python eval_from_files.py ./streaming_eval_res
done

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
