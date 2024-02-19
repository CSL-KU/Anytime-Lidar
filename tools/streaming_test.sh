#!/bin/bash
. nusc_sh_utils.sh

export DO_EVAL="1"
export CALIBRATION=0
export DATASET_PERIOD=50

mkdir -p streaming_eval_res
rm -f eval_dict_*.json "eval.pkl"

for DATASET_SEL in $(seq 0 4); do
	export DATASET_RANGE=$(($DATASET_SEL*30))-$(((DATASET_SEL+1)*30))
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

	export DO_DYN_SCHED="1"
	for m in ${!CFG_FILES[@]}; do
		export CFG_FILE=${CFG_FILES[$m]}
		export CKPT_FILE=${CKPT_FILES[$m]}
		prefix=$(echo $CFG_FILE | cut -d '_' -f 4)
		fpath="streaming_eval_res/d${DATASET_SEL}_${prefix}_eval_dict.json"
		if [ -f $fpath ]; then
			printf "Skipping $fpath test.\n"
		else
			./run_tests.sh singlems 2 10.000
			mv -f eval_dict_*.json $fpath
			#fpath=$(echo $fpath | sed 's/json/pkl/g')
			#mv -f 'eval.pkl' $fpath
		fi
	done

	# finally, evaluate the anytime model
	export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml"
	export CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth"

	#export DO_DYN_SCHED="0" # Later on try this
	for deadline in $(seq 0.075 0.025 0.150); do
		prefix=valo0075dl${deadline}
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

	export CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel01_res3d_centerpoint_anytime_16.yaml"
	export CKPT_FILE="../models/cbgs_voxel01_res3d_centerpoint_anytime_16.pth"
	for deadline in $(seq 0.075 0.025 0.100); do
		prefix=valo01dl${deadline}
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

	export DO_DYN_SCHED="0"
	export E2E_REL_DEADLINE_S=0.100
	prefix=valo01dl${deadline}nodyn
	fpath="streaming_eval_res/d${DATASET_SEL}_${prefix}_eval_dict.json"
	if [ -f $fpath ]; then
		printf "Skipping $fpath test.\n"
	else
		./run_tests.sh singlems 2 $deadline
		mv -f eval_dict_*.json $fpath
		#fpath=$(echo $fpath | sed 's/json/pkl/g')
		#mv -f 'eval.pkl' $fpath
	fi

	#python eval_from_files.py ./streaming_eval_res
done

unset CALIBRATION DATASET_PERIOD DATASET_RANGE CFG_FILE CKPT_FILE DO_DYN_SCHED
