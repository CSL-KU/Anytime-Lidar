if [ -z $1 ]; then
    printf "Give cmd line arg, profile, methods, or slices"
    exit
fi

#export IGNORE_DL_MISS=0
export DO_EVAL=0
export E2E_REL_DEADLINE_S=0.0
export CALIBRATION=0
export OMP_NUM_THREADS=4
TASKSET="taskset -c 4-7"

PROF_CMD=""
if [ $1 == 'profile' ]; then
    PROF_CMD="nsys profile -w true \
            --trace cuda,nvtx \
            --process-scope=process-tree"
    # osrt and cudnn doesn't work :(
    #--sampling-trigger=timer,sched,cuda \

    # if want to trace stage2 only
    #NUM_SAMPLES=5
    #ARGS="$ARGS -c nvtx \
    #   --capture-range-end=repeat-shutdown:$NUM_SAMPLES \
    #   -p RPNstage2@* \
    #   -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
    #   --sampling-frequency=50000 --cuda-memory-usage=true"
fi


# Imprecise model
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_multihead_imprecise.yaml"
#CKPT_FILE="../models/cbgs_pp_multihead_imprecise.pth"

#SECOND CBGS
#CFG_FILE="./cfgs/nuscenes_models/cbgs_second_multihead.yaml"
#CKPT_FILE="../models/cbgs_second_multihead_nds6229_updated.pth"

# PointPillars Single Head
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_singlehead.yaml"
#CKPT_FILE="../models/cbgs_dyn_pp_singlehead/default/ckpt/checkpoint_epoch_20.pth"

#PointPillars Multihead
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_multihead_3br.yaml"
#CKPT_FILE="../models/pp_multihead_nds5823_updated.pth"

# Centerpoint-pointpillar
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml"
#CKPT_FILE="../models/cbgs_dyn_pp_centerpoint_12_5_data.pth"

# Centerpoint-pointpillar-anytime
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint_anytime_16x16.yaml"
#CKPT_FILE="../models/cbgs_dyn_pp_centerpoint_anytime_16x16_12_5_data.pth"

# Centerpoint-voxel01
#CFG_FILE="./cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint.yaml"
#CKPT_FILE="../models/cbgs_voxel01_centerpoint_nds_6454.pth"

# Centerpoint-voxel01-anytime
#CFG_FILE="./cfgs/nuscenes_models/cbgs_voxel01_res3d_centerpoint_anytime_16x16.yaml"
#CKPT_FILE="../models/cbgs_voxel01_centerpoint_anytime_16x16.pth"

# Centerpoint-voxel0075
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint.yaml"
#CKPT_FILE="../models/cbgs_voxel0075_centerpoint_5swipes.pth"

# Centerpoint-voxel0075-anytime-v2
CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml"
CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth"

# Centerpoint-voxel0075-anytime-v1
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_v1.yaml"
#CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_v1.pth"
#TASKSET="taskset 0x3f"
#export OMP_NUM_THREADS=2
#export USE_ALV1=1

# Centerpoint-KITTI-voxel
#CFG_FILE="./cfgs/kitti_models/centerpoint.yaml"
#CKPT_FILE="../models/centerpoint_kitti.pth"

# VoxelNeXt
#NDS:     0.5501 End-to-end,202.77,315.00,347.92,354.18,363.76,22.32
#CFG_FILE="./cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
#CFG_FILE="./cfgs/nuscenes_models/cbgs_voxel0075_voxelnext_anytime.yaml"
#CKPT_FILE="../models/voxelnext_nuscenes_kernel1.pth"

# PillarNet
#CFG_FILE="./cfgs/nuscenes_models/cbgs_pillar0075_res2d_centerpoint.yaml"
#CKPT_FILE="../models/cbgs_voxel0075_centerpoint_nds_6648.pth"

#DATASET="nuscenes_dataset.yaml"
#DATASET="nuscenes_mini_dataset.yaml"
#ARG="s/_BASE_CONFIG_: cfgs\/dataset_configs.*$"
#ARG=$ARG"/_BASE_CONFIG_: cfgs\/dataset_configs\/$DATASET/g"
#sed -i "$ARG" $CFG_FILE

#CMD="nice --20 $PROF_CMD $TASKSET python test.py --cfg_file=$CFG_FILE \
#   --ckpt $CKPT_FILE --batch_size=1 --workers 0"
CMD="chrt -r 90 $PROF_CMD $TASKSET python test.py --cfg_file=$CFG_FILE \
        --ckpt $CKPT_FILE --batch_size=1 --workers 0"

#export CUBLAS_WORKSPACE_CONFIG=":4096:2"
set -x
if [ $1 == 'profile' ]; then
    #export CUDA_LAUNCH_BLOCKING=1
    $CMD --set "MODEL.DEADLINE_SEC" $2
    #export CUDA_LAUNCH_BLOCKING=0
elif [ $1 == 'methods' ]; then
    export IGNORE_DL_MISS=0
    export DO_EVAL=0
    export E2E_REL_DEADLINE_S=0.0 # not streaming
    export CALIBRATION=0

    rm eval_dict_*
    OUT_DIR=exp_data_nsc
    mkdir -p $OUT_DIR

    CFG_FILES=( \
           "./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint.yaml" \
	   "./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_v1.yaml" \
	   "./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml" \
	   "./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml" \
	   "./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml")
    CKPT_FILES=( \
            "../models/cbgs_voxel0075_centerpoint_5swipes.pth" \
	    "../models/cbgs_voxel0075_res3d_centerpoint_anytime_v1.pth" \
            "../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth" \
            "../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth" \
	    "../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth")

    for m in ${!CFG_FILES[@]}
    do
        CFG_FILE=${CFG_FILES[$m]}
        CKPT_FILE=${CKPT_FILES[$m]}

	if [ $m == 1 ]; then
		# need to change test.py
		TSKST="taskset 0x3f"
		MTD=10
		export OMP_NUM_THREADS=2
		export USE_ALV1=1
	else
		TSKST="taskset -c 4-7"
		MTD=$m
		export OMP_NUM_THREADS=4
		export USE_ALV1=0
	fi	
        #CMD="nice --20 $TASKSET python test.py --cfg_file=$CFG_FILE \
        #   --ckpt $CKPT_FILE --batch_size=1 --workers 0"
	CMD="chrt -r 90 $TSKST python test.py --cfg_file=$CFG_FILE \
		--ckpt $CKPT_FILE --batch_size=1 --workers 0"

        for s in $(seq $2 $3 $4)
        do
            OUT_FILE=$OUT_DIR/eval_dict_m"$m"_d"$s".json
            if [ -f $OUT_FILE ]; then
                printf "Skipping $OUT_FILE test.\n"
            else
	        $CMD --set "MODEL.DEADLINE_SEC" $s "MODEL.METHOD" $MTD
                # rename the output and move the corresponding directory
		fpath=$OUT_DIR/eval_dict_m"$m"_d"$s".json
                mv -f eval_dict_*.json $fpath
		mv -f 'eval.pkl' $(echo $fpath | sed 's/json/pkl/g')
            fi
        done
    done
elif [ $1 == 'proj_calibm' ]; then
	export IGNORE_DL_MISS=1
	export DO_EVAL=0
	export E2E_REL_DEADLINE_S=0.0 # not streaming
	export CALIBRATION=1
	export USE_ALV1=0

	rm eval_dict_*
	OUT_DIR=exp_data_proj
	mkdir -p $OUT_DIR

	CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml"
	CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth"

	CMD="chrt -r 90 $TSKST python test.py --cfg_file=$CFG_FILE \
		--ckpt $CKPT_FILE --batch_size=1 --workers 0"

	. nusc_sh_utils.sh # for link_data
	for proj_coeff in $(seq 1.0 0.5 2.0)
	do
		for period in $(seq 200 100 500)
		do
			link_data $period
			for budget in $(seq 0.150 0.050 0.250)
			do
			    fpath=$OUT_DIR/eval_dict_m$2_b${budget}_p${period}_pc${proj_coeff}.json
			    if [ -f $fpath ]; then
				printf "Skipping $fpath test.\n"
			    else
				printf "Running $fpath test.\n"
				$CMD --set "MODEL.DEADLINE_SEC" $budget "MODEL.METHOD" $2 \
					"MODEL.PROJECTION_COEFF" $proj_coeff
				# rename the output and move the corresponding directory
                                mv -f eval_dict_*.json $fpath
                                mv -f 'eval.pkl' $(echo $fpath | sed 's/json/pkl/g')
			    fi
			done
		done
	done
elif [ $1 == 'single' ]; then
    $CMD  --set "MODEL.DEADLINE_SEC" $2
elif [ $1 == 'singlem' ]; then
    $CMD  --set "MODEL.METHOD" $2 "MODEL.DEADLINE_SEC" $3
elif [ $1 == 'singlems' ]; then
    $CMD  --set "MODEL.METHOD" $2 "MODEL.DEADLINE_SEC" $3 "MODEL.STREAMING_EVAL" True
elif [ $1 == 'singlemsp' ]; then
    $CMD  --set "MODEL.METHOD" $2 "MODEL.DEADLINE_SEC" $3 "MODEL.STREAMING_EVAL" True "MODEL.PROJECTION_COEFF" $4
elif [ $1 == 'calibm' ]; then
    export CALIBRATION=1
    $CMD  --set "MODEL.METHOD" $2
fi
