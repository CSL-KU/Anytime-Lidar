#!/bin/bash
if [ -z $1 ]; then
    printf "Give cmd line arg, profile, methods, or slices"
    exit
fi

. nusc_sh_utils.sh

export IGNORE_DL_MISS=${IGNORE_DL_MISS:-0}
export DO_EVAL=${DO_EVAL:-1}
export E2E_REL_DEADLINE_S=${E2E_REL_DEADLINE_S:-0.0}
export CALIBRATION=${CALIBRATION:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export TASKSET=${TASKSET:-"taskset -c 2-7"}
export USE_AMP=${USE_AMP:-"false"}

if [ -z $CFG_FILE ] && [ -z $CKPT_FILE ]; then
    #PointPillars Multihead *
    #CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_multihead_3br.yaml"
    #CKPT_FILE="../models/pp_multihead_nds5823_updated.pth"

    #SECOND CBGS *
    #CFG_FILE="./cfgs/nuscenes_models/cbgs_second_multihead.yaml"
    #CKPT_FILE="../models/cbgs_second_multihead_nds6229_updated.pth"
    
    # Centerpoint-pointpillar * 
    #CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint_trt.yaml"
    #CKPT_FILE="../models/cbgs_pp_centerpoint_nds6070.pth"
    
    # Centerpoint-voxel0075 *
    #CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_trt.yaml"
    #CKPT_FILE="../models/cbgs_voxel0075_centerpoint_nds_6648.pth"

    # Centerpoint-voxel01 *
    CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel01_res3d_centerpoint_trt.yaml"
    CKPT_FILE="../models/cbgs_voxel01_centerpoint_nds_6454.pth"

    # VoxelNeXt *
    #CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_voxelnext.yaml"
    #CKPT_FILE="../models/voxelnext_nuscenes_kernel1.pth"
 
    # PillarNet ??
    #CFG_FILE="./cfgs/nuscenes_models/cbgs_pillar0075_res2d_centerpoint.yaml"
    #CKPT_FILE="../models/.pth"

    ########################################
 
    
    # Centerpoint-voxel02
    #CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel02_res3d_centerpoint.yaml"
    #CFG_FILE="./cfgs/nuscenes_models/cbgs_voxel02_res3d_centerpoint.yaml"
    #CKPT_FILE="../models/cbgs_voxel02_centerpoint_5swipes.pth"
    
    # Centerpoint-voxel0075-VALO
    #CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml"
    #CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth"

    # Centerpoint-VALO-autoware
    #CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel01_res3d_centerpoint_anytime_aw.yaml"
    #CKPT_FILE="../models/cbgs_voxel01_res3d_centerpoint_anytime_aw_amp_5sweeps.pth"
    #CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_aw.yaml"
    ###############################3
    
    # Centerpoint-voxel0075-anytime-v1
    #CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_v1.yaml"
    #CKPT_FILE="../models/cbgs_voxel0075_res3d_centerpoint_anytime_v1.pth"
    #TASKSET="taskset 0x3f"
    #export OMP_NUM_THREADS=2
    #export USE_ALV1=1
    
    # Centerpoint-KITTI-voxel
    #CFG_FILE="./cfgs/kitti_models/centerpoint.yaml"
    #CKPT_FILE="../models/centerpoint_kitti.pth"
   
    #DSVT mymodel
    #CFG_FILE="./cfgs/nuscenes_models/dsvt_plain_1f_onestage_nusences_aw.yaml"
    #CFG_FILE="./cfgs/nuscenes_models/dsvt_plain_1f_onestage_nusences_aw_no_iou_valo.yaml"
    #CFG_FILE="./cfgs/nuscenes_models/dsvt_plain_1f_onestage_nusences_aw_no_iou.yaml"
    #CKPT_FILE="../models/dsvt_plain_1f_onestage_nusences_aw_no_iou_10sweeps.pth"
    #CKPT_FILE="../models/dsvt_plain_1f_onestage_nusences_aw_2_untrained.pth"

    #CFG_FILE="./cfgs/nuscenes_models/dsvt_plain_1f_onestage_nusc_chm_trt.yaml"
    #CKPT_FILE="../models/dsvt_plain_1f_onestage_nusc_chm_ep8.pth"

fi

#CMD="$PROF_CMD $TASKSET python test.py --cfg_file=$CFG_FILE \
#   --ckpt $CKPT_FILE --batch_size=32 --workers 0"
if [ $USE_AMP == 'true' ]; then
	AMP="--use_amp"
else
	AMP=""
fi

PROF_CMD="nsys profile -w true" # --trace cuda,nvtx"
CMD="python test.py --cfg_file=$CFG_FILE \
        --ckpt $CKPT_FILE --batch_size=1 --workers 0 $AMP"

#export CUBLAS_WORKSPACE_CONFIG=":4096:2"
set -x
if [ $1 == 'ros2' ]; then
    chrt -r 90 python inference_ros2.py --cfg_file=$CFG_FILE \
            --ckpt $CKPT_FILE --set "MODEL.METHOD" $2 "MODEL.DEADLINE_SEC" $3
elif [ $1 == 'ros2_nsys' ]; then
    chrt -r 90 $PROF_CMD python inference_ros2.py --cfg_file=$CFG_FILE \
            --ckpt $CKPT_FILE --set "MODEL.METHOD" $2 "MODEL.DEADLINE_SEC" $3
elif [ $1 == 'profilem' ]; then
    #export CUddDA_LAUNCH_BLOCKING=1
    $PROF_CMD $CMD --set "MODEL.METHOD" $2 "MODEL.DEADLINE_SEC" $3
    #export CUDA_LAUNCH_BLOCKING=0
elif [ $1 == 'methods' ] || [ $1 == 'methods_dyn' ]; then
    export IGNORE_DL_MISS=0
    export DO_EVAL=0
    export E2E_REL_DEADLINE_S=0.0 # not streaming
    export CALIBRATION=0

    rm eval_dict_*
    OUT_DIR=exp_data_nsc_${1}
    mkdir -p $OUT_DIR

    CFG_FILES=( \
	"./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint.yaml" \
	"./cfgs/nuscenes_models/cbgs_dyn_voxel01_res3d_centerpoint.yaml" \
	"./cfgs/nuscenes_models/cbgs_dyn_voxel02_res3d_centerpoint.yaml" \
	"./cfgs/nuscenes_models/cbgs_dyn_voxel0075_voxelnext.yaml" \
	"./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml" \
	"./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml" \
	"./cfgs/nuscenes_models/cbgs_dyn_voxel0075_voxelnext_anytime.yaml" \
	"./cfgs/nuscenes_models/cbgs_dyn_voxel0075_voxelnext_anytime.yaml" \
	"./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml" \
	"./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml" \
	"./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_v1.yaml" \
	"./cfgs/nuscenes_models/cbgs_dyn_voxel01_res3d_centerpoint_anytime_16.yaml" \
	"./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_anytime_18.yaml" \
	"./cfgs/nuscenes_models/dsvt_anytime.yaml" \
	"./cfgs/nuscenes_models/dsvt.yaml" )

    CKPT_FILES=( \
	"../models/cbgs_voxel0075_centerpoint_5swipes.pth" \
	"../models/cbgs_voxel01_centerpoint_5swipes.pth" \
	"../models/cbgs_voxel02_centerpoint_5swipes.pth" \
	"../models/voxelnext_nuscenes_kernel1.pth" \
	"../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth" \
	"../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth" \
	"../models/voxelnext_nuscenes_kernel1.pth" \
	"../models/voxelnext_nuscenes_kernel1.pth" \
	"../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth" \
	"../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth" \
	"../models/cbgs_voxel0075_res3d_centerpoint_anytime_v1.pth" \
	"../models/cbgs_voxel01_res3d_centerpoint_anytime_16.pth" \
	"../models/cbgs_voxel0075_res3d_centerpoint_anytime_18.pth"  \
	"../models/DSVT_Nuscenes_val.pth" \
	"../models/DSVT_Nuscenes_val.pth" )

    for m in ${!CFG_FILES[@]}
    do
        CFG_FILE=${CFG_FILES[$m]}
        CKPT_FILE=${CKPT_FILES[$m]}

	if [ $m == 10 ]; then
		# need to change test.py
		TSKST="taskset 0x3f"
		MTD=10
		export OMP_NUM_THREADS=2
		export USE_ALV1=1
	else
		TSKST="taskset -c 2-7"
		MTD=$m
		export OMP_NUM_THREADS=4
		export USE_ALV1=0
	fi	
        #CMD="nice --20 $TASKSET python test.py --cfg_file=$CFG_FILE \
        #   --ckpt $CKPT_FILE --batch_size=1 --workers 0"
	CMD="chrt -r 90 $TSKST python test.py --cfg_file=$CFG_FILE \
		--ckpt $CKPT_FILE --batch_size=1 --workers 0 $AMP"

	if [ $1 == 'methods' ]; then
		for s in $(seq $2 $3 $4)
		do
		    OUT_FILE=$OUT_DIR/eval_dict_m"$m"_d"$s".json
		    if [ -f $OUT_FILE ]; then
			printf "Skipping $OUT_FILE test.\n"
		    else
			$CMD --set "MODEL.DEADLINE_SEC" $s "MODEL.METHOD" $MTD
			# rename the output and move the corresponding directory
			mv -f eval_dict_*.json $OUT_FILE
			mv -f 'eval.pkl' $(echo $OUT_FILE | sed 's/json/pkl/g')
		    fi
		done
	else
	    OUT_FILE=$OUT_DIR/eval_dict_m"$m"_dyndl.json
	    if [ -f $OUT_FILE ]; then
		printf "Skipping $OUT_FILE test.\n"
	    else
		$CMD --set "MODEL.DEADLINE_SEC" 100.0 "MODEL.METHOD" $MTD
		# rename the output and move the corresponding directory
		mv -f eval_dict_*.json $OUT_FILE
		mv -f 'eval.pkl' $(echo $OUT_FILE | sed 's/json/pkl/g')
	    fi
	fi
    done
elif [ $1 == 'single' ]; then
    $CMD  --set "MODEL.DEADLINE_SEC" $2
elif [ $1 == 'singlem' ]; then
    $CMD  --set "MODEL.METHOD" $2 "MODEL.DEADLINE_SEC" $3
elif [ $1 == 'singlemt' ]; then
    $CMD  --set "MODEL.METHOD" $2 "MODEL.DEADLINE_SEC" $3 "MODEL.TILE_COUNT" $4
elif [ $1 == 'singlems' ]; then
    $CMD  --set "MODEL.METHOD" $2 "MODEL.DEADLINE_SEC" $3 "MODEL.STREAMING_EVAL" True
elif [ $1 == 'singlemsp' ]; then
    $CMD  --set "MODEL.METHOD" $2 "MODEL.DEADLINE_SEC" $3 "MODEL.STREAMING_EVAL" True "MODEL.PROJECTION_COEFF" $4
fi
