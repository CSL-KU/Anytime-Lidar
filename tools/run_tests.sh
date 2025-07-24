#!/bin/bash
if [ -z $1 ]; then
    printf "Give cmd line arg, profile, methods, or slices"
    exit
fi

. nusc_sh_utils.sh

export IGNORE_DL_MISS=${IGNORE_DL_MISS:-0}
export DO_EVAL=${DO_EVAL:-1}
export CALIBRATION=${CALIBRATION:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export TASKSET=${TASKSET:-"taskset -c 2-7"}
export USE_AMP=${USE_AMP:-"false"}
export PMODE=${PMODE:-"UNKNOWN_POWER_MODE"}

mkdir -p deploy_files/trt_engines/$PMODE

if [ -z $CFG_FILE ] && [ -z $CKPT_FILE ]; then
    #CFG_FILE="./cfgs/nuscenes_models/mural_pillarnet_0100_4res.yaml"
    #CKPT_FILE="../models/mural_pillarnet_0100_4res_e20.pth"

    #CFG_FILE="./cfgs/nuscenes_models/mural_pillarnet_0100_0128_0200.yaml"
    #CKPT_FILE="../models/mural_pillarnet_0100_0128_0200_e20.pth"

    #CFG_FILE="./cfgs/nuscenes_models/mural_pp_centerpoint_0200_0256_0400.yaml"
    #CKPT_FILE="../models/mural_pp_centerpoint_0200_0256_0400_e20.pth"

    #CFG_FILE="./cfgs/nuscenes_models/valo_pillarnet_0100.yaml"
    #CKPT_FILE="../models/pillarnet0100_e20.pth"

    CFG_FILE="./cfgs/nuscenes_models/valo_pointpillars_cp_0200.yaml"
    CKPT_FILE="../models/PointPillarsCP0200_e20.pth"

    #CFG_FILE="./cfgs/nuscenes_models/pillarnet0100.yaml"
    #CKPT_FILE="../models/pillarnet0100_e20.pth"

    #CFG_FILE="./cfgs/nuscenes_models/PointPillarsCP0200.yaml"
    #CKPT_FILE="../models/PointPillarsCP0200_e20.pth"
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
            --ckpt $CKPT_FILE --set "MODEL.METHOD" 0 "MODEL.DEADLINE_SEC" 10.0
elif [ $1 == 'str_ros2' ]; then
    chrt -r 90 python strinf_ros2.py --cfg_file=$CFG_FILE \
            --ckpt $CKPT_FILE --set "MODEL.METHOD" 0 "MODEL.DEADLINE_SEC" 10.0
elif [ $1 == 'ros2_nsys' ]; then
    chrt -r 90 $PROF_CMD python inference_ros2.py --cfg_file=$CFG_FILE \
            --ckpt $CKPT_FILE --set "MODEL.METHOD" $2 "MODEL.DEADLINE_SEC" $3
elif [ $1 == 'profilem' ]; then
    #export CUDA_LAUNCH_BLOCKING=1
    $PROF_CMD $CMD --set "MODEL.METHOD" $2 "MODEL.DEADLINE_SEC" $3
    #export CUDA_LAUNCH_BLOCKING=0
elif [ $1 == 'methods' ] || [ $1 == 'methods_dyn' ]; then
  export IGNORE_DL_MISS=0
  export DO_EVAL=0
  export CALIBRATION=0

  rm eval_dict_*
  OUT_DIR=exp_data_nsc_${1}_${5}
  mkdir -p $OUT_DIR

  if [ $5 == "Pillarnet" ]; then
    CFG_FILES=( \
      "./cfgs/nuscenes_models/pillarnet0100.yaml"
      "./cfgs/nuscenes_models/pillarnet0128.yaml"
      "./cfgs/nuscenes_models/pillarnet0200.yaml"
      "dummy"
      "dummy"
      "./cfgs/nuscenes_models/valo_pillarnet_0100.yaml"
      "./cfgs/nuscenes_models/mural_pillarnet_0100_0128_0200.yaml"
      "./cfgs/nuscenes_models/mural_pillarnet_0100_0128_0200.yaml"
      "./cfgs/nuscenes_models/mural_pillarnet_0100_0128_0200.yaml"
      "./cfgs/nuscenes_models/mural_pillarnet_0100_0128_0200.yaml"
      "./cfgs/nuscenes_models/mural_pillarnet_0100_0128_0200.yaml"
      "dummy"
      "dummy"
      "./cfgs/nuscenes_models/valo_pillarnet_0100.yaml" )
  
    CKPT_FILES=( \
      "../models/pillarnet0100_e20.pth"
      "../models/pillarnet0128_e20.pth"
      "../models/pillarnet0200_e20.pth"
      "dummy"
      "dummy"
      "../models/pillarnet0100_e20.pth"
      "../models/mural_pillarnet_0100_0128_0200_e20.pth"
      "../models/mural_pillarnet_0100_0128_0200_e20.pth"
      "../models/mural_pillarnet_0100_0128_0200_e20.pth"
      "../models/mural_pillarnet_0100_0128_0200_e20.pth"
      "../models/mural_pillarnet_0100_0128_0200_e20.pth"
      "dummy"
      "dummy"
      "../models/pillarnet0100_e20.pth" )
  fi

  if [ $5 == "PointpillarsCP" ]; then
    CFG_FILES=( \
      "./cfgs/nuscenes_models/PointPillarsCP0200.yaml"
      "./cfgs/nuscenes_models/PointPillarsCP0256.yaml"
      "./cfgs/nuscenes_models/PointPillarsCP0400.yaml"
      "dummy"
      "dummy"
      "./cfgs/nuscenes_models/valo_pointpillars_cp_0200.yaml"
      "./cfgs/nuscenes_models/mural_pp_centerpoint_0200_0256_0400.yaml"
      "./cfgs/nuscenes_models/mural_pp_centerpoint_0200_0256_0400.yaml"
      "./cfgs/nuscenes_models/mural_pp_centerpoint_0200_0256_0400.yaml"
      "./cfgs/nuscenes_models/mural_pp_centerpoint_0200_0256_0400.yaml"
      "./cfgs/nuscenes_models/mural_pp_centerpoint_0200_0256_0400.yaml"
      "dummy"
      "dummy"
      "./cfgs/nuscenes_models/valo_pointpillars_cp_0200.yaml" )
  
    CKPT_FILES=( \
      "../models/PointPillarsCP0200_e20.pth"
      "../models/PointPillarsCP0256_e20.pth"
      "../models/PointPillarsCP0400_e20.pth"
      "dummy"
      "dummy"
      "../models/PointPillarsCP0200_e20.pth"
      "../models/mural_pp_centerpoint_0200_0256_0400_e20.pth"
      "../models/mural_pp_centerpoint_0200_0256_0400_e20.pth"
      "../models/mural_pp_centerpoint_0200_0256_0400_e20.pth"
      "../models/mural_pp_centerpoint_0200_0256_0400_e20.pth"
      "../models/mural_pp_centerpoint_0200_0256_0400_e20.pth"
      "dummy"
      "dummy"
      "../models/PointPillarsCP0200_e20.pth" )
  fi

  for m in ${!CFG_FILES[@]}
  do
    CFG_FILE=${CFG_FILES[$m]}
    CKPT_FILE=${CKPT_FILES[$m]}

    TSKST="taskset -c 2-7"
    MTD=$m

    if [ $CFG_FILE == "dummy" ]; then
      continue
    fi

    export OMP_NUM_THREADS=4
    export USE_ALV1=0
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
    chrt -r 90 $CMD  --set "MODEL.METHOD" $2 "MODEL.DEADLINE_SEC" $3 
elif [ $1 == 'singlev' ]; then
    $CMD  --set "MODEL.METHOD" 0 "MODEL.DEADLINE_SEC" 10.0 "DATA_CONFIG.DATA_PROCESSOR.1.VOXEL_SIZE" "[$2, $2, 0.2]"
fi
