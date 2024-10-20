#!/bin/bash

rm -f test_logs.txt
touch test_logs.txt


#BASELINES
CFG_FILES=( \
"./cfgs/nuscenes_models/cbgs_dyn_pillar0075_res2d_centerpoint_trt.yaml" \
"./cfgs/nuscenes_models/cbgs_dyn_pillar01_res2d_centerpoint_trt.yaml" \
"./cfgs/nuscenes_models/cbgs_dyn_pillar015_res2d_centerpoint_trt.yaml" \
"./cfgs/nuscenes_models/cbgs_dyn_pillar02_res2d_centerpoint_trt.yaml" \
"./cfgs/nuscenes_models/cbgs_dyn_pillar03_res2d_centerpoint_trt.yaml" \
)

CKPT_FILES=( \
"../models/cbgs_pillar0075_res2d_centerpoint_nds_6694.pth" \
"../models/cbgs_pillar01_res2d_centerpoint_nds_6585.pth" \
"../models/cbgs_pillar015_res2d_centerpoint_nds_6389.pth" \
"../models/cbgs_pillar02_res2d_centerpoint_nds_6150.pth" \
"../models/cbgs_pillar03_res2d_centerpoint_nds_5604.pth" \
)

DEADLINES=( 7.5 10.0 15.0 20.0 30.0 )

#for m in ${!CFG_FILES[@]}
#do
#    CFG_FILE=${CFG_FILES[$m]} CKPT_FILE=${CKPT_FILES[$m]} \
#            CALIB_DEADLINE_MILLISEC=${DEADLINES[$m]} ./run_tests.sh ros2 4 10.0
#done

OUT_DIR=results_streval/pillarnet015
for DL in $(seq 65 10 95)
do
    printf "DEADLINE $DL\n"
    m=2
    TMP=${CFG_FILES[$m]}
    CFG_FILE="${TMP/trt/valo}" CKPT_FILE=${CKPT_FILES[$m]} \
            CALIB_DEADLINE_MILLISEC=$DL ./run_tests.sh ros2 4 10.0
done

mkdir -p $OUT_DIR
mv eval_data_* $OUT_DIR
python streval_from_files.py $OUT_DIR
mv segment_precision_info.json $OUT_DIR
python fine_grained_mAP_calc.py $OUT_DIR
