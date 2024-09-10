#!/bin/bash

rm -f test_logs.txt
touch test_logs.txt

OUT_DIR=streval_dirs/cp_pp_valo

#BASELINES
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel0075_res3d_centerpoint_trt.yaml" \
#    CKPT_FILE="../models/cbgs_voxel0075_centerpoint_nds_6648.pth" \
#    FINE_GRAINED_EVAL=1 CALIB_DEADLINE_MILLISEC=200 ./run_tests.sh ros2 4 10.0
#
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_voxel01_res3d_centerpoint_trt.yaml" \
#    CKPT_FILE="../models/cbgs_voxel01_centerpoint_nds_6454.pth" \
#    FINE_GRAINED_EVAL=1 CALIB_DEADLINE_MILLISEC=150 ./run_tests.sh ros2 4 10.0
#
#CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint_trt.yaml" \
#    CKPT_FILE="../models/cbgs_pp_centerpoint_nds6070.pth" \
#    FINE_GRAINED_EVAL=1 CALIB_DEADLINE_MILLISEC=100 ./run_tests.sh ros2 4 10.0
#

for DL in $(seq 65 10 115)
do
	CFG_FILE="./cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint_valo.yaml" \
	    CKPT_FILE="../models/cbgs_pp_centerpoint_nds6070.pth" \
	    FINE_GRAINED_EVAL=1 CALIB_DEADLINE_MILLISEC=$DL ./run_tests.sh ros2 4 10.0
done
mv eval_data_* $OUT_DIR
python streval_from_files.py $OUT_DIR
mv segment_precision_info.json $OUT_DIR
python fine_grained_mAP_calc.py $OUT_DIR
