export PCDET_PATH="$HOME/shared/Anytime-Lidar"

CMD="taskset fc chrt -r 90 python test2.py"

MODEL=MURAL_0075_3res

export IGNORE_DL_MISS=1
export DATASET_PERIOD=50
mkdir -p sampled_dets
for C in 1 0
do
  export CALIBRATION=$C
  FIXED_RES_IDX=2 $CMD $MODEL 10.0 build_gt_database
  for RES in $(seq 0 2)
  do
    FIXED_RES_IDX=$RES $CMD $MODEL 10.0 streaming noforecast
  done
done
