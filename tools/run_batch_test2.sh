export PCDET_PATH="$HOME/shared/Anytime-Lidar"

CMD="taskset fc chrt -r 90 python test2.py"

export DATASET_PERIOD=50
export OMP_NUM_THREADS=4

MODEL=MURAL_0100_0128_0200

run_baselines()
{
  export CALIBRATION=0
  $CMD Pillarnet0100 10.0 streaming noforecast
  $CMD Pillarnet0128 10.0 streaming noforecast
  $CMD Pillarnet0200 10.0 streaming noforecast
}

calibrate()
{
  export IGNORE_DL_MISS=0  # for calibration
  CALIBRATION=1 FIXED_RES_IDX=-1 $CMD $MODEL 10.0 offline noforecast
}

run_dynamicres()
{
  export IGNORE_DL_MISS=1
  CALIBRATION=0 FIXED_RES_IDX=-1 $CMD $MODEL 10.0 streaming noforecast
}

# Create dataset
build_dataset()
{
  mkdir -p sampled_dets
  export IGNORE_DL_MISS=1
  for C in 1 0
  do
    export CALIBRATION=$C
    FIXED_RES_IDX=0 $CMD $MODEL 10.0 build_gt_database
    for RES in $(seq 0 5)
    do
      FIXED_RES_IDX=$RES $CMD $MODEL 10.0 streaming noforecast
    done
  done
}

run_baselines
