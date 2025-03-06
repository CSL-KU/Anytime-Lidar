export PCDET_PATH="$HOME/shared/Anytime-Lidar"

CMD="taskset fc chrt -r 90 python test2.py"

#For calib:
#export CALIBRATION=1
#export DATASET_PERIOD=100
#link_data 100
#FIXED_RES_IDX='-1' $CMD VALOR 10.0 offline forecast
#return

export IGNORE_DL_MISS=1
#export CALIBRATION=0
export DATASET_PERIOD=50
#link_data "50.valcalib_50"
for C in 0 1
do
  export CALIBRATION=$C
  for RES in $(seq 0 2)
  do
    FIXED_RES_IDX=$RES $CMD MURAL 10.0 streaming noforecast
  done
done
return

#FIXED_RES_IDX=2 $CMD MURAL 10.0 offline noforecast
#$CMD MURAL 10.0 streaming noforecast
#return

#for DP in 300 200 100
#do
#  export DATASET_PERIOD=$DP
#  if [ ! -d nusc_tables_and_dicts/${DP}.val150 ]; then
#    ./nusc_dataset_prep.sh
#    cd nusc_tables_and_dicts
#    mv $DP ${DP}.val150
#    cd ..
#  fi
#  link_data ${DP}.val150
#  DL="0.${DP}"
#  # Make sure test2.py uses periodic as the sched_algo
#  FIXED_RES_IDX=-1 $CMD VALOR $DL streaming forecast
#  for RES in $(seq 0 4)
#  do
#    FIXED_RES_IDX=$RES $CMD VALOR $DL streaming forecast
#  done
#done

##DATASET GEN
export DATASET_PERIOD=50
for C in $(seq 85 150)
do
  export CALIBRATION=$C
  link_data "50.valcalib_$C"
  python ./eval_utils/gen_frames_meta.py # have to do it for tracking
  for RES in $(seq 0 4)
  do
    FIXED_RES_IDX=$RES $CMD VALOR 10.0 streaming noforecast
  done
#  for DL in 300 200 100
#  do
#    FIXED_RES_IDX=-1 $CMD VALOR 0.$DL streaming forecast
#  done
done

#for RES in $(seq 0 4)
#do
#  FIXED_RES_IDX=$RES $CMD VALOR 10.0 streaming forecast
#done

##############################################
##############################################
##############################################
##############################################
#for RES in "010" "015" "020" "024" "030"
#do
#  CMD="taskset fc chrt -r 90 python test2.py Pillarnet$RES 10.0"
#  #$CMD "offline"
#  $CMD "streaming" "noforecast"
#done
