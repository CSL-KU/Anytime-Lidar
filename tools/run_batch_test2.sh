export PCDET_PATH="$HOME/shared/Anytime-Lidar"
export IGNORE_DL_MISS=1

CMD="taskset fc chrt -r 90 python test2.py"

#For calib:
#export CALIBRATION=1
#export DATASET_PERIOD=100
#link_data 100
#FIXED_RES_IDX='-1' $CMD VALOR 10.0 offline forecast
#return

export CALIBRATION=0
export DATASET_PERIOD=50
#link_data 50.val75

FIXED_RES_IDX=-1 $CMD VALOR 10.0 streaming forecast

##DATASET GEN
export DATASET_PERIOD=50
for C in $(seq 1 75)
do
  export CALIBRATION=$C
  link_data "50.valcalib_$C"
  for RES in $(seq 0 4)
  do
    FIXED_RES_IDX=$RES $CMD VALOR 10.0 streaming forecast
  done
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
