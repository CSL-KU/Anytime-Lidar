export CALIBRATION=0
export DATASET_PERIOD=50
export PCDET_PATH="$HOME/shared/Anytime-Lidar"

for RES in "010" "015" "020" "024" "030"
do
  CMD="taskset fc chrt -r 90 python test2.py Pillarnet$RES 10.0"
  #$CMD "offline"
  $CMD "streaming" "noforecast"
done
