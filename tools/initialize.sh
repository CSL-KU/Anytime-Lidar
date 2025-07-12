#!/bin/bash
pushd $PCDET_PATH/data
ln -s ~/nuscenes
popd

pushd $PCDET_PATH/pcdet/trt_plugins/slice_and_batch_nhwc
mkdir -p build && cd build && cmake .. && make
popd

mkdir -p $PCDET_PATH/tools/deploy_files/onnx_files
mkdir -p $PCDET_PATH/tools/deploy_files/trt_engines/$PMODE
mkdir -p $PCDET_PATH/tools/calib_files
mkdir -p $PCDET_PATH/tools/../../latest_exp_plots
mkdir -p $PCDET_PATH/tools/../../calib_plots

export DATASET_PERIOD="250"
./nusc_dataset_prep.sh
