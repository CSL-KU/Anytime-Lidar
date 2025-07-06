#!/bin/bash

pushd $PCDET_PATH/pcdet/trt_plugins/slice_and_batch_nhwc
mkdir -p build && cd build && cmake .. && make
popd

pushd $PCDET_PATH/data
ln -s ~/nuscenes
popd
