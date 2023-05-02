#!/bin/bash

clear_data()
{
	pushd ../data/nuscenes/v1.0-mini
	rm -rf gt_database* *pkl
	popd
}

copy_data()
{
	clear_data
	data_path="./nusc_generated_data/$1/$2"
	echo "Copying from "$data_path
	cp -r $data_path/* ../data/nuscenes/v1.0-mini
}

gen_data()
{
	clear_data
	pushd ..
	python -m pcdet.datasets.nuscenes.nuscenes_dataset \
		--func create_nuscenes_infos \
		--cfg_file tools/cfgs/dataset_configs/nuscenes_mini_dataset.yaml \
		--version v1.0-mini
	popd
	sleep 1
}

link_tables_and_dicts()
{
	. nusc_revert_tables.sh
	. nusc_link_tables.sh $1/tables
	for f in token_to_anns.json token_to_pos.json
	do
		rm -f $f
		ln -s $1/$f
	done
}

test_prep()
{
	link_tables_and_dicts "nusc_tables_and_dicts/$1"
	cp splits.py.test $SPLITS_PY_DIR/splits.py
	gen_data
}

SPLITS_PY_DIR="/root/nuscenes-devkit/python-sdk/nuscenes/utils"

## CALIBRATION
link_tables_and_dicts "nusc_tables_and_dicts/250"
cp splits.py.calib $SPLITS_PY_DIR/splits.py
gen_data
./run_tests.sh calib

## TEST
#test_prep 250
#./run_tests.sh methods 0.250 -0.025 0.225
#test_prep 200
#./run_tests.sh methods 0.200 -0.025 0.175
#test_prep 150
#./run_tests.sh methods 0.150 -0.025 0.125
#test_prep 100
#./run_tests.sh methods 0.100 -0.025 0.075

##Plot
#for s in 0 1 2 3
#do
#	python3 log_plotter.py exp_data_nsc/ $s
#done
