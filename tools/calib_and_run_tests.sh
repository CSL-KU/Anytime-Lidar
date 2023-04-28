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

link_tables_and_dicts "nusc_tables_and_dicts/250"

## CALIBRATION
str='mini_train, mini_val = mini_val, mini_train'
sed_str_calib='s/#'$str'/'$str'/g'
sed --follow-symlinks -i "$sed_str_calib" splits.py # calib
copy_data 250 calib
./run_tests.sh calib
#python find_correlation.py calib_raw_data.json

## TEST
#sed_str_test='s/'$str'/#'$str'/g'
#sed --follow-symlinks -i "$sed_str_test" splits.py # test
#copy_data 250 test

## Test 150 ms
#./run_tests.sh methods 0.450 -0.050 0.200
#
## Gen test data 100ms
#link_tables_and_dicts "nusc_tables_and_dicts/100"
#gen_data
#
## Test 100 ms
#./run_tests.sh methods 0.100 -0.010 0.050
#
##Plot
#for s in 0 1 2 3
#do
#	python3 log_plotter.py exp_data_nsc/ $s
#done
