#!/bin/bash

rm token_to_*.json
./nusc_revert_tables.sh
for period in "50" "100" "150" "200" "250" "300" "350"
do
	TABLES_PATH="nusc_tables_and_dicts/$period"
	rm -rf $TABLES_PATH
	mkdir -p "$TABLES_PATH/tables"
	python nusc_dataset_utils.py populate_annos_v2 50
	mv -f sample.json sample_data.json instance.json \
		sample_annotation.json scene.json "$TABLES_PATH/tables"
	./nusc_link_tables.sh "$TABLES_PATH/tables"
	if [ $period != 50 ]; then
		python nusc_dataset_utils.py prune_annos $period
		mv -f sample.json sample_data.json instance.json \
			sample_annotation.json scene.json "$TABLES_PATH/tables"
	fi
	python nusc_dataset_utils.py generate_dicts
	mv -f token_to_anns.json token_to_pos.json $TABLES_PATH
	./nusc_revert_tables.sh
done
