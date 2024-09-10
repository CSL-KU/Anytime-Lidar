import _init_path
import json
import pickle
import sys
import glob
import gc 
import os
import datetime

from nuscenes.nuscenes import NuScenes
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils


def get_dataset(cfg):
    log_file = ('./tmp_results/log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    #log_config_to_file(cfg, logger=logger)
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
        dist=False, workers=0, logger=logger, training=False
    )

    return logger, test_set

def main():
    # load nusc object beforehand to speed up execution
    inp_dir = sys.argv[1]
    dataset_version = sys.argv[2] if len(sys.argv) > 2 else 'v1.0-trainval'
    root_path = "../data/nuscenes/" + dataset_version
    loaded_nusc = NuScenes(version=dataset_version, dataroot=root_path, verbose=True)
   
    eval_dict_paths = glob.glob(inp_dir + "/eval_data_*.pkl")

    os.environ['FINE_GRAINED_EVAL'] = '1'
    for ed_path in eval_dict_paths:
        print('Loading eval dict for', ed_path)
        with open(ed_path, 'rb') as handle:
            eval_d = pickle.load(handle)

        if 'result_str' in eval_d:
            print('Skipping this one it is already evaluated')
            continue # done already

        cfg = eval_d['cfg']
        det_annos = eval_d['det_annos']
        annos_in_glob = eval_d['annos_in_glob_coords']
        calib_deadline_ms = eval_d['calib_deadline_ms']
        os.environ['CALIB_DEADLINE_MILLISEC'] = str(calib_deadline_ms)

        logger, dataset = get_dataset(cfg)
        try:
            result_str, result_dict = dataset.evaluation(
                det_annos, dataset.class_names,
                eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
                output_path='./tmp_results',
                loaded_nusc=loaded_nusc,
                #nusc_annos_outp=nusc_annos,
                boxes_in_global_coords=annos_in_glob,
                #det_elapsed_musec=det_elapsed_musec,
            )

            print(result_str)
            eval_d['result_str'] = result_str
            eval_d['result_dict'] = result_dict

            print('Dumping updated eval dict')
            with open(ed_path, 'wb') as f:
                pickle.dump(eval_d, f)
        except:
            print('Could not do the eval of ', ed_path)
        
        print('Collecting garbage')
        gc.collect()

if __name__ == '__main__':
    main()

