import _init_path
import json
import pickle
import sys
import glob
import gc 

from nuscenes.nuscenes import NuScenes

# This script runs evaluation on the detection results and updates
# the eval dicts

def main():
    # load nusc object beforehand to speed up execution
    inp_dir = sys.argv[1]
    dataset_version = sys.argv[2] if len(sys.argv) > 2 else 'v1.0-trainval'
    root_path = "../data/nuscenes/" + dataset_version
    loaded_nusc = NuScenes(version=dataset_version, dataroot=root_path, verbose=True)
   
    eval_dict_paths = glob.glob(inp_dir + "/*.json")
    for ed_path in eval_dict_paths:
        with open(ed_path, 'r') as handle:
            eval_d = json.load(handle)

        if 'result_str' in eval_d['eval_results_dict']:
            continue # done already

        det_path = ed_path[:-4] + 'pkl'
        with open(det_path, 'rb') as f:
            det_data = pickle.load(f)

        dataset, det_annos, class_names, eval_metric, \
                final_output_dir, _, det_elapsed_musec = det_data
        result_str, result_dict = dataset.evaluation(
                det_annos, class_names,
                eval_metric=eval_metric,
                output_path=final_output_dir,
                loaded_nusc=loaded_nusc,
            #    nusc_annos_outp=nusc_annos,
            #   det_elapsed_musec=det_elapsed_musec,
        )
        print(result_str)

        eval_d['eval_results_dict'].update(result_dict)
        eval_d['eval_results_dict']['result_str'] = result_str

        with open(ed_path, 'w') as handle:
            json.dump(eval_d, handle, indent=4)
        
        gc.collect()

if __name__ == '__main__':
    main()

