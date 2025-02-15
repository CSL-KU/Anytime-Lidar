#!/usr/bin/python3
import glob
import pickle
import numpy as np
from alive_progress import alive_bar
import concurrent.futures
import time

def read_data(pth):
    with open(pth, 'rb') as f:
        eval_d = pickle.load(f)

    mAP_NDS = eval_d['result_str'].split('\n')[-3:]
    mAP = float(mAP_NDS[0].split(' ')[-1])
    NDS = float(mAP_NDS[1].split(' ')[-1])

    calibxdotpkl = pth.split('_')[-1]
    calib_id = int(calibxdotpkl.split('.')[0][5:])
    res = int(eval_d['resolution'])
    main_stats = np.array([calib_id, res, mAP, NDS])

    num_class = 10
    
    # dataset can be generated only with global best resolution objects
    det_objects = eval_d['objects']
    egovels = eval_d['egovels']
    egovel_inds = [i for i, ev in enumerate(egovels) if ev is not None]
    data_tuples = []
    for idx in range(1, len(egovel_inds)-1):
        ev = egovels[egovel_inds[idx]]
        pred_dict = det_objects[egovel_inds[idx+1]] #[0]
        if pred_dict is None:
            continue
        pred_dict = pred_dict[0]
        boxes = pred_dict['pred_boxes']
        if boxes.size(0) == 0:
            continue
        
        obj_velos = boxes[:, 7:9].numpy()
        rel_velos = obj_velos - ev
        obj_velos = np.linalg.norm(obj_velos, axis=1)
        rel_velos = np.linalg.norm(rel_velos, axis=1)

        vel_10p = np.percentile(obj_velos, 10)
        vel_mean = np.mean(obj_velos)
        vel_90p = np.percentile(obj_velos, 90)
        vel_99p = np.percentile(obj_velos, 99)

        rvel_10p = np.percentile(rel_velos, 10)
        rvel_mean = np.mean(rel_velos)
        rvel_90p = np.percentile(rel_velos, 90)
        rvel_99p = np.percentile(rel_velos, 99)

        vel_data = np.array([np.linalg.norm(ev), vel_10p, vel_mean, vel_90p, vel_99p,
                rvel_10p, rvel_mean, rvel_90p, rvel_99p])
        labels = pred_dict['pred_labels']
        num_objs_dist = np.bincount(labels.numpy()-1, minlength=num_class)
        data_tuple = np.concatenate((vel_data, num_objs_dist, [num_objs_dist.sum()]))
        data_tuples.append(data_tuple)

    data_tuples = np.array(data_tuples)
    return (main_stats, data_tuples)

def get_best_res(evals, metric='mAP'):
    accs = np.array([e[metric] for e in evals])
    max_idx = np.argmax(accs)
    return max_idx, accs[max_idx]
    
numres = 5
num_calibs_scenes = 75
merge_evals=True

if merge_evals:
    all_evals = [[None for r in range(numres)] for c in range(num_calibs_scenes)]
    eval_dict_paths = glob.glob("sampled_dets/res*.pkl")
    num_procs = 8
    cur_idx = 0
    with alive_bar(len(eval_dict_paths), force_tty=True, max_cols=160, manual=True) as bar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as executor:
            while cur_idx < len(eval_dict_paths):
                paths = eval_dict_paths[cur_idx:(min(cur_idx+num_procs, len(eval_dict_paths)))]
                futs = [executor.submit(read_data, pth) for pth in paths]
                #for fut in futures:
                for fut in futs:
                    main_stats, data_tuples = fut.result()
                    calib_id = int(main_stats[0])
                    res = int(main_stats[1])
                    eval_d = {'mAP': main_stats[2],
                              'NDS': main_stats[3],
                              'tuples': data_tuples
                    }
                    all_evals[calib_id-1][res] = eval_d
                cur_idx += len(paths)
                bar(cur_idx/len(eval_dict_paths))

    with open('sampled_dets/dataset.pkl', 'wb') as f:
        pickle.dump(all_evals, f)
else:
    with open('sampled_dets/dataset.pkl', 'rb') as f:
        all_evals = pickle.load(f)

global_best_res = 2
mAP_stats = np.zeros(numres)
#NDS_stats = np.zeros(numres)
# For each calibration scene
all_train_inputs, all_train_labels = [], []
all_test_inputs, all_test_labels = [], []
for evals in all_evals:
    best_res, mAP = get_best_res(evals)
    mAP_stats[best_res] += 1
    #best_res, NDS = get_best_res(evals, 'NDS')
    tuples = evals[global_best_res]['tuples']

    if mAP_stats[best_res] % 5 == 0: # make 80% train 20% test 
        all_test_inputs.append(tuples)
        all_test_labels.append(np.full(tuples.shape[0], best_res))
    else:
        all_train_inputs.append(tuples)
        all_train_labels.append(np.full(tuples.shape[0], best_res))

    mAP_stats[best_res] += 1

print('Best resolution stats:')
print(mAP_stats)

X_train = np.concatenate(all_train_inputs, axis=0)
X_test = np.concatenate(all_test_inputs, axis=0)
y_train = np.concatenate(all_train_labels, axis=0)
y_test = np.concatenate(all_test_labels, axis=0)

# Using relvel 90th percentile, num cars, num peds, num barriers, num tcones works ok!
feature_names = ['ev', 'ov10p', 'ovmean', 'ov90p', 'ov99p',
                 'rv10p', 'rvmean', 'rv90p', 'rv99p',
                 'car','truck', 'construction_vehicle', 'bus', 'trailer',
                 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
                 'num_all_objs']

features_to_keep = ['rv90p', 'car', 'pedestrian'] #, 'num_all_objs'] #, 'traffic_cone']
mask = [feature_names.index(f) for f in features_to_keep]

print('mask:', mask)
X_train = X_train[:, mask]
X_test = X_test[:, mask]

print('Train data:')
print(X_train.shape, y_train.shape)
print('Test data:')
print(X_test.shape, y_test.shape)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Split the data into training and testing sets (80% train, 20% test)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X_train, X_test, y_train, y_test = X, X, y, y
# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier(
    n_estimators=64,  # number of trees
    max_depth=None,    # maximum depth of trees
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=40
)

# Train the model
rf_classifier.fit(X_train, y_train)
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)

# Make predictions on test set
y_pred = rf_classifier.predict(X_test)

#speed test
t1 = time.monotonic()
for i in range(100):
    rf_classifier.predict(X_test[i:i+1])
t2 = time.monotonic()
tdiff = (t2 - t1) * 1000 / 100
print('Inference time:', tdiff, 'ms')

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': range(X_train.shape[1]),
    'importance': rf_classifier.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

#from sklearn.model_selection import cross_val_score
#
## Perform 5-fold cross-validation
#cv_scores = cross_val_score(rf_classifier, X, y, cv=5)
#print("\nCross-validation scores:", cv_scores)
#print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


