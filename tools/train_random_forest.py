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
    exec_times_ms = eval_d['exec_times_ms']
    egovel_inds = [i for i, ev in enumerate(egovels) if ev is not None]
    data_tuples = []

    # start from 5 since first 5 sample are calibration
    for idx in range(5, len(egovel_inds)-1):
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

        vel_10p, vel_90p, vel_99p = np.percentile(obj_velos, [10, 90, 99])
        vel_mean = np.mean(obj_velos)

        rvel_10p, rvel_90p, rvel_99p = np.percentile(rel_velos, [10, 90, 99])
        rvel_mean = np.mean(rel_velos)

        vel_data = np.array([np.linalg.norm(ev), vel_10p, vel_mean, vel_90p, vel_99p,
                rvel_10p, rvel_mean, rvel_90p, rvel_99p])
        labels = pred_dict['pred_labels']
        num_objs_dist = np.bincount(labels.numpy()-1, minlength=num_class)
        etime = exec_times_ms[egovel_inds[idx]]
        assert etime is not None
        data_tuple = np.concatenate((vel_data, num_objs_dist, [num_objs_dist.sum(), etime]))
        data_tuples.append(data_tuple)

    data_tuples = np.array(data_tuples)
    return (main_stats, data_tuples)

def get_best_res(evals, metric='mAP'):
    accs = np.array([e[metric] for e in evals])
    max_idx = np.argmax(accs)
    return max_idx, accs[max_idx]
    
numres = 5
num_calibs_scenes = 75
merge_evals=False

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
window_length=1
for evals in all_evals:
    best_res, mAP = get_best_res(evals)
    mAP_stats[global_best_res] += 1
    #best_res, NDS = get_best_res(evals, 'NDS')
    tuples = evals[global_best_res]['tuples']

    if window_length > 1:
        new_tuples = []
        for i in range(tuples.shape[0] - (window_length) + 1):
            new_tuples.append(tuples[i:(i+window_length)].ravel())
        tuples = np.stack(new_tuples)

    if mAP_stats[best_res] % 5 == 0: # make 80% train 20% test
        all_test_inputs.append(tuples)
        all_test_labels.append(np.full(tuples.shape[0], best_res))
    else:
        all_train_inputs.append(tuples)
        all_train_labels.append(np.full(tuples.shape[0], best_res))

    mAP_stats[best_res] += 1

print('Best resolution stats:')
print(mAP_stats)

X_train = np.concatenate(all_train_inputs, axis=0).astype(float)
X_test = np.concatenate(all_test_inputs, axis=0).astype(float)
y_train = np.concatenate(all_train_labels, axis=0).astype(int)
y_test = np.concatenate(all_test_labels, axis=0).astype(int)

#Filter nans
mask = np.logical_not(np.isnan(X_train).any(1))
X_train = X_train[mask]
y_train = y_train[mask]
mask = np.logical_not(np.isnan(X_test).any(1))
X_test = X_test[mask]
y_test = y_test[mask]

# Using relvel 90th percentile, num cars, num peds, num barriers, num tcones works ok!
feature_names = ['ev', 'ov10p', 'ovmean', 'ov90p', 'ov99p',
                 'rv10p', 'rvmean', 'rv90p', 'rv99p',
                 'car','truck', 'construction_vehicle', 'bus', 'trailer',
                 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
                 'num_all_objs', 'exec_time_ms']

do_masking = True
if do_masking:
    features_to_keep = ['rv99p', 'car', 'pedestrian', 'num_all_objs', 'exec_time_ms']
    mask = np.array([feature_names.index(f) for f in features_to_keep])
    mask = np.concatenate([mask+i*len(feature_names) for i in range(window_length)])
    print('mask:', mask)
    X_train = X_train[:, mask]
    X_test = X_test[:, mask]

#    newfeat1 = np.expand_dims(X_train[:, 1] / X_train[:, 3], 1)
#    X_train = np.concatenate((X_train, newfeat1), axis=1)
#    newfeat1 = np.expand_dims(X_test[:, 1] / X_test[:, 3], 1)
#    X_test = np.concatenate((X_test, newfeat1), axis=1)


print('Train data shapes and labels distribution:')
print(X_train.shape, y_train.shape, np.bincount(y_train))
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
use_RFECV = False
use_grid_search = False
if use_RFECV:
    # Perform RFECV on all features
    from sklearn.feature_selection import RFECV
    base_rf = RandomForestClassifier(n_estimators=64, random_state=40)
    rf_classifier= RFECV(
        estimator=base_rf,
        step=1,
        cv=5,
        scoring='accuracy'
    )
elif use_grid_search:
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'n_estimators': [64, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=40),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    rf_classifier = grid_search

else:
    rf_classifier = RandomForestClassifier(
        n_estimators=64,  # number of trees
        max_depth=10,    # maximum depth of trees
        max_features='sqrt',
        min_samples_split=10,
        min_samples_leaf=2,
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

if use_RFECV:
    print("Optimal number of features:", rf_classifier.n_features_)
    print("Selected features:", [feature_names[i] for i in range(len(feature_names)) if rf_classifier.support_[i]])
else:
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
