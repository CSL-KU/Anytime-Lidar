#!/usr/bin/python3
import glob
import pickle
import numpy as np
from alive_progress import alive_bar
import concurrent.futures
import time

numres = 5
num_calibs_scenes = 100
#merge_evals=True
merge_evals=False
NUM_BINS=10

def read_data(pth):
    global NUM_BINS
    MAX_CAR_VEL = 15
    MAX_REL_VEL = 2*MAX_CAR_VEL
    with open(pth, 'rb') as f:
        eval_d = pickle.load(f)

    result_str = eval_d['result_str'].split('\n')

    cls_AP_scores = []
    for i, line in enumerate(result_str):
        if line[:3] == "***":
            cls = line.split(' ')[0][3:]
            mean_cls_AP = float(result_str[i+1].split(' ')[-1])
            cls_AP_scores.append(mean_cls_AP)

    mAP = float(result_str[-3:][0].split(' ')[-1])
    NDS = float(result_str[-3:][1].split(' ')[-1])

    calibxdotpkl = pth.split('_')[-1]
    calib_id = int(calibxdotpkl.split('.')[0][5:])
    res = int(eval_d['resolution'])
    main_stats = np.array([calib_id, res, mAP, NDS, *cls_AP_scores])

    num_class = 10
    
    # dataset can be generated only with global best resolution objects
    det_objects = eval_d['objects']
    egovels = eval_d['egovels']
    exec_times_ms = eval_d['exec_times_ms']
    egovel_inds = [i for i, ev in enumerate(egovels) if ev is not None]
    data_tuples = []

    for idx in range(1, len(egovel_inds)-1):
        time_tpl = exec_times_ms[egovel_inds[idx]]
        if time_tpl is None:
            continue
        etime, sim_time_ms = time_tpl
        if sim_time_ms < 800:
            continue # avoid first few samples since their input was not full point cloud

        ev = egovels[egovel_inds[idx]]
        if np.isnan(ev).any():
            continue
        pred_dict = det_objects[egovel_inds[idx+1]] #[0]
        if pred_dict is None:
            continue
        pred_dict = pred_dict[0]
        boxes = pred_dict['pred_boxes']
        if boxes.size(0) == 0:
            continue
        
        obj_velos = boxes[:, 7:9].numpy()
        mask = np.logical_not(np.isnan(obj_velos).any(1))
        obj_velos = obj_velos[mask]
        rel_velos = obj_velos - ev
        obj_velos = np.linalg.norm(obj_velos, axis=1)
        rel_velos = np.linalg.norm(rel_velos, axis=1)

        objvel_dist = np.bincount((obj_velos/MAX_CAR_VEL*NUM_BINS).astype(int),
                                  minlength=NUM_BINS)[:NUM_BINS]
        relvel_dist = np.bincount((rel_velos/MAX_REL_VEL*NUM_BINS).astype(int),
                                  minlength=NUM_BINS)[:NUM_BINS]

        vel_data = np.concatenate(([np.linalg.norm(ev)], objvel_dist, relvel_dist))
        labels = pred_dict['pred_labels']
        num_objs_dist = np.bincount(labels.numpy()-1, minlength=num_class)

        obj_sizes = boxes[:, 3:6].numpy()
        obj_sizes = obj_sizes.prod(axis=1)

        obj_pos = boxes[:, :2].numpy()[mask]
        obj_pos = np.linalg.norm(obj_pos, axis=1)

        MAX_DIST = 82
        objpos_dist = np.bincount((obj_pos/MAX_DIST*NUM_BINS).astype(int),
                                  minlength=NUM_BINS)[:NUM_BINS]

        data_tuple = np.concatenate((vel_data, num_objs_dist, objpos_dist, [etime]))
        data_tuples.append(data_tuple)

    data_tuples = np.array(data_tuples)
    return (main_stats, data_tuples)

def get_all_res(evals, metric='mAP', normalize=False):
    if metric == 'class_AP_scores':
        class_AP_scores_all = np.array([e[metric] for e in evals])
        winners = np.zeros(5)
        for i in range(10):
            if np.count_nonzero(class_AP_scores_all[:, i]) > 0:
                winners[np.argmax(class_AP_scores_all[:, i])] += 1
        return winners
    else:
        return np.array([e[metric] for e in evals])

def get_best_res(evals, metric='mAP'):
    accs = get_all_res(evals, metric)
    max_idx = np.argmax(accs)
    return max_idx, accs[max_idx]

def mask_files(path_list, num_calibs_scenes):
    mask = [False] * len(path_list)
    for i, pth in enumerate(path_list):
        calib_id = int(pth.split('_')[-1].split('.')[0][5:])
        mask[i] = (calib_id <= num_calibs_scenes)
    return [pth for m, pth in zip(mask, path_list) if m]

if merge_evals:
    all_evals = [[None for r in range(numres)] for c in range(num_calibs_scenes)]
    eval_dict_paths = glob.glob("sampled_dets/res*.pkl")
    eval_dict_paths = mask_files(eval_dict_paths, num_calibs_scenes)
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
                              'class_AP_scores': main_stats[4:],
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
res_sel_stats_train, res_sel_stats_test = np.zeros(numres), np.zeros(numres)
# For each calibration scene
all_train_inputs, all_train_labels, all_train_soft_labels = [], [], []
all_test_inputs, all_test_labels, all_test_soft_labels = [], [], []
window_length=1
skipped_scenes = 0
for calib_id, evals in enumerate(all_evals):
    all_acc = get_all_res(evals)
    best_res, mAP = get_best_res(evals, 'mAP')
    #best_res2, NDS = get_best_res(evals, 'NDS')
    #if best_res != best_res2:
    #    skipped_scenes += 1
    #    continue
    tuples = evals[global_best_res]['tuples']

    if window_length > 1:
        new_tuples = []
        for i in range(tuples.shape[0] - (window_length) + 1):
            new_tuples.append(tuples[i:(i+window_length)].ravel())
        tuples = np.stack(new_tuples)

    #if res_sel_stats[best_res] % 5 == 0: # if want to do split of 80% to 20%
    #if (res_sel_stats_train + res_sel_stats_test)[best_res] % 2 == 0:
    if calib_id >= 75:
        #if res_sel_stats_test[best_res] < 9:
        all_test_inputs.append(tuples) #[:2])
        all_test_labels.append(np.full(tuples.shape[0], best_res))
        all_test_soft_labels.append(np.stack([all_acc for i in range(tuples.shape[0])]))
        res_sel_stats_test[best_res] += 1
    else:
        #if res_sel_stats_train[best_res] < 9:
        all_train_inputs.append(tuples)
        all_train_labels.append(np.full(tuples.shape[0], best_res))
        all_train_soft_labels.append(np.stack([all_acc for i in range(tuples.shape[0])]))
        res_sel_stats_train[best_res] += 1

print(f'Skipped {skipped_scenes} scenes')
print('Best resolution stats for train and test:')
print(res_sel_stats_train)
print(res_sel_stats_test)
print('Num train scenes:', len(all_train_inputs))
print('Num test scenes:', len(all_test_inputs))
X_train = np.concatenate(all_train_inputs, axis=0).astype(float)
X_test = np.concatenate(all_test_inputs, axis=0).astype(float)
y_train = np.concatenate(all_train_labels, axis=0).astype(int)
y_test = np.concatenate(all_test_labels, axis=0).astype(int)
y_train_soft = np.concatenate(all_train_soft_labels, axis=0).astype(float)
y_test_soft = np.concatenate(all_test_soft_labels, axis=0).astype(float)

#Filter nans
mask = np.logical_not(np.isnan(X_train).any(1))
X_train = X_train[mask]
y_train = y_train[mask]
y_train_soft = y_train_soft[mask]
mask = np.logical_not(np.isnan(X_test).any(1))
X_test = X_test[mask]
y_test = y_test[mask]
y_test_soft = y_test_soft[mask]

# Using relvel 90th percentile, num cars, num peds, num barriers, num tcones works ok!
objveldist = [f'objvel_bin{i}' for i in range(NUM_BINS)]
relveldist = [f'relvel_bin{i}' for i in range(NUM_BINS)]
objposdist = [f'objpos_bin{i}' for i in range(NUM_BINS)]
obj_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
             'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
feature_names = ['ev', *objveldist, *relveldist, *obj_names, *objposdist, 'exec_time_ms']

#ci = feature_names.index('car')
#X_train = np.concatenate((X_train, X_train[:,ci:ci+10].sum(1, keepdims=True)),axis=1)
#X_test = np.concatenate((X_test, X_test[:,ci:ci+10].sum(1, keepdims=True)),axis=1)
#feature_names.append('num_obj')

do_masking = False
if do_masking:
    #features_to_keep = ['dsmean', 'exec_time_ms']
    features_to_keep = [*objveldist, *relveldist , 'exec_time_ms']
    #features_to_keep = [*relveldist, 'exec_time_ms', 'num_obj']
    mask = np.array([feature_names.index(f) for f in features_to_keep])
    mask = np.concatenate([mask+i*len(feature_names) for i in range(window_length)])
    print('mask:', mask)
    X_train = X_train[:, mask]
    X_test = X_test[:, mask]
    feature_names = features_to_keep

print('Chosen feature names:')
print(feature_names)

print('Train data shapes and labels distribution:')
print(X_train.shape, y_train.shape, np.bincount(y_train))
print('Test data shapes and labels distribution:')
print(X_test.shape, y_test.shape, np.bincount(y_test))

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
        n_estimators=200,  # number of trees
        max_depth=10,    # maximum depth of trees
        max_features='log2',
        min_samples_split=5,
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
for i in range(50):
    rf_classifier.predict(X_test[i:i+1])
t2 = time.monotonic()
tdiff = (t2 - t1) * 1000 / 50
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

import sys
#sys.exit()

print("Training soft label distribution:")
print(y_train_soft.mean(axis=0))
print("\nTest soft label distribution:")
print(y_test_soft.mean(axis=0))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FeatureNormalizer:
    def __init__(self):
        self.ego_vel_mean = None
        self.ego_vel_std = None
        self.dist_means = None
        self.dist_stds = None
        self.exec_time_mean = None
        self.exec_time_std = None
        
    def fit(self, features):
        # Ego velocity stats
        self.ego_vel_mean = torch.mean(features[:, 0])
        self.ego_vel_std = torch.std(features[:, 0]) + 1e-7
        
        # Reshape distributions and compute stats
        dist_features = features[:, 1:-1].reshape(len(features), -1, 10)
        self.dist_means = torch.mean(dist_features, dim=0)  # shape: (4, 10)
        self.dist_stds = torch.std(dist_features, dim=0) + 1e-7
        
        # Execution time stats
        self.exec_time_mean = torch.mean(features[:, -1])
        self.exec_time_std = torch.std(features[:, -1]) + 1e-7
        
    def transform(self, features):
        #if isinstance(features, np.ndarray):
        #    features = torch.from_numpy(features).float()
            
        normalized = torch.zeros_like(features)
        # Normalize ego velocity
        normalized[:, 0] = (features[:, 0] - self.ego_vel_mean) / self.ego_vel_std
        
        # Normalize distributions
        dist_features = features[:, 1:-1].reshape(len(features), -1, 10)
        for i in range(4):  # 4 distributions
            normalized[:, 1+i*10:1+(i+1)*10] = (dist_features[:, i] - self.dist_means[i]) / self.dist_stds[i]
        
        # Normalize execution time
        normalized[:, -1] = (features[:, -1] - self.exec_time_mean) / self.exec_time_std
        
        return normalized

class AttentionBlock(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, feature_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class ModelSelector(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input_size = 42  # 1 + 4*10 + 1 = 42 features
        
        # Attention for different feature groups
        self.ego_vel_attention = AttentionBlock(1)
        self.obj_vel_attention = AttentionBlock(10)
        self.rel_vel_attention = AttentionBlock(10)
        self.class_attention = AttentionBlock(10)
        self.distance_attention = AttentionBlock(10)
        self.exec_time_attention = AttentionBlock(1)
        
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 5)
        )
        
    def forward(self, x):
        # Split features and apply attention
        ego_vel = self.ego_vel_attention(x[:, 0:1])
        obj_vel = self.obj_vel_attention(x[:, 1:11])
        rel_vel = self.rel_vel_attention(x[:, 11:21])
        class_dist = self.class_attention(x[:, 21:31])
        distance = self.distance_attention(x[:, 31:41])
        exec_time = self.exec_time_attention(x[:, 41:42])
        
        # Concatenate all features
        x = torch.cat([ego_vel, obj_vel, rel_vel, class_dist, distance, exec_time], dim=1)
        return self.model(x)

class SceneDataset(Dataset):
    def __init__(self, features, labels, normalizer=None):
        self.features = features
        self.labels = labels
        self.normalizer = normalizer
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        if self.normalizer:
            x = self.normalizer.transform(x.reshape(1, -1)).squeeze(0)
        return x, self.labels[idx]

def train_model(X_train, X_test, y_train, y_test, num_epochs=100, batch_size=32):
    for t in (X_train, y_train, X_test, y_test):
        print(t.size(), t.dtype)
    # Initialize normalizer and fit on training data
    normalizer = FeatureNormalizer()
    normalizer.fit(X_train)

    # Create datasets
    train_dataset = SceneDataset(X_train, y_train, normalizer)
    test_dataset = SceneDataset(X_test, y_test, normalizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model and training components
    model = ModelSelector().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

    best_acc = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss #.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum() #.item()

        train_loss = train_loss.item()
        train_correct = train_correct.item()

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pt')

        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}')
            print(f'Test Acc: {test_acc:.4f}')

    return model, normalizer

tensors1 = [torch.from_numpy(t).float().cuda() for t in (X_train, X_test)]
tensors2 = [torch.from_numpy(t).long().cuda() for t in (y_train, y_test)]
torch.set_num_threads(8)
model, normalizer = train_model(*(tensors1 + tensors2))

# Example usage:
"""
# Assuming your data is prepared as:
# X_train: numpy array of shape (N, 42) - all features
# y_train: numpy array of shape (N,) - labels (0-4 for 5 models)
# X_test: numpy array of shape (M, 42)
# y_test: numpy array of shape (M,)

model, normalizer = train_model(X_train, y_train, X_test, y_test)

# For inference:
def predict(model, normalizer, features):
    model.eval()
    with torch.no_grad():
        normalized_features = normalizer.transform(features.reshape(1, -1))
        outputs = model(normalized_features)
        return torch.argmax(outputs).item() + 1  # +1 to get model ID 1-5
"""
