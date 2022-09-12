#!/usr/bin/python3

import json
import pickle
import sys
import csv
import numpy as np
import torch
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split

fnames = sys.argv[1:]

num_voxels = []
num_tiles = []
psched_time = []

for fname in fnames:
    with open(fname, 'r') as handle:
        eval_dict = json.load(handle)

    num_voxels.extend(eval_dict["num_voxels"])
    num_tiles.extend(eval_dict["num_tiles"])
    psched_time.extend(eval_dict["PostSched"])

# Define the neural network
#class PostSchedWCETPred(torch.nn.Module):
#    def __init__(self):
#        super(PostSchedWCETPred, self).__init__()
#        self.fc1 = torch.nn.Linear(2, 1)
#
#    def forward(self, x):
#        x = self.fc1(x)
#        return x

#https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40
class StandardScaler:
    def __init__(self, mean=0., std=0., epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. 
	The module does not expect the
        tensors to be of any specific shape; as long as the features are the last 
	dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. 
		The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = torch.tensor(mean, dtype=torch.float)
        self.std = torch.tensor(std, dtype=torch.float)
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, values):
        return values * (self.std + self.epsilon) + self.mean

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std= self.std.cuda()
        return self

    def get_params(self):
        return (self.mean, self.std, self.epsilon)
    
    def set_params(self, params):
        self.mean, self.std, self.epsilon = params

def network_arch_search(num_voxels, num_tiles, psched_time):
    archs = []
    
    archs.append(torch.nn.Linear(2, 1))
    archs.append(torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        torch.nn.ReLU())
    for n in range(1,10):
        archs.append(torch.nn.Sequential(
            torch.nn.Linear(2, n),
            torch.nn.ReLU(),
            torch.nn.Linear(n, 1))
    for n in range(1,10):
        archs.append(torch.nn.Sequential(
            torch.nn.Linear(2, n),
            torch.nn.ReLU(),
            torch.nn.Linear(n, 1),
            torch.nn.ReLU())

    all_stats = []
    for arch in archs:
        for epochs in (10000, 15000, 20000, 25000, 30000):
            stats = train_pred_wcet_net(num_voxels, num_tiles, psched_time,
                arch, epochs)
            all_stats.append(stats)

    print(all_stats)
        

@torch.enable_grad()
def train_pred_wcet_net(num_voxels, num_tiles, psched_time, arch, epochs=10000):
    inputs = torch.tensor([[nv, nt] for nv, nt in zip(num_voxels, num_tiles)], \
            dtype=torch.float)

    outputs = torch.tensor([[tm] for tm in psched_time], dtype=torch.float)

    #inputs_train, inputs_val, outputs_train, outputs_val = \
    #        train_test_split(inputs, outputs, test_size=0.1)

    pred_net_scaler_in = StandardScaler()
    inputs_n  = pred_net_scaler_in.fit_transform(inputs)
    #inputs_train_n  = torch.tensor(scaler_in.fit_transform(inputs_train), dtype=torch.float32)
    #inputs_val_n    = torch.tensor(scaler_in.fit_transform(inputs_val), dtype=torch.float32)

    #self.pred_net_scaler_out = StandardScaler()
    #outputs_n = torch.tensor(self.pred_net_scaler_out.fit_transform(outputs),
    #        dtype=torch.float)

    # Initialize the neural network
    print(arch)
    post_sched_pred_net = arch #PostSchedWCETPred()
    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(post_sched_pred_net.parameters(), lr=0.001)

    # Train the neural network
    for epoch in range(epochs):
        # Forward pass for training set
        predicted_outputs = post_sched_pred_net(inputs_n)
        loss_train = criterion(predicted_outputs, outputs)

        # Backward pass and optimization for training set
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # Print the losses every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss_train.item()}")

    # Test the neural network
    predicted_output = post_sched_pred_net(inputs_n)
    #predicted_output = self.pred_net_scaler_out.inverse_transform(\
    #        predicted_output_norm)
    x = torch.cat((inputs.int(), (predicted_output*1000).int()), dim=1)
    print(x[:200])
    diff = (predicted_output - outputs).flatten().detach().numpy()
    #print(diff)
    perc95 = np.percentile(diff, 95, method='lower')
    perc99 = np.percentile(diff, 99, method='lower')
    pred_net_time_stats = {
            'min': float(min(diff)), 
            'avrg': float(sum(diff)/len(diff)), 
            '95perc': float(perc95),
            '99perc': float(perc99),
            'max': float(max(diff))}
    print('Time prediction stats:')
    print(pred_net_time_stats)

    # Open a file and use dump()
    fname = f"calib_data.pkl"
    with open(fname, 'wb') as handle:
        # A new file will be created
        data = [
            post_sched_pred_net.state_dict(),
            pred_net_scaler_in.get_params(),
    #        self.pred_net_scaler_out,
            pred_net_time_stats]
        pickle.dump(data, handle)

    return pred_net_time_stats

#train_pred_wcet_net(num_voxels, num_tiles, psched_time)
network_arch_search(num_voxels, num_tiles, psched_time)
