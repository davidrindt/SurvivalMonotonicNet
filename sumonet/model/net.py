import numpy as np
from ray import tune
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch.optim as optim
import os
from functools import partial
import pycox
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.transforms import ToTensor


totensor = ToTensor()


def load_data(dataset):

    if dataset == 'metabric':
        df = pycox.datasets.metabric.read_df()
    else:
        print('Dataset not found')

    cov, time, event = preprocess(df)

    full_dataset = MyDataset(cov, time, event)
    train_size = int(np.floor(0.8 * len(full_dataset)))
    test_size = len(full_dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    return train_set, test_set


def scale(arr, scaling_type):

    assert scaling_type in ['standard', 'minmax']

    if scaling_type == 'standard':
        scaler = StandardScaler()

    else:
        scaler = MinMaxScaler()

    df = scaler.fit_transform(arr)

    return df

def preprocess(df, scaling_type_cov='standard', scaling_type_time='standard'):

    cov = df.drop(['duration', 'event'], axis=1).to_numpy()
    time, event = df['duration'].to_numpy()[:, None], df['event'].to_numpy()[:, None]

    cov = scale(cov, scaling_type_cov)
    time = scale(time, scaling_type_time)

    return cov, time, event


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, cov, time, event):
        self.cov = torch.from_numpy(cov)
        self.time = torch.from_numpy(time)
        self.event = torch.from_numpy(event)

    def __len__(self):
        return len(self.event)

    def __getitem__(self, idx):
        return self.cov[idx], self.time[idx], self.event[idx]


class Net(nn.Module):
    def __init__(self, config):
        self.config = config
        self.layers = self.config['layers']
        self.hidden = nn.ModuleList()
        self.activation = getattr(nn, config['activation'])()
        for input_size, output_size in zip(self.layers, self.layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))


    def forward(self, x):
        return None


if __name__ == '__main__':

    train_set, test_set = load_data('metabric')
    print(len(train_set))
    print(len(test_set))
    print(test_set[1])