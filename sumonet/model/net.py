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
    """
    :param dataset: str - name of the dataset
    :return: - torch datasets - a train and test set
    """

    # Load the data
    if dataset == 'metabric':
        df = pycox.datasets.metabric.read_df()
    else:
        print('Dataset not found')
    cov, time, event = preprocess(df)

    # Turn into a torch Dataset and split into train and test
    full_dataset = SurvivalDataset(cov, time, event)
    train_size = int(np.floor(0.8 * len(full_dataset)))
    test_size = len(full_dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    return train_set, test_set


def scale(arr, scaling_type):
    """
    :param arr: np.ndarry - to be scaled
    :param scaling_type: - standard or minmax scaling
    :return: ndarray - scaled array
    """

    # Select the scaler from sklearn
    assert scaling_type in ['standard', 'minmax']
    if scaling_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    # Do the scaling
    arr = scaler.fit_transform(arr)

    return arr


def preprocess(df, scaling_type_cov='standard', scaling_type_time='standard'):
    """
    :param df: pd.dataframe - to be preprocessed
    :param scaling_type_cov: str - how to scale the covariates
    :param scaling_type_time:  str - how to scale the times
    :return: np.ndarray - cov, time, event
    """

    # Select the covariates, times, events
    cov = df.drop(['duration', 'event'], axis=1).to_numpy()
    time, event = df['duration'].to_numpy()[:, None], df['event'].to_numpy()[:, None]

    # Perform the scaling
    cov = scale(cov, scaling_type_cov)
    time = scale(time, scaling_type_time)

    return cov, time, event


class SurvivalDataset(torch.utils.data.Dataset):
    def __init__(self, cov, time, event):
        super().__init__()
        self.cov_dim = np.shape(cov)[1]
        self.cov = torch.from_numpy(cov)
        self.time = torch.from_numpy(time)
        self.event = torch.from_numpy(event)


    def __len__(self):
        return len(self.event)

    def __getitem__(self, idx):
        return self.cov[idx], self.time[idx], self.event[idx]


class CovNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_widths = self.config['cov_net_widths']
        self.linear_transforms = nn.ModuleList()
        self.activation = getattr(F, config['activation'])
        for input_size, output_size in zip(self.layer_widths, self.layer_widths[1:]):
            self.linear_transforms.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        for linear_transform in self.linear_transforms:
            x = self.activation(linear_transform(x))
        return x


def get_bounding_operation(bounding_operation_name):
    if bounding_operation_name == 'abs':
        return torch.abs
    elif bounding_operation_name == 'square':
        return torch.square


class BoundedLinear(nn.Linear):
    def __init__(self, input_size, output_size, bounding_operation_name):
        super().__init__(input_size, output_size)
        self.bounding_operation = get_bounding_operation(bounding_operation_name)
        self.bounded_weight = self.bounding_operation(self.weight)

    def forward(self, x):
        return F.linear(x, self.bounded_weight, self.bias)


class MixedLinear(nn.Linear):
    def __init__(self, input_size, output_size, bounding_operation_name):
        super().__init__(input_size, output_size)
        self.bounding_operation = get_bounding_operation(bounding_operation_name)
        self.bounded_weight = self.bounding_operation(self.weight[:, 0])
        self.unrestricted_weight = self.weight[:, 1:]

    def forward(self, x, t):
        print(f' weight {self.weight}, bounded weight {self.bounded_weight}, unrestricted weight, {self.unrestricted_weight}')
        return F.linear(x, self.unrestricted_weight, self.bias) + F.linear(t, self.bounded_weight)


class MixedNet(nn.Module):
    """
    The mixed net consists of first a MixedLinear layer, and then BoundedLinear layers.
    """
    def __init__(self, config):
        super().__init__()
        self.layer_widths = config['mixed_net_widths']


if __name__ == '__main__':
    # Initiate train, test set
    train_set, test_set = load_data('metabric')
    print(len(test_set))
    cov, time, event = train_set[1:3]
    print('dim', test_set[1, 0].shape[1])

    # Define the config file
    config = {'cov_net_widths': [9, 10, 1],
              'activation': 'tanh'
              }

    # Initiate the net
    net = CovNet(config)
    print(net(cov))
    print(net.linear_transforms)
    # Initiate the bounded_layer
    bounded_linear = BoundedLinear(9, 1, 'square')
    print(bounded_linear(cov))

    print(bounded_linear.weight)
    print(bounded_linear.bounded_weight)

    # Initiate the mixed layer
    mixed_linear = MixedLinear(10, 1, 'abs')
    print(mixed_linear(cov, time))