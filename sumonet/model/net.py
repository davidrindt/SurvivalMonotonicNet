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
from sklearn.model_selection import train_test_split
import sklearn
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
    df = preprocess(df)

    # Split the dataset
    train, val, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
    train, val, test = SurvivalDataset(train), SurvivalDataset(val), SurvivalDataset(test)
    return train, val, test




def preprocess(df, scaling_type_cov='StandardScaler', scaling_type_time='StandardScaler'):
    """
    :param df: pd.dataframe - to be preprocessed
    :param scaling_type_cov: str - how to scale the covariates
    :param scaling_type_time:  str - how to scale the times
    :return: np.ndarray - cov, time, event
    """
    time_scaler = getattr(sklearn.preprocessing, scaling_type_time)()
    cov_scaler = getattr(sklearn.preprocessing, scaling_type_cov)()

    # Select the covariates, times, events
    for col in df.columns:
        if col == 'event':
            continue
        elif col == 'time':
            df[col] = time_scaler.fit_transform(df[col].to_numpy()[:, None])
        else:
            df[col] = cov_scaler.fit_transform(df[col].to_numpy()[:, None])

    return df


class SurvivalDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.cov = torch.from_numpy(df.drop(['duration', 'event'], axis=1).to_numpy())
        self.event_time = torch.from_numpy(df['duration'].to_numpy()[:, None])
        self.event = torch.from_numpy(df['event'].to_numpy()[:, None])

    def __len__(self):
        return len(self.event)

    def __getitem__(self, idx):
        return self.cov[idx], self.event_time[idx], self.event[idx]


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
    train_set, val_set, test_set = load_data('metabric')
    cov, event_time, event = train_set[1:3]

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
    print(mixed_linear(cov, event_time))