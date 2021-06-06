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

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)

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
        df = None
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
        self.cov_dim = self.cov.shape[1]
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
        self.activation = getattr(torch, config['activation'])
        for input_size, output_size in zip(self.layer_widths, self.layer_widths[1:]):
            self.linear_transforms.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        # print('cov', x)
        for linear_transform in self.linear_transforms:
            x = self.activation(linear_transform(x))
        return x


class BoundedLinear(nn.Linear):
    def __init__(self, input_size, output_size, bounding_operation_name='abs'):
        super().__init__(input_size, output_size)
        self.bounding_operation = getattr(torch, bounding_operation_name)
        self.bounded_weight = self.bounding_operation(self.weight)

    def forward(self, x):
        return F.linear(x, self.bounded_weight, self.bias)


class MixedLinear(nn.Linear):
    def __init__(self, input_size, output_size, bounding_operation_name='abs'):
        super().__init__(input_size, output_size)
        self.bounding_operation = getattr(torch, bounding_operation_name)
        self.bounded_weight = self.bounding_operation(self.weight[:, 0][:, None])
        self.unrestricted_weight = self.weight[:, 1:]

    def forward(self, x, t):
        return F.linear(x, self.unrestricted_weight) + F.linear(t, self.bounded_weight) + self.bias


class MixedNet(nn.Module):
    """
    The mixed net consists of first a MixedLinear layer, and then BoundedLinear layers.
    """

    def __init__(self, config):
        super().__init__()
        self.layer_widths = config['mixed_net_widths']
        self.bounded_transforms = nn.ModuleList()
        self.activation = getattr(torch, config['activation'])
        self.mixed_linear = MixedLinear(self.layer_widths[0], self.layer_widths[1])
        for input_size, output_size in zip(self.layer_widths[1:], self.layer_widths[2:]):
            self.bounded_transforms.append(BoundedLinear(input_size, output_size))

    def forward(self, x, t):
        y = self.activation(self.mixed_linear(x, t))
        L = len(self.bounded_transforms)
        for l, bounded_linear in enumerate(self.bounded_transforms):
            if l < L - 1:
                y = self.activation(bounded_linear(y))
            else:
                y = bounded_linear(y)
        return y


class TotalNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cov_net = CovNet(self.config)
        self.mixed_net = MixedNet(self.config)


    def forward_h(self, x, t):
        x = self.cov_net(x)
        x = self.mixed_net(x, t)
        return x

    def forward_S(self, x, t):
        x = self.forward_h(x, t)
        return 1 - torch.sigmoid(x)

    def forward_f_approx(self, x, t):
        h_t = self.forward_h(x, t)
        h_t_plus = self.forward_h(x, t + self.config['epsilon'])
        h_derivative_approx = (h_t_plus - h_t) / self.config['epsilon']
        f_approx = torch.sigmoid(h_t) * (1 - torch.sigmoid(h_t)) * h_derivative_approx
        return f_approx

    def forward_f_exact(self, x, t):
        y = - self.forward_S(x, t)
        f = torch.autograd.grad(outputs=y, inputs=t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        return f

    def forward(self, x, t, d):
        event_mask = (d == 1).ravel()
        x_obs, t_obs = x[event_mask, :], t[event_mask, :]
        x_cens, t_cens = x[~ event_mask, :], t[~ event_mask, :]
        t_obs.requires_grad = True
        forward_f = self.forward_f_exact if self.config['exact'] else self.forward_f_approx
        S = self.forward_S(x_cens, t_cens)
        f = forward_f(x_obs, t_obs)
        return S,  f


def get_cov_widths(cov_dim, layer_width_cov, num_layers):
    widths = [layer_width_cov for _ in range(num_layers + 1)]
    widths[0] = cov_dim
    return widths


def get_mixed_widths(layer_width_cov, layer_width_mixed, num_layers):
    widths = [layer_width_mixed for _ in range(num_layers + 1)]
    widths[0] = layer_width_cov + 1
    widths[-1] = 1
    return widths


def log_loss(S, f):
    cat = torch.cat((S.flatten(), f.flatten()))
    return - torch.mean(torch.log(cat))


if __name__ == '__main__':
    # Initiate train, test set
    train_set, val_set, test_set = load_data('metabric')
    cov, event_time, event = train_set[1:10]
    print('cov dim', train_set.cov_dim)
    # Define the config file
    config = {'cov_net_widths': [9, 7, 7],
              'activation': 'tanh',
              'mixed_net_widths': [8, 3, 1],
              'epsilon': 1e-5,
              'exact': True
              }

    # Initiate the net
    # net = CovNet(config)
    # print(net(cov))
    # print(net.linear_transforms)
    #
    # # Initiate the bounded_layer
    # bounded_linear = BoundedLinear(9, 3, 'square')
    # print(bounded_linear(cov))
    #
    # print(bounded_linear.weight)
    # print(bounded_linear.bounded_weight)
    #
    # # Initiate the mixed layer
    # mixed_linear = MixedLinear(10, 3, 'abs')
    # print(mixed_linear(cov, event_time))
    #
    # # Initiate the mixed net
    # mixed_net = MixedNet(config)
    # mixed_net(cov, event_time)

    # Initiate the total net
    config['cov_net_widths'] = get_cov_widths(9, 3, 1)
    config['mixed_net_widths'] = get_mixed_widths(3, 3, 1)
    total_net = TotalNet(config)
    # print(total_net.forward_h(cov, event_time))
    # print(total_net.forward_S(cov, event_time))
    # print('f approx', total_net.forward_f_approx(cov, event_time))
    S, f = total_net(cov, event_time, event)
    print('forward', S, f)
    print('loss', log_loss(S, f))
    loss = log_loss(S, f)
    loss.backward()

