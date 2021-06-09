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
from tqdm import tqdm

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
    print('lens', len(train), len(val), len(test))
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
        # self.layer_widths = self.config['cov_net_widths']
        self.layer_widths = get_cov_widths(config['cov_dim'], config['width_cov'], config['num_layers_cov'])
        self.linear_transforms = nn.ModuleList()
        self.activation = getattr(torch, config['activation'])
        self.dropout = nn.Dropout(self.config['dropout'])
        for input_size, output_size in zip(self.layer_widths, self.layer_widths[1:]):
            self.linear_transforms.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        for linear_transform in self.linear_transforms:
            x = self.dropout(get_batch_norm(self.activation(linear_transform(x)), self.config['batch_norm']))
        return x


class BoundedLinear(nn.Linear):
    def __init__(self, input_size, output_size, bounding_operation_name='abs', dropout=0.9):
        super().__init__(input_size, output_size)
        self.bounding_operation = getattr(torch, bounding_operation_name)
        self.dropout = nn.Dropout(dropout)
        # self.bounded_weight = self.bounding_operation(self.weight)

    def forward(self, x):
        return F.linear(x, self.bounding_operation(self.weight), self.bias)


class MixedLinear(nn.Module):
    def __init__(self, input_size, output_size, bounding_operation_name='abs'):
        super().__init__()
        self.bounding_operation = getattr(torch, bounding_operation_name)
        self.bounded_linear = BoundedLinear(1, output_size, bounding_operation_name)
        self.linear = nn.Linear(input_size - 1, output_size, bias=False)

    def forward(self, t, x):
        return self.bounded_linear(t) + self.linear(x)



class MixedNet(nn.Module):
    """
    The mixed net consists of first a MixedLinear layer, and then BoundedLinear layers.
    """

    def __init__(self, config):
        super().__init__()
        # self.layer_widths = config['mixed_net_widths']
        self.config = config
        self.layer_widths = get_mixed_widths(config['width_cov'], config['width_mixed'], config['num_layers_mixed'])
        self.bounded_transforms = nn.ModuleList()
        self.activation = getattr(torch, config['activation'])
        self.mixed_linear = MixedLinear(self.layer_widths[0], self.layer_widths[1])
        self.dropout = nn.Dropout(config['dropout'])
        for input_size, output_size in zip(self.layer_widths[1:], self.layer_widths[2:]):
            self.bounded_transforms.append(BoundedLinear(input_size, output_size))

    def forward(self, t, x):
        y = self.dropout(get_batch_norm(self.activation(self.mixed_linear(t, x)), self.config['batch_norm']))

        L = len(self.bounded_transforms)
        for l, bounded_linear in enumerate(self.bounded_transforms):
            if l < L - 1:
                y = self.dropout(get_batch_norm(self.activation(bounded_linear(y)), self.config['batch_norm']))
            else:
                y = bounded_linear(y)
        return y


class TotalNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cov_net = CovNet(self.config)
        self.mixed_net = MixedNet(self.config)

    def forward_h(self, t, x):
        x = self.cov_net(x)
        x = self.mixed_net(t, x)
        return x

    def forward_S(self, t, x):
        x = self.forward_h(t, x)
        return 1 - torch.sigmoid(x)

    def forward_f_approx(self, t, x):
        eps = 1e-6
        h_t = self.forward_h(t, x)
        h_t_plus = self.forward_h(t + eps, x)
        h_derivative_approx = (h_t_plus - h_t) / eps
        f_approx = torch.sigmoid(h_t) * (1 - torch.sigmoid(h_t)) * h_derivative_approx
        return f_approx

    def forward_f_exact(self, t, x):
        t.requires_grad = True
        y = self.forward_S(t, x)
        f, = torch.autograd.grad(outputs=y, inputs=t, grad_outputs=torch.ones_like(y), create_graph=True)
        return - f

    def forward(self, t, x, d):
        event_mask = (d == 1).ravel()
        x_obs, t_obs = x[event_mask, :], t[event_mask, :]
        x_cens, t_cens = x[~ event_mask, :], t[~ event_mask, :]
        forward_f = self.forward_f_exact if self.config['exact'] else self.forward_f_approx
        S = self.forward_S(t_cens, x_cens)
        f = forward_f(t_obs, x_obs)
        return S,  f

def get_batch_norm(x, batch_norm):
    if batch_norm:
        bn = nn.BatchNorm1d(x.shape[1])
        result = bn(x)
        # print('result', result)
        return result
    else:
        return x

def get_cov_widths(cov_dim, layer_width_cov, num_layers):
    widths = [layer_width_cov for _ in range(num_layers + 1)]
    widths[0] = cov_dim
    return widths


def get_mixed_widths(layer_width_cov, layer_width_mixed, num_layers):
    widths = [layer_width_mixed for _ in range(num_layers + 1)]
    widths[0] = layer_width_cov + 1
    widths[-1] = 1
    return widths


def log_loss_mean(S, f):
    cat = torch.cat((S.flatten(), f.flatten()))
    eps = 1e-7
    return - torch.mean(torch.log(cat + eps))


def log_loss_sum(S, f):
    cat = torch.cat((S.flatten(), f.flatten()))
    eps = 1e-7
    return - torch.sum(torch.log(cat + eps))


def train_sumo_net(config, train, val, checkpoint_dir='checkpoints'):

    # Define the checkpoint path
    path = os.path.join(checkpoint_dir, 'checkpoint')
    if not os.path.exists(path):
        os.makedirs(checkpoint_dir)

    # Define the network
    net = TotalNet(config)

    # Get the device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    # Set the optimizer
    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Get the train and val loader
    train_loader = torch.utils.data.DataLoader(train, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=config['batch_size'], shuffle=True)
    best_val_loss = np.inf

    # Define the train loop
    for epoch in range(config['num_epochs']):
        running_loss, epoch_steps = 0.0, 0

        net.train()
        for data in train_loader:

            # Get the data
            cov, event_time, event = data
            # cov, event_time, event = cov.to(device), event_time.to(device), event.to(device)

            # Zero the gradient
            optimizer.zero_grad()

            # Compute probabilities, likelihood, gradients, and take step
            S, f = net(event_time, cov, event)
            loss = log_loss_mean(S, f)
            loss.backward()
            optimizer.step()

            # Track the loss
            running_loss += loss.item()
            epoch_steps += 1
            # if epoch_steps % 50 == 0:
            #     print(f'epoch {epoch} epoch_steps {epoch_steps} loss {running_loss / epoch_steps}')


        # Get the validation loss
        val_loss, val_steps = 0, 0

        net.eval()
        for data in val_loader:

            # Get the data
            cov, event_time, event = data
            # cov, event_time, event = cov.to(device), event_time.to(device), event.to(device)

            # Zero the gradient
            optimizer.zero_grad()

            # Compute probabilities and the loss
            S, f = net(event_time, cov, event)
            val_loss += log_loss_sum(S, f)

        # print(f'epoch {epoch} val loss {val_loss / len(val)} train loss {loss}')

        if val_loss < best_val_loss:
            print(f' --epoch {epoch} best val loss {val_loss/ len(val)} -- ')
            best_val_loss = val_loss
            torch.save((net.state_dict(), optimizer.state_dict()), path)


    print('finished training')







cov_dim = {'metabric': 9}

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    # Initiate train, test set
    train, val, test = load_data('metabric')
    cov, event_time, event = train[1:4]
    # print('cov dim', train.cov_dim)
    # Define the config file
    config = {'activation': 'tanh', 'epsilon': 1e-5, 'exact': True, 'lr': 1e-2, 'num_layers_mixed': 3,
               'num_layers_cov': 3, 'width_cov': 32, 'width_mixed': 32, 'num_layers_cov': 3, 'data': 'metabric',
              'cov_dim': cov_dim['metabric'], 'batch_size': 128, 'num_epochs': 100, 'dropout': 0.5,
              'weight_decay': 1e-4, 'batch_norm': False}

    # print('cov', cov)
    # print(cov.mean(axis=0))
    # batch_norm = nn.BatchNorm1d(9)
    # bn_cov = batch_norm(cov)
    # print('bn cov', bn_cov)


    # # Initiate the net
    # net = CovNet(config)
    # print(net(cov))
    # print(net.linear_transforms)
    # #
    # # # # Initiate the bounded_layer
    # bounded_linear = BoundedLinear(9, 3, 'square')
    # print(bounded_linear(cov))
    # print(bounded_linear.weight)
    # #
    # # # Initiate the mixed layer
    # mixed_linear = MixedLinear(10, 3, 'abs')
    # print(mixed_linear(event_time, cov))
    # #
    # # # Initiate the mixed net
    # config['width_cov'] = 9
    # mixed_net = MixedNet(config)
    # print('layer widths', mixed_net.layer_widths)
    # print('mixed net', mixed_net(event_time, cov))


    # Initiate the total net
    # total_net = TotalNet(config)
    # total_net.eval()
    # print(total_net.forward_h(event_time, cov))
    # print(total_net.forward_S(event_time, cov))
    # print('f approx', total_net.forward_f_approx(event_time, cov))
    # print('f exact', total_net.forward_f_exact(event_time, cov))
    # print(f'difference {total_net.forward_f_approx(cov, event_time) - total_net.forward_f_exact(cov, event_time)}')
    # S, f = total_net(cov, event_time, event)
    # # print('forward', S, f)
    # # print('loss', log_loss(S, f))
    # loss = log_loss(S, f)
    # loss.backward()
    train_sumo_net(config, train, val)
