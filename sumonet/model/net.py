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
from sumonet.datasets.load_data import load_data
seed = 2
torch.manual_seed(seed)
np.random.seed(seed)



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
        print('xxx', x)
        print('type', type(x))
        for linear_transform in self.linear_transforms:
            print('a', linear_transform(x))
            x = self.dropout(self.activation(linear_transform(x)))
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



def train_sumo_net(config, train, val, tuning=False):

    # Define the network
    net = TotalNet(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    # Set the criterion and optimizer
    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])


    # Load the datasets
    trainloader = torch.utils.data.DataLoader(train, batch_size=config['batch_size'], shuffle=True)
    valloader = torch.utils.data.DataLoader(val, batch_size=config['batch_size'], shuffle=True)
    best_val_loss = np.inf

    for epoch in range(config['num_epochs']):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader):

            # Get the data
            cov, event_time, event = data
            cov, event_time, event = cov.to(device), event_time.to(device), event.to(device)

            # Zero the gradient
            optimizer.zero_grad()

            # Compute loss, gradients, and take step
            S, f = net(event_time, cov, event)
            loss = log_loss_mean(S, f)
            loss.backward()
            optimizer.step()

            # Track the loss
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (epoch_steps + 1)))
                running_loss = 0
                epoch_steps = 0

        # Validation loss
        val_loss = 0.0
        val_steps = 0

        # Loop over valloader
        for i, data in enumerate(valloader):
            # Get the data
            cov, event_time, event = data
            cov, event_time, event = cov.to(device), event_time.to(device), event.to(device)

            # Zero the gradient
            optimizer.zero_grad()

            # Compute the percentage correct
            S, f = net(event_time, cov, event)
            loss = log_loss_mean(S, f)

            val_loss += loss.cpu().detach().numpy()
            val_steps += 1

        if val_loss < best_val_loss:
            print(f'new best val loss {val_loss}')
            best_val_loss = val_loss

        if tuning:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), path)
            tune.report(loss=(val_loss / val_steps))

    print('Finished training')





if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    # Initiate train, test set

    # Define the config file
    config = {'activation': 'tanh',
              'epsilon': 1e-5,
              'exact': True,
              'lr': 1e-2,
              'num_layers_mixed': 3,
               'num_layers_cov': 3,
              'width_cov': 32,
              'width_mixed': 32,
              'num_layers_cov': 3,
              'data': 'checkerboard',
              'cov_dim':1,
              'batch_size': 128,
              'num_epochs': 100,
              'dropout': 0.5,
              'weight_decay': 1e-4,
              'batch_norm': False,
              'scaling_type_time':'StandardScaler',
              'scaling_type_cov':'StandardScaler'
    }

    train, val, test = load_data(config['data'], config)
    cov, event_time, event = train[1:4]

    train_sumo_net(config, train, val)
