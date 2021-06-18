import torch
import numpy as np
import matplotlib.pyplot as plt

from sumonet.model.net import train_sumo_net
from sumonet.model.net import TotalNet
from sumonet.data.load_data import load_data

if __name__ == '__main__':

    # Define the config
    config = {'activation': 'tanh',
              'epsilon': 1e-05,
              'exact': True,
              'lr': 0.01,
              'num_layers_mixed': 3,
              'num_layers_cov': 3,
              'width_cov': 8,
              'width_mixed': 8,
              'data': 'checkerboard',
              'cov_dim': 1,
              'batch_size': 128,
              'num_epochs': 100,
              'dropout': 0.0,
              'weight_decay': 0.0,
              'batch_norm': False,
              'scaling_type_time': 'MinMaxScaler',
              'scaling_type_cov': 'MinMaxScaler'}

    train, val, test = load_data(config, n=10000)
    best_state_dict = train_sumo_net(config, train, val)
    net = TotalNet(config)
    net.eval()
    net.load_state_dict(best_state_dict)

    # Get the true survival curves
    n = 100
    t = torch.linspace(0, 1, n)[:, None]
    S0 = net.forward_S(t, torch.zeros(n, 1)).detach().numpy()
    S1 = net.forward_S(t, torch.ones(n, 1)).detach().numpy()

    # Plot the curves
    plt.plot(t, S0)
    plt.plot(t, S1)
    plt.show()