from sumonet.model.net import train_sumo_net, load_data, TotalNet, MixedNet, MixedLinear, BoundedLinear
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Load the result
    with open(os.path.join('results', 'checkerboard_experiment.p'), 'rb') as f:
        result = pickle.load(f)

    config, model = result['best_config'], result['best_model_state_dict']

    net = TotalNet(config)
    net.load_state_dict(model)
    net.eval()

    # Calculate some numbers
    n = 100
    x0 = torch.zeros(n, 1)
    x1 = torch.ones(n, 1)
    t = torch.linspace(0, 1, n)[:, None]

    # Get the survival curves
    S0 = net.forward_S(t, x0).detach().numpy()
    S1 = net.forward_S(t, x1).detach().numpy()

    # Plot the curves
    plt.plot(t, S0)
    plt.plot(t, S1)
    plt.show()

