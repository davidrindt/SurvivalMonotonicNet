import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sumonet.model.net import TotalNet
from sumonet.data.load_data import Checkerboard, sample_synthetic_dataset
from sumonet.data.utils import survival_scatter_plot


if __name__ == '__main__':
    # The true dataset is as follows:
    dist = Checkerboard()

    # Get the true survival curves
    n = 100
    t = torch.linspace(0, 1, n)[:, None]
    S0_true = [dist.surv_given_x(s, 0) for s in t.flatten()]
    S1_true = [dist.surv_given_x(s, 0.99) for s in t.flatten()]
    plt.plot(t, S0_true)
    plt.plot(t, S1_true)
    plt.show()

    # Load the result
    with open(os.path.join('results', 'checkerboard_experiment.p'), 'rb') as f:
        result = pickle.load(f)

    config, model = result['best_config'], result['best_model_state_dict']

    print(f'best config {config}')

    net = TotalNet(config)
    net.load_state_dict(model)
    net.eval()

    # Get the survival curves
    S0 = net.forward_S(t, torch.zeros(n, 1)).detach().numpy()
    S1 = net.forward_S(t, torch.ones(n, 1)).detach().numpy()

    # Plot the curves
    plt.plot(t, S0)
    plt.plot(t, S1)
    plt.show()


    # Get some data
    df = sample_synthetic_dataset('checkerboard')
    survival_scatter_plot(np.array(df.x), np.array(df.duration), np.array(df.event))
