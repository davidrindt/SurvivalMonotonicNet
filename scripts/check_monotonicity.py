from sumonet.model.net import train_sumo_net, load_data, TotalNet, MixedNet, MixedLinear, BoundedLinear
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import os
import torch
import numpy as np
import pickle

if __name__ == '__main__':

    # Load the result
    with open(os.path.join('results', 'checkerboard_experiment.p'), 'rb') as f:
        loaded_result = pickle.load(f)

    loaded_best_trial = loaded_result.get_best_trial('loss', 'min', 'last')
    loaded_best_trained_model = TotalNet(loaded_best_trial.config)
    best_checkpoint_dir = loaded_best_trial.checkpoint.value
    loaded_model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, 'checkpoint'))
    loaded_best_trained_model.load_state_dict(loaded_model_state)
    loaded_best_trained_model.eval()

    # Calculate some numbers
    n = 100
    x_0 = torch.zeros(n, 1)
    t = torch.linspace(0, 1, n)[:, None]

    S_0 = loaded_best_trained_model.forward_h(t, x_0)
    print(f' S_0 {S_0}')