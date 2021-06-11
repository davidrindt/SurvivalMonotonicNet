from sumonet.model.hyperopt_run import hyperopt_run
from sumonet.datasets.load_data import load_data
import torch
import numpy as np
from ray import tune

if __name__ == '__main__':
    seed = 2
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = dict(activation='tanh',
                  epsilon=1e-5,
                  exact=True,
                  lr=tune.choice([0.01, 0.001]),
                  num_layers_mixed=tune.choice([3]),
                  num_layers_cov=tune.choice([1, 3, 5]),
                  width_cov=tune.choice([4, 8, 16]),
                  width_mixed=tune.choice([4, 8, 16]),
                  data='checkerboard',
                  cov_dim=1,
                  batch_size=tune.choice([32, 64, 128]),
                  num_epochs=100,
                  dropout=tune.choice([0., 0.2, 0.5]),
                  weight_decay=tune.choice([0, 1e-4, 1e-3]),
                  batch_norm=False,
                  scaling_type_time='StandardScaler',
                  scaling_type_cov='StandardScaler')

    hyperopt_run(config, load_data('checkerboard', config, n=10000))