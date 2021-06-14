from sumonet.model.hyperopt_run import hyperopt_run
from sumonet.datasets.load_data import load_data
from sumonet.model.net import TotalNet
import torch
import numpy as np
from ray import tune
import os
import pickle

if __name__ == '__main__':
    # Initialize some variables and define the checkpoint directory
    seed = 2
    torch.manual_seed(seed)
    np.random.seed(seed)
    checkpoint_dir = 'checkpoints'

    # Define the config
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
                  scaling_type_time='MinMaxScaler',
                  scaling_type_cov='MinMaxScaler')

    result = hyperopt_run(config, load_data(config, n=25000), num_samples=20)
    print(result)
    # Save the result
    with open(os.path.join('results', 'checkerboard_experiment.p'), 'wb') as f:
        pickle.dump(result, f)
    #
    # #
    # # # Get the best trial and print some numbers
    # # best_trial = result.get_best_trial('loss', 'min', 'last')
    # # best_trained_model = TotalNet(best_trial.config)
    # # best_checkpoint_dir = best_trial.checkpoint.value
    # # model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, 'checkpoint'))
    # # best_trained_model.load_state_dict(model_state)
    # # best_trained_model.eval()
    # #
    # # Load the result
    # with open(os.path.join('results', 'checkerboard_experiment.p'), 'rb') as f:
    #     loaded_result = pickle.load(f)
    #
    # loaded_best_trial = loaded_result.get_best_trial('loss', 'min', 'last')
    # loaded_best_trained_model = TotalNet(loaded_best_trial.config)
    # best_checkpoint_dir = loaded_best_trial.checkpoint.value
    # loaded_model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, 'checkpoint'))
    # loaded_best_trained_model.load_state_dict(loaded_model_state)
    # loaded_best_trained_model.eval()
    #
    # # Calculate some numbers
    # n = 100
    # x_0 = torch.zeros(n, 1)
    # t = torch.linspace(0, 1, n)[:, None]
    #
    # loaded_S_0 = loaded_best_trained_model.forward_h(t, x_0)
    #
    # print('ss', loaded_S_0 )
