from sumonet.model.net import train_sumo_net, load_data, TotalNet
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import os
import torch
import numpy as np


def main(config, num_samples=6, gpus_per_trial=0):

    train, val, test = load_data('metabric')

    # Define the scheduler and reporter
    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=300,
        grace_period=1,
        reduction_factor=2
    )
    reporter = CLIReporter(
        metric_columns=['loss', 'training_iteration']
    )

    # Run the training
    result = tune.run(
        partial(train_sumo_net, train=train, val=val, tuning=True),
        resources_per_trial={'cpu': 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    # Get the best trial and print some numbers
    best_trial = result.get_best_trial('loss', 'min', 'last')
    print(f'Best trial config {best_trial.config}')
    print('Best trial final validation loss: {}'.format(best_trial.last_result['loss']))
    # print('Best trail final validation accuracy: {}'.format(best_trial.last_result['accuracy']))

    # Load the best model
    best_trained_model = TotalNet(best_trial.config)
    device = 'cpu'
    best_trained_model.to(device)
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, 'checkpoint'))
    best_trained_model.load_state_dict(model_state)
    #
    # test_acc = test_accuracy(best_trained_model, device)
    #
    # print(f'Best trial test accuracy {test_acc}')



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
                       data='metabric',
                       cov_dim=9,
                       batch_size=tune.choice([32, 64, 128]),
                       num_epochs=100,
                       dropout=tune.choice([0., 0.2, 0.5]),
                       weight_decay=tune.choice([0, 1e-4, 1e-3]),
                       batch_norm=False)

    main(config)

