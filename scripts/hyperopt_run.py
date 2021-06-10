from sumonet.model.net import train_sumo_net, load_data, TotalNet
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import os

def hyperopt_run(hyperconfig, num_samples=2, max_num_epochs=100, gpus_per_trial=2):
    checkpoint_dir = 'checkpoints'

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["dropout"],
        metric_columns=["loss", "training_iteration"])

    result = tune.run(
        partial(train_sumo_net),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=hyperconfig,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)


    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = TotalNet(best_trial.config)
    print(f'Best trained model state dict {best_trained_model.state_dict()}')

if __name__ == '__main__':
    train, val, test = load_data('metabric')

    # config = {'activation': 'tanh', 'epsilon': 1e-5, 'exact': True, 'lr': 1e-2, 'num_layers_mixed': 3,
    #            'num_layers_cov': 3, 'width_cov': 32, 'width_mixed': 32, 'num_layers_cov': 3, 'data': 'metabric',
    #           'batch_size': 128, 'num_epochs': 100, 'dropout': 0.5,
    #           'weight_decay': 1e-4, 'batch_norm': False}

    hyperconfig = dict(activation='tanh',
                       epsilon=1e-5,
                       exact=True,
                       lr=tune.choice([0.01, 0.001]),
                       num_layers_mixed=tune.choice([3]),
                       num_layers_cov=tune.choice([1, 3]),
                       width_cov=tune.choice([4, 8]),
                       width_mixed=tune.choice([4]),
                       data='metabric',
                       cov_dim=9,
                       batch_size=tune.choice([32, 64, 128]),
                       num_epochs=100,
                       dropout=tune.choice([0., 0.2, 0.5]),
                       weight_decay=tune.choice([0, 1e-4, 1e-3]),
                       batch_norm=False)

    hyperopt_run(hyperconfig=hyperconfig)
    # train_sumo_net(config, train, val)