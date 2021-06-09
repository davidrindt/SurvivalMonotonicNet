from sumonet.model.net import train_sumo_net, load_data


if __name__ == '__main__':
    train, val, test = load_data('metabric')

    config = {'activation': 'tanh', 'epsilon': 1e-5, 'exact': True, 'lr': 1e-2, 'num_layers_mixed': 3,
               'num_layers_cov': 3, 'width_cov': 32, 'width_mixed': 32, 'num_layers_cov': 3, 'data': 'metabric',
              'batch_size': 128, 'num_epochs': 100, 'dropout': 0.5,
              'weight_decay': 1e-4, 'batch_norm': False}

    train_sumo_net(config, train, val)