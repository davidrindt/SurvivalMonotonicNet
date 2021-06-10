# import numpy as np
# from ray import tune
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import random_split
# import torchvision
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
# import torch.optim as optim
# import os
# from functools import partial
#
#
# # https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
#
#
# def load_data(data_dir='data/'):
#     IMAGE_SIZE = 16
#
#     transform = transforms.Compose([
#         transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#         transforms.ToTensor(),
#         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     trainset = torchvision.datasets.MNIST(
#         root=data_dir, train=True, download=True, transform=transform
#     )
#
#     testset = torchvision.datasets.MNIST(
#         root=data_dir, train=False, download=True, transform=transform
#     )
#
#     trainset = torch.utils.data.Subset(trainset, torch.arange(5000))
#     testset = torch.utils.data.Subset(testset, torch.arange(1000))
#
#     print('TYPE', type(trainset))
#
#     return trainset, testset
#
#
# class Net(nn.Module):
#     def __init__(self, out1=16, out2=32, l1=120, l2=84):
#         print('types', type(l1), type(l2))
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=out1, kernel_size=5, padding=2)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=5, stride=1, padding=2)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2)
#         self.fc1 = nn.Linear(out2 * 16, l1)
#         self.fc2 = nn.Linear(l1, l2)
#         self.fc3 = nn.Linear(l2, 10)
#
#     def forward(self, x):
#         x = self.maxpool1(F.relu(self.conv1(x)))
#         x = self.maxpool2(F.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# def train_mnist(config, checkpoint_dir=None, data_dir=None):
#     # Define the network
#     net = Net(config['l1'], config['l2'])
#
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if torch.cuda.device_count() > 1:
#             net = nn.DataParallel(net)
#     net.to(device)
#
#     # Set the criterion and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9)
#
#     # Check if there is a checkpoint
#     # if checkpoint_dir:
#     #     model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, 'checkpoint'))
#     #     net.load_state_dict(model_state)
#     #     optimizer.load_state_dict(optimizer_state)
#
#     # Load the datasets
#     trainset, testset = load_data(data_dir)
#     test_abs = int(len(trainset) * 0.8)
#     train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
#     print(len(train_subset), len(val_subset))
#     trainloader = torch.utils.data.DataLoader(train_subset, batch_size=int(config['batch_size']), shuffle=True)
#     valloader = torch.utils.data.DataLoader(val_subset, batch_size=int(config['batch_size']), shuffle=True)
#     best_val_loss = np.inf
#
#     print('num epochs', config['num_epochs'])
#     for epoch in range(10):
#         print(f'epoch {epoch}')
#         running_loss = 0.0
#         epoch_steps = 0
#         for i, data in enumerate(trainloader):
#
#             # Get the data
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             # Zero the gradient
#             optimizer.zero_grad()
#
#             # Compute loss, gradients, and take step
#             predictions = net(inputs)
#             loss = criterion(predictions, labels)
#             loss.backward()
#             optimizer.step()
#
#             # Track the loss
#             running_loss += loss.item()
#             epoch_steps += 1
#             if i % 2000 == 1999:
#                 print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (epoch_steps + 1)))
#                 running_loss = 0
#                 epoch_steps = 0
#
#         # Validation loss
#         val_loss = 0.0
#         val_steps = 0
#         total = 0
#         correct = 0
#
#         # Loop over valloader
#         for i, data in enumerate(valloader):
#             with torch.no_grad():
#                 # Get the data
#                 inputs, labels = data
#                 inputs, labels = inputs.to(device), labels.to(device)
#
#                 # Compute the percentage correct
#                 predictions = net(inputs)
#                 _, predicted = torch.max(predictions.data, dim=1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#
#                 # Compute the loss
#                 loss = criterion(predictions, labels)
#                 val_loss += loss.cpu().numpy()
#                 val_steps += 1
#         print(f'val loss {val_loss}')
#         print(f'val accuracy {correct / total}')
#
#         if val_loss < best_val_loss:
#             print(f'new best val loss {val_loss}')
#             best_val_loss = val_loss
#
#         with tune.checkpoint_dir(epoch) as checkpoint_dir:
#             path = os.path.join(checkpoint_dir, "checkpoint")
#             torch.save((net.state_dict(), optimizer.state_dict()), path)
#
#         tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
#     print('Finished training')
#
#
# def test_accuracy(net, device='cpu'):
#     _, testset = load_data()
#     testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
#
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     return correct / total
#
#
def main(config, num_samples=30, gpus_per_trial=0):
    data_dir = os.path.abspath('./data')
    checkpoint_dir = 'checkpoint_dir'

    # load_data(data_dir)

    # Define the scheduler and reporter
    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=300,
        grace_period=1,
        reduction_factor=2
    )
    reporter = CLIReporter(
        # parameter_columns=['l1', 'l2', 'lr', 'batch_size'],
        metric_columns=['loss', 'training_iteration']
    )

    # Run the training
    result = tune.run(
        partial(train_mnist, checkpoint_dir=checkpoint_dir, data_dir=data_dir),
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

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)

totensor = ToTensor()


def load_data(dataset):
    """
    :param dataset: str - name of the dataset
    :return: - torch datasets - a train and test set
    """

    # Load the data
    if dataset == 'metabric':
        df = pycox.datasets.metabric.read_df()
    else:
        df = None
        print('Dataset not found')
    df = preprocess(df)

    # Split the dataset
    train, val, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
    train, val, test = SurvivalDataset(train), SurvivalDataset(val), SurvivalDataset(test)
    print('HERE HERE', len(train), len(val), len(test))
    print(train, val, test)
    return train, val, test


def preprocess(df, scaling_type_cov='StandardScaler', scaling_type_time='StandardScaler'):
    """
    :param df: pd.dataframe - to be preprocessed
    :param scaling_type_cov: str - how to scale the covariates
    :param scaling_type_time:  str - how to scale the times
    :return: np.ndarray - cov, time, event
    """
    time_scaler = getattr(sklearn.preprocessing, scaling_type_time)()
    cov_scaler = getattr(sklearn.preprocessing, scaling_type_cov)()

    # Select the covariates, times, events
    for col in df.columns:
        if col == 'event':
            continue
        elif col == 'time':
            df[col] = time_scaler.fit_transform(df[col].to_numpy()[:, None])
        else:
            df[col] = cov_scaler.fit_transform(df[col].to_numpy()[:, None])

    return df


class SurvivalDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.cov = torch.from_numpy(df.drop(['duration', 'event'], axis=1).to_numpy())
        self.cov_dim = self.cov.shape[1]
        self.event_time = torch.from_numpy(df['duration'].to_numpy()[:, None])
        self.event = torch.from_numpy(df['event'].to_numpy()[:, None])

    def __len__(self):
        return len(self.event)

    def __getitem__(self, idx):
        return self.cov[idx], self.event_time[idx], self.event[idx]


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
        for linear_transform in self.linear_transforms:
            x = self.dropout(get_batch_norm(self.activation(linear_transform(x)), self.config['batch_norm']))
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


# def train_sumo_net(config):
#     net = TotalNet(config)
#
#     # Get the device
#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = 'cuda:0'
#         if torch.cuda.device_count() > 1:
#             net = nn.DataParallel(net)
#     net.to(device)
#
#     print('here')
#     train, val, test = load_data(config['data'])
#     train, val, test = train.to(device), val.to(device), test.to(device)
#
#     print('AAAAAA')
#
#     # Set the optimizer
#     optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
#
#     # Get the train and val loader
#     train_loader = torch.utils.data.DataLoader(train, batch_size=config['batch_size'], shuffle=True)
#     val_loader = torch.utils.data.DataLoader(val, batch_size=config['batch_size'], shuffle=True)
#     best_val_loss = np.inf
#
#     print('CCCCCC')
#     # Define the train loop
#     for epoch in range(config['num_epochs']):
#         running_loss, epoch_steps = 0.0, 0
#
#         net.train()
#         for data in train_loader:
#
#             # Get the data
#             cov, event_time, event = data
#             cov, event_time, event = cov.to(device), event_time.to(device), event.to(device)
#
#             # Zero the gradient
#             optimizer.zero_grad()
#
#             # Compute probabilities, likelihood, gradients, and take step
#             S, f = net(event_time, cov, event)
#             loss = log_loss_mean(S, f)
#             loss.backward()
#             optimizer.step()
#
#             # Track the loss
#             running_loss += loss.item()
#             epoch_steps += 1
#             # if epoch_steps % 50 == 0:
#             #     print(f'epoch {epoch} epoch_steps {epoch_steps} loss {running_loss / epoch_steps}')
#
#
#         # Get the validation loss
#         val_loss, val_steps = 0, 0
#
#         net.eval()
#         for data in val_loader:
#
#             # Get the data
#             cov, event_time, event = data
#             cov, event_time, event = cov.to(device), event_time.to(device), event.to(device)
#
#             # Zero the gradient
#             optimizer.zero_grad()
#
#             # Compute probabilities and the loss
#             S, f = net(event_time, cov, event)
#             val_loss += log_loss_sum(S, f).cpu().detach().numpy()
#
#         # print(f'epoch {epoch} val loss {val_loss / len(val)} train loss {loss}')
#
#         if val_loss < best_val_loss:
#             print(f'epoch {epoch} train loss {running_loss / epoch_steps} best val loss {val_loss/ len(val)}')
#             best_val_loss = val_loss
#             # torch.save((net.state_dict(), optimizer.state_dict()), path)
#
#
#         with tune.checkpoint_dir(epoch) as checkpoint_dir:
#             path = os.path.join(checkpoint_dir, "checkpoint")
#             torch.save((net.state_dict(), optimizer.state_dict()), path)
#
#         tune.report(loss=val_loss)
#
#     print('finished training')

def train_mnist(config, checkpoint_dir=None, data_dir=None):
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

    # Check if there is a checkpoint
    # if checkpoint_dir:
    #     model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, 'checkpoint'))
    #     net.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)

    # Load the datasets
    train, val, test = load_data('metabric')
    trainloader = torch.utils.data.DataLoader(train, batch_size=config['batch_size'], shuffle=True)
    valloader = torch.utils.data.DataLoader(val, batch_size=config['batch_size'], shuffle=True)
    best_val_loss = np.inf

    print('num epochs', config['num_epochs'])
    for epoch in range(config['num_epochs']):
        print(f'epoch {epoch}')
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader):

            # Get the data
            cov, event_time, event = data
            cov, event_time, event = cov.to(device), event_time.to(device), event.to(device)

            # Zero the gradient
            optimizer.zero_grad()

            # Compute loss, gradients, and take step
            #             # Compute probabilities, likelihood, gradients, and take step
            #             S, f = net(event_time, cov, event)
            #             loss = log_loss_mean(S, f)
            #             loss.backward()
            #             optimizer.step()
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
        # total = 0
        correct = 0

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
        print(f'val loss {val_loss}')
        # print(f'val accuracy {correct / total}')
        correct=1
        if val_loss < best_val_loss:
            print(f'new best val loss {val_loss}')
            best_val_loss = val_loss

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))
    print('Finished training')
#

if __name__ == '__main__':
    # Define the config dictionary
    # config = {
    #     "l1": tune.choice([64, 128]),
    #     "l2": tune.choice([16, 64, 128]),
    #     "lr": tune.choice([0.1, 0.01, 0.005]),
    #     "batch_size": tune.choice([4, 16, 64]),
    #     'num_epochs': 10
    # }


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
    # config = {
    #     "l1": 2 ** 5,
    #     "l2": 2 ** 5,
    #     "lr": 0.01,
    #     "batch_size": 16,
    #     'num_epochs': 10
    # }

    #
    # train_mnist(config, checkpoint_dir='checkpoint_dir/', data_dir='data/')
    # load_data()

    main(config)

