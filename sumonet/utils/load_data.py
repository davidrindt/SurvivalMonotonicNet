import numpy as np
from ray import tune
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torchvision
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch.optim as optim
import os
from functools import partial
import pycox


def load_data(data_name):
    """
    :param data_name: name of the dataset
    :return:
    """
    if data_name == 'metabric':
        df = pycox.datasets.metabric.read_df()
        print(df)


if __name__ == '__main__':
    load_data('metabric')