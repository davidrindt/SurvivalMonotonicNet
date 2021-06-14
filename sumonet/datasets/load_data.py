import pycox
from pycox import datasets
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
import scipy
import math

def load_data(config, n=1000, seed=1):
    """
    :param dataset: str - name of the dataset
    :return: - torch datasets - a train and test set
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = config['data']

    # Load the data
    if dataset == 'metabric':
        df = pycox.datasets.metabric.read_df()
    elif dataset in ['weibulls', 'normals', 'checkerboard']:
        df = sample_synthetic_dataset(dataset, n)
    else:
        df = None
        print('Dataset not found')

    # Preprocess the dataframe
    df = preprocess(df, config)

    # Split the dataset
    train, val, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
    train, val, test = SurvivalDataset(train), SurvivalDataset(val), SurvivalDataset(test)
    return train, val, test



def preprocess(df, config):
    """
    :param df: pd.dataframe - to be preprocessed
    :param scaling_type_cov: str - how to scale the covariates
    :param scaling_type_time:  str - how to scale the times
    :return: np.ndarray - cov, time, event
    """
    time_scaler = getattr(sklearn.preprocessing, config['scaling_type_time'])()
    cov_scaler = getattr(sklearn.preprocessing, config['scaling_type_cov'])()

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



class Weibulls:  # a weibull distribution with scale set to 1

    def __init__(self, a=2, b=6):
        self.a = a
        self.b = b

    def shape_given_x(self, x):
        return self.a + self.b * x

    def surv_given_x(self, t, x):
        shape = self.shape_given_x(x)
        return np.exp(-t ** shape)

    def sample(self, n):
        x = np.random.uniform(0, 1, size=n).astype(np.float32)
        t = np.zeros(n)
        for i in range(n):
            t[i] = np.random.weibull(a=self.shape_given_x(x[i]))
        c = np.random.exponential(1.5, size=n)
        return x.astype(np.float32), t.astype(np.float32), c.astype(np.float32)


class Checkerboard:

    def __init__(self, grid_width=1, grid_length=1, num_tiles_width=4, num_tiles_length=6):
        self.grid_width = grid_width
        self.grid_length = grid_length
        self.num_tiles_width = num_tiles_width
        self.num_tiles_length = num_tiles_length
        self.tile_width = self.grid_width / self.num_tiles_width
        self.tile_length = self.grid_length / self.num_tiles_length

    def find_class(self, x):
        return math.floor(x / self.tile_width) % 2

    def surv_given_x(self, t, x):
        c = math.floor(t / (self.tile_length * 2))
        res = t % (2 * self.tile_length)
        if self.find_class(x) == 0:
            c += min(1, res / self.tile_length)
        elif self.find_class(x) == 1:
            c += max(0, (res - self.tile_length) / self.tile_length)
        return 1 - max(0, min(c / (self.num_tiles_length / 2), 1))

    def sample(self, n):
        x = np.random.uniform(0, self.grid_width, size=n)
        t = np.array([self.find_class(xi) for xi in x]) * self.tile_length
        t += np.random.choice(np.arange(0, self.num_tiles_length, step=2), size=n) * self.tile_length
        t += np.random.uniform(low=0, high=self.tile_length, size=n)
        c = np.random.exponential(1.5, size=n)

        return x.astype(np.float32), t.astype(np.float32), c.astype(np.float32)


class Normals:

    def __init__(self, mean=100, var_slope=6):
        self.mean = mean
        self.var_slope = var_slope

    def surv_given_x(self, t, x):
        return 1 - scipy.stats.norm.cdf(t, loc=self.mean, scale=1 + self.var_slope * x)

    def sample(self, n):
        x = np.random.uniform(size=n)
        t = np.zeros(shape=n)
        for i in range(n):
            t[i] = np.random.normal(loc=self.mean, scale=1 + self.var_slope * x[i])
        c = np.random.normal(loc=100, scale=6, size=n)
        return x.astype(np.float32), t.astype(np.float32), c.astype(np.float32)


def sample_synthetic_dataset(dataset, n=1000):

    # Get the correct distribution
    if dataset == 'normals':
        dist = Normals()
    elif dataset == 'weibulls':
        dist = Weibulls()
    elif dataset == 'checkerboard':
        dist = Checkerboard()

    # Sample
    x, t, c = dist.sample(n)
    d = np.int64(c > t)
    z = np.minimum(t, c)

    # Create a df
    df = pd.DataFrame(dict(x=x, duration=z, event=d))

    return df


if __name__ == '__main__':
    # df = sample_synthetic_dataset('weibulls', 1000)
    train, val, test = load_data('weibulls')

    # train_loader = torch.utils.data.DataLoader(train)
    cov, event_time, event = train[1:30]
    print(cov)
    print(cov.shape)