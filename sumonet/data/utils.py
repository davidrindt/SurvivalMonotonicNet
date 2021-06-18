import numpy as np
import matplotlib.pyplot as plt


def survival_scatter_plot(x, z, d):
    select = np.where(d == 1)
    unselect = np.where(d == 0)
    l1 = plt.scatter(x[select], z[select], c='navy', label='observed')
    l2 = plt.scatter(x[unselect], z[unselect], facecolors='none', edgecolors='r', label='censored')
    plt.xlabel('covariate')
    plt.ylabel('time')
    plt.legend()
    plt.show()