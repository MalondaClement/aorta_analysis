#
# utils/plots.py
#
# Clément Malonda
#

import numpy as np
import matplotlib.pyplot as plt

def plot_curves(title, datas, legends, path, grid=True) :
    assert len(datas) == len(legends)
    for i in range(len(datas)) :
         plt.plot(np.linspace(1, len(datas[i]), len(datas[i])), datas[i])
    plt.title(title)
    plt.grid(grid)
    plt.legend(legends)
    plt.savefig(path)
