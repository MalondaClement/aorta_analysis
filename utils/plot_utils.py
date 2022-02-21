#
# utils/plot_utils.py
#
# Clément Malonda
#

import numpy as np
import matplotlib.pyplot as plt

def plot_curve(title, data1, data2=None, x_title=None, y_title=None):
    plt.plot(np.linspace(1, len(data1), len(data1)), data1)
    if data2 is not None :
        plt.plot(np.linspace(1, len(data2), len(data2)), data2)
    plt.title(title)
    plt.show()
