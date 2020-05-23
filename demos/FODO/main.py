import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense


from utils import load_TM_map



import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent.parent)
from tm_pnn.layers.Taylor_Map import TaylorMap


def load_weights(dim):
    weights = {}
    for element in ['b', 'd', 'qdk', 'qfk', 'sd', 'sf']:
        weights[element] = load_TM_map(filepath=f'maps/{element}.txt', dim=dim)
    return weights


def get_model(dim, order):
    weights = load_weights(dim)

    fodo = Sequential()

    lattice = ['qfk', 'd', 'b', 'd', 'qdk', 'sd', 'qdk', 'd', 'b', 'd', 'qfk', 'sf']

    for element in lattice:
        element_map = TaylorMap(output_dim = dim, order = order, input_shape = (dim,))
        fodo.add(element_map)
        element_map.set_weights([w.copy() for w in weights[element]])

    return fodo

def main():
    dim = 3
    order = 3

    fodo = get_model(dim, order)

    x0 = np.arange(start=0, stop=0.21, step=0.01)

    X0 = np.zeros((len(x0), dim))
    X0[:, 0] = x0

    _, ax = plt.subplots(1,2,figsize=(15,5))

    for k in [0, 1]:
        X0[:, 2] = k # parameter as additional state

        X = []
        X.append(X0)

        for i in range(2000):
            X.append(fodo.predict(X[-1]))

        X = np.array(X)

        ax[k].plot(X[:, :, 0], X[:, :, 1], linestyle='none', marker='*', markersize = 0.2)
        ax[k].set_xlim([-0.25, 0.25])

        # plt.title('Trained TM-PNN with one-turn', fontsize=14)

        ax[k].set_xlabel('x (m)', fontsize=14)
        ax[k].set_ylabel('x\' (rad)', fontsize=14)
        # plt.gca().set_yticks([-0.05, 0, 0.05])
        ax[k].tick_params(axis='both', which='major', labelsize=14)

    ax[1].set_ylim([-0.6, 0.6])

    plt.show()

    return 0

if __name__ == "__main__":
    main()