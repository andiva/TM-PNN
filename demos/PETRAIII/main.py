import numpy as np
from scipy.optimize import minimize
import progressbar
import matplotlib.pyplot as plt


import tensorflow as tf
tf.keras.backend.clear_session()

from ocelot2tf import Ocelot2TFAdapter


def main():
    adapter = Ocelot2TFAdapter(dim=4, Ls=1e-5, progress_bar=lambda N: progressbar.ProgressBar(max_value=N), separators=['Hcor', 'Vcor'], exclude_names=["pkhs", "pkvs_", "pkvsa_"],
                           start_with='screenmon', end_with=None, add_last_output=True)

    print()
    print('TM-PNN layers:', len(adapter.model.layers)-1)
    print('BPM:', len(adapter.model.outputs))
    print('Horizontal correctors:', len(adapter.info['Hcor']))
    print('Vertical correctors:', len(adapter.info['Vcor']))


    X_input = np.array([0.001, 0.0002, -0.001, 0.0002]).reshape(1,-1)



    X = np.array(adapter.model.predict(X_input)).reshape(-1, 4)
    _, axs = plt.subplots(2,1, sharey=False, sharex=True, figsize=(25,10))

    axs[0].plot(adapter.lengths, X[:, 0])
    axs[1].plot(adapter.lengths, X[:, 2])

    for i in range(2):
        axs[i].grid()
        axs[i].set_ylim([-0.01, 0.02])

    axs[0].set_ylabel('x', fontsize=14)
    axs[1].set_ylabel('y', fontsize=14)
    axs[1].set_xlabel('Length, m', fontsize=14)
    plt.show()



    return 0

if __name__ == "__main__":
    main()