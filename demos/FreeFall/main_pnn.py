import numpy as np
import matplotlib.pyplot as plt

from physics_based import air_resistance2, simulate
from physics_based import _g as g

from keras import Input, Model
from keras import backend as K
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.regularizers import L1L2

import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent.parent)
from tm_pnn.layers.Taylor_Map import TaylorMap



def create_mTMPNN(N, inputDim, outputDim, order=3):
    ''' Creates polynomial neural network based on Taylor map'''
    input = Input(shape=(inputDim,))
    m = input
    tm = TaylorMap(output_dim = outputDim, order=order,
                   input_shape = (inputDim,)
                  )

    outs=[]
    for i in range(N):
        m = tm(m)
        outs.append(m)

    opt = optimizers.SGD(clipvalue=1e-10)
    model = Model(inputs=input, outputs=outs)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


def iterative_predict(model, X0, N, reshape = False):
    ans = np.empty((N, X0.shape[0], X0.shape[1]))
    X = X0
    for i in range(N):
        if reshape:
            X = X.reshape(X0.shape[0], 1, X0.shape[1])
        X = model.predict(X)[-1]
        ans[i] = X

    return np.vstack((X0[np.newaxis,:], ans))


def main():
    # K.set_floatx('float64')
    m = 100
    k = 0.392

    X0 = np.array([[0, 1/m], [0, 1/(m*0.64)], [0, 1/(m*1.44)], [0, 1/(m*0.3)], [0, 1/(m*2)], [0, 1/(m*0.1)]])
    h = 0.5
    N = 30
    # N = 1

    t = np.arange(0, N*h+h, h)


    f0 = lambda X: air_resistance2(X, k=k)
    v0 = simulate(f0, X0[0], h, N)
    v1 = simulate(f0, X0[1], h, N)
    v2 = simulate(f0, X0[2], h, N)
    v3 = simulate(f0, X0[3], h, N)
    v4 = simulate(f0, X0[4], h, N)
    v5 = simulate(f0, X0[5], h, N)


    # f1 = lambda X: air_resistance(X, k=k, m=0.64*m)
    # v1 = simulate(f1, X0[0], h, N)


    # lstm = create_LSTM(1, 1, n_units=5)
    # lstm.fit(v0[:-1].reshape((-1, 1, v0.shape[1])), v0[1:], epochs=1000, verbose=1)
    # X_lstm = iterative_predict(lstm, X0, N, reshape=True)

    # trN = 10
    pnn = create_mTMPNN(N, 2, 2, order=3)
    W = pnn.layers[1].get_weights()
    # initial values:
    init_w =  [4.864129, 0.99798, -0.19208]
    W[0][0,0] = init_w[0]
    W[1][0,0] = init_w[1]
    W[3][1,0] = init_w[2]
    pnn.layers[1].set_weights(W)

    pnn.fit(X0[:2], [np.vstack((v_1, v_2)) for v_1,v_2 in zip(v0[1:],v1[1:])], epochs=100, verbose=1)
    for w in pnn.layers[1].get_weights():
        print(w)

    X_pnn = np.array(pnn.predict(X0))

    plt.plot(t, v0[:,0], label='true, m = 100')
    plt.plot(t, v1[:,0], label='true, m = 64')
    plt.plot(t-1, v2[:,0], label='true, m = 144')
    plt.plot(t+1, v3[:,0], label='true, m = 30')
    plt.plot(t-2, v4[:,0], label='true, m = 200')
    plt.plot(t+2, v5[:,0], label='true, m = 10')

    plt.gca().set_prop_cycle(None)
    plt.plot(t[1:], X_pnn[:, 0, 0], linestyle='--', marker='o', fillstyle='none', label='TM, m = 100 (training)')
    plt.plot(t[1:], X_pnn[:, 1, 0], linestyle='--', marker='o', fillstyle='none', label='TM, m = 64 (training)')
    plt.plot(t[1:]-1, X_pnn[:, 2, 0], linestyle='--', label='TM, m = 144 (unseen)')
    plt.plot(t[1:]+1, X_pnn[:, 3, 0], linestyle='--', label='TM, m = 30 (unseen)')
    plt.plot(t[1:]-2, X_pnn[:, 4, 0], linestyle='--', label='TM, m = 200 (unseen)')
    plt.plot(t[1:]+2, X_pnn[:, 5, 0], linestyle='--', label='TM, m = 10 (unseen)')

    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=14)

    plt.subplots_adjust(right=0.7)
    plt.grid()
    plt.gca().tick_params(axis='both', which='major', labelsize=14)

    plt.xlabel('Time, sec', fontsize=14)
    plt.ylabel('Velocity, m/sec', fontsize=14)
    plt.show()
    return 0

if __name__ == "__main__":
    main()