import numpy as np
import matplotlib.pyplot as plt

from dynamics import simulate
from utils import create_PNN, create_LSTM, iterative_predict


def plot(ax, Xtrain, Xtest, title, alpha=1, labels = ['train', 'test'], linestyle='-'):
    ax.set_ylabel(title)

    if alpha<1:
        linestyle = '--'

    ax.plot(Xtrain[:, 0], alpha=alpha, linestyle=linestyle, label=labels[0])
    ax.plot(Xtest[:, 0], alpha=alpha, linestyle=linestyle, label=labels[1])
    ax.legend(loc='upper right')
    return


def main():
    N = 100
    ndim = 2
    order = 3 # nonlinear order in PNN

    X0 = np.array([[0.1, 0],
                   [0.5, 0]
                  ])

    X_train = simulate(X0[0], 0.1, N)
    X_test  = simulate(X0[1], 0.1, N)

    lstm = create_LSTM(ndim, ndim, n_units=15)
    lstm.fit(X_train[:-1].reshape((-1, 1, X_train.shape[1])),
             X_train[1:], epochs=500, batch_size=50, verbose=1)
    X_lstm = iterative_predict(lstm, X0, N, reshape=True)

    pnn = create_PNN(ndim, ndim, order=order)
    pnn.fit(X_train[:-1], X_train[1:], epochs=500, verbose=1)
    X_pnn = iterative_predict(pnn, X0, N)


    _, axs = plt.subplots(3,1,True)
    axs = axs.ravel()

    plot(axs[0], X_train, X_test, title='ODE', alpha=1)


    plot(axs[1], X_train, X_test, title='LSTM', alpha=0.5, labels=[None, None])
    axs[1].set_prop_cycle(None)
    plot(axs[1], X_lstm[:, 0], X_lstm[:, 1], title='LSTM')

    plot(axs[2], X_train, X_test, title='TM-PNN', alpha=0.5, labels=[None, None])
    axs[2].set_prop_cycle(None)
    plot(axs[2], X_pnn[:, 0], X_pnn[:, 1], title='TM-PNN')

    axs[0].set_title('Oscillation of angle (φ[rad])')
    axs[-1].set_xlabel('Time, sec')

    plt.figure()
    fixed_lstm = iterative_predict(lstm, np.zeros((1, 2)), 800, reshape=True)
    fixed_pnn  = iterative_predict(pnn,  np.zeros((1, 2)), 800)

    plt.plot(fixed_lstm[:,0,0])
    plt.plot(fixed_pnn[:,0,0])

    plt.title('Prediction of fixed point')
    plt.ylabel('φ[rad]')
    plt.xlabel('Time, sec')
    plt.legend(['LSTM', 'TM-PNN'])
    plt.show()
    return 0


if __name__ == "__main__":
    main()