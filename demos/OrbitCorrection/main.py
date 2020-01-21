import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_elementwise_model, get_bpm_model
from utils import _misalignements


def plot_traks(ax, model, lengths, N=11, dim=2):
    X0 = np.zeros((N, dim))
    X0[:, 1] = np.linspace(-8e-6, 8e-6, N)
    X0[N//2]*=0 # reference track with initial conditions x(0)=0, x'(0)=0

    X = np.array(model.predict(X0))
    X = np.vstack((X0[np.newaxis, :], X))

    ax.plot(lengths, X[:, :, 0], alpha=0.7)
    ax.plot(lengths, X[:, N//2, 0], c='k')

    return X


def set_correctors(model, corrections):
    cors = iter(corrections)
    for layer in model.layers[1:]:
        if hasattr(layer, 'tag') and layer.tag == 'Marker':
            W = layer.get_weights()
            W[0][0,1] = next(cors)
            layer.set_weights(W)
    return


def MAE_BPM(model, corrections):
    set_correctors(model, corrections)
    X = np.array(model.predict(np.array([0.0, 0.0]).reshape(1, -1)))[:, 0, 0] # get only x location
    return np.abs(X).max()


def optimal_control(model, controls_number):
    x0 = np.zeros(controls_number)
    res = minimize(lambda x: MAE_BPM(model, x), x0, method='Nelder-Mead', options={'maxiter': 100, 'disp': True, 'fatol': 1e-6, 'xatol': 1e-6})

    print(f'optimal control MAE on BPMs: {MAE_BPM(model, res.x)}')
    return res.x


def main():
    dim = 2
    order = 2
    model_ideal, lengths, bpm_inds, mrk_inds = get_elementwise_model(dim, order, None, sigm=0) # ideal lattice without misalignements
    # model_real, lengths, bpm_inds, mrk_inds = get_elementwise_model(dim, order, _misalignements) # real lattice with misalignements
    model_real, lengths, bpm_inds, mrk_inds = get_elementwise_model(dim, order, None) # real lattice with misalignements

    _, axs = plt.subplots(3,2, sharex=True)
    axs = axs.ravel()

    plot_traks(axs[0], model_ideal, lengths, dim=dim)
    plot_traks(axs[2], model_real, lengths, dim=dim)

    X_input = np.array([0.0, 0.0]).reshape(1, -1)
    X_train = np.array(model_real.predict(X_input))[:, 0, :]

    axs[1].plot(lengths[1:], X_train[:, 0], alpha=0.7, color='k')
    axs[1].plot((lengths[1:])[bpm_inds], X_train[bpm_inds, 0], color='k', marker='*', markersize=5, linestyle='none')


    model_real_bpm = get_bpm_model(model_real)
    c_true = optimal_control(model_real_bpm, len(mrk_inds))
    plot_traks(axs[4], model_real, lengths, dim=dim)

    model_trained_bpm = get_bpm_model(model_ideal)
    # model_trained_bpm.fit(X_input, [x.reshape(1,-1) for x in X_train[bpm_inds]], epochs=100, verbose=1)
    # plot_traks(axs[3], model_ideal, lengths, dim=dim)
    model_trained_bpm.fit(X_input, [x.reshape(1,-1) for x in X_train[bpm_inds]], epochs=2, verbose=1)
    plot_traks(axs[3], model_ideal, lengths, dim=dim)
    model_trained_bpm.fit(X_input, [x.reshape(1,-1) for x in X_train[bpm_inds]], epochs=99, verbose=1)
    plot_traks(axs[5], model_ideal, lengths, dim=dim)

    c_pred = optimal_control(model_trained_bpm, len(mrk_inds))
    # plot_traks(axs[5], model_real, lengths, dim=dim)

    print('BPM erros:')
    print(MAE_BPM(model_trained_bpm, c_pred))
    print(MAE_BPM(model_real_bpm, c_pred))
    print(MAE_BPM(model_real_bpm, 0*c_pred))


    #-plot attributes:---------------------------------------------------------
    for ax in axs:
        ax.set_ylim([-3.5e-4, 3.5e-4])

    for bi in bpm_inds:
        for i in [1, 2]:
            axs[i].axvline(lengths[bi], linestyle='--', color='k', alpha=0.2)

    for mi in mrk_inds:
        axs[4].axvline(lengths[mi], linestyle='--', color='r', alpha=0.2)

    axs[0].set_title('Ideal dynamics by design')
    axs[2].set_title('Real dynamics with imperfections')
    axs[4].set_title('Optimal control of beam')

    axs[1].set_title('Training data: 11 points of beam location')
    axs[3].set_title('Dynamics recovering after 2 training epochs')
    axs[5].set_title('Dynamics recovering after 100 training epochs')


    axs[2].set_ylabel('x, meters')
    axs[4].set_xlabel('Length along the lattice, meters')
    plt.show()
    return 0


if __name__ == "__main__":
    sns.set()
    main()