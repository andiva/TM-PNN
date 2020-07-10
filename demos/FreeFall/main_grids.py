import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as MSE
from keras import Input, Model
from keras import optimizers
from keras import backend as K

# fix paths:
import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent.parent)
from tm_pnn.layers.Taylor_Map import TaylorMap
#
# from Taylor_Map import TaylorMap


def analytic(t, m, k, g=9.8):
    return np.sqrt(m*g/k)*np.tanh(t*np.sqrt(g*k/m))


def create_TMPNN(N, inputDim, outputDim, order=3):
    ''' Creates polynomial neural network based on Taylor map'''
    input = Input(shape=(inputDim,))
    m = input
    tm = TaylorMap(output_dim = outputDim, order=order,
                   input_shape = (inputDim,),)

    outs=[]
    for i in range(N):
        m = tm(m)
        outs.append(m)

    opt = optimizers.SGD(clipvalue=1e-11)
    opt = optimizers.SGD(clipvalue=1e-9)
    # opt = optimizers.RMSprop(clipvalue=1e-10)
    model = Model(inputs=input, outputs=outs)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model

from sklearn.linear_model import LinearRegression

def init_weights(v_data, im):

    counts = v_data.shape[1]
    counts2 = counts-1
    print(counts)
    X = np.empty((2*(counts-1), 3))
    Y = np.empty((2*(counts-1), 1))

    for i in range(v_data.shape[0]):
        v = v_data[i]
        x = np.ones((counts, 3))
        x[:,1] = v
        x[:,2] = v*v*im[i]

        X[i*counts2:(i+1)*counts2] = x[:-1]
        Y[i*counts2:(i+1)*counts2] = v[1:].reshape(-1,1)

    lin_model = LinearRegression(fit_intercept=False)

    lin_model.fit(X, Y)
    return lin_model.coef_[0]

def main():
    # K.set_floatx('float64')
    m = 100
    k = 0.392

    X0 = np.array([[0, 1/m],       #train
                   [0, 1/(m*0.64)],#train
                   [0, 1/(m*2)],   #test (unseen)
                   [0, 1/(m*1.44)],#test (unseen)
                   [0, 1/(m*0.3)], #test (unseen)
                   [0, 1/(m*0.1)], #test (unseen)
                   ])

    T = 15 # total time
    grid_num = [15, 30, 60] # grid size

    _, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax = ax.ravel()
    for n, N in enumerate(grid_num): # foreach grid_num
        h = T/N

        t = np.arange(0, N*h+h, h)

        v_analytic = np.zeros((X0.shape[0], len(t)))
        for i, m_inv in enumerate(X0[:, 1]): # foreach m
            v_analytic[i] = analytic(t, 1/m_inv, k)


        ax[n].plot(t, v_analytic.T) # plot analytical solutions
        #-----------------------------------------------------------------------
        init = init_weights(v_analytic[:2], X0[:2, 1])
        print(h, init)
        #-----------------------------------------------------------------------
        # train TM-PNN
        pnn = create_TMPNN(N, 2, 2, order=3)
        W = pnn.layers[1].get_weights()
        #init = np.array([4.864129, 0.99798733, -0.19208]) # initial approach for h = 0.5
        #W[0][0,0] = init[0]*h/0.5 #normalize to actual timestamp h
        #W[1][0,0] = init[1]       #normalize to actual timestamp h
        #W[3][1,0] = init[2]*h/0.5 #normalize to actual timestamp h

        W[0][0,0] = init[0]
        W[1][0,0] = init[1]
        W[3][1,0] = init[2]

        pnn.layers[1].set_weights(W)


        pnn.fit(X0[:2],
                [np.vstack((v, X0[:2, 1])).T for v in v_analytic[:2, 1:].T],
                epochs=1, verbose=1)
        #-----------------------------------------------------------------------
        X_pnn = np.array(pnn.predict(X0))

        ax[n].set_prop_cycle(None)
        ax[n].plot(t[1:], X_pnn[:,:,0], linestyle='--')
        ax[n].set_title(f'grid size N={N}')
        ax[n].set_ylabel('v')
        v_true = v_analytic.T[1:]
        v_predict = X_pnn[:, :, 0]

        # l2 = np.linalg.norm(v_true - v_predict, axis=0)
        l2 = MSE(v_true, v_predict, multioutput='raw_values')


        incr_order = [2,3,0,1,4,5] # re-order masses
        m = 1/X0[[2,3,0,1,4,5], 1]
        ax[3].plot(m, l2[incr_order], label=f'N={N}, h={h}')
        ax[3].legend()


    ax[2].set_xlabel('Time')
    ax[3].set_title("Mean Squared Error")
    ax[3].set_ylabel('MSE')
    ax[3].set_xlabel("Mass")
    ax[3].grid()
    plt.show()

    return 0

if __name__ == "__main__":
    main()