import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import Input, Model
from keras import backend as K
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.regularizers import L1L2

from physics_based import rk4

import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent.parent)
from tm_pnn.layers.Taylor_Map import TaylorMap



def f(t, X, a=-2, b=-1, d=-1, g=-1):
    x = X[0]
    y = X[1]
    return np.array([ a*x - b*x*y, d*x*y - g*y])


def integrate(X0, dt, N, a=-2, b=-1, d=-1, g=-1):
    ans = np.empty((N, 2))
    t = 0
    X = X0
    print(X0 - np.array([g/d, a/b]))
    for i in range(N):
        k1 = f(t, X)
        k2 = f(t+dt/2.0, X+dt*k1/2.0)
        k3 = f(t+dt/2.0, X+dt*k2/2.0)
        k4 = f(t+dt, X + dt*k3)

        X = X + dt*(k1+2*k2+2*k3+k4)/6.0
        ans[i] = X
        t += dt

    # normalization to the fixed point:
    ans -= np.array([g/d, a/b])
    return ans

N = 465
dt = 0.01

X_train = integrate(np.array([1.5, 2.5]), dt, N)
X_test_outer  = integrate(np.array([1.8, 2.8]), dt, N)
X_test_inner  = integrate(np.array([1.1, 2.1]), dt, N)
X_test_fixed  = integrate(np.array([1.0, 2.0]), dt, N)


time = np.arange(N)*dt

if False:
    f, ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].plot(X_train[:,0], X_train[:,1], linestyle='-.', label='train data')
    ax[0].plot(X_test_fixed[:,0], X_test_fixed[:,1], marker='o', linestyle='none', markersize=2, label='test: fixed point')
    ax[0].plot(X_test_outer[:,0], X_test_outer[:,1], label='test: outer track')
    ax[0].plot(X_test_inner[:,0], X_test_inner[:,1], label='test: inner track')

    time = np.arange(N)*dt

    ax[1].plot(time, X_train[:,0], linestyle='--', label='train data')
    ax[1].plot(time, X_test_fixed[:,0], label='test: fixed point')
    ax[1].plot(time, X_test_outer[:,0], label='test: outer track')
    ax[1].plot(time, X_test_inner[:,0], label='test: inner track')

    for k in [0,1]:
        handles, labels = ax[k].get_legend_handles_labels()
        ax[k].legend(handles, labels)

    # ax[0].set_title('Dynamics in phase space')
    # ax[1].set_title('Dynamics in time space')

    ax[0].legend(fontsize=14)
    ax[1].legend(fontsize=14)

    ax[0].grid()
    ax[1].grid()

    ax[0].tick_params(axis='both', which='major', labelsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=14)

    ax[0].set_xlabel('x', fontsize=14)
    ax[0].set_ylabel('y', fontsize=14)
    ax[1].set_xlabel('Time, s', fontsize=14)
    ax[1].set_ylabel('x', fontsize=14)

    plt.show()
    quit()



def createTMPNN(inputDim=2, outputDim=2, order=2):
    model = Sequential()
    model.add(TaylorMap(output_dim = outputDim, order=order,
                      input_shape = (inputDim,)))
    opt = keras.optimizers.Adamax(lr=0.02, beta_1=0.99,
                                  beta_2=0.99999, epsilon=1e-1, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


def createMLP(inputDim, outputDim):
    model = Sequential()
    model.add(Dense(5, input_dim=inputDim, init='uniform', activation='sigmoid'))
    # model.add(Dense(4, init='uniform', activation='sigmoid'))
    model.add(Dense(outputDim, init='uniform', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    return model


def createLSTM(inputDim, outputDim):
    model = Sequential()
    model.add(LSTM(5, input_dim=inputDim, input_length=1,
                   # bias_regularizer=L1L2(0.1, 0.1),
                   # kernel_regularizer=L1L2(0.1, 0.1),
                   # recurrent_regularizer=L1L2(0.6, 0.6)
                    ))
    model.add(Dense(outputDim, init='uniform', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    return model


def iterative_predict(model, X0, N, reshape = False):
    ans = np.empty((N, 2))
    X = X0.reshape(-1,2)
    for i in range(N):
        if reshape:
            X = model.predict(X.reshape(1,1,2))
        else:
            X = model.predict(X)
        ans[i] = X
    return np.vstack((X0, ans[:-1]))


num_epoch = 2000


pnn = createTMPNN(2,2)
pnn.fit(X_train[:-1], X_train[1:], epochs=num_epoch, batch_size=50, verbose=0)
pnn.fit(X_train[:-1], X_train[1:], epochs=1, batch_size=50, verbose=1)
print('TMPNN is built')

mlp = createMLP(2,2)
mlp.fit(X_train[:-1], X_train[1:], epochs=num_epoch, batch_size=50, verbose=0)
mlp.fit(X_train[:-1], X_train[1:], epochs=1, batch_size=50, verbose=1)
print('MLP is built')


lstm = createLSTM(2,2)
lstm.fit(X_train[:-1].reshape((-1, 1, X_train.shape[1])), X_train[1:], epochs=num_epoch, batch_size=50, verbose=0)
lstm.fit(X_train[:-1].reshape((-1, 1, X_train.shape[1])), X_train[1:], epochs=1, batch_size=50, verbose=1)
print('LSTM is built')




# draw true train and test data
f, ax = plt.subplots(2,3,figsize=(15,10))
for i in range(3):
    ax[0, i].plot(X_train[:,0], X_train[:,1], linestyle='-.', alpha=0.2)
    ax[0, i].plot(X_test_fixed[:,0], X_test_fixed[:,1], marker='o', linestyle='none', markersize=2, alpha=0.2)
    ax[0, i].plot(X_test_outer[:,0], X_test_outer[:,1], alpha=0.2)
    ax[0, i].plot(X_test_inner[:,0], X_test_inner[:,1], alpha=0.2)

    ax[1, i].plot(time, X_train[:,0], linestyle='-.', label='train data', alpha=0.2)
    ax[1, i].plot(time, X_test_fixed[:,0], label='test: fixed point', alpha=0.2)
    ax[1, i].plot(time, X_test_outer[:,0], label='test: outer track', alpha=0.2)
    ax[1, i].plot(time, X_test_inner[:,0], label='test: inner track', alpha=0.2)


for i in range(2):
    ax[i,0].set_title('Lie transform')
    ax[i,1].set_title('MLP')
    ax[i,2].set_title('LSTM')



ax[1,0].set_prop_cycle(None)
ax[1,1].set_prop_cycle(None)
ax[1,2].set_prop_cycle(None)


# then predict via different neural networks
reshapes = [False, False, True]
for i, model in enumerate([pnn, mlp, lstm]):
    X_train_predict      = iterative_predict(model, X_train[0], N, reshape = reshapes[i])
    X_test_outer_predict = iterative_predict(model, X_test_outer[0], N, reshape  = reshapes[i])
    X_test_inner_predict = iterative_predict(model, X_test_inner[0], N, reshape  = reshapes[i])
    X_test_fixed_predict = iterative_predict(model, X_test_fixed[0], N, reshape  = reshapes[i])

    ax[0,i].set_prop_cycle(None)
    ax[1,i].set_prop_cycle(None)

    l = ['train', 'fixed point', 'outer track', 'inner track']
    for a, X_predict in enumerate([X_train_predict, X_test_fixed_predict, X_test_outer_predict, X_test_inner_predict]):
        ax[0, i].plot(X_predict[:,0], X_predict[:,1], linestyle='--', label=f'predict: {l[a]}')
        ax[1, i].plot(time, X_predict[:,0], linestyle='--')

for i in range(2):
    for j in range(3):
        ax[i,j].legend(fontsize=14)
        ax[i,j].legend(fontsize=14)

        ax[i,j].grid()

        ax[i,j].tick_params(axis='both', which='major', labelsize=14)

    ax[0,0].set_xlabel('x', fontsize=14)
    # ax[0].set_ylabel('y', fontsize=14)
    # ax[1].set_xlabel('Time, s', fontsize=14)
    # ax[1].set_ylabel('x', fontsize=14)

for j  in range(3):
    ax[0,j].set_xlabel('x', fontsize=14)
    ax[0,j].set_ylabel('y', fontsize=14)
    ax[1,j].set_xlabel('Time, s', fontsize=14)
    ax[1,j].set_ylabel('x', fontsize=14)

plt.show()
