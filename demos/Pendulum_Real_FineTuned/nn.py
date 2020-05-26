import numpy as np

import keras
from keras import backend as K
from keras import regularizers
from keras import optimizers
from keras.models import Sequential, Model, Input
from keras.layers import Dense, LSTM
from keras import backend as K

from keras.regularizers import L1L2

import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent.parent)

from tm_pnn.layers.Taylor_Map import TaylorMap
from tm_pnn.regularization.symplectic import dim2_order3


def create_TMPNN(inputDim=2, outputDim=2, order=3):
    ''' Creates polynomial neural network based on Taylor map'''
    model = Sequential()
    model.add(TaylorMap(output_dim = outputDim, order=order,
                        input_shape = (inputDim,)
                        #weights_regularizer=lambda W: dim2_order3(0.009, W),
              ))

    opt = optimizers.Adamax(clipvalue=1e-3)
    model.compile(loss='mean_squared_error', optimizer=opt)
    return model


def create_multioutput_TMPNN(N_outs, W=None, dim=2, outputDim=2, order=3):
    first_input = Input(shape=(dim, ))
    m = first_input
    outs = []

    pnn_layer = TaylorMap(output_dim = dim, order = order, input_shape = (dim,),
                          initial_weights=W)

    for i in range(N_outs):
        m = pnn_layer(m)
        outs.append(m)

    model = Model(inputs=[first_input], outputs=outs)

    # model.compile(optimizer=optimizers.Adam(clipvalue=0.5e-5, clipnorm=0.5e-5),
    # model.compile(optimizer=optimizers.Adam(clipvalue=1e-6, clipnorm=1e-6),
    model.compile(optimizer=optimizers.Adam(clipvalue=1e-8, clipnorm=1e-8),
    # model.compile(optimizer=optimizers.Adamax(),
                  loss=lambda x_true, x_pred: K.mean(K.square(x_pred[:, 0] - x_true[:, 0])))#+K.mean(K.square(x_pred[:, 1] - x_true[:, 1])))
    return model


def create_LSTM(lsinputDim, outputDim, n_units=225):
    model = Sequential()
    model.add(LSTM(n_units, input_shape=(1,2),
                   #bias_regularizer=L1L2(0.1, 0.1),
                   #kernel_regularizer=L1L2(0.001, 0.001),
                   #recurrent_regularizer=L1L2(0.9, 0.9)
             ))
    model.add(Dense(outputDim, kernel_initializer='uniform', activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adamax')
    return model


def iterative_predict(model, X0, N, reshape = False):
    ans = np.empty((N, X0.shape[0], X0.shape[1]))
    X = X0
    for i in range(N):
        if reshape:
            X = X.reshape(X0.shape[0], 1, X0.shape[1])
        X = model.predict(X)
        ans[i] = X

    return np.vstack((X0[np.newaxis,:], ans))

