import numpy as np
import pickle
import keras
from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model

import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent.parent)

from tm_pnn.layers.Taylor_Map import TaylorMap
from tm_pnn.regularization.symplectic import get_reg_term_2_2 as sympl_reg

from demos.PETRAIV.ocelot_lattice import get_transfermaps

_misalignements = [-1.3695576896492115e-05, 2.0035806729622158e-05, -8.88409800933373e-06, -1.049521063892591e-05,
                    1.3602575362196903e-05, -4.595850753574459e-06, 1.3659452596163099e-06, -4.360252174124462e-06,
                    1.8967831854026095e-06, -8.435597123581483e-06, 1.2725265379628313e-05, 5.40320889296336e-06,
                   -6.15591535694958e-06, 5.4174836280781275e-06, -1.0436204026881908e-05, 4.101442572899299e-06]


def get_sequential_model(dim, order, dx_list, sigm):
    model = Sequential()
    dim = 2
    order = 2
    lengths = [0]
    bpm_inds = []
    mrk_inds = []
    if dx_list is None:
        dx_list = np.random.normal(0, sigm, 1000) # hardcoded 1000 - max. number of elements in an acceleratror
    dx_list = iter(dx_list)

    for i, (R, T, name, length) in enumerate(get_transfermaps(dim=dim)):
        element_map = TaylorMap(output_dim = dim, order = order, input_shape = (dim,),
                                weights=[np.zeros((1,dim)), R.T, T.T],
                                weights_regularizer=lambda W: sympl_reg(0.009, W))
        element_map.tag = name

        dx = 0
        if name == 'Monitor':
            bpm_inds.append(i)
        elif name == 'Marker':
            mrk_inds.append(i)

        if name == 'Quadrupole':
            w0 = np.zeros((1,dim))
            dx = next(dx_list)

            print(f'missalignement is inserted: dx = {dx}')
            w0[0,0] = dx
            I1 = TaylorMap(output_dim = dim, order = 1, input_shape = (dim,),
                           weights=[w0, np.eye(dim)],
                           weights_regularizer=lambda W: sympl_reg(0.009, W))
            I1.tag='I'
            model.add(I1)

        model.add(element_map)
        lengths.append(length)

        if name == 'Quadrupole':
            W = element_map.get_weights()
            W[0][0,0] = -dx
            element_map.set_weights(W)

    print(f'true number of layers: {i+1}')

    lengths = np.cumsum(np.array(lengths))
    print('sequential model is built')
    return model, lengths, bpm_inds, mrk_inds


def get_elementwise_model(dim=2, order=2, dx_list=None, sigm=1e-5):
    model, lengths, bpm_inds, mrk_inds  = get_sequential_model(dim, order, dx_list, sigm)
    model = Model(inputs=model.input, outputs=[el.output for el in model.layers if el.tag != 'I'])
    return model, lengths, bpm_inds, mrk_inds


def get_bpm_model(model):
    model = Model(inputs=model.input, outputs=[el.output for el in model.layers if hasattr(el, 'tag') and el.tag == 'Monitor'])

    def x_loss(y_true, y_pred):
        return K.mean(K.square(y_pred[:, 0] - y_true[:, 0])) # MSE for x location

    opt = keras.optimizers.Adam(clipvalue=1e-10) # small gradients
    model.compile(loss=x_loss, optimizer=opt)
    return model


def main():
    return 0

if __name__ == "__main__":
    main()