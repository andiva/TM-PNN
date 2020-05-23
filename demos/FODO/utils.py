import numpy as np
import sympy as sp


def getPhaseStateVector(dim):
    state = []
    for i in range(dim):
        state.append(sp.Symbol('x%s'% str(i+1).zfill(2)))
    return sp.Array(state)


def load_TM_map(filepath, dim=6):
    state = getPhaseStateVector(dim=dim)
    # print('load weights: ', filepath)
    state_power = np.array([1])
    order = 0

    weights = []
    W = []
    with open(filepath, 'r') as file:
        for row in file:
            row = row.replace('\r', '')
            row = row.replace('\n', '')

            if(0==len(row)):
                W = np.array(W)
                W_ext = np.zeros((len(state), len(state_power)))
                str_power = state_power.astype(str)
                reduced_str_power = []
                sl = []
                for i, el in enumerate(str_power):
                    if el not in reduced_str_power:
                        reduced_str_power.append(el)
                        sl.append(i)

                if len(W.shape)==1:
                    W = W.reshape((len(state), 1))
                W_ext[:, sl] = W
                weights.append(W_ext.T)
                state_power = np.kron(state_power, state)
                W = []
                continue

            if row[0] != '%':
                row = list(map(float, row.split(' ')))
                W.append(row)
    return weights