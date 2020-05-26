import numpy as np
from ocelot import *

from sklearn.linear_model import LinearRegression

import keras
from keras.models import Model, Input
from keras import backend as K

from p3x_v23 import ring

import sys
from pathlib import Path
sys.path[0] = str(Path(sys.path[0]).parent.parent)
from tm_pnn.layers.Taylor_Map import TaylorMap


def get_transfermaps(sequence, dim=2):
    method = MethodTM()
    method.global_method = SecondTM

    lattice = MagneticLattice(sequence,  method=method)

    for i, tm in enumerate(get_map(lattice, lattice.totalLen, Navigator(lattice))):
        name = type(lattice.sequence[i]).__name__
        if hasattr(lattice.sequence[i], 'cs_id'):
            tag_id = lattice.sequence[i].cs_id
        else:
            tag_id = []

        if name in ['Marker']:
            continue

        if hasattr(tm, 'r_z_no_tilt'):
            R = tm.r_z_no_tilt(tm.length, 0)[:dim, :dim]
        else:
            R = tm.R(0)[:dim, :dim]

        if hasattr(tm, 't_mat_z_e'):
            T = tm.t_mat_z_e(tm.length, 0)[:dim, :dim, :dim].reshape((dim, -1))
        else:
            T = None

        yield R, T, name, lattice.sequence[i].l, tag_id


def concatenate(R1, T1, R2, T2):
    dim = R1.shape[0]
    if T2 is None:
        T2 = np.zeros((dim, dim*dim))

    R = np.dot(R2, R1)
    R1_2 = np.kron(R1, R1)
    T = np.dot(R2, T1) + np.dot(T2, R1_2)
    return R, T


def normalize_lattice(lattice, start_with):
    if start_with is None:
        return lattice
    names_list = [el.cs_id for el in lattice]
    for i, names in enumerate(names_list):
        if start_with in names:
            break

    return lattice[i:] + lattice[:i]


def get_model(dim = 2, Ls = 0.009, N=None, progress_bar=None, correctors=['Hcor', 'Vcor'], exclude_names=["pkhs"], start_with='screenmon', end_with=None, add_last_output=True):
    first_input = Input(shape=(dim, ))
    m1 = first_input
    outs = []
    info = {key:[] for key in correctors}
    info['Monitor'] = []
    Length = [0]


    W1 = np.eye(dim)
    W2 = np.zeros((dim, dim*dim))
    cur_length = 0

    order = 2
    K=1

    ring_normalized = normalize_lattice(ring, start_with)

    if N==None:
        N = len(list(get_transfermaps(ring_normalized, dim)))

    if progress_bar!=None:
        bar = progress_bar(N)

    is_end = False
    for i, (R,T,name,length,id) in enumerate(get_transfermaps(ring_normalized, dim)):
        is_excluded = []
        for id_tag in id:
            is_excluded.append(True in [id_tag.startswith(tag) for tag in exclude_names])
        is_excluded = True in is_excluded

        if name not in correctors or is_excluded: # if element is not located separately in lattice (correctors)
            W1, W2 = concatenate(W1, W2, R, T) # concatenate it with previous
            cur_length+=length

        if (name in info and not is_excluded): # stop concatenation
            #add concatenated section
            element_map = TaylorMap(output_dim = dim, order = order, input_shape = (dim,),
                                    initial_weights=[np.zeros((1,dim)), W1.T, W2.T],
                                    # weights_regularizer=lambda W: sympl_reg(Ls, W),
                                    name=f'TMC_{i}')
            if name not in correctors:
                element_map.tag_name = name
                element_map.tag_id = id

            m1 = element_map(m1)
            Length.append(cur_length)

            # add corrector
            if name in correctors:
                K+=1
                W1 = np.eye(dim)
                W2 = np.zeros((dim, dim*dim))

                W1, W2 = concatenate(W1, W2, R, T)
                element_map = TaylorMap(output_dim = dim, order = order, input_shape = (dim,),
                                        initial_weights=[np.zeros((1,dim)), W1.T, W2.T],
                                        # weights_regularizer=lambda W: sympl_reg(Ls, W),
                                        name=f'TM_{i}')
                element_map.trainable = False
                element_map.tag_name = name
                element_map.tag_id = id

                m1 = element_map(m1)
                Length.append(length)

            if name in info:
                info[name].append(K) # add element location

            if name == 'Monitor':
                outs.append(m1)

            K+=1
            W1 = np.eye(dim)
            W2 = np.zeros((dim, dim*dim))
            cur_length = 0

        if end_with in id:
            break


        if progress_bar!=None:
            bar.update(i)
        if(i>=N):
            break

    if cur_length>0:
        # to do: remove copypaste
        element_map = TaylorMap(output_dim = dim, order = order, input_shape = (dim,),
                                    initial_weights=[np.zeros((1,dim)), W1.T, W2.T],
                                    # weights_regularizer=lambda W: sympl_reg(Ls, W),
                                    name=f'TMC_{i}')
        m1 = element_map(m1)
        Length.append(cur_length)
        if add_last_output:
            outs.append(m1) # add last layer as output

    m1 = Model(inputs=[first_input], outputs=outs)
    info['Length'] = np.cumsum(np.array(Length))

    return m1, info


class Adapter:
    def tracks(self, N=3, xp_lims = [-1e-5, 1e-5], yp_lims = [-5e-6, 5e-6]):
        model = self.model
        # dim = model.inputs[0].shape[1].value # TF1
        dim = model.inputs[0].shape[1] # TF2

        X0 = np.zeros((N*dim//2, dim))
        X0[:N, 1] = np.linspace(xp_lims[0], xp_lims[1], N)
        X0[N//2]*=0 # reference track with initial conditions x(0)=0, x'(0)=0
        if dim > 2:
            X0[N:, 3] = np.linspace(yp_lims[0], yp_lims[1], N)
            X0[N+N//2]*=0 # reference track with initial conditions y(0)=0, y'(0)=0

        X = np.array(model.predict(X0))
        # X = np.vstack((X0[np.newaxis, :], X))
        return X

    def set_cor_by_tag(self, correctors, tag):
        pass

    def set_correctors(self, h_cors=None, v_cors=None):
        if h_cors is not None:
            self.set_cor_by_tag(h_cors, 'Hcor')
        if v_cors is not None:
            self.set_cor_by_tag(v_cors, 'Vcor')


class Ocelot2TFAdapter(Adapter):
    def __init__(self, dim=2, Ls = 0.009, N=None, progress_bar=None, separators=['Hcor', 'Vcor'], exclude_names=["pkhs", "pkvs_", "pkvsa_"], start_with='screenmon', end_with=None, add_last_output=True):
        model, info = get_model(dim=dim, Ls=Ls, N=N, progress_bar=progress_bar, correctors=separators, start_with=start_with, exclude_names=exclude_names, end_with=end_with, add_last_output=add_last_output)
        self.progress_bar = progress_bar
        self.model = model
        self.info = info
        self.dim = dim
        self.lengths = info['Length'][info['Monitor']]
        if add_last_output and len(info['Monitor']) < len(model.outputs):
            self.lengths = np.append(self.lengths, info['Length'][-1])

        for tag in ['Hcor', 'Vcor']:
            names = self.get_variable_names(tag)
            self.info[tag] = self.info[tag][:len(names)]

    def get_variable_names(self, tag):
        ids = []
        if tag not in self.info:
            return []
        for i in self.info[tag]:
            if i < len(self.model.layers):
                id = self.model.layers[i].tag_id
                if len(id)>1:
                    print(f'first id is taken in {id}')
                ids.append(id[0])
        return ids

    def set_cor_by_tag(adapter, correctors, tag):
        model = adapter.model
        info = adapter.info
        if tag=='Hcor':
            ci = 1
        else:
            ci = 3

        for i, el in enumerate(info[tag]):
            W = model.layers[el].get_weights()
            if W[0].shape[1]==2: #####
                ci=1             #####
            W[0][0,ci] = correctors[i]
            model.layers[el].set_weights(W)
        return

    def build_response_matrix(self, X_input=None, dc=1e-6):
        num_cor = len(self.info['Hcor'])
        controls_number = len(self.info['Hcor']) + len(self.info['Vcor'])
        model = self.model

        C = np.eye(controls_number)*dc
        B = np.empty((controls_number, 246*2))

        c = C[1]
        self.set_correctors(0*c[:num_cor], 0*c[num_cor:])
        if X_input is None:
            X_input = np.zeros((1, self.dim))
        x0 = np.array(model.predict(X_input))[:, 0, [0,2]].ravel()
        # x0 = X1[:, [0,2]].ravel()

        progress_bar = self.progress_bar
        if progress_bar!=None:
            bar = progress_bar(controls_number)

        for i,c in enumerate(C):
            self.set_correctors(c[:num_cor], c[num_cor:])
            xpred = np.array(model.predict(X_input))[:, 0, [0,2]].ravel()
            B[i] = xpred - x0
            if progress_bar!=None:
                bar.update(i)

        reg = LinearRegression(fit_intercept=False)
        reg.fit(C, B)
        print(reg.score(C, B))

        self.set_correctors(0*c[:num_cor], 0*c[num_cor:])
        M = reg.coef_
        self.R = np.linalg.pinv(M, rcond=1e-2)
        return

    def orbit2zero(self, X, Y, N=None):
        model = self.model
        R = self.R
        BPM = np.vstack((X, Y)).T.ravel()
        if N is None:
            N=len(BPM)//2

        c_svd = np.dot(R[:, :2*N], -BPM[:2*N])
        # c_svd = np.dot(R, -BPM)
        return c_svd


def main():
    adapter = Ocelot2TFAdapter(dim=2, N=100)

    model = adapter.model
    print(len(model.layers))
    bpms = adapter.get_variable_names('Monitor')
    print(bpms)
    # print(adapter.misalignments)

    return 0


if __name__ == "__main__":
    main()