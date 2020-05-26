import numpy as np

_L = 1
_g = 9.8


def f(X):
    phi, dphi = X[0], X[1]
    return np.array([dphi, -_g*np.sin(phi)/_L])


def simulate(X0, h, N):
    X = np.empty((N+1, len(X0)))
    X[0] = X0
    for i in range(N):
        x = X[i]
        k1 = h*f(x)
        k2 = h*f(x + 0.5*k1)
        k3 = h*f(x + 0.5*k2)
        k4 = h*f(x + k3)

        X[i+1] = X[i] + (k1 + 2*k2 + 2*k3 + k4)/6

    return X