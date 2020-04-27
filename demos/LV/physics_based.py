import numpy as np

_g = 9.8

def f(X):
    x,y = X[0], X[1]
    return np.array([y + x*y, -2*x - x*y])


def rk4(X0, stop=4.6, step=0.1):
    t = np.arange(start=0, stop=stop, step=step)
    v = np.zeros((len(t), len(X0)))
    v[0] = X0
    for i in range(1, len(t)):
        v0 = v[i-1]
        k1 = step*f(v0)
        k2 = step*f(v0 + 0.5*k1)
        k3 = step*f(v0 + 0.5*k2)
        k4 = step*f(v0 + k3)

        v[i] = v0 + (k1 + 2*k2 + 2*k3 + k4)/6

    return t, v
