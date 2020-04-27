import numpy as np

_g = 9.8

def air_resistance(X, k, m):
    v = X[0]
    return np.array([_g - k*v*v/m])

def air_resistance2(X, k):
    v = X[0]
    m = 1/X[1]

    return np.array([_g - k*v*v/m, 0])


def simulate(f, X0, h, N):
    X = np.empty((N+1, len(X0)))
    X[0] = X0
    for i in range(N):
        x = X[i]
        k1 = h*f(x)

        # X[i+1] = X[i] + k1

        k2 = h*f(x + 0.5*k1)
        k3 = h*f(x + 0.5*k2)
        k4 = h*f(x + k3)

        X[i+1] = X[i] + (k1 + 2*k2 + 2*k3 + k4)/6

    return X

def euler(v0, k, m, stop=15, step=0.5):

    t = np.arange(start=0, stop=stop, step=step)
    v = np.zeros_like(t)
    v[0] = v0
    for i in range(1, len(t)):
        v0 = v[i-1]
        dv = _g - k*v0*v0/m

        v[i] = v0 + dv*step
    print(f'estep = {step}: {[_g*step, 1, -step*k/m]}')
    return t, v

def rk4(v0, k, m, stop=15, step=0.5):

    f = lambda v: _g - k*v*v/m

    t = np.arange(start=0, stop=stop, step=step)
    v = np.zeros_like(t)
    v[0] = v0
    for i in range(1, len(t)):
        v0 = v[i-1]
        k1 = step*f(v0)
        k2 = step*f(v0 + 0.5*k1)
        k3 = step*f(v0 + 0.5*k2)
        k4 = step*f(v0 + k3)

        v[i] = v0 + (k1 + 2*k2 + 2*k3 + k4)/6

    # print(f'rkstep = {step}: {[_g*step, 1, -step*k/m]}')
    return t, v


def exact(t, k, m):
    mu = k/m

    return np.sqrt(_g/mu)*np.tanh(np.sqrt(_g*mu)*t)

if __name__ == "__main__":
    import sympy as sp
    v0 = sp.Symbol('x')
    f = lambda v: 1 + v*v

    step=0.1
    k1 = step*f(v0)
    k2 = step*f(v0 + 0.5*k1)
    k3 = step*f(v0 + 0.5*k2)
    k4 = step*f(v0 + k3)

    v0 = v0 + (k1 + 2*k2 + 2*k3 + k4)/6
    print(v0.expand())