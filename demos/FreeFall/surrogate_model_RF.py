import numpy as np
import matplotlib.pyplot as plt

from physics_based import air_resistance2, simulate
from physics_based import _g as g

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR


def create_RF(n=100):
    model = ExtraTreesRegressor(n_estimators=n)
    return model


def iterative_predict(model, X0, N, reshape = False):
    ans = np.empty((N, X0.shape[0], X0.shape[1]))
    X = X0
    m = X0[0,1]
    for i in range(N):
        X = model.predict(X)
        X[0,1]=m
        ans[i] = X

    return np.vstack((X0[np.newaxis,:], ans))


def main():
    m = 100
    k = 0.392

    X0 = np.array([[0, 1/m], [0, 1/(m*0.64)], [0, 1/(m*1.44)], [0, 1/(m*0.3)], [0, 1/(m*2)], [0, 1/(m*0.1)]])
    h = 0.5
    N = 30

    t = np.arange(0, N*h+h, h)


    f0 = lambda X: air_resistance2(X, k=k)
    v0 = simulate(f0, X0[0], h, N)
    v1 = simulate(f0, X0[1], h, N)
    v2 = simulate(f0, X0[2], h, N)
    v3 = simulate(f0, X0[3], h, N)
    v4 = simulate(f0, X0[4], h, N)
    v5 = simulate(f0, X0[5], h, N)

    rf = create_RF(n=50)
    rf.fit( np.vstack((v0[:-1], v1[:-1])),
            np.vstack((v0[1:], v1[1:])))

    X_rf0 = iterative_predict(rf, X0[0].reshape(1,2), N)
    X_rf1 = iterative_predict(rf, X0[1].reshape(1,2), N)
    X_rf2 = iterative_predict(rf, X0[2].reshape(1,2), N)
    X_rf3 = iterative_predict(rf, X0[3].reshape(1,2), N)
    X_rf4 = iterative_predict(rf, X0[4].reshape(1,2), N)
    X_rf5 = iterative_predict(rf, X0[5].reshape(1,2), N)



    plt.plot(t, v0[:,0], label='true, m = 100')
    plt.plot(t, v1[:,0], label='true, m = 64')
    plt.plot(t-1, v2[:,0], label='true, m = 144')
    plt.plot(t+1, v3[:,0], label='true, m = 30')
    plt.plot(t-2, v4[:,0], label='true, m = 200')
    plt.plot(t+2, v5[:,0], label='true, m = 10')

    plt.gca().set_prop_cycle(None)
    plt.plot(t, X_rf0[:, :, 0], linestyle='--', marker='o', fillstyle='none', label='RFR, m = 100 (training)')
    plt.plot(t, X_rf1[:, :, 0], linestyle='--', marker='o', fillstyle='none', label='RFR, m = 64 (training)')
    plt.plot(t-1, X_rf2[:, :, 0], linestyle='--', label='RFR, m = 144 (unseen)')
    plt.plot(t+1, X_rf3[:, :, 0], linestyle='--', label='RFR, m = 30 (unseen)')
    plt.plot(t-2, X_rf4[:, :, 0], linestyle='--', label='RFR, m = 200 (unseen)')
    plt.plot(t+2, X_rf5[:, :, 0], linestyle='--', label='RFR, m = 10 (unseen)')

    plt.ylim([-2, 72])
    plt.legend(fontsize=14)

    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=14)
    plt.subplots_adjust(right=0.7)
    plt.grid()
    plt.gca().tick_params(axis='both', which='major', labelsize=14)

    plt.xlabel('Time, sec', fontsize=14)
    plt.ylabel('Velocity, m/sec', fontsize=14)
    plt.show()

    return 0

if __name__ == "__main__":
    main()