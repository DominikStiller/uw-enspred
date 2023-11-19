import numpy as np
import scipy


def run_oscillator(t_end=200, dt=0.1, sigma=0.001, initial=[1, 0]):
    m = 20
    c = 0.4
    k = 2

    L = np.array([[0, 1], [-k / m, -c / m]])
    G = scipy.linalg.expm(L * dt)

    print("G:", G)
    print("Eigenvalues of G:", np.linalg.eigvals(G))
    print("L:", L)
    print("Eigenvalues of L:", np.linalg.eigvals(L))

    tt = np.arange(0, t_end, dt)
    xx = np.empty((2, len(tt)))
    xx[:, 0] = initial

    for i, t in enumerate(tt[1:]):
        # xx[:, i + 1] = np.linalg.matrix_power(G, i + 1) @ xx[:, 0]
        xx[:, i + 1] = xx[:, i] + L @ xx[:, i] * dt

    yy = xx + np.random.normal(scale=sigma, size=xx.shape)

    return tt, xx, yy
