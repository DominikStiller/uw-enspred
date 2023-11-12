import numpy as np


def run_oscillator(t_end=200, dt=0.1, sigma=0.01, initial=[1, 0]):
    m = 20
    c = 0.4
    k = 2

    L = np.array([[0, 1], [-k / m, -c / m]])

    print("G:", np.exp(L * dt))
    print("L:", L)
    print("Eigenvalues of L:", np.linalg.eigvals(L))

    tt = np.arange(0, t_end, dt)
    xx = np.empty((2, len(tt)))
    xx[:, 0] = initial

    for i, t in enumerate(tt[1:]):
        xx[:, i + 1] = xx[:, i] + L @ xx[:, i] * dt

    yy = xx + np.random.normal(scale=sigma, size=xx.shape)

    return tt, xx, yy
