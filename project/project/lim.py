import numpy as np
from numpy.typing import ArrayLike
import xarray as xr
from xarray import DataArray

from project.util import stack_state, estimate_cov, inverse, matrix_power


class LIM:
    def __init__(self):
        self.tau0 = None
        self.G_tau0: DataArray = None
        self.C_t0: DataArray = None
        self.mean = None
        self.state_coords = None

    def fit_np(self, data):
        self.tau0 = 0.1

        self.mean = data.mean(axis=1)[:, np.newaxis]
        data -= self.mean

        data_t0 = data[:, :-1]
        data_tau0 = data[:, 1:]
        C_tau0 = data_tau0 @ data_t0.T
        C_t0 = data_t0 @ data_t0.T
        self.G_tau0 = C_tau0 @ np.linalg.inv(C_t0)

    def fit(self, data: DataArray):
        data = stack_state(data)
        self.state_coords = data.state

        self.mean = data.mean("time")
        data -= self.mean

        taus = data["time"].diff("time").values
        assert all(np.isclose(taus[0], taus))  # must have uniform dt
        self.tau0 = taus[0].item()

        data_t0 = data.isel(time=slice(None, -1))
        data_tau0 = data.isel(time=slice(1, None))

        C_tau0 = estimate_cov(data_tau0, data_t0)
        C_t0 = estimate_cov(data_t0, data_t0)

        self.G_tau0 = xr.dot(C_tau0, inverse(C_t0), dims=[])
        self.C_t0 = C_t0

    def print_properties(self):
        print("G1:", self.G_tau0)
        print("L:", np.log(self.G_tau0) / self.tau0)
        print("Eigenvalues of L:", np.log(np.linalg.eigvals(self.G_tau0)) / self.tau0)

    def forecast(self, initial: np.array, n_steps):
        tt = np.arange(0, n_steps * self.tau0, self.tau0)
        xx = np.empty([len(initial), n_steps])
        xx[:, 0] = initial

        for i in range(n_steps - 1):
            xx[:, i + 1] = matrix_power(self.G_tau0, i + 1) @ initial
            # xx[:, i + 1] = np.linalg.matrix_power(self.G_tau0, i + 1) @ initial

        xx += self.mean

        # ds = xr.Dataset({
        #     "x": (["time"], yy[0,:]),
        #     "v": (["time"], yy[1,:]),
        # }, coords={
        #     "time": tt
        # })

        # TODO unstack

        return tt, xx
