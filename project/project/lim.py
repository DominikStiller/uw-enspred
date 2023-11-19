import numpy as np
import scipy
from numpy.typing import ArrayLike
import xarray as xr
from xarray import DataArray, Dataset

from project.util import stack_state, unstack_state, estimate_cov, inverse, matrix_power


class LIM:
    def __init__(self):
        self.tau0 = None
        self.G_tau0: ArrayLike = None
        self.C_t0: ArrayLike = None
        self.mean = None
        self.state_coords = None
        self.Nx = None  # state length

    def fit(self, data):
        taus = data["time"].diff("time").values
        assert all(np.isclose(taus[0], taus))  # must have uniform dt
        self.tau0 = taus[0].item()

        data = stack_state(data)
        self.state_coords = data.state
        self.Nx = len(self.state_coords)

        data_np = data.values

        self.mean = data_np.mean(axis=1)[:, np.newaxis]
        data_np -= self.mean

        data_t0 = data_np[:, :-1]
        data_tau0 = data_np[:, 1:]

        C_tau0 = data_tau0 @ data_t0.T
        C_t0 = data_t0 @ data_t0.T
        C_t0 /= data_np.shape[1] - 1
        C_tau0 /= data_np.shape[1] - 1

        self.G_tau0 = C_tau0 @ np.linalg.inv(C_t0)

    def print_properties(self):
        L = scipy.linalg.logm(self.G_tau0) / self.tau0

        print("G1:", self.G_tau0)
        print("Eigenvalues of G:", np.linalg.eigvals(self.G_tau0))
        print("L:", L)
        print("Eigenvalues of L:", np.linalg.eigvals(L))

    def forecast(self, initial: Dataset, n_steps):
        initial = stack_state(initial)
        assert initial.state.equals(
            self.state_coords
        ), "Initial state must match training data."
        initial_np = np.squeeze(initial.values)

        forecast_np = np.zeros((self.Nx, n_steps + 1))
        time = np.arange(0, (n_steps + 1) * self.tau0, self.tau0)
        forecast_np[:, 0] = initial_np

        for i in range(n_steps - 1):
            forecast_np[:, i + 1] = (
                np.linalg.matrix_power(self.G_tau0, i + 1) @ initial_np
            )

        forecast_np += self.mean

        forecast = xr.DataArray(
            forecast_np,
            dims=["state", "time"],
            coords={"state": self.state_coords, "time": time},
        )
        forecast = unstack_state(forecast)

        return forecast

    def forecast_np(self, initial: np.array, n_steps):
        time = np.arange(0, n_steps * self.tau0, self.tau0)
        forecast = np.empty([self.Nx, n_steps])
        forecast[:, 0] = initial

        for i in range(n_steps - 1):
            forecast[:, i + 1] = np.linalg.matrix_power(self.G_tau0, i + 1) @ initial

        forecast += self.mean

        return time, forecast
