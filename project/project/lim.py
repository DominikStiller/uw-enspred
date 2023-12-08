import pickle
from pathlib import Path
from typing import Union

import dask.array
import numpy as np
import scipy
import xarray as xr
from numpy.typing import NDArray
from tqdm import tqdm
from xarray import Dataset, DataArray

from project.logger import get_logger
from project.util import stack_state, unstack_state, TimeConverter, get_timestamp

logger = get_logger(__name__)


class LIM:
    def __init__(self):
        self.tau0 = None
        self.G_tau0: NDArray = None
        self.C_tau0: NDArray = None
        self.L: NDArray = None
        self.mean: NDArray = None
        self.state_coords = None
        self.Nx = None  # state length
        self.time_converter = TimeConverter()

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        outfile = directory / f"lim-{get_timestamp()}.pkl"
        logger.info(f"Saving LIM to {outfile}")
        pickle.dump(self, outfile.open("wb"))

    @classmethod
    def load(cls, file: Path):
        return pickle.load(file.open("rb"))

    def fit(self, data: Union[Dataset, DataArray]):
        self.time_converter.fit(data.time.data)
        time = self.time_converter.forwards(data.time.data)
        taus = np.diff(time)
        self.tau0 = taus.mean()
        # Must have uniform dt
        assert all(np.isclose(taus, self.tau0)), "Sample times must be uniform"

        if isinstance(data, Dataset):
            data = stack_state(data)
        self.state_coords = data.state
        data = data.data  # extract Dask array
        self.Nx = len(self.state_coords)

        self.mean = data.mean(axis=1)[:, np.newaxis].compute()
        data -= self.mean

        data_t0 = data[:, :-1]
        data_tau0 = data[:, 1:]

        C_tau0 = data_tau0 @ data_t0.T
        C_t0 = data_t0 @ data_t0.T
        C_t0 /= data.shape[1] - 1
        C_tau0 /= data.shape[1] - 1

        self.G_tau0 = (C_tau0 @ np.linalg.inv(C_t0)).compute()
        # Take real part because logm introduces spurious imaginary parts
        self.L = np.real(scipy.linalg.logm(self.G_tau0) / self.tau0)

    def print_properties(self):
        print("G1:", self.G_tau0)
        print("Eigenvalues of G:", np.linalg.eigvals(self.G_tau0))
        print("L:", self.L)
        print("Eigenvalues of L:", np.linalg.eigvals(self.L))

    def forecast(self, initial: Union[Dataset, DataArray], n_steps, t0):
        unstack = False
        if isinstance(initial, Dataset):
            initial = stack_state(initial)
            unstack = True

        assert np.array_equal(
            self.state_coords.data, initial.state.data
        ), "Initial state must match training data."
        initial_np = np.squeeze(initial.values)

        forecast_np = np.zeros((self.Nx, n_steps + 1))
        time = self.time_converter.forwards(t0) + np.arange(0, (n_steps + 1) * self.tau0, self.tau0)
        forecast_np[:, 0] = initial_np

        G = self.G_tau0

        for i in tqdm(range(n_steps)):
            forecast_np[:, i + 1] = G @ initial_np
            G = self.G_tau0 @ G

        # forecast_np += self.mean

        forecast = xr.DataArray(
            self.mean + dask.array.from_array(forecast_np),
            dims=["state", "time"],
            coords={"state": self.state_coords, "time": self.time_converter.backwards(time)},
        )
        if unstack:
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
