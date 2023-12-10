import pickle
from pathlib import Path
from typing import Union

import dask.array
import numpy as np
import xarray as xr
from numpy.linalg import inv, eig, eigvals, pinv, eigh
from numpy.typing import NDArray
from tqdm import tqdm
from xarray import Dataset, DataArray

from project.logger import get_logger
from project.util import stack_state, unstack_state, TimeConverter, get_timestamp, is_dask_array

logger = get_logger(__name__)


class LIM:
    def __init__(self, max_neg_evals=5):
        self.max_neg_evals = max_neg_evals
        self.tau = None
        self.G_tau: NDArray = None
        self.L: NDArray = None
        self.Q_evals: NDArray = None
        self.Q_evecs: NDArray = None
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
        self.tau = taus.mean()
        # Must have uniform dt
        assert all(np.isclose(taus, self.tau)), "Sample times must be uniform"

        if isinstance(data, Dataset):
            data = stack_state(data)
        self.state_coords = data.state
        data = data.data  # extract Dask array
        self.Nx = len(self.state_coords)

        self.mean = data.mean(axis=1)[:, np.newaxis]
        if is_dask_array(self.mean):
            self.mean = self.mean.compute()

        data -= self.mean
        data_0 = data[:, :-1]
        data_tau = data[:, 1:]

        C_tau = (data_tau @ data_0.T) / (data.shape[1] - 1)
        C_0 = (data_0 @ data_0.T) / (data.shape[1] - 1)

        self._fit_dynamics(C_0, C_tau)
        self._fit_noise(C_0)

    def _fit_dynamics(self, C_0, C_tau):
        self.G_tau = C_tau @ inv(C_0)
        if is_dask_array(self.G_tau):
            self.G_tau = self.G_tau.compute()

        G_eval, G_evects = eig(self.G_tau)
        L_evals = np.log(G_eval) / self.tau
        self.L = G_evects @ np.diag(L_evals) @ pinv(G_evects)

    def _fit_noise(self, C_0):
        # Adapted from https://github.com/frodre/pyLIM/blob/master/pylim/LIM.py
        Q = -(self.L @ C_0 + C_0 @ self.L.conj().T)
        if is_dask_array(Q):
            Q = Q.compute()

        # Check if Q is Hermetian
        if not np.isclose(Q, Q.conj().T, atol=1e-10).all():
            raise ValueError("Q is not Hermetian (Q should equal Q.H)")

        q_evals, q_evecs = eigh(Q)
        sort_idx = q_evals.argsort()
        q_evals = q_evals[sort_idx][::-1]
        q_evecs = q_evecs[:, sort_idx][:, ::-1]
        num_neg = (q_evals < 0).sum()

        if num_neg > 0:
            num_left = len(q_evals) - num_neg
            if num_neg > self.max_neg_evals:
                logger.debug(
                    f"Found {num_neg:d} modes with negative eigenvalues in the noise covariance term, Q."
                )
                raise ValueError(
                    f"More than {self.max_neg_evals:d} negative eigenvalues of Q detected. "
                    "Consider further dimensional reduction."
                )

            logger.info(
                f"Removing negative eigenvalues and rescaling {num_left:d} remaining eigenvalues of Q."
            )
            pos_q_evals = q_evals[q_evals > 0]
            scale_factor = q_evals.sum() / pos_q_evals.sum()
            logger.info(f"Q eigenvalue rescaling: {scale_factor:1.2f}")

            q_evals = q_evals[:-num_neg] * scale_factor
            q_evecs = q_evecs[:, :-num_neg]

        self.Q_evals = q_evals
        self.Q_evecs = q_evecs

    def print_properties(self):
        print("G1:", self.G_tau)
        print("Eigenvalues of G:", eigvals(self.G_tau))
        print("L:", self.L)
        print("Eigenvalues of L:", eigvals(self.L))
        print("Eigenvalues of Q:", self.Q_evals)

    def forecast_deterministic(self, initial: Union[Dataset, DataArray], n_steps, t0):
        unstack = False
        if isinstance(initial, Dataset):
            initial = stack_state(initial)
            unstack = True

        assert np.array_equal(
            self.state_coords.data, initial.state.data
        ), "Initial state dimension must match training data state dimension."
        initial_np = np.squeeze(initial.values)

        time = self.time_converter.forwards(t0) + np.arange(0, (n_steps + 1) * self.tau, self.tau)

        forecast_np = np.zeros((self.Nx, n_steps + 1))
        forecast_np[:, 0] = initial_np

        ####

        G = self.G_tau
        for i in range(1, n_steps + 1):
            forecast_np[:, i] = G @ initial_np
            G = self.G_tau @ G

        ###

        forecast = xr.DataArray(
            self.mean + dask.array.from_array(forecast_np),
            dims=["state", "time"],
            coords={"state": self.state_coords, "time": self.time_converter.backwards(time)},
        )
        if unstack:
            forecast = unstack_state(forecast)

        return forecast

    def forecast_stochastic(
        self, initial: Union[Dataset, DataArray], n_steps, n_ensemble, n_int_steps_per_tau, t0
    ):
        """
        Forecast using stochastic integration.

        Args:
            initial: Initial conditions
            n_steps: number of "tau"-length periods to forecast
            n_ensemble: number of ensemble members
            n_int_steps_per_tau: number of integration steps per "tau"-length period
            t0: time of initial conditions

        Returns:
            Ensemble forecast
        """
        unstack = False
        if isinstance(initial, Dataset):
            initial = stack_state(initial)
            unstack = True

        assert np.array_equal(
            self.state_coords.data, initial.state.data
        ), "Initial state dimension must match training data state dimension."
        initial_np = initial.values.squeeze()

        time = self.time_converter.forwards(t0) + np.arange(0, (n_steps + 1) * self.tau, self.tau)

        forecast_np = np.zeros((n_ensemble, self.Nx, n_steps + 1))
        forecast_np[:, :, 0] = np.broadcast_to(initial_np, (n_ensemble, self.Nx))

        ####

        rng = np.random.default_rng(546503548)
        dt = 1 / n_int_steps_per_tau
        n_int_steps = int(n_int_steps_per_tau * n_steps)
        num_evals = self.Q_evals.shape[0]

        state_1 = forecast_np[:, :, 0].T
        state_mid = state_1
        state_2 = None

        # Do stochastic integration
        for i in tqdm(range(n_int_steps + 1)):
            deterministic = (self.L @ state_1) * dt
            stochastic = self.Q_evecs @ (
                np.sqrt(self.Q_evals[:, np.newaxis] * dt) * rng.normal(size=(num_evals, n_ensemble))
            )
            state_2 = state_1 + deterministic + stochastic
            state_mid = (state_1 + state_2) / 2
            state_1 = state_2
            if i % n_int_steps_per_tau == 0:
                forecast_np[:, :, i // n_int_steps_per_tau] = state_mid.T.real

        ####

        forecast = xr.DataArray(
            self.mean + dask.array.from_array(forecast_np),
            dims=["ens", "state", "time"],
            coords={
                "ens": np.arange(n_ensemble),
                "state": self.state_coords,
                "time": self.time_converter.backwards(time),
            },
        )
        if unstack:
            forecast = unstack_state(forecast)

        return forecast

    def forecast_np(self, initial: np.array, n_steps):
        time = np.arange(0, n_steps * self.tau, self.tau)
        forecast = np.empty([self.Nx, n_steps])
        forecast[:, 0] = initial

        for i in range(n_steps - 1):
            forecast[:, i + 1] = np.linalg.matrix_power(self.G_tau, i + 1) @ initial

        forecast += self.mean

        return time, forecast
