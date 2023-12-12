import numpy as np
import xarray as xr
from numpy.linalg import inv
from tqdm import tqdm

from project.logger import get_logger
from project.util import stack_state, unstack_state

logger = get_logger(__name__)


def create_initial_ensemble_from_perturbations(initial, n_ens, std=0.005):
    initial = stack_state(initial, sample_dim=None)

    rng = np.random.default_rng(562151)

    ens = []
    for i in range(n_ens):
        member = initial * rng.normal(1, std, size=initial.shape)
        member = member.expand_dims("ens").assign_coords(ens=[i])
        ens.append(member)

    ens = xr.concat(ens, "ens").transpose("ens", ...)
    ens = unstack_state(ens)

    return ens


def create_initial_ensemble_from_sample(ds_all, n_ens, year_start):
    sample = np.random.default_rng(682652).choice(np.arange(len(ds_all.time)), n_ens, replace=False)
    logger.info(f"Drawing IC from years {sample}")
    ens = ds_all.isel(time=sample).rename(time="ens").assign_coords(ens=range(n_ens))
    ens = ens.expand_dims("time").assign_coords(time=[year_start]).squeeze()
    return ens


class SerialEnSRF:
    def assimilate(self, prior_ds, obs, sigma_obs):
        prior_ds = stack_state(prior_ds, sample_dim="ens")
        state_coords = prior_ds.state
        ens_coords = prior_ds.ens

        prior = prior_ds.data

        # TODO allow multiple obs at the same location and different fields than tas
        for i in tqdm(range(len(obs.location))):
            obs_i = obs.isel(location=i)["tas"]
            obs_state_idx = state_coords.indexes["state"].get_loc(("tas",) + obs_i.location.item())
            prior = self._assimilate_single(prior, obs_i.values, obs_state_idx, sigma_obs)

        posterior = xr.DataArray(prior, coords=dict(state=state_coords, ens=ens_coords))
        if "time" in prior_ds.coords:
            posterior = posterior.expand_dims("time").assign_coords(time=[prior_ds.time])

        return posterior.unstack().to_dataset("field")

    def _assimilate_single(self, prior, obs, obs_state_idx, sigma_obs):
        Nx, Ne = prior.shape

        x_hat = prior.mean(axis=1, keepdims=True)
        X = prior - x_hat

        Z = X[obs_state_idx, :][np.newaxis, :]  # identical to H @ X for H with single 1
        sigma_p_hat_sq = (Z @ Z.T) / (Ne - 1)

        C = Z.T @ Z / ((sigma_obs**2 + sigma_p_hat_sq) * (Ne - 1))

        # Update perturbations
        alpha = 1 / (1 + np.sqrt(sigma_obs**2 / (sigma_obs**2 + sigma_p_hat_sq)))
        T = np.identity(Ne) - alpha * C
        Xa = X @ T

        # Update mean
        # (Ne - 1) is missing in lecture notes
        # print(sigma_p_hat_sq, sigma_obs**2)
        D_hat = sigma_p_hat_sq + np.array([[sigma_obs**2]])
        x_hat_a = x_hat + X @ Z.T @ inv(D_hat) / (Ne - 1) @ (
            obs - x_hat[obs_state_idx, :][np.newaxis, :]
        )

        posterior = Xa + x_hat_a

        return posterior
