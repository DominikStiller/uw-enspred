import numpy as np
import xarray as xr
from numpy.linalg import inv

from project.util import stack_state, unstack_state


class SerialEnSRF:
    def assimilate(self, prior, obs, sigma_obs):
        prior = stack_state(prior, sample_dim="ens")
        assert prior.dims == ("state", "ens")
        state_coords = prior.state
        ens_coords = prior.ens

        prior = prior.data.compute()

        # TODO allow multiple obs at the same location and different fields than tas
        for i in range(len(obs.location)):
            obs_i = obs.isel(location=i)["tas"]
            obs_state_idx = state_coords.indexes["state"].get_loc(("tas",) + obs_i.location.item())
            print(obs_state_idx)
            prior = self._assimilate_single(prior, obs_i.values, obs_state_idx, sigma_obs)

        posterior = xr.DataArray(prior, coords=dict(state=state_coords, ens=ens_coords))

        return unstack_state(posterior)

    def _assimilate_single(self, prior, obs, obs_state_idx, sigma_obs):
        Nx, Ne = prior.shape

        x_hat = prior.mean(axis=1, keepdims=True)
        X = prior - x_hat

        Z = X[obs_state_idx, :][np.newaxis, :]  # identical to H @ X
        sigma_p_hat_sq = (Z @ Z.T) / (Ne - 1)

        C = Z.T @ Z / ((sigma_obs**2 + sigma_p_hat_sq) * (Ne - 1))

        # Update perturbations
        alpha = 1 / (1 + np.sqrt(sigma_obs**2 / (sigma_obs**2 + sigma_p_hat_sq)))
        T = np.identity(Ne) - alpha * C
        # This is the slowest step
        Xa = X @ T

        # Update mean
        # (Ne - 1) is missing in lecture notes
        D_hat = sigma_p_hat_sq + np.array([[sigma_obs]])
        x_hat_a = x_hat + X @ Z.T @ inv(D_hat) / (Ne - 1) @ (
            obs - x_hat[obs_state_idx, :][np.newaxis, :]
        )

        posterior = Xa + x_hat_a

        return posterior
