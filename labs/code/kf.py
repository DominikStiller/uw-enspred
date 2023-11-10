import numpy as np
import xarray as xr


def stack_state(ds):
    return ds.to_array("field").stack(elem=("lat", "lon", "field")).transpose()


def unstack_state(ds):
    return ds.unstack("elem").to_dataset("field")


def inverse(X: xr.DataArray):
    return xr.DataArray(np.linalg.inv(X.values), coords=X.coords)


class SingleObsEnKF:
    def __init__(self, sigma_obs, H_fn):
        self.sigma_obs = sigma_obs
        self.R = xr.DataArray([[self.sigma_obs**2]], coords={"ob": [0], "ob2": [0]})
        self.H_fn = H_fn

    def assimilate(self, prior_samples, y):
        prior_samples = stack_state(prior_samples)
        Ne = len(prior_samples.ensemble)

        H = self.H_fn(prior_samples.elem)

        x_hat = prior_samples.mean("ensemble")
        X = prior_samples - x_hat

        Z = H @ X
        sigma_p_hat_sq = (Z @ Z.T) / (Ne - 1)

        # Renaming dimensions is necessary to force outer product
        C = (
            Z.rename({"ensemble": "ensemble1"}).T
            @ Z.rename({"ensemble": "ensemble2"})
            / ((self.sigma_obs**2 + sigma_p_hat_sq) * (Ne - 1))
        )

        # Update perturbations
        alpha = 1 / (
            1 + np.sqrt(self.sigma_obs**2 / (self.sigma_obs**2 + sigma_p_hat_sq))
        )
        T = np.identity(Ne) - alpha * C
        # This is the slowest step
        Xa = (X @ T.rename({"ensemble1": "ensemble"})).rename({"ensemble2": "ensemble"})

        # Update mean
        # (Ne - 1) is missing in lecture notes
        D_hat = sigma_p_hat_sq + self.R
        x_hat_a = (
            x_hat + X @ Z.T @ inverse(D_hat) / (Ne - 1) @ (y - H @ x_hat)
        ).squeeze()

        prior_var = unstack_state(X.var("ensemble"))
        posterior_var = unstack_state(Xa.var("ensemble"))

        x_hat = unstack_state(x_hat)
        x_hat_a = unstack_state(x_hat_a)

        return x_hat, prior_var, x_hat_a, posterior_var
