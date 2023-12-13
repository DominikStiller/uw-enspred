from pathlib import Path

import numpy as np
import xarray as xr

from project.kf import SerialEnSRF, create_initial_ensemble_from_sample
from project.lim import LIM
from project.logger import get_logger, logging_disabled
from project.spaces import PhysicalSpaceForecastSpaceMapper
from project.util import get_timestamp

logger = get_logger(__name__)

if __name__ == "__main__":
    year_start = 850
    year_end = 1050
    n_obs = 200
    n_ens = 100
    sigma_obs = 1

    outdir = Path("/home/enkf6/dstiller/enspred/runs") / get_timestamp()
    (outdir / "prior_physical").mkdir(parents=True)
    (outdir / "posterior_physical").mkdir(parents=True)

    logger.info(f"Saving to {outdir}")

    obs = xr.open_mfdataset(
        Path("/home/enkf6/dstiller/enspred/obs/2023-12-12T14-35-41").glob("**/*.nc")
    ).set_xindex(["lon", "lat"])
    obs = obs.sel(time=slice(year_start, year_end)).isel(
        location=np.random.default_rng(84513).choice(
            np.arange(len(obs.location)), n_obs, replace=False
        )
    )
    obs.reset_index("location").to_netcdf(outdir / "obs.nc")

    mapper = PhysicalSpaceForecastSpaceMapper.load(
        Path("/home/enkf6/dstiller/enspred/mapper/mapper-2023-12-08T00-12-39.pkl")
    )
    lim = LIM.load(Path("/home/enkf6/dstiller/enspred/lim/lim-2023-12-12T13-46-52.pkl"))
    kf = SerialEnSRF()

    prior_physical = create_initial_ensemble_from_sample(
        xr.open_mfdataset(
            Path("/home/enkf6/dstiller/enspred/annual_averages/2023-12-09T23-33-06").glob("**/*.nc")
        ),
        n_ens,
        year_start,
    ).compute()

    for year in range(year_start, year_end):
        logger.info(f"===== YEAR {year} =====")

        prior_physical.to_netcdf(outdir / "prior_physical" / f"{year}.nc")

        logger.info("Assimilating observations")
        posterior_physical = kf.assimilate(
            prior_physical, obs.sel(time=prior_physical.time), sigma_obs
        )
        posterior_physical.to_netcdf(outdir / "posterior_physical" / f"{year}.nc")

        logger.info("Mapping posterior from physical to reduced space")
        with logging_disabled():
            posterior_reduced = mapper.forward_ensemble(posterior_physical).compute()

        logger.info("Forecasting in reduced space")
        forecast_reduced = lim.forecast_stochastic(
            posterior_reduced, 1, n_ens, 1440, posterior_reduced.time.item()
        ).sel(time=slice(year + 1, year + 2))

        logger.info("Mapping forecast from reduced to physical space")
        with logging_disabled():
            forecast_physical = mapper.backward_ensemble(forecast_reduced).compute()

        prior_physical = forecast_physical

    logger.info(f"Saved to {outdir}")
