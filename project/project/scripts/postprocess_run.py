from pathlib import Path

import dask
import xarray as xr
from dask.distributed import Client

from project.io import save_mfdataset
from project.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)

    rundir = Path("/home/enkf6/dstiller/enspred/runs/2023-12-12T14-58-07")

    logger.info("Loading prior")
    prior_phyical = xr.open_mfdataset(
        (rundir / "prior_physical").glob("**/*.nc"), combine="nested", concat_dim="time"
    )

    logger.info("Computing ensemble mean of prior")
    prior_phyical_mean = prior_phyical.mean("ens")
    save_mfdataset(prior_phyical_mean, rundir / "prior_physical_mean", add_timestamp=False)

    logger.info("Computing ensemble variance of prior")
    prior_phyical_var = prior_phyical.var("ens")
    save_mfdataset(prior_phyical_var, rundir / "prior_physical_var", add_timestamp=False)

    logger.info("Loading posterior")
    posterior_phyical = xr.open_mfdataset(
        (rundir / "posterior_physical").glob("**/*.nc"), combine="nested", concat_dim="time"
    )

    logger.info("Computing ensemble mean of posterior")
    posterior_phyical_mean = posterior_phyical.mean("ens")
    save_mfdataset(posterior_phyical_mean, rundir / "posterior_physical_mean", add_timestamp=False)

    logger.info("Computing ensemble variance of posterior")
    posterior_phyical_var = posterior_phyical.var("ens")
    save_mfdataset(posterior_phyical_var, rundir / "posterior_physical_var", add_timestamp=False)
