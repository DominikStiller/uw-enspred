import dask
from dask.distributed import Client

from project.io import IntakeESMLoader, save_mfdataset
from project.logger import get_logger
from project.util import get_data_path, average_annually

logger = get_logger(__name__)

if __name__ == "__main__":
    client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)

    data_path = get_data_path()

    loader = IntakeESMLoader(
        "past1000",
        "MRI-ESM2-0",
        ["tas", "zg500", "tos"],
    )
    ds = loader.load_dataset()

    logger.info("Averaging annually")
    ds = average_annually(ds).chunk(chunks=dict(time=100))

    save_mfdataset(ds, data_path / "annual_averages_verification")
    logger.info("Computing of annual averages completed")
