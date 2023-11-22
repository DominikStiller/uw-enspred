import dataclasses
import warnings
from typing import Optional
import xesmf as xe

import intake
import xarray as xr

from project.project.grid import Regridder
from project.project.logger import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class CMIP6Variable:
    name: str  # will be renamed to this
    id: str  # id in CMIP dataset
    table: str
    pl: Optional[float] = None


VARIABLES = {
    "zg500": CMIP6Variable("zg500", "zg", "Amon", 500e2),
    "pr": CMIP6Variable("pr", "pr", "Amon"),
    "ts": CMIP6Variable("ts", "ts", "Amon"),
    "psl": CMIP6Variable("psl", "psl", "Amon"),
    "rsut": CMIP6Variable("rsut", "rsut", "Amon"),
    "rlut": CMIP6Variable("rlut", "rlut", "Amon"),
    "tas": CMIP6Variable("tas", "tas", "Amon"),
    "tos": CMIP6Variable("tos", "tos", "Omon"),
    "zos": CMIP6Variable("zos", "zos", "Omon"),
}


class GLADELoader:
    def __init__(self, experiment_id: str, model_id: str, variables: list[str]):
        self.experiment_id = experiment_id
        self.model_id = model_id
        self.variables = list(map(VARIABLES.get, variables))
        self.cat = None
        self.regridder = Regridder(xe.util.grid_global(2, 2, lon1=359))

    def load_dataset(self):
        if self.cat is None:
            logger.debug("Opening catalog")
            self.cat = intake.open_esm_datastore(
                "/glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json"
            )

        logger.info("Loading dataset from GLADE")
        datasets = []
        for variable in self.variables:
            logger.debug(f" - {variable}")
            query_results = self.cat.search(
                experiment_id=self.experiment_id,
                source_id=self.model_id,
                member_id="r1i1p1f1",
                table_id=variable.table,
                variable_id=variable.id,
                grid_label="gn",
                time_range="000101-005012",  # TODO remove
                # time_range="185001-186912",  # TODO remove
            )

            if not all(query_results.nunique().drop(["time_range", "path"]) <= 1):
                raise "Multiple datasets found for query"

            with warnings.catch_warnings(action="ignore"):
                # Some datasets raise a warning about multiple fill values
                dataset = query_results.to_dataset_dict(progressbar=False)
            assert len(dataset) == 1
            dataset = (
                next(iter(dataset.values()))
                .drop_vars(
                    ["dcpp_init_year", "member_id", "time_bnds", "lat_bnds", "lon_bnds"]
                )
                .squeeze()
            )

            if variable.pl is not None:
                dataset = (
                    dataset.sel(plev=variable.pl)
                    .drop_vars("plev")
                    .rename_vars({variable.id: variable.name})
                )

            dataset = self.regridder.regrid(dataset.realm, dataset)

            datasets.append(dataset)

        # datasets = xr.combine_by_coords(datasets, combine_attrs="drop")

        return datasets


if __name__ == "__main__":
    loader = GLADELoader(
        "piControl",
        "CESM2-FV2",
        [
            "zg500",
            "pr",
            "ts",
            "psl",
            "rsut",
            "rlut",
            "tas",
            "tos",
            "zos",
        ],
    )
    # loader = GLADELoader("historical", "MPI-ESM1-2-LR", ["zg500", "pr"])
    mpidata = loader.load_dataset()
    print(mpidata)
