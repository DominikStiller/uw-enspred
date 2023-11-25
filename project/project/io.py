import dataclasses
import warnings
from typing import Optional, Callable, Self
import xesmf as xe
import platform

import intake
import xarray as xr

from project.grid import Regridder
from project.logger import get_logger
from project.units import convert_to_si_units, convert_thetaot700_to_ohc700

logger = get_logger(__name__)


@dataclasses.dataclass
class CMIP6Variable:
    name: str  # will be renamed to this
    id: str  # id in CMIP dataset
    table: str
    pl: Optional[float] = None
    postprocess_fn: Optional[Callable[[xr.DataArray], xr.DataArray]] = None

    @staticmethod
    def from_list(vars_str: list[str]) -> list["CMIP6Variable"]:
        vars = []
        for var in vars_str:
            if var in VARIABLES:
                vars.append(VARIABLES.get(var))
            else:
                raise ValueError(f"Unknown variable name {var}")
        return vars


VARIABLES = {
    "zg500": CMIP6Variable("zg500", "zg", "Amon", pl=500e2),
    "pr": CMIP6Variable("pr", "pr", "Amon"),
    "ts": CMIP6Variable("ts", "ts", "Amon"),
    "psl": CMIP6Variable("psl", "psl", "Amon"),
    "rsut": CMIP6Variable("rsut", "rsut", "Amon"),
    "rlut": CMIP6Variable("rlut", "rlut", "Amon"),
    "tas": CMIP6Variable("tas", "tas", "Amon"),
    "tos": CMIP6Variable("tos", "tos", "Omon"),
    "zos": CMIP6Variable("zos", "zos", "Omon"),
    "sos": CMIP6Variable("sos", "sos", "Omon"),
    "thetaot700": CMIP6Variable("thetaot700", "thetaot700", "Emon"),
    "ohc700": CMIP6Variable(
        "ohc700", "thetaot700", "Emon", postprocess_fn=convert_thetaot700_to_ohc700
    ),
}


VARS_AND_DIMS_TO_DROP = [
    "dcpp_init_year",
    "member_id",
    "time_bnds",
    "lat_bnds",
    "lon_bnds",
    "height",
    "depth",
    "vertices",
    "vertices_latitude",
    "vertices_longitude",
]

# These have to be set again when doing computations
ATTRS_TO_KEEP = [
    "standard_name",
    "long_name",
    "units",
    "realm",
]


def get_catalog_location():
    hostname = platform.node()
    if hostname in ["enkf"]:
        return "/home/enkf6/dstiller/CMIP6/catalog.json"
    elif hostname in ["casper-login1"] or hostname.startswith("crhtc"):
        return "/glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json"
    else:
        raise ValueError("Unknown host, please specify catalog location")


class IntakeESMLoader:
    def __init__(
        self,
        experiment_id: str,
        model_id: str,
        variables: list[str],
        catalog_location: str = None,
    ):
        self.experiment_id = experiment_id
        self.model_id = model_id
        self.variables = CMIP6Variable.from_list(variables)
        self.cat = None
        self.regridder = Regridder(xe.util.grid_global(2, 2, lon1=359, cf=True))
        self.catalog_location = catalog_location or get_catalog_location()

    def load_dataset(self):
        if self.cat is None:
            logger.debug(f"Opening catalog {self.catalog_location}")
            self.cat = intake.open_esm_datastore(self.catalog_location)

        logger.info("Loading dataset from catalog")
        dataarrays = []
        for variable in self.variables:
            logger.debug(f" - {variable}")
            query_results = self.cat.search(
                experiment_id=self.experiment_id,
                source_id=self.model_id,
                member_id="r1i1p1f1",
                table_id=variable.table,
                variable_id=variable.id,
                grid_label="gn",
                time_range="700101-702012",  # TODO remove
                # time_range="000101-005012",  # TODO remove
                # time_range="185001-186912",  # TODO remove
            )

            # Ensure there is no ambiguity about dataset (i.e. exactly one is found)
            if len(query_results) == 0:
                raise LookupError("No datasets found for query")

            if not all(query_results.nunique().drop(["time_range", "path"]) <= 1):
                raise LookupError("Multiple datasets found for query")

            with warnings.catch_warnings(action="ignore"):
                # Some datasets raise a warning about multiple fill values
                dataset = query_results.to_dataset_dict(progressbar=False)
            assert len(dataset) == 1
            dataset = (
                next(iter(dataset.values()))
                .drop_vars(
                    VARS_AND_DIMS_TO_DROP,
                    errors="ignore",
                )
                .squeeze()
            )

            # Select single pressure level if necessary
            if variable.pl is not None:
                dataset = dataset.sel(plev=variable.pl).drop_vars("plev")

            dataset = dataset.rename_vars({variable.id: variable.name})

            # Convert single-variable dataset to DataArray such that attrs are preserved
            dataset_attrs = dataset.attrs
            realm = dataset.realm
            dataarray = dataset[variable.name].assign_attrs(dataset_attrs)
            dataarray.attrs = filter(
                lambda kv: kv[0] in ATTRS_TO_KEEP, dataarray.attrs.items()
            )

            dataarray = convert_to_si_units(dataarray)
            if variable.postprocess_fn is not None:
                dataarray = variable.postprocess_fn(dataarray)

            dataarray = self.regridder.regrid(realm, dataarray)

            dataarrays.append(dataarray)

        dataarrays = xr.combine_by_coords(dataarrays)

        return dataarrays


if __name__ == "__main__":
    loader = IntakeESMLoader(
        "past2k",
        "MPI-ESM1-2-LR",
        [
            "zg500",
            "pr",
            "psl",
            "rsut",
            "rlut",
            "tas",
            "tos",
            "zos",
            "sos",
            "thetaot700",
        ],
    )
    # loader = IntakeESMLoader("historical", "MPI-ESM1-2-LR", ["zg500", "pr"])
    mpidata = loader.load_dataset()
    print(mpidata)
