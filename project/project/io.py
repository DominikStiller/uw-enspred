import dataclasses
import platform
import warnings
from pathlib import Path
from typing import Optional, Callable

import cftime
import intake
import xarray as xr
from xarray import SerializationWarning

from project.grid import (
    Regridder,
    GLOBAL_GRID,
    mask_greenland_and_antarctica,
    mask_poles,
)
from project.logger import get_logger
from project.units import convert_to_si_units, convert_thetaot700_to_ohc700
from project.util import get_timestamp

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=SerializationWarning)


@dataclasses.dataclass
class CMIP6Variable:
    name: str  # will be renamed to this
    id: str  # id in CMIP dataset
    table: str
    pl: Optional[float] = None
    postprocess_fns: list[Callable[[xr.DataArray], xr.DataArray]] = None

    @staticmethod
    def from_list(vars_str: list[str]) -> list["CMIP6Variable"]:
        vars = []
        for var in vars_str:
            if var in VARIABLES:
                vars.append(VARIABLES.get(var))
            else:
                raise ValueError(f"Unknown variable name {var}")
        return vars


# Atmospheric variables tend towards zero at poles
# Ocean variables are non-zero above some regions of Greenland and Antarctica
# We need to mask these to prevent distortion
VARIABLES = {
    "zg500": CMIP6Variable("zg500", "zg", "Amon", pl=500e2, postprocess_fns=[mask_poles]),
    "pr": CMIP6Variable("pr", "pr", "Amon", postprocess_fns=[mask_poles]),
    "ts": CMIP6Variable("ts", "ts", "Amon", postprocess_fns=[mask_poles]),
    "psl": CMIP6Variable("psl", "psl", "Amon", postprocess_fns=[mask_poles]),
    "rsut": CMIP6Variable("rsut", "rsut", "Amon", postprocess_fns=[mask_poles]),
    "rlut": CMIP6Variable("rlut", "rlut", "Amon", postprocess_fns=[mask_poles]),
    "tas": CMIP6Variable("tas", "tas", "Amon", postprocess_fns=[mask_poles]),
    "tos": CMIP6Variable("tos", "tos", "Omon", postprocess_fns=[mask_greenland_and_antarctica]),
    "zos": CMIP6Variable("zos", "zos", "Omon", postprocess_fns=[mask_greenland_and_antarctica]),
    "sos": CMIP6Variable("sos", "sos", "Omon", postprocess_fns=[mask_greenland_and_antarctica]),
    "thetaot700": CMIP6Variable(
        "thetaot700",
        "thetaot700",
        "Emon",
        postprocess_fns=[mask_greenland_and_antarctica],
    ),
    "ohc700": CMIP6Variable(
        "ohc700",
        "thetaot700",
        "Emon",
        postprocess_fns=[mask_greenland_and_antarctica, convert_thetaot700_to_ohc700],
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
    elif (
        hostname in ["casper-login1"]
        or hostname.startswith("crhtc")
        or hostname.startswith("casper")
    ):
        # return "/glade/collections/cmip/catalog/intake-esm-datastore/catalogs/glade-cmip6.json"
        return "/glade/work/dstiller/CMIP6/catalog.json"
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
        self.regridder = Regridder(GLOBAL_GRID)
        self.catalog_location = catalog_location or get_catalog_location()

    def load_dataset(self, timerange: list[str] = None):
        if self.cat is None:
            logger.debug(f"Opening catalog {self.catalog_location}")
            self.cat = intake.open_esm_datastore(self.catalog_location)

        logger.info("Loading dataset from catalog")
        dataarrays = {}
        for variable in self.variables:
            logger.debug(f" - {variable}")

            query = dict(
                experiment_id=self.experiment_id,
                source_id=self.model_id,
                member_id="r1i1p1f1",
                table_id=variable.table,
                variable_id=variable.id,
                grid_label="gn",
            )
            if timerange:
                query["time_range"] = timerange

            query_results = self.cat.search(**query)

            # Ensure there is no ambiguity about dataset (i.e. exactly one is found)
            if len(query_results) == 0:
                raise LookupError("No datasets found for query")
            if not all(query_results.nunique().drop(["time_range", "path"]) <= 1):
                raise LookupError("Multiple datasets found for query")

            # Chunk size should be a multiple of 120 (number of timesteps per file)
            # Otherwise multiple processes would access the same file, leading to errors
            dataset = query_results.to_dask(
                progressbar=False, xarray_open_kwargs=dict(chunks=dict(time=5040))
            )
            dataset = dataset.drop_vars(
                VARS_AND_DIMS_TO_DROP,
                errors="ignore",
            ).squeeze()

            # Select single pressure level if necessary
            if variable.pl is not None:
                dataset = dataset.sel(plev=variable.pl).drop_vars("plev")

            dataset = dataset.rename_vars({variable.id: variable.name})

            # Convert single-variable dataset to DataArray such that attrs are preserved
            dataset_attrs = dataset.attrs
            realm = dataset.realm
            dataarray = dataset[variable.name].assign_attrs(dataset_attrs)
            dataarray.attrs = filter(lambda kv: kv[0] in ATTRS_TO_KEEP, dataarray.attrs.items())

            dataarray = self.regridder.regrid(realm, dataarray)

            dataarray = convert_to_si_units(dataarray)
            if variable.postprocess_fns:
                for fn in variable.postprocess_fns:
                    dataarray = fn(dataarray)

            dataarrays[variable.name] = dataarray

        dataarrays = dict(sorted(dataarrays.items()))

        # Merge all fields
        dataarrays = xr.combine_by_coords(dataarrays.values())

        return dataarrays


def save_mfdataset(ds: xr.Dataset, directory: Path, compute=True, add_timestamp=True):
    if add_timestamp:
        directory /= get_timestamp()
    directory.mkdir(parents=True)

    if isinstance(ds["time"].values.flat[0], cftime.datetime):
        year = ds["time"].dt.year
    else:
        year = ds["time"]
    century = ((year - year[0]) / 100).astype(int)

    indexes, datasets = zip(*ds.groupby(century))
    paths = [directory / f"{i}.nc" for i in indexes]

    logger.info(f"Saving dataset to {directory}")
    if compute:
        xr.save_mfdataset(datasets, paths)
    else:
        return xr.save_mfdataset(datasets, paths, compute=False)


def save_dataset(ds: xr.Dataset, directory: Path, compute=True):
    directory /= get_timestamp()
    directory.mkdir(parents=True)

    logger.info(f"Saving dataset to {directory}")
    if compute:
        ds.to_netcdf(directory / "data.nc")
    else:
        return ds.to_netcdf(directory / "data.nc", compute=False)
