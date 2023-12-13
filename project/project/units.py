import xarray as xr


def convert_to_si_units(da: xr.DataArray):
    units = da.units
    if units == "degC":
        da += 273.15
        da.attrs["units"] = "K"
    elif units == "0.001":
        da /= 1000
        da.attrs["units"] = "1"

    return da


def convert_thetaot_to_ohc(thetaot_avg: xr.DataArray, depth: float) -> xr.DataArray:
    """
    Calculates the ocean heat content from the average potential temperature in a given layer.

    Args:
        thetaot_avg: average potential temperature [K]
        depth: depth over which the given potential temperature is the average [m]

    Returns:
        Ocean heat content
    """
    assert thetaot_avg.units == "K"

    rho = 1025  # kg/m^3
    cp = 3850  # J/(kg K)

    ohc = rho * cp * thetaot_avg * depth
    ohc = ohc.assign_attrs(
        dict(
            standard_name="sea_water_potential_temperature_expressed_as_heat_content",
            long_name=f"Ocean heat content (0-{depth} m)",
            units="J m-2",
            realm="ocean",
        )
    )

    return ohc


def convert_thetaot700_to_ohc700(thetaot_avg: xr.DataArray) -> xr.DataArray:
    # Cannot use functools.partial with ipython autoreload
    return convert_thetaot_to_ohc(thetaot_avg, 700)
