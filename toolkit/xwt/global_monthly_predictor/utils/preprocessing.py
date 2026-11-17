"""
Preprocessing of the SST / MSL predictor fields before the PCA:
land mask, removal of anomalous months, quadratic detrend, deseasonalization
and standardization + latitude weighting.
"""

import numpy as np
import xarray as xr
import regionmask


def apply_land_mask(ds):
    """Set land points (Natural Earth 110m) to NaN."""
    land = regionmask.defined_regions.natural_earth_v4_1_0.land_110.mask(
        ds.longitude, ds.latitude
    )
    return ds.where(land.isnull())


def remove_anomalous_months(ds, sst_min=12, msl_min=1007):
    """
    Drop months whose spatial mean is unrealistic (e.g. May 2005, Dec 2013).
    Returns the filtered dataset and the spatial-mean time series.
    """
    means = ds.mean(dim=["latitude", "longitude"])
    means = means.where(means.sst > sst_min, drop=True)
    means = means.where(means.msl > msl_min, drop=True)
    return ds.sel(time=means.time.values), means


def polynomial_trend(y, deg=2):
    """Polynomial trend of a 1D series (fitted on non-NaN values)."""
    y = np.asarray(y)
    x = np.arange(len(y))
    mask = np.isnan(y)
    z = np.polyfit(x[~mask], y[~mask], deg)
    return np.polyval(z, x)


def detrend_1d(y, deg=2):
    """Remove a polynomial (parabolic by default) trend, keeping the initial level."""
    y = np.asarray(y)
    if np.all(np.isnan(y)):
        return y
    trend = polynomial_trend(y, deg)
    return (y - trend + trend[0]).astype(y.dtype)


def detrend_dataset(ds, variables=("sst", "msl"), deg=2):
    """Apply `detrend_1d` along time at every grid point for each variable."""
    out = ds.copy()
    for var in variables:
        out[var] = xr.apply_ufunc(
            detrend_1d,
            ds[var],
            kwargs={"deg": deg},
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[ds[var].dtype],
        )
    return out


def deseasonalize(ds):
    """Remove the monthly climatology, keeping the overall time mean."""
    climatology = ds.groupby("time.month").mean("time")
    return ds.groupby("time.month") - climatology + ds.mean("time")


def standardize_lat_weighted(da):
    """Standardize in time at each grid point and weight by cos(latitude)."""
    return (da - da.mean(dim="time")) / da.std(dim="time") * np.cos(
        np.deg2rad(da["latitude"])
    )
