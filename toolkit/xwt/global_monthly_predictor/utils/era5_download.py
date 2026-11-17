"""
Download of ERA5 daily single-level fields from the Earth Data Hub (DestinE), one file per year.

Each year is written to `<out_dir>/<prefix>_<year>.nc.part` and renamed to `.nc` only when the
write has finished, so an interrupted run never leaves a file that looks complete. Years whose
file already exists with all the days available on the server are skipped, so the download can be
re-run at any time and only missing or incomplete years (e.g. the current one) are downloaded.
"""

import os
import os.path as op
import time
from datetime import datetime

import numpy as np
import xarray as xr

URL = "https://edh:{pat}@data.earthdatahub.destine.eu/era5/era5-single-levels-atmosphere-daily-utc-v0.zarr"


def _log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def _n_times(path, tdim):
    """Number of time steps of a local file (None if it cannot be read)."""
    try:
        with xr.open_dataset(path, decode_times=False) as d:
            return d.sizes[tdim]
    except Exception:
        return None


def open_remote(pat, url=URL):
    """Open the remote Zarr store (only metadata is downloaded)."""
    return xr.open_dataset(url.format(pat=pat), chunks={}, engine="zarr")


def download_by_year(pat, out_dir, variables=("sst", "msl"), years=None, prefix="era5_daily_sst_msl",
                     complevel=1, retries=3, wait=60, url=URL):
    """
    Download `variables` year by year to `out_dir`.

    years : iterable of years (default: all the years available on the server)
    Returns a dict {year: 'skipped' | 'downloaded' | 'failed'}.
    """
    os.makedirs(out_dir, exist_ok=True)
    ds = open_remote(pat, url)[list(variables)]
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    all_years = ds[tdim].dt.year.values
    years = sorted(set(all_years)) if years is None else list(years)
    _log(f"{len(years)} years to check ({years[0]}-{years[-1]}), variables {list(variables)}, output {out_dir}")

    status = {}
    for year in years:
        n_remote = int((all_years == year).sum())
        path = op.join(out_dir, f"{prefix}_{year}.nc")
        if n_remote == 0:
            _log(f"{year}: not available on the server")
            continue

        n_local = _n_times(path, tdim) if op.isfile(path) else None
        if n_local == n_remote:
            _log(f"{year}: already complete ({n_local} days), skipped")
            status[year] = "skipped"
            continue
        if n_local is not None:
            _log(f"{year}: local file has {n_local} of {n_remote} days, downloading again")

        part = path + ".part"
        for attempt in range(1, retries + 1):
            try:
                t0 = time.time()
                sub = ds.sel({tdim: str(year)})
                enc = {v: {"zlib": True, "complevel": complevel} for v in variables} if complevel else None
                sub.to_netcdf(part, encoding=enc)
                if _n_times(part, tdim) != n_remote:
                    raise IOError("written file is incomplete")
                os.replace(part, path)
                size = op.getsize(path) / 1e9
                _log(f"{year}: downloaded {n_remote} days, {size:.2f} GB in {(time.time() - t0) / 60:.1f} min")
                status[year] = "downloaded"
                break
            except Exception as e:
                _log(f"{year}: attempt {attempt}/{retries} failed: {type(e).__name__}: {e}")
                if op.exists(part):
                    os.remove(part)
                if attempt < retries:
                    time.sleep(wait)
                    ds = open_remote(pat, url)[list(variables)]
                else:
                    status[year] = "failed"
    n = {s: sum(v == s for v in status.values()) for s in ("downloaded", "skipped", "failed")}
    _log(f"Finished: {n}")
    return status


def open_years(out_dir, prefix="era5_daily_sst_msl", **kwargs):
    """Open all the yearly files as a single dataset."""
    return xr.open_mfdataset(op.join(out_dir, f"{prefix}_*.nc"), combine="by_coords", **kwargs)
