from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr


def load_era5_winds(
    file_path: str,
    lon_to_180: bool = True,
    time_dim: str = "time",
) -> xr.Dataset:
    """
    Load an ERA5 10 m wind dataset and normalize its coordinates.

    Parameters
    ----------
    file_path : str
        Path to the ERA5 NetCDF file. Expected to contain the variables
        'u10' and 'v10' with dimensions (time, lat, lon).
    lon_to_180 : bool, optional
        If True, convert longitudes from the ERA5 [0, 360] convention to
        [-180, 180] and sort the dataset accordingly. Default is True.
    time_dim : str, optional
        Name of the time dimension. Default is 'time'.

    Returns
    -------
    xr.Dataset
        Dataset with 'u10' and 'v10', longitudes in the requested convention
        and sorted in increasing time order.

    Raises
    ------
    ValueError
        If the expected wind variables are not present in the dataset.

    Notes
    -----
    ERA5 stores longitudes in [0, 360]. Keeping them that way silently breaks
    any later comparison against bathymetry or SWAN grids expressed in
    [-180, 180], so the conversion is applied by default.
    """

    ds = xr.open_dataset(file_path)

    missing = [var for var in ("u10", "v10") if var not in ds.data_vars]
    if missing:
        raise ValueError(
            f"Missing expected wind variables {missing} in {file_path}. "
            f"Found: {list(ds.data_vars)}"
        )

    if lon_to_180 and "lon" in ds.coords and float(ds.lon.max()) > 180.0:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")

    return ds.sortby(time_dim)


def wind_speed_direction(
    ds: xr.Dataset,
    u_var: str = "u10",
    v_var: str = "v10",
) -> xr.Dataset:
    """
    Add wind speed and meteorological wind direction to a wind dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the zonal and meridional wind components.
    u_var : str, optional
        Name of the zonal (eastward) wind component. Default is 'u10'.
    v_var : str, optional
        Name of the meridional (northward) wind component. Default is 'v10'.

    Returns
    -------
    xr.Dataset
        A copy of the input dataset with two extra variables:
        - 'wspd': wind speed (m/s)
        - 'wdir': wind direction (degrees, meteorological convention)

    Notes
    -----
    'wdir' follows the meteorological convention: the direction the wind is
    blowing *from*, measured clockwise from North. This is the convention SWAN
    expects with the NAUTICAL keyword.
    """

    ds = ds.copy()
    u, v = ds[u_var], ds[v_var]

    ds["wspd"] = np.sqrt(u**2 + v**2)
    ds["wspd"].attrs = {"long_name": "wind speed", "units": "m/s"}

    ds["wdir"] = (270.0 - np.rad2deg(np.arctan2(v, u))) % 360.0
    ds["wdir"].attrs = {
        "long_name": "wind direction (coming from)",
        "units": "degrees",
        "convention": "meteorological / nautical",
    }

    return ds


def domain_mean_wind(
    ds: xr.Dataset,
    u_var: str = "u10",
    v_var: str = "v10",
    spatial_dims: Tuple[str, ...] = ("lat", "lon"),
) -> pd.DataFrame:
    """
    Collapse a gridded wind field into a domain-averaged speed and direction.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the wind components on a (time, lat, lon) grid.
    u_var : str, optional
        Name of the zonal wind component. Default is 'u10'.
    v_var : str, optional
        Name of the meridional wind component. Default is 'v10'.
    spatial_dims : tuple of str, optional
        Dimensions to average over. Default is ('lat', 'lon').

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by time with columns 'wspd' (m/s) and 'wdir'
        (degrees, meteorological convention).

    Notes
    -----
    The components are averaged *before* computing speed and direction, so the
    result is the vector mean of the field rather than the mean of the
    magnitudes. This is only used for diagnostic plots: the PCA/MDA selection
    works on the full field, not on this summary.
    """

    u = ds[u_var].mean(dim=list(spatial_dims))
    v = ds[v_var].mean(dim=list(spatial_dims))

    return pd.DataFrame(
        {
            "wspd": np.sqrt(u.values**2 + v.values**2),
            "wdir": (270.0 - np.rad2deg(np.arctan2(v.values, u.values))) % 360.0,
        },
        index=pd.to_datetime(ds.time.values),
    )


def build_mda_ranking(
    pcs_df: pd.DataFrame,
    centroid_real_indices: Union[List[int], np.ndarray],
) -> pd.DataFrame:
    """
    Build the full MDA ranking table from the centroid indices.

    Parameters
    ----------
    pcs_df : pd.DataFrame
        Principal components indexed by time, as returned by `PCA.pcs_df`.
    centroid_real_indices : list of int or np.ndarray
        Positional indices of the selected centroids, in MDA selection order,
        as returned by `MDA.centroid_real_indices`.

    Returns
    -------
    pd.DataFrame
        Table indexed by MDA rank (starting at 1) with columns:
        - 'time': the timestamp of the selected case
        - 'index': the positional index in the original dataset
        - one column per principal component

    Notes
    -----
    MDA is a greedy algorithm: rank 1 is the seed, and each subsequent rank is
    the point most dissimilar to everything already selected. This makes the
    ranking *nested* — the first N rows are exactly the selection you would get
    by running MDA with `num_centers=N`. Saving the full ranking once means any
    later subset (10, 100, 1000 cases) is a slice, not a re-run.
    """

    idx = np.asarray(centroid_real_indices, dtype=int)
    ranking = pcs_df.iloc[idx].copy()
    ranking.insert(0, "index", idx)
    ranking.insert(0, "time", pd.to_datetime(ranking.index))
    ranking.index = pd.RangeIndex(start=1, stop=len(ranking) + 1, name="rank")

    return ranking


def save_times_txt(times: Union[pd.DatetimeIndex, np.ndarray], file_path: str) -> None:
    """
    Write a list of timestamps to a plain text file, one per line.

    Parameters
    ----------
    times : pd.DatetimeIndex or np.ndarray
        Timestamps to write.
    file_path : str
        Destination path.

    Notes
    -----
    Timestamps are written with second resolution in ISO format
    (``YYYY-MM-DDTHH:MM:SS``), matching the format read back by
    `load_times_txt` and by the downstream RBF notebook.
    """

    values = pd.to_datetime(np.asarray(times)).values.astype("datetime64[s]")
    np.savetxt(file_path, values.astype(str), fmt="%s")


def load_times_txt(file_path: str) -> pd.DatetimeIndex:
    """
    Read a list of timestamps written by `save_times_txt`.

    Parameters
    ----------
    file_path : str
        Path to the text file.

    Returns
    -------
    pd.DatetimeIndex
        The timestamps contained in the file.
    """

    return pd.DatetimeIndex(pd.to_datetime(np.loadtxt(file_path, dtype=str, ndmin=1)))


def apply_mask(
    ds_wave: xr.Dataset,
    bathy: xr.Dataset,
    replace_vars: Optional[List[str]] = None,
    number_times: Optional[int] = None,
) -> xr.Dataset:
    """
    Apply the land/sea mask to a gridded wave dataset.

    Parameters
    ----------
    ds_wave : xr.Dataset
        Wave fields with dimensions (lat, lon, time, tide). Must contain
        'Hsig'; the dimension order is normalised internally.
    bathy : xr.Dataset
        Bathymetry on the same grid, with a 'depth' or an 'elevation' variable.
    replace_vars : list of str, optional
        Variables to mask. Default is ['Tm02', 'Dir', 'Hsig'].
    number_times : int, optional
        Keep only the first N times when interpolating. Default is None (all).

    Returns
    -------
    xr.Dataset
        The masked dataset, with a 'depth' variable added and the outer row
        and column trimmed. The mask itself is not returned: it is built on the
        intermediate dataset only, as in the original implementation.

    Raises
    ------
    ValueError
        If `bathy` has neither 'depth' nor 'elevation'.

    Notes
    -----
    The mask marks every cell where SWAN produced no wave height at any time,
    which is how land shows up in the output. Before masking, the domain edges
    are forced to zero and the interior NaNs are filled by nearest-neighbour
    interpolation, so isolated dry cells inside the estuary do not punch holes
    in the field.

    Unlike the original in `hywindsea_functions.py`, this works on a copy: the
    dataset passed in is left untouched.
    """

    # The original mutated its argument, which makes re-running a notebook cell
    # give a different answer the second time.
    ds_wave = ds_wave.copy(deep=True)
    ds_wave = ds_wave.transpose("lat", "lon", "time", "tide")

    # 1. Attach depth
    if "depth" in bathy:
        ds_wave["depth"] = bathy.depth
    elif "elevation" in bathy:
        ds_wave["depth"] = bathy.elevation
    else:
        raise ValueError("'bathy' must contain either 'depth' or 'elevation'.")

    # 2. Mask: NaN wherever Hsig is NaN at every time  -> (lat, lon, tide)
    hs_mask = np.where(np.isnan(ds_wave["Hsig"].max(dim="time")), np.nan, 1.0)

    # 3. Variables to process
    if replace_vars is None:
        replace_vars = ["Tm02", "Dir", "Hsig"]

    # 4. Domain edges
    lon, lat = ds_wave["lon"], ds_wave["lat"]
    border_mask = (
        (lon != lon.min().item())
        & (lon != lon.max().item())
        & (lat != lat.min().item())
    )

    # 5. Force the edges to zero
    for var in replace_vars:
        ds_wave[var] = ds_wave[var].where(border_mask, 0)

    # 6. Optionally limit the number of times
    ds_wave_int = (
        ds_wave.isel(time=slice(0, number_times)) if number_times else ds_wave
    )

    # 7. Fill interior gaps
    ds_wave_int = ds_wave_int.interpolate_na(
        dim="lon", method="nearest"
    ).interpolate_na(dim="lat", method="nearest")

    # 8-9. Apply the mask
    ds_wave_int["mask"] = (("lat", "lon", "tide"), hs_mask)
    ds_wave_final = ds_wave.copy()
    for var in replace_vars:
        ds_wave_final[var] = ds_wave_int[var] * ds_wave_int["mask"]

    # 10. Trim the outer ring
    return ds_wave_final.isel(lat=slice(1, -1), lon=slice(1, -1))


def direction_to_uv(ds: xr.Dataset, var: str) -> xr.Dataset:
    """
    Split a directional variable into its unit-circle components.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the directional variable, in degrees.
    var : str
        Name of the directional variable, e.g. 'Dir'.

    Returns
    -------
    xr.Dataset
        Dataset with two variables, ``f"{var}_u"`` and ``f"{var}_v"``, holding
        the sine and cosine of the direction.

    Notes
    -----
    A direction is an angle, but PCA treats it as an ordinary number: it works
    on means and variances, and 359 deg and 1 deg are 358 apart arithmetically
    while being 2 apart physically. A cell oscillating around North therefore
    looks wildly variable, gets a mean near South, and the reconstruction can
    return directions that never occurred.

    The sine and cosine are continuous across that discontinuity, so PCA can
    work on them safely. Recombine afterwards with `uv_to_direction`.

    `MDA` and `RBF` do this internally through their `directional_variables`
    argument; `PCA` has no such argument, so it has to be done by hand.
    """

    rad = np.deg2rad(ds[var])

    return xr.Dataset({f"{var}_u": np.sin(rad), f"{var}_v": np.cos(rad)})


def uv_to_direction(ds: xr.Dataset, var: str) -> xr.DataArray:
    """
    Recombine unit-circle components back into a direction in degrees.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset holding ``f"{var}_u"`` and ``f"{var}_v"``.
    var : str
        Base name of the directional variable, e.g. 'Dir'.

    Returns
    -------
    xr.DataArray
        The direction in degrees, in [0, 360).
    """

    direction = np.rad2deg(np.arctan2(ds[f"{var}_u"], ds[f"{var}_v"])) % 360.0
    direction.name = var

    return direction


def angular_difference(
    predicted: Union[np.ndarray, xr.DataArray],
    truth: Union[np.ndarray, xr.DataArray],
) -> np.ndarray:
    """
    Signed difference between two directions, wrapped to [-180, 180].

    Parameters
    ----------
    predicted : np.ndarray or xr.DataArray
        Predicted directions in degrees.
    truth : np.ndarray or xr.DataArray
        Reference directions in degrees.

    Returns
    -------
    np.ndarray
        The difference in degrees, in [-180, 180].

    Notes
    -----
    Subtracting directions directly scores 359 deg against 1 deg as an error of
    358 instead of 2, which inflates every error metric computed on angles.
    """

    return (np.asarray(predicted) - np.asarray(truth) + 180.0) % 360.0 - 180.0


def make_pca_metric(pca_wave, var: str, directional: bool = False):
    """
    Build a metric callable for `KFold_cross_validation_RBF`.

    Parameters
    ----------
    pca_wave : bluemath_tk.datamining.pca.PCA
        The fitted PCA of the wave variable, used to rebuild the field from PCs.
    var : str
        Name of the wave variable, e.g. 'Hsig_rescaled'.
    directional : bool, optional
        True when `var` is an angle stored as sine/cosine components, so the
        field is recombined with `uv_to_direction` and the error wrapped to
        [-180, 180]. Default is False.

    Returns
    -------
    callable
        A function of (df_true, df_pred) returning a dict with 'rmse',
        'median_abs' and 'bias'.

    Notes
    -----
    The error is measured on the reconstructed field, not on the PCs: a given
    PC error means very different things depending on the component it belongs
    to. Both sides are passed through `inverse_transform`, so what is measured
    is the error the RBF introduces, without the PCA truncation error that is
    the same for every training-set size.
    """

    def _inverse(df: pd.DataFrame) -> xr.Dataset:
        return pca_wave.inverse_transform(
            PCs=xr.Dataset(
                {"PCs": (("time", "n_component"), df.values)},
                coords={"time": df.index.values},
            )
        )

    def metric(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
        df_true, df_pred = df_true.align(df_pred, join="inner")
        true, pred = _inverse(df_true), _inverse(df_pred)

        if directional:
            error = angular_difference(
                uv_to_direction(pred, var), uv_to_direction(true, var)
            )
        else:
            error = (pred[var] - true[var]).values

        valid = ~np.isnan(error)

        return {
            "rmse": float(np.sqrt(np.nanmean(error[valid] ** 2))),
            "median_abs": float(np.nanmedian(np.abs(error[valid]))),
            "bias": float(np.nanmean(error[valid])),
        }

    return metric


def summarize_kfold(kfold_results: dict, metric_name: str = "rmse") -> Tuple[float, float]:
    """
    Collapse the folds of one K-Fold run into a mean and a standard deviation.

    Parameters
    ----------
    kfold_results : dict
        The dictionary returned by `KFold_cross_validation_RBF`, keyed by fold.
    metric_name : str, optional
        Which entry of the fold metric to summarise. Default is 'rmse'.

    Returns
    -------
    tuple of float
        The mean and the standard deviation across folds.
    """

    values = [fold["metric"][metric_name] for fold in kfold_results.values()]

    return float(np.mean(values)), float(np.std(values))


def selection_dispersion(
    space: Union[pd.DataFrame, np.ndarray],
    selected: Union[List[int], np.ndarray],
    normalize: bool = False,
) -> dict:
    """
    Describe how a subset of cases is spread through the space it was drawn from.

    Parameters
    ----------
    space : pd.DataFrame or np.ndarray
        The full candidate cloud, one row per case. Distances are measured in
        this space as given, so pass the same one the selection was made in.
    selected : list of int or np.ndarray
        Positional indices of the selected cases.
    normalize : bool, optional
        Min-max scale every column first, giving each dimension equal weight.
        Default is False, which matches `MDA.fit(normalize_data=False)` — its
        default, where the leading principal component dominates the distance
        because it carries the largest amplitude.

    Returns
    -------
    dict
        ``separation`` — mean distance from each selected case to its nearest
        neighbour *inside the selection*. MDA maximises this, so it is high for
        an MDA set and low for a random draw, which clumps wherever the cloud is
        dense.

        ``coverage`` — mean distance from every case in the cloud to the nearest
        selected case. Lower is better covered, but a random draw scores well
        here by following the density, so read it together with the next one.

        ``worst_coverage`` — the largest such distance: the corner of the cloud
        left furthest from anything selected. This is the one MDA is really
        minimising.

    Notes
    -----
    Together the three numbers are a fingerprint of *how* a selection was made,
    which is useful when the selection is known but the procedure is not.
    """

    from scipy.spatial import cKDTree

    values = np.asarray(
        space.values if isinstance(space, pd.DataFrame) else space, dtype=float
    )
    if normalize:
        span = values.max(axis=0) - values.min(axis=0)
        span[span == 0] = 1.0
        values = (values - values.min(axis=0)) / span

    points = values[np.asarray(selected, dtype=int)]
    tree = cKDTree(points)

    within, _ = tree.query(points, k=2)
    to_cloud, _ = tree.query(values, k=1)

    return {
        "separation": float(within[:, 1].mean()),
        "coverage": float(to_cloud.mean()),
        "worst_coverage": float(to_cloud.max()),
    }


def max_diss(
    data: Union[pd.DataFrame, np.ndarray],
    num_centers: int,
    normalize: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Maximum Dissimilarity Algorithm, in the teslakit formulation.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Candidate cases, one row each.
    num_centers : int
        How many cases to select.
    normalize : bool, optional
        Min-max scale every column before measuring distances, so each one
        counts the same. Default is True.
    seed : int, optional
        Row to start from. Default is None, which starts at the largest value of
        the first column.

    Returns
    -------
    np.ndarray
        Positional indices of the selected cases, in selection order. The result
        is nested: the first N of a longer run are the same as a run of N.

    Notes
    -----
    This reproduces ``MaxDiss_Simplified_NoThreshold`` from teslakit, which is
    what the original HyWindSea case selection used. It differs from
    ``bluemath_tk.datamining.mda.MDA`` in two ways that both change the result:
    that one measures distances on the raw columns unless ``normalize_data`` is
    set, and it starts from the row with the largest sum rather than the largest
    first column. Either difference on its own is enough to send the greedy path
    somewhere else and cut the agreement between two selections to about a
    quarter.
    """

    values = np.asarray(
        data.values if isinstance(data, pd.DataFrame) else data, dtype=float
    )
    if normalize:
        span = values.max(axis=0) - values.min(axis=0)
        span[span == 0] = 1.0
        values = (values - values.min(axis=0)) / span

    if seed is None:
        seed = int(np.argmax(values[:, 0]))

    selected = [seed]
    distance = ((values - values[seed]) ** 2).sum(axis=1)

    for _ in range(num_centers - 1):
        nearest = int(np.argmax(distance))
        selected.append(nearest)
        distance = np.minimum(
            distance, ((values - values[nearest]) ** 2).sum(axis=1)
        )

    return np.array(selected)
