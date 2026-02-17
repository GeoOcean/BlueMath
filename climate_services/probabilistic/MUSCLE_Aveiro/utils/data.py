from typing import List, Tuple

import numpy as np
import xarray as xr
from bluemath_tk.predictor.xwt import get_dynamic_estela_predictor
from dask.diagnostics import ProgressBar


def load_atmospheric_predictor(
    variables: List[str], region: Tuple[float], estela: bool = False
) -> xr.Dataset:
    """
    Load atmospheric predictor data for the given region using GeoOcean's THREDDS server.

    Parameters
    ----------
    variables : List[str]
        List of variables to load.
    region : Tuple[float]
        Region to load. Format: (min_lon, min_lat, max_lon, max_lat).
    estela : bool, optional
        Whether to use the ESTELA predictor. Default is False.

    Returns
    -------
    era5 : xarray.Dataset
        Dataset with the loaded variables.

    Notes
    -----
    For more data, please check: https://github.com/GeoOcean/BlueMath/blob/main/toolkit/data/ERA5_DestinE_SingleLevels.ipynb
    """

    with ProgressBar():
        era5 = xr.open_dataset(
            "https://geoocean.sci.unican.es/thredds/dodsC/geoocean/era5-msl"
        )
        era5 = (
            era5[variables]
            .sel(
                latitude=slice(region[3], region[1], 4),
                longitude=slice(region[0], region[2], 4),
            )
            .load()
        )
        era5["time"] = np.char.decode(era5.time.values, encoding="utf-8").astype(
            "datetime64"  # Convert to datetime64 fot time operations
        )

    if estela:
        estela = xr.open_dataset("inputs/estela_sea.nc")
        era5 = get_dynamic_estela_predictor(data=era5, estela=estela)

    return era5
