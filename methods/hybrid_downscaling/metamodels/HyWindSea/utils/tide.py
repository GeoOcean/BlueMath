"""Tide level for the metamodel: which of the five SWAN levels applies when.

The metamodel is fitted at five discrete tide levels, so running it over a period
means answering one question per hour — which level is closest right now — and
that comes from a harmonic prediction of the tide.

The expensive part is fitting the harmonic constituents, and it only has to
happen once: they describe the gauge, not the forecast. `tide_coefficients`
caches them, which turns every later prediction into a few milliseconds.
"""

import os
import pickle
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

#: Tide levels the metamodel was fitted at, in metres above mean sea level.
LEVELS: Sequence[float] = (-1.5, 0.0, 1.5, 2.5, 3.5)

#: How each level is spelled in the model file names.
LEVEL_NAMES: Dict[float, str] = {
    -1.5: "neg1.5m",
    0.0: "pos0m",
    1.5: "pos1.5m",
    2.5: "pos2.5m",
    3.5: "pos3.5m",
}

#: Santander tide gauge.
GAUGE_LAT = 43.46

_REDMAR_COLUMNS = (
    ["YY", "MM", "DD", "HH", "Niv_H", "Mar_H", "Res_H"]
    + [f"Niv_{minute:02d}" for minute in range(0, 60, 5)]
    + ["Pro"]
)


def read_redmar(file_path: str, column: str = "Mar_H") -> pd.Series:
    """
    Read a REDMAR hourly record.

    Parameters
    ----------
    file_path : str
        Path to the REDMAR text file.
    column : str, optional
        Which level to return. 'Mar_H' is the astronomical tide, 'Niv_H' the
        total observed level and 'Res_H' the residual. Default is 'Mar_H'.

    Returns
    -------
    pd.Series
        The chosen level in **metres above its own mean**, indexed by time. The
        file stores centimetres and -9999 for missing values; both are handled.

    Notes
    -----
    Fitting the harmonics to `Mar_H` rather than `Niv_H` keeps the surge out of
    the analysis, which is what makes the prediction a tide prediction.
    """

    table = pd.read_csv(
        file_path, sep=r"\s+", skiprows=8, names=_REDMAR_COLUMNS, na_values=-9999
    )
    table.index = pd.to_datetime(
        dict(year=table.YY, month=table.MM, day=table.DD, hour=table.HH)
    )

    level = table[column] / 100.0

    return level - level.mean()


def tide_coefficients(
    file_path: str,
    cache_path: Optional[str] = None,
    lat: float = GAUGE_LAT,
    column: str = "Mar_H",
    verbose: bool = True,
) -> dict:
    """
    Harmonic constituents for the gauge, fitted once and cached.

    Parameters
    ----------
    file_path : str
        REDMAR record to fit.
    cache_path : str, optional
        Where to keep the fitted constituents. Read if it exists, written if it
        does not. Pass None to fit every time.
    lat : float, optional
        Gauge latitude, used for the nodal corrections. Default is Santander's.
    column : str, optional
        Level column to fit. Default is 'Mar_H', the astronomical tide.
    verbose : bool, optional
        Report whether the cache was used and how long the fit took.

    Returns
    -------
    dict
        The `utide` coefficient structure, ready for `predict_tide`.

    Notes
    -----
    Two choices make this fast. The fit runs with ``conf_int="none"``: the
    confidence intervals are a Monte Carlo estimate that nothing downstream
    reads, and on a 30-year hourly record they dominate the cost — enough to ask
    for several GB of memory. And the result is cached, because the constituents
    belong to the gauge, not to the period being predicted: fitting takes
    around twenty seconds, reading the cache takes none, and predicting any
    number of hours from it takes milliseconds either way.
    """

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as handle:
            coefficients = pickle.load(handle)
        if verbose:
            print(f"constituents read from {cache_path}")
        return coefficients

    import time

    import utide

    level = read_redmar(file_path, column=column)

    start = time.time()
    coefficients = utide.solve(
        level.index,
        level.values,
        lat=lat,
        method="ols",
        conf_int="none",
        verbose=False,
    )
    if verbose:
        print(f"fitted {len(coefficients['name'])} constituents to "
              f"{len(level):,} hours in {time.time() - start:.0f} s")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump(coefficients, handle)
        if verbose:
            print(f"cached to {cache_path}")

    return coefficients


def predict_tide(
    coefficients: dict, times: Union[pd.DatetimeIndex, Sequence]
) -> pd.Series:
    """
    Harmonic tide prediction at the given times.

    Parameters
    ----------
    coefficients : dict
        From `tide_coefficients`.
    times : pd.DatetimeIndex or sequence
        Instants to predict.

    Returns
    -------
    pd.Series
        Tide level in metres above mean sea level, indexed by time.
    """

    import utide

    times = pd.DatetimeIndex(times)
    prediction = utide.reconstruct(times, coefficients, verbose=False)

    return pd.Series(prediction.h, index=times, name="tide")


def nearest_level(
    tide: pd.Series, levels: Sequence[float] = LEVELS
) -> pd.DataFrame:
    """
    Assign each instant the fitted tide level closest to it.

    Parameters
    ----------
    tide : pd.Series
        Predicted tide in metres, indexed by time.
    levels : sequence of float, optional
        Levels the metamodel was fitted at. Default is `LEVELS`.

    Returns
    -------
    pd.DataFrame
        Indexed by time, with `tide` as predicted, `level` the level assigned,
        and `case` its name in the model files.

    Notes
    -----
    The levels are not evenly spaced, so "closest" is worth taking literally
    rather than rounding: the gap between 0 and 1.5 m is snapped at 0.75, but
    the one between 1.5 and 2.5 m at 2.0.
    """

    levels = np.asarray(levels, dtype=float)
    assigned = levels[np.abs(tide.values[:, None] - levels[None, :]).argmin(axis=1)]

    return pd.DataFrame(
        {
            "tide": tide.values,
            "level": assigned,
            "case": [LEVEL_NAMES[level] for level in assigned],
        },
        index=pd.DatetimeIndex(tide.index),
    )
