"""Turning reconstructed wave fields into warnings.

The metamodel gives a wave field; a warning system needs three more things, and
this module holds them:

* **Where.** A handful of named places rather than 11,363 cells, because a
  warning is about a berth or a beach, not about a grid.
* **How much is a lot.** Thresholds taken from the reconstructed climatology at
  each place, not from a number chosen in advance — 0.4 m is unremarkable at the
  mouth of the bay and exceptional inside a marina, and only the climatology
  knows the difference.
* **For how long.** A single hour above a threshold is noise; what makes a berth
  unworkable is agitation that lasts.
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

#: Places the warnings are issued for, from the exposed mouth of the bay to the
#: sheltered inner reaches. Coordinates are approximate and snapped to the
#: nearest wet cell by `snap_to_sea`.
POINTS: Dict[str, Tuple[float, float]] = {
    "Sardinero": (-3.7480, 43.4700),        # open coast, the most exposed
    "Bocana": (-3.7789, 43.4620),           # entrance channel
    "Puertochico": (-3.7899, 43.4600),      # marina, and the tide gauge
    "Estacion Maritima": (-3.8009, 43.4590),
    "El Puntal": (-3.7779, 43.4550),
    "Somo": (-3.7604, 43.4480),
    "Raos": (-3.8179, 43.4340),             # commercial docks
    "Astillero": (-3.8134, 43.4020),        # inner bay, the most sheltered
}

#: Warning levels, quietest first.
ALERT_LEVELS: Tuple[str, ...] = ("calm", "yellow", "orange", "red")

#: Used where the model leaves a cell dry. Several of these places are
#: intertidal — at low water there is no sea state to warn about, which is not
#: the same as a calm one.
DRY = "dry"

#: Climatological percentiles separating them, one fewer than the levels.
ALERT_PERCENTILES: Tuple[float, ...] = (90.0, 99.0, 99.9)

#: Colour per level. Green is deliberately muted and red saturated: the eye
#: should land on the warnings, not on the quiet hours.
ALERT_COLORS: Dict[str, str] = {
    "calm": "#8FBF8F",
    "yellow": "#E9C46A",
    "orange": "#E76F51",
    "red": "#9B1D20",
    DRY: "#C7CCD1",
}


def snap_to_sea(
    field: xr.DataArray, points: Dict[str, Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Move each requested point to the nearest cell the model actually wets.

    Parameters
    ----------
    field : xr.DataArray
        Any reconstructed field on the bay grid, used only for its land mask:
        cells that are NaN are land.
    points : dict, optional
        Name to (lon, lat). Defaults to `POINTS`.

    Returns
    -------
    pd.DataFrame
        Indexed by name, with the requested `lon`/`lat`, the `cell_lon`/`cell_lat`
        used instead, and how far the point moved in metres.

    Notes
    -----
    The offset is worth reading. A few tens of metres is the grid spacing and
    means the point was already right; hundreds of metres means the coordinate
    was on land and the warning is really about somewhere else.
    """

    points = points or POINTS
    field = field.transpose("lat", "lon")

    lon_grid, lat_grid = np.meshgrid(field.lon.values, field.lat.values)
    wet = np.isfinite(field.values)
    wet_lon, wet_lat = lon_grid[wet], lat_grid[wet]

    rows = {}
    for name, (lon, lat) in points.items():
        distance = np.sqrt(
            ((wet_lon - lon) * np.cos(np.deg2rad(lat))) ** 2 + (wet_lat - lat) ** 2
        )
        nearest = int(np.argmin(distance))
        rows[name] = {
            "lon": lon,
            "lat": lat,
            "cell_lon": float(wet_lon[nearest]),
            "cell_lat": float(wet_lat[nearest]),
            "moved_m": float(distance[nearest] * 111320),
        }

    return pd.DataFrame(rows).T


def reconstruct(
    wind_pcs: pd.DataFrame,
    tide_state: pd.DataFrame,
    models: dict,
    variables: Sequence[str],
    cells: Optional[pd.DataFrame] = None,
    chunk: int = 4096,
) -> "xr.Dataset | Dict[str, pd.DataFrame]":
    """
    Wave fields for a period, each hour at the tide level closest to it.

    Parameters
    ----------
    wind_pcs : pd.DataFrame
        Wind principal components, indexed by time.
    tide_state : pd.DataFrame
        From `utils.tide.nearest_level`: a `case` column naming the tide level
        for each hour, indexed the same way.
    models : dict
        `models[case][variable]` is `(wave_pca, rbf, coefficients)`.
    variables : sequence of str
        Which wave variables to reconstruct.
    cells : pd.DataFrame, optional
        From `snap_to_sea`. When given, only those cells are kept and the result
        is one DataFrame per variable, time by place. When absent the whole field
        comes back as a Dataset — fine for a week, far too much for a decade.
    chunk : int, optional
        Hours reconstructed at a time. Default is 4096.

    Returns
    -------
    xr.Dataset or dict of pd.DataFrame
        Depending on `cells`.
    """

    from utils.rbf_cache import rbf_predict

    collected = {variable: [] for variable in variables}

    for case, group in tide_state.groupby("case"):
        block = pd.DatetimeIndex(group.index)

        for start in range(0, len(block), chunk):
            piece = block[start:start + chunk]

            for variable in variables:
                wave_pca, rbf, coefficients = models[case][variable]
                predicted = rbf_predict(rbf, wind_pcs.loc[piece], coefficients)
                field = wave_pca.inverse_transform(
                    PCs=xr.Dataset(
                        {"PCs": (("time", "n_component"), np.asarray(predicted))},
                        coords={"time": piece.values},
                    )
                )
                name = list(field.data_vars)[0] if len(field.data_vars) == 1 else None

                if cells is None:
                    collected[variable].append(field)
                else:
                    at_points = pd.DataFrame(
                        {
                            place: field[name].sel(
                                lon=row.cell_lon, lat=row.cell_lat, method="nearest"
                            ).values
                            for place, row in cells.iterrows()
                        },
                        index=piece,
                    ) if name else _points_from_uv(field, cells, piece)
                    collected[variable].append(at_points)

    if cells is None:
        return xr.merge([
            xr.concat(pieces, dim="time").sortby("time")
            for pieces in collected.values()
        ])

    return {
        variable: pd.concat(pieces).sort_index()
        for variable, pieces in collected.items()
    }


def _points_from_uv(field, cells, times):
    """Direction comes back as sine and cosine; recombine at the points."""

    values = {}
    for place, row in cells.iterrows():
        u = field["u_wave"].sel(lon=row.cell_lon, lat=row.cell_lat, method="nearest")
        v = field["v_wave"].sel(lon=row.cell_lon, lat=row.cell_lat, method="nearest")
        values[place] = np.rad2deg(np.arctan2(u.values, v.values)) % 360.0

    return pd.DataFrame(values, index=times)


def thresholds(
    climatology: pd.DataFrame, percentiles: Sequence[float] = ALERT_PERCENTILES
) -> pd.DataFrame:
    """
    Warning thresholds, one set per place, from its own reconstructed history.

    Parameters
    ----------
    climatology : pd.DataFrame
        Long reconstruction, time by place.
    percentiles : sequence of float, optional
        Percentiles separating the levels. Default is `ALERT_PERCENTILES`.

    Returns
    -------
    pd.DataFrame
        Indexed by place, one column per percentile, in the units of the input.
    """

    return pd.DataFrame(
        {f"p{value:g}": climatology.quantile(value / 100.0) for value in percentiles}
    )


def alert_levels(
    series: pd.DataFrame,
    limits: pd.DataFrame,
    persistence: int = 1,
    levels: Sequence[str] = ALERT_LEVELS,
) -> pd.DataFrame:
    """
    Assign a warning level to every hour and place.

    Parameters
    ----------
    series : pd.DataFrame
        Forecast wave height, time by place.
    limits : pd.DataFrame
        From `thresholds`, indexed by place.
    persistence : int, optional
        Hours a level must hold before it is issued. Default is 1, which issues
        every exceedance. Raising it suppresses the single-hour spikes that
        would make a warning flicker without changing whether a berth is
        workable.
    levels : sequence of str, optional
        Level names, quietest first.

    Returns
    -------
    pd.DataFrame
        The level name for each hour and place.

    Notes
    -----
    Persistence is applied by holding a level only where it survives a rolling
    minimum of `persistence` hours, so an isolated peak falls back to the level
    below rather than being deleted.

    Hours where the model leaves the cell dry come back as `DRY` rather than
    calm: at low water several of these places have no water in them, and
    reporting that as a quiet sea state would be wrong.
    """

    ranked = pd.DataFrame(0, index=series.index, columns=series.columns)

    for place in series.columns:
        for step, column in enumerate(limits.columns, start=1):
            ranked[place] += (series[place] > limits.loc[place, column]).astype(int)

    if persistence > 1:
        held = ranked.rolling(persistence, min_periods=persistence).min()
        # Look forwards as well as backwards: an exceedance counts if it sits
        # anywhere inside a run long enough, not only at the end of one.
        forward = ranked[::-1].rolling(persistence, min_periods=persistence).min()[::-1]
        ranked = pd.concat([held, forward]).groupby(level=0).max().fillna(0)
        ranked = ranked.astype(int)

    assigned = ranked.apply(lambda column: [levels[value] for value in column])

    return assigned.where(series.notna(), DRY)


def worst_hour(levels: pd.DataFrame, series: pd.DataFrame) -> pd.Timestamp:
    """
    The hour to put on the front page: highest level anywhere, then highest wave.

    Parameters
    ----------
    levels : pd.DataFrame
        From `alert_levels`.
    series : pd.DataFrame
        The wave heights those levels came from.

    Returns
    -------
    pd.Timestamp
    """

    order = {name: index for index, name in enumerate(ALERT_LEVELS)}
    rank = levels.apply(lambda column: [order.get(v, 0) for v in column])
    top = rank.max(axis=1)
    candidates = top[top == top.max()].index

    return series.loc[candidates].max(axis=1).idxmax()


def contingency(
    predicted: pd.Series, observed: pd.Series, limit: float
) -> Dict[str, float]:
    """
    How a threshold would have performed against an observed record.

    Parameters
    ----------
    predicted, observed : pd.Series
        Aligned on their index before counting; only hours present in both are
        used.
    limit : float
        The threshold, applied to both.

    Returns
    -------
    dict
        Counts (`hits`, `false_alarms`, `misses`, `correct_negatives`) and the
        three scores that matter for a warning: `pod` (probability of detection,
        the fraction of real events caught), `far` (false alarm ratio, the
        fraction of warnings that were wrong) and `csi` (critical success index,
        which penalises both).

    Notes
    -----
    There is no threshold that is good at both: raising it buys a lower false
    alarm ratio with missed events. The point of the table is to say where on
    that trade-off a choice sits, rather than to hide it.
    """

    both = pd.concat({"p": predicted, "o": observed}, axis=1).dropna()
    warned, happened = both["p"] > limit, both["o"] > limit

    hits = int((warned & happened).sum())
    false_alarms = int((warned & ~happened).sum())
    misses = int((~warned & happened).sum())
    correct = int((~warned & ~happened).sum())

    return {
        "n": len(both),
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "correct_negatives": correct,
        "pod": hits / (hits + misses) if hits + misses else np.nan,
        "far": false_alarms / (hits + false_alarms) if hits + false_alarms else np.nan,
        "csi": hits / (hits + misses + false_alarms)
        if hits + misses + false_alarms else np.nan,
    }
