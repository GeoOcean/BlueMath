from datetime import timedelta
from typing import List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import CubicSpline


def detect_outliers_window(
    data_series: pd.Series, deviation_threshold: float = 3, window_size_days: float = 2
) -> List[int]:
    """
    Detect outliers in a time series based on a rolling window median and standard deviation.

    Parameters
    ----------
    data_series : pd.Series
        Time-indexed series of values (e.g., sea level) to analyze.
    deviation_threshold : float, optional
        Number of standard deviations a point must differ from the window median
        to be flagged as an outlier. Default is 3.
    window_size_days : float, optional
        Total length of the rolling window in days (centered on each point).
        Default is 2.

    Returns
    -------
    List[int]
        List of indices in `data_series` identified as outliers.

    Notes
    -----
    The function uses a centered rolling window approach where each data point
    is compared against the median and standard deviation of surrounding data
    within the specified time window.
    """

    outliers = []
    half_window = pd.Timedelta(window_size_days / 2, "d")

    for i, (time, value) in enumerate(data_series.items()):
        window_start = time - half_window
        window_end = time + half_window
        window = data_series.loc[window_start:window_end]

        if window.empty or len(window) < 2:
            continue

        med_value = window.median()
        std_dev = window.std()

        if std_dev == 0:  # avoid division by zero
            continue

        if abs(value - med_value) > deviation_threshold * std_dev:
            outliers.append(i)

    return outliers


def detrend_series(series: pd.Series) -> Tuple[pd.Series, str]:
    """
    Remove linear trend from a time-indexed series using linear regression.

    Parameters
    ----------
    series : pd.Series
        Time-indexed data series to detrend.

    Returns
    -------
    detrended_series : pd.Series
        The detrended data series.
    trend_str : str
        String representation of the linear trend: "slope * t + intercept".

    Notes
    -----
    The function fits a linear regression to the time series and subtracts
    the fitted trend from the original data.
    """

    series_clean = series.dropna()
    if len(series_clean) < 2:
        raise ValueError(
            "Series must contain at least 2 non-null values for detrending"
        )

    t_numeric = np.arange(len(series_clean))

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        t_numeric, series_clean
    )

    trend_str = f"{slope:.6f} * t + {intercept:.6f}"
    detrended_series = series_clean - (slope * t_numeric + intercept)

    return detrended_series, trend_str


class HarmonicAnalysis:
    """
    Class for harmonic analysis visualization and coefficient plotting.

    This class provides methods to visualize harmonic analysis results,
    including amplitude and phase information for tidal constituents.
    """

    def __init__(self, coef, efreq: Optional[Union[float, List[float]]] = None):
        """
        Initialize the HarmonicAnalysis class.

        Parameters
        ----------
        coef : object
            Object containing harmonic analysis results, with attributes:
            - coef.A: Amplitudes
            - coef.g: Phase lags (degrees)
            - coef.name: Names of constituents
            - coef.aux.frq: Frequencies
        efreq : float or list of float, optional
            Frequency limit(s) for visual reference on plot.
        """

        self.coef = coef
        self.efreq = efreq

    def plot_coefs(
        self, alim: float = 0.02, show: bool = True, title: Optional[str] = None
    ) -> None:
        """
        Plot the coefficients obtained from harmonic analysis.

        Parameters
        ----------
        alim : float, optional
            Minimum amplitude for plotting coefficient name. Default is 0.02.
        show : bool, optional
            Flag to control whether to display the plot. Default is True.
        title : str, optional
            Title of the plot. Default is None.

        Notes
        -----
        The plot shows amplitude vs frequency with phase information
        encoded in the color of the vertical lines.
        """

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.vlines(
            self.coef.aux.frq,
            ymin=0,
            ymax=self.coef.A,
            color=cm.twilight(self.coef.g / 360),
        )

        for i, c in enumerate(self.coef.name):
            if self.coef.A[i] > alim:
                ax.text(
                    self.coef.aux.frq[i],
                    self.coef.A[i] + 0.002,
                    c,
                    ha="center",
                    c="k",
                    size=8,
                )

        ax.set_xticks(self.coef.aux.frq)
        ax.set_xticklabels(self.coef.aux.frq)
        ax.set(
            xlabel="Frequency [$h^{-1}$]",
            ylabel="Amplitude [m]",
            ylim=[0, None],
            xscale="log",
        )

        if self.efreq is not None:
            ax.vlines(
                self.efreq,
                ymin=0,
                ymax=ax.get_ylim()[1],
                ls="--",
                color="k",
                lw=1,
                label="Frequency limit",
            )
            ax.legend()

        ax.grid(axis="y")

        cbar = plt.colorbar(cm.ScalarMappable(cmap="twilight"), ax=ax)
        cbar.set_ticks(np.linspace(0, 1, 7))
        cbar.set_ticklabels([str(el) for el in range(0, 361, 60)])
        cbar.set_label("Phase lag [degrees]")

        if title:
            fig.suptitle(title)

        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax


def extract_high_water_tides(
    tides: List[pd.Series], min_hours_between: float = 6
) -> List[pd.DataFrame]:
    """
    Extract high water tides using the turning point method.

    This function identifies high water peaks in tidal data using cubic spline
    interpolation to find turning points, then filters out peaks that are too
    close together.

    Parameters
    ----------
    tides : List[pd.Series]
        List of pandas Series with datetime index containing predicted sea level data.
    min_hours_between : float, optional
        Minimum time between valid high waters in hours. Default is 6.

    Returns
    -------
    List[pd.DataFrame]
        List of DataFrames containing valid high water tides for each tide series.
        Each DataFrame contains columns for time and sea level values.

    Notes
    -----
    The function uses cubic spline interpolation to find turning points in the
    tidal signal, then applies a filtering step to remove peaks that are too
    close together based on the minimum time threshold.
    """

    def drop_close_highs(df_orig: pd.DataFrame, h: float) -> pd.DataFrame:
        """
        Remove high water peaks that are too close together.

        Parameters
        ----------
        df_orig : pd.DataFrame
            DataFrame with time and sea level columns.
        h : float
            Minimum hours between peaks.

        Returns
        -------
        pd.DataFrame
            DataFrame with close peaks removed.
        """

        df = df_orig.copy()
        df["time_diff"] = df["time"].diff()

        while (df["time_diff"].iloc[1:] <= timedelta(hours=h)).any():
            rows_to_drop = []
            for i in range(len(df) - 1):
                time_diff = df.iloc[i + 1]["time"] - df.iloc[i]["time"]
                if time_diff <= timedelta(hours=h):
                    # Keep the higher peak
                    if df.iloc[i + 1][0] > df.iloc[i][0]:
                        rows_to_drop.append(i)
                    else:
                        rows_to_drop.append(i + 1)
            df = df.drop(rows_to_drop).reset_index(drop=True)
            df["time_diff"] = df["time"].diff()

        return df

    high_water_tides = []

    for series in tides:
        cstime = series.index
        cs = CubicSpline(cstime, series.values.flatten())
        turning_idxs = np.where(np.diff(np.sign(np.diff(cs(cstime)))) < 0)[0] + 1
        turning_times = series.index[turning_idxs]

        high_water = pd.DataFrame(
            series.iloc[turning_idxs].values.flatten(), index=turning_times
        )
        high_water["time"] = high_water.index.to_series()
        high_water = high_water.reset_index(drop=True)

        high_water = drop_close_highs(high_water, min_hours_between)
        high_water = high_water[high_water[0] > 0]  # Remove false/negative peaks

        high_water_tides.append(high_water)

    return high_water_tides


def find_measured_high_water(
    detrended_SL: List[pd.Series],
    high_water_tides: List[Union[pd.Series, pd.DataFrame]],
    time_window_hours: float = 6,
) -> List[pd.Series]:
    """
    Find measured high water peaks around predicted high water tides.

    This function searches for actual high water peaks in measured sea level data
    within a specified time window around predicted high water times.

    Parameters
    ----------
    detrended_SL : List[pd.Series]
        List of pandas Series containing detrended sea level data.
    high_water_tides : List[Union[pd.Series, pd.DataFrame]]
        List of pandas Series or DataFrames with predicted high water times and values.
    time_window_hours : float, optional
        Time window in hours around each predicted high water timestamp
        to search for maxima. Default is 6.

    Returns
    -------
    List[pd.Series]
        List of pandas Series with measured high water peaks (values indexed
        by timestamp of maxima).

    Notes
    -----
    The function searches for the maximum value within the specified time window
    around each predicted high water time and returns the actual measured peaks.
    """

    time_window = pd.Timedelta(hours=time_window_hours)
    high_water_measured = []

    for k, detrended_series in enumerate(detrended_SL):
        # Ensure high_water_tides[k] is a Series with times as index and values as predicted high water
        if isinstance(high_water_tides[k], pd.DataFrame):
            high_water_tides[k] = pd.Series(
                high_water_tides[k][0].values, index=high_water_tides[k]["time"]
            )

        maxima = []
        maxima_times = []

        for timestamp in high_water_tides[k].index:
            start_time = timestamp - time_window
            end_time = timestamp + time_window

            window_data = detrended_series[start_time:end_time]

            if not window_data.empty and len(window_data) > 0:
                max_value = window_data.max()
                max_time = window_data.idxmax()
                maxima.append(max_value)
                maxima_times.append(max_time)

        if maxima:  # Only create series if we found maxima
            maxima_series = pd.Series(maxima, index=maxima_times)
            high_water_measured.append(maxima_series)
        else:
            high_water_measured.append(pd.Series(dtype=float))

    return high_water_measured
