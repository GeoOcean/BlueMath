import heapq
import re
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy.signal
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import kendalltau, pearsonr, spearmanr


def extract_top_event_types(
    event_types: Dict[str, Dict[str, Dict[str, np.ndarray]]],
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Extract the top 1 most prevalent event types per station, including ties.

    Parameters
    ----------
    event_types : dict
        Event types structured by type -> station -> index -> np.array

    Returns
    -------
    tuple
        top_event_types : dict
            Top event types per station (including ties)
        unique_top_types : list
            Unique event types across all stations
    """

    # Count occurrences of each event type per station
    station_type_counts = defaultdict(Counter)
    for event_type, stations in event_types.items():
        for station, events in stations.items():
            station_type_counts[station][event_type] += len(events)

    # Convert counts to percentages
    station_type_percentages = {}
    for station, counts in station_type_counts.items():
        total = sum(counts.values())
        percentages = {etype: (count / total) * 100 for etype, count in counts.items()}
        station_type_percentages[station] = percentages

    # Extract top 1 most prevalent event types, including ties
    top_event_types = {}
    for station, percentages in station_type_percentages.items():
        sorted_types = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        top_types = []
        if sorted_types:
            top_percentage = sorted_types[0][1]
            top_types.extend(
                [etype for etype, pct in sorted_types if pct == top_percentage]
            )
        top_event_types[station] = top_types

    # Get all unique top event types
    unique_top_types = list(
        {etype for types in top_event_types.values() for etype in types}
    )

    return top_event_types, unique_top_types


def extract_all_event_timeseries_and_high_tides(
    NTR: List[pd.DataFrame],
    high_tide: List[pd.DataFrame],
    extremes_NTR: List[pd.DataFrame],
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Extract NTR and high tide time series for all events.

    Parameters
    ----------
    NTR : list of pd.DataFrame
        Non-tidal residual time series for each station.
    high_tide : list of pd.DataFrame
        High tide time series for each station.
    extremes_NTR : list of pd.DataFrame
        Extreme event dataframes with 'start' and 'end' columns.

    Returns
    -------
    tuple
        NTR_timeseries : list of pd.DataFrame
            NTR segments for each event.
        high_tides_timeseries : list of pd.DataFrame
            High tide segments for each event (±24h extension).
    """

    NTR_timeseries = []
    high_tides_timeseries = []

    for ntr_df, ht_df, extremes in zip(NTR, high_tide, extremes_NTR):
        for _, row in extremes.iterrows():
            # NTR segment within event bounds
            ntr_segment = ntr_df[
                (ntr_df.index >= row["start"]) & (ntr_df.index <= row["end"])
            ]
            NTR_timeseries.append(ntr_segment)

            # High tide segment with ±24h extension
            start_ext = row["start"] - timedelta(hours=24)
            end_ext = row["end"] + timedelta(hours=24)
            ht_segment = ht_df[(ht_df.index >= start_ext) & (ht_df.index <= end_ext)]
            high_tides_timeseries.append(ht_segment)

    return NTR_timeseries, high_tides_timeseries


def extract_event_type_datetimes(
    event_types: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    extremes_NTR: List[pd.DataFrame],
) -> Tuple[List[List[pd.Timestamp]], List[List[Any]], List[Dict[int, int]]]:
    """
    Extract datetime information for each event type and compute unique event dates and yearly counts.

    Parameters
    ----------
    event_types : dict
        Event types structured by type -> station -> index -> np.array.
    extremes_NTR : list of pd.DataFrame
        Contains the datetime indices.

    Returns
    -------
    tuple
        datetime_clusters : list of list of pd.Timestamp
            Datetime values per event type.
        unique_dates : list of list of datetime.date
            Unique dates per event type.
        unique_yearly_count : list of dict
            Yearly counts of events per type.
    """

    # Flatten datetime index and build lookup
    datetime = pd.DatetimeIndex(np.concatenate([df.index for df in extremes_NTR]))
    array_extremes = np.concatenate([df.values for df in extremes_NTR])
    assert len(datetime) == array_extremes.shape[0], "Datetime length mismatch"
    idx_to_dt = {str(i): dt for i, dt in enumerate(datetime)}

    datetime_clusters, unique_dates, yearly_counts = [], [], []

    for stations in event_types.values():
        # Gather datetime for all indices in this event type
        dts = [idx_to_dt[k] for s in stations.values() for k in s if k in idx_to_dt]
        datetime_clusters.append(dts)

        # Unique dates
        dates = sorted(set(dt.date() for dt in dts))
        unique_dates.append(dates)

        # Count events per year with ±2-day grouping
        per_year = defaultdict(list)
        for d in dates:
            per_year[d.year].append(d)

        counts = {}
        for y, dlist in per_year.items():
            dlist.sort()
            count, ref = 0, None
            for d in dlist:
                if not ref or (d - ref).days > 3:
                    count += 1
                    ref = d
            counts[y] = count
        yearly_counts.append(counts)

    return datetime_clusters, unique_dates, yearly_counts


def plot_yearly_event_counts(
    event_types: Dict[str, Any],
    unique_yearly_count: List[Dict[int, int]],
    num_rows: int = 2,
    num_cols: int = 5,
) -> None:
    """
    Plot yearly event counts for each storm surge type.

    Parameters
    ----------
    event_types : dict
        Event types with keys (e.g. 'type_1', 'type_2', ...).
    unique_yearly_count : list of dict
        Yearly counts corresponding to the keys in prevalent_types (unsorted).
    num_rows : int, optional
        Number of rows in subplot grid. Default is 2.
    num_cols : int, optional
        Number of columns in subplot grid. Default is 5.
    """

    # Get the unsorted list of event type keys
    original_types = list(event_types.keys())

    # Create a mapping from type name to yearly count
    type_to_yearly = dict(zip(original_types, unique_yearly_count))

    # Sort type names
    sorted_type_names = sorted(
        event_types.keys(), key=lambda x: int(re.search(r"\d+", x).group())
    )

    # Get the yearly count in sorted order
    sorted_yearly_counts = [type_to_yearly[etype] for etype in sorted_type_names]

    # Setup plot
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 4), dpi=600)
    axs = axs.flatten()

    for i, (etype, yearly_counts) in enumerate(
        zip(sorted_type_names, sorted_yearly_counts)
    ):
        years = list(yearly_counts.keys())
        counts = list(yearly_counts.values())

        axs[i].bar(years, counts, color="#6AAD91", width=1)
        axs[i].set_title(f"{etype}", fontsize=8, fontweight="bold", pad=3)
        axs[i].set_xlim(1950, 2018)
        axs[i].set_ylim(0, 5)
        axs[i].grid(True, linewidth=0.3)
        axs[i].tick_params(axis="x", labelsize=8, colors="black")
        axs[i].tick_params(axis="y", labelsize=8, colors="black")
        axs[i].set_yticks([1, 2, 3, 4])
        axs[i].set_xticks([1960, 1980, 2000, 2018])
        axs[i].set_xticklabels(["'60", "'80", "'00", "'18"])

        if i % num_cols != 0:
            axs[i].set_yticklabels([])

        if i // num_cols != num_rows - 1:
            axs[i].set_xticklabels([])

        for spine in axs[i].spines.values():
            spine.set_edgecolor("gray")
            spine.set_linewidth(0.3)

    fig.text(
        -0.01,
        0.5,
        "Number of events per year",
        va="center",
        rotation="vertical",
        fontsize=10,
        color="black",
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Year",
        va="center",
        ha="center",
        fontsize=10,
        color="black",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()


def download_nao_index(
    url: str = "https://www.psl.noaa.gov/data/correlation/nao.data",
) -> pd.DataFrame:
    """
    Download and return the raw NAO data as a DataFrame.

    Parameters
    ----------
    url : str, optional
        URL to download NAO data from. Default is NOAA PSL NAO data URL.

    Returns
    -------
    pd.DataFrame
        Raw NAO data with Year column and monthly values.
    """

    response = requests.get(url)
    lines = response.text.strip().split("\n")[1:-3]  # Skip header and footer
    data = [list(map(float, line.split())) for line in lines]

    return pd.DataFrame(data, columns=["Year"] + list(range(1, 13)))


def compute_annual_nao_means(
    df: pd.DataFrame, missing_val: float = -99.9
) -> Dict[int, float]:
    """
    Compute yearly average NAO index from the monthly data.

    Parameters
    ----------
    df : pd.DataFrame
        NAO data with Year column and monthly values.
    missing_val : float, optional
        Value to treat as missing data. Default is -99.9.

    Returns
    -------
    dict
        Dictionary with year as key and annual mean NAO as value.
    """

    df = df.replace(missing_val, np.nan)

    return {
        int(row["Year"]): row.iloc[1:].mean(skipna=True) for _, row in df.iterrows()
    }


def filter_years(
    data: Dict[int, float], start: int = 1950, end: int = 2017
) -> Dict[int, float]:
    """
    Filter a dictionary to include only years within the given range.

    Parameters
    ----------
    data : dict
        Dictionary with year as key and values.
    start : int, optional
        Start year (inclusive). Default is 1950.
    end : int, optional
        End year (inclusive). Default is 2017.

    Returns
    -------
    dict
        Filtered dictionary with years in the specified range.
    """

    return {year: val for year, val in data.items() if start <= year <= end}


def aggregate_cluster_events(clusters: List[Dict[int, int]]) -> Dict[int, int]:
    """
    Aggregate events per year across all clusters.

    Parameters
    ----------
    clusters : list of dict
        List of dictionaries with year as key and event count as value.

    Returns
    -------
    dict
        Aggregated event counts per year across all clusters.
    """

    total = {}
    for cluster in clusters:
        for year, count in cluster.items():
            total[year] = total.get(year, 0) + count

    return total


def compute_correlations(events: Dict[int, int], nao: Dict[int, float]) -> None:
    """
    Compute Pearson, Spearman, and Kendall correlations between events and NAO index.

    Parameters
    ----------
    events : dict
        Dictionary with year as key and event count as value.
    nao : dict
        Dictionary with year as key and NAO index as value.

    Raises
    ------
    ValueError
        If insufficient data for correlation calculation.
    """

    shared_years = sorted(set(events) & set(nao))
    if len(shared_years) < 2:
        raise ValueError("Insufficient data for correlation calculation.")

    event_values = [events[year] for year in shared_years]
    nao_values = [nao[year] for year in shared_years]

    pearson_corr, pearson_p = pearsonr(event_values, nao_values)
    spearman_corr, spearman_p = spearmanr(event_values, nao_values)
    kendall_corr, kendall_p = kendalltau(event_values, nao_values)

    print(f"Pearson:  {pearson_corr:.3f} (p={pearson_p:.4f})")
    print(f"Spearman: {spearman_corr:.3f} (p={spearman_p:.4f})")
    print(f"Kendall:  {kendall_corr:.3f} (p={kendall_p:.4f})")


def plot_scatter_nao_vs_events(
    events: Dict[int, int], nao: Dict[int, float], title: str = "NAO vs Event Frequency"
) -> None:
    """
    Plot a scatter plot between NAO index and event counts.

    Parameters
    ----------
    events : dict
        Dictionary with year as key and event count as value.
    nao : dict
        Dictionary with year as key and NAO index as value.
    title : str, optional
        Plot title. Default is "NAO vs Event Frequency".

    Raises
    ------
    ValueError
        If not enough data to plot scatter.
    """

    years = sorted(set(events) & set(nao))
    if len(years) < 2:
        raise ValueError("Not enough data to plot scatter.")

    x = [nao[year] for year in years]
    y = [events[year] for year in years]

    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, color="#6AAD91", edgecolors="black")
    plt.xlabel("NAO Index")
    plt.ylabel("Number of Events")
    plt.title(title, fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", linewidth=0.3)
    plt.tight_layout()
    plt.show()


def cumulative_intensity(time_series: pd.Series, threshold_99: float) -> float:
    """
    Compute cumulative intensity over threshold for a single time series.

    Parameters
    ----------
    time_series : pd.Series
        Time series data with datetime index.
    threshold_99 : float
        99th percentile threshold value.

    Returns
    -------
    float
        Cumulative intensity above threshold.
    """

    time = time_series.index
    time_numeric = (time - time[0]).total_seconds() / 3600.0  # hours

    above_threshold = time_series >= threshold_99

    if above_threshold.any():
        ts_filtered = time_series.copy()
        ts_filtered[~above_threshold] = 0
        intensity = np.trapz(ts_filtered, time_numeric)
        return intensity
    else:
        return 0.0


def compute_cumulative_intensities_types(
    event_types: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    NTR_timeseries: List[pd.DataFrame],
    stations: pd.DataFrame,
) -> Dict[str, List[float]]:
    """
    Compute cumulative intensity per event type across all stations.

    Parameters
    ----------
    event_types : dict
        Event types structured by type -> station -> index -> np.array.
    NTR_timeseries : list of pd.DataFrame
        NTR time series for each event.
    stations : pd.DataFrame
        Station data with 'station' and 'p99' columns.

    Returns
    -------
    dict
        Dictionary with event type as key and list of intensities as value
    """

    cumulative_intensity_trap = {}

    # Map station names to 99th percentile thresholds
    station_thresholds = dict(zip(stations["station"], stations["p99"]))

    for event_type, station_dict in event_types.items():
        type_intensities = []

        for station_name, events_dict in station_dict.items():
            threshold_99 = station_thresholds.get(station_name)
            if threshold_99 is None:
                continue  # skip if no threshold

            for key_str in events_dict.keys():
                key = int(key_str)
                time_series = NTR_timeseries[key]

                intensity = cumulative_intensity(time_series, threshold_99)
                type_intensities.append(intensity)

        cumulative_intensity_trap[event_type] = type_intensities

    return cumulative_intensity_trap


def get_top_events(
    cumulative_intensity_trap: Dict[str, List[float]],
    event_types: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    top_n: int = 10,
) -> List[Tuple[float, str, str, str]]:
    """
    Extract top N events by intensity across all event types.

    Parameters
    ----------
    cumulative_intensity_trap : dict
        Dictionary with event type as key and list of intensities as value.
    event_types : dict
        Event types structured by type -> station -> index -> np.array.
    top_n : int, optional
        Number of top events to return. Default is 10.

    Returns
    -------
    list of tuple
        List of (intensity, event_type, station, key_str) tuples.
    """

    flat_intensities = []

    for event_type, intensities in cumulative_intensity_trap.items():
        station_dict = event_types[event_type]

        # Flatten keys in the station dict (station_name -> keys)
        keys_flat = []
        for station_name, keys_dict in station_dict.items():
            keys_flat.extend([(station_name, key_str) for key_str in keys_dict.keys()])

        # Now intensities and keys_flat correspond index-wise
        for i, intensity in enumerate(intensities):
            station_name, key_str = keys_flat[i]
            flat_intensities.append((intensity, event_type, station_name, key_str))

    # Get top N events by intensity
    top_events = heapq.nlargest(top_n, flat_intensities, key=lambda x: x[0])

    return top_events


def plot_event_cumulative_intensity(
    event_types: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    NTR_timeseries: List[pd.DataFrame],
    threshold_99: float,
    key: int,
) -> None:
    """
    Plot time series for a given event key with threshold highlighting.

    Parameters
    ----------
    event_types : dict
        Event types structured by type -> station -> index -> np.array.
    NTR_timeseries : list of pd.DataFrame
        NTR time series for each event.
    threshold_99 : float
        Threshold value for highlighting.
    key : int
        Event key (corresponding to keys in event_types).

    Raises
    ------
    ValueError
        If key not found in event_types.
    """

    # Find the station and event_type that contains the given key
    found = False
    for event_type, station_dict in event_types.items():
        for station_name, events_dict in station_dict.items():
            if str(key) in events_dict:
                found = True
                break
        if found:
            break
    if not found:
        raise ValueError(f"Key {key} not found in event_types")

    # Extract full time series for this key
    time_series = NTR_timeseries[key]

    # Determine which values exceed threshold
    event_series_masked = time_series.where(time_series >= threshold_99)

    # Plot setup
    plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[2, 1])
    ax1 = plt.subplot(gs[0, :])

    # Plot full series and highlight above-threshold event segment
    ax1.plot(
        time_series.index,
        time_series.values,
        label="Original Time Series",
        color="#6F1D77",
        linewidth=1,
    )
    ax1.plot(
        event_series_masked.index,
        event_series_masked.values,
        label="Above Threshold",
        color="#431148",
        linewidth=5,
        zorder=6,
    )
    ax1.axhline(y=threshold_99, color="black", linestyle="--", linewidth=0.8, zorder=6)

    # Style
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_ylabel("Non-tidal residual [m]", fontsize=8, fontweight="bold", labelpad=10)
    ax1.tick_params(axis="both", labelsize=8)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("'%y %b %d"))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    for i, label in enumerate(ax1.get_xticklabels()):
        if i % 3 != 0:
            label.set_visible(False)

    ax1.text(
        ax1.get_xlim()[0] + 0.04 * (ax1.get_xlim()[1] - ax1.get_xlim()[0]),
        threshold_99,
        "99th",
        color="black",
        fontsize=8,
        ha="left",
        va="center",
        backgroundcolor="white",
        zorder=6,
        fontweight="bold",
    )

    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1, hspace=0.3)
    plt.show()


def smooth_signal(
    series: pd.Series, window_hours: int = 12
) -> Tuple[np.ndarray, int, float]:
    """
    Smooth a time series using a moving average window.

    Parameters
    ----------
    series : pd.Series
        Time series to smooth.
    window_hours : int, optional
        Window size in hours. Default is 12.

    Returns
    -------
    tuple
        smoothed : np.ndarray
            Smoothed time series.
        window_size : int
            Window size in samples.
        time_diff : float
            Time difference between samples in seconds.
    """

    time_diff = (series.index[1] - series.index[0]).total_seconds()
    window_size = int((window_hours * 3600) / time_diff)
    smoothed = np.convolve(
        series.values, np.ones(window_size) / window_size, mode="same"
    )

    return smoothed, window_size, time_diff


def find_peaks_in_smoothed(
    smoothed: np.ndarray, time_index: pd.DatetimeIndex, time_step_sec: float
) -> List[Tuple[int, pd.Timestamp]]:
    """
    Find peaks in smoothed time series with prominence filtering.

    Parameters
    ----------
    smoothed : np.ndarray
        Smoothed time series.
    time_index : pd.DatetimeIndex
        Time index for the series.
    time_step_sec : float
        Time step in seconds.

    Returns
    -------
    list of tuple
        List of (peak_index, peak_date) tuples.
    """

    wlen_max = int((72 * 3600) / time_step_sec)
    wlen_min = int((12 * 3600) / time_step_sec)
    wlen_samples = max(wlen_min, min(wlen_max, len(smoothed)))

    peaks, props = scipy.signal.find_peaks(smoothed, prominence=0.02, wlen=wlen_samples)
    peak_dates = time_index[peaks]
    prominences = props["prominences"]

    peaks_with_prom = sorted(zip(peaks, prominences, peak_dates), key=lambda x: -x[1])

    seen = []
    final_peaks = []
    for peak, prom, pdate in peaks_with_prom:
        if all(abs((pdate - prev).days) >= 0.52 for prev in seen):
            final_peaks.append((peak, pdate))
            seen.append(pdate)

    return final_peaks


def extract_original_maxima_near_peaks(
    series: pd.Series,
    smoothed_peaks: List[Tuple[int, pd.Timestamp]],
    search_radius: int,
) -> List[Tuple[pd.Timestamp, float]]:
    """
    Extract original maxima near smoothed peaks.

    Parameters
    ----------
    series : pd.Series
        Original time series.
    smoothed_peaks : list of tuple
        List of (peak_index, peak_date) tuples from smoothed series.
    search_radius : int
        Search radius in samples around each peak.

    Returns
    -------
    list of tuple
        List of (max_time, max_val) tuples.
    """

    maxima = []

    for peak_idx, _ in smoothed_peaks:
        left = max(0, peak_idx - search_radius)
        right = min(len(series), peak_idx + search_radius)
        window = series.iloc[left:right]
        if not window.empty:
            max_time = window.idxmax()
            max_val = window.max()
            maxima.append((max_time, max_val))

    return maxima


def plot_all_events(processed_events: List[Dict[str, Any]]) -> None:
    """
    Plot all processed events in a grid layout.

    Parameters
    ----------
    processed_events : list of dict
        List of event dictionaries with 'series', 'smoothed', 'peaks', and 'maxima' keys.
    """

    n = len(processed_events)
    n_cols = 5
    n_rows = int(np.ceil(n / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    axs = axs.flatten()

    for i, event in enumerate(processed_events):
        ax = axs[i]
        ax.plot(
            event["series"].index, event["smoothed"], color="blue", label="Smoothed"
        )
        ax.plot(
            event["series"].index,
            event["series"].values,
            color="green",
            alpha=0.5,
            label="Original",
        )

        for peak_idx, _ in event["peaks"]:
            ax.plot(event["series"].index[peak_idx], event["smoothed"][peak_idx], "rx")

        for max_time, max_val in event["maxima"]:
            ax.plot(max_time, max_val, "gx")

        ax.tick_params(labelsize=6)

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()


def find_nearest_high_tide(
    peak_time: pd.Timestamp, high_tide_times: List[pd.Timestamp]
) -> Tuple[pd.Timestamp, float]:
    """
    Find the nearest high tide time to a peak time.

    Parameters
    ----------
    peak_time : pd.Timestamp
        The time of the peak.
    high_tide_times : list of pd.Timestamp
        The high tide times to compare against.

    Returns
    -------
    tuple
        nearest_high_tide : pd.Timestamp
            The closest high tide time.
        time_diff_hours : float
            Time difference in hours (peak_time - nearest_high_tide).
    """

    peak_time = pd.Timestamp(peak_time)
    high_tide_times = pd.to_datetime(high_tide_times)

    nearest_idx = np.argmin(np.abs(high_tide_times - peak_time))
    nearest_high_tide = high_tide_times[nearest_idx]

    time_diff_hours = (peak_time - nearest_high_tide).total_seconds() / 3600

    return nearest_high_tide, time_diff_hours


def plot_time_diff_histogram(time_differences: List[float], bins: int = 30) -> None:
    """
    Plot histogram of time differences between peaks and high tides.

    Parameters
    ----------
    time_differences : list of float
        Time differences in hours.
    bins : int, optional
        Number of histogram bins. Default is 30.
    """

    figsize = (8, 4)
    dpi = 600
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.hist(
        time_differences,
        bins=bins,
        color="#C28B3C",
        edgecolor="#C28B3C",
        alpha=0.7,
        density=True,
        zorder=2,
    )
    sns.kdeplot(time_differences, color="#996515", linewidth=1, ax=ax)

    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax.set_ylim([0, 0.3])

    plt.show()
