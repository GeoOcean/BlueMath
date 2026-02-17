from collections import Counter
from typing import Any, Callable, Dict, List, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist as dist
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler


def plot_stations(
    lat: List[float], lon: List[float], coast_extent: Tuple[float, float, float, float]
) -> None:
    """
    Plot station locations on a map with coastlines and borders.

    Parameters
    ----------
    lat : array-like
        Latitude coordinates of stations.
    lon : array-like
        Longitude coordinates of stations.
    coast_extent : tuple
        Map extent as (lon_min, lon_max, lat_min, lat_max).
    """

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(8, 6)
    )

    # Add coastlines and borders
    ax.add_feature(
        cfeature.COASTLINE.with_scale("10m"), edgecolor="white", linewidth=0.5, zorder=3
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("10m"),
        linestyle="-",
        linewidth=1,
        edgecolor="white",
        zorder=4,
    )
    ax.add_feature(
        cfeature.LAND.with_scale("10m"), facecolor="#f2e2a0", linewidth=0.5, zorder=3
    )

    ax.set_extent(coast_extent, crs=ccrs.PlateCarree())

    # Plot stations location
    for i, (lat_val, lon_val) in enumerate(zip(lat, lon)):
        ax.text(
            lon_val,
            lat_val,
            str(i + 1),
            transform=ccrs.PlateCarree(),
            color="black",
            fontsize=12,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=5,
            path_effects=[PathEffects.withStroke(linewidth=2, foreground="white")],
        )

    plt.axis("off")
    plt.show()


def normalize_per_duration(series: pd.Series) -> pd.Series:
    """
    Normalize a time series by its duration.

    Parameters
    ----------
    series : pd.Series
        Time series with datetime index to normalize.

    Returns
    -------
    pd.Series
        Normalized series with index scaled to [0, 1].
    """

    dur = (series.index[-1] - series.index[0]).total_seconds() / 3600
    normalized_index = (series.index - series.index[0]).total_seconds() / (dur * 3600)
    series.index = normalized_index

    return series


def get_max_resample_interval(
    extremes_list: List[pd.DataFrame], NTR_list: List[pd.Series]
) -> Tuple[float, pd.Series, int, float, float]:
    """
    Determine the maximum time interval used for re-sampling storm surge hydrographs.

    Parameters
    ----------
    extremes_list : list of pd.DataFrame
        Each dataframe contains event information with columns ['max', 'duration',
        'start', 'end'], and datetime index as event peaks.
    NTR_list : list of pd.Series
        List of non-tidal residual time series (datetime-indexed).

    Returns
    -------
    tuple
        max_interval : float
            Maximum ∆t * D value across all events.
        timeseries_max_interval : pd.Series
            Corresponding NTR time series for the max interval event.
        index_max_interval : int
            Index of the station with the max interval event.
        dt_max_interval : float
            Sampling interval of the event with the max ∆t * D.
        duration_max_interval : float
            Duration (in hours) of the event with the max ∆t * D.
    """

    max_interval = 0
    timeseries_max_interval = None

    for j, extremes in enumerate(extremes_list):
        extremes = extremes.copy()
        extremes["t_peak"] = extremes.index
        extremes.reset_index(drop=True, inplace=True)

        max_interval_element = 0

        for i in range(len(extremes)):
            timeseries = NTR_list[j][
                (NTR_list[j].index >= extremes["start"][i])
                & (NTR_list[j].index <= extremes["end"][i])
            ]
            if len(timeseries) < 2:
                continue

            dt = extremes["duration"][i] / (len(timeseries) - 1)
            interval_value = dt * extremes["duration"][i]

            if interval_value > max_interval_element:
                max_interval_element = interval_value
                timeseries_max_element = timeseries
                index_max_element = j
                dt_max_element = dt
                duration_max_element = extremes["duration"][i]

        if max_interval_element > max_interval:
            max_interval = max_interval_element
            timeseries_max_interval = timeseries_max_element
            index_max_interval = index_max_element
            dt_max_interval = dt_max_element
            duration_max_interval = duration_max_element

    return (
        max_interval,
        timeseries_max_interval,
        index_max_interval,
        dt_max_interval,
        duration_max_interval,
    )


def extract_hydrographs(
    extremes_list: List[pd.DataFrame],
    NTR_list: List[pd.Series],
    max_interval: float,
    normalize_func: Callable[[pd.Series], pd.Series],
) -> List[np.ndarray]:
    """
    Extract and resample normalized hydrographs from non-tidal residual (NTR) data.

    Parameters
    ----------
    extremes_list : list of pd.DataFrame
        List of dataframes containing event info with 'start' and 'end' columns.
    NTR_list : list of pd.Series
        List of non-tidal residual time series.
    max_interval : float
        Maximum interval value for resampling.
    normalize_func : callable
        Function to normalize a time series per event duration.

    Returns
    -------
    list of np.ndarray
        List where each element is a 2D array (events x resampled time steps).
    """

    hydrographs_list = []
    max_dt = 1 / max_interval
    new_index = np.arange(0, 1 + max_dt, max_dt)

    for j, extremes in enumerate(extremes_list):
        extremes = extremes.copy()
        extremes["t_peak"] = extremes.index
        extremes.reset_index(drop=True, inplace=True)

        regridded_NTR = []

        for i in range(len(extremes)):
            timeseries = NTR_list[j][
                (NTR_list[j].index >= extremes["start"][i])
                & (NTR_list[j].index <= extremes["end"][i])
            ]

            normalized = normalize_func(timeseries)
            index_joined = normalized.index.join(new_index, how="outer")
            regridded_temp = (
                normalized.reindex(index_joined)
                .interpolate(method="slinear")
                .reindex(new_index)
            )
            regridded_NTR.append(regridded_temp)

        matrix_NTR = np.array(regridded_NTR)
        matrix_NTR = matrix_NTR[:, :-1]
        hydrographs_list.append(matrix_NTR)

    return hydrographs_list


def prepare_weighted_features(
    extremes_list: List[pd.DataFrame],
    hydrographs_list: List[np.ndarray],
    ntr_weight: float = 0.5,
    duration_weight: float = 0.5,
) -> np.ndarray:
    """
    Concatenate and standardize hydrographs and durations, apply weighting,
    and combine into a single feature matrix.

    Parameters
    ----------
    extremes_list : list of pd.DataFrame
        List of DataFrames containing extreme events with 'duration' in column index 1.
    hydrographs_list : list of np.ndarray
        List of 2D arrays with interpolated hydrographs.
    ntr_weight : float, optional
        Proportion of importance given to NTR features. Default is 0.5.
    duration_weight : float, optional
        Proportion of importance given to duration. Default is 0.5.

    Returns
    -------
    np.ndarray
        Combined weighted feature matrix for clustering.
    """

    # Concatenate all events and durations
    array_extremes = np.concatenate([df.values for df in extremes_list])
    durations = array_extremes[:, 1]

    array_NTR = np.concatenate(hydrographs_list)

    # Standardize separately
    scaler = StandardScaler()
    scaled_NTR = scaler.fit_transform(array_NTR)
    scaled_duration = scaler.fit_transform(durations.reshape(-1, 1))

    # Apply weighting
    ntr_weight_factor = np.sqrt(ntr_weight / scaled_NTR.shape[1])
    duration_weight_factor = np.sqrt(duration_weight)

    weighted_NTR = scaled_NTR * ntr_weight_factor
    weighted_duration = scaled_duration * duration_weight_factor

    # Combine into single feature matrix
    features = np.column_stack((weighted_NTR, weighted_duration))

    return features


def plot_pca_variance(
    pca: Any, n_components: int = 62, figsize: Tuple[int, int] = (10, 8), dpi: int = 600
) -> None:
    """
    Plot cumulative explained variance from a fitted PCA object.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        Fitted sklearn PCA object.
    n_components : int, optional
        Number of components to display on x-axis. Default is 62.
    figsize : tuple, optional
        Figure size. Default is (10, 8).
    dpi : int, optional
        Dots per inch (image resolution). Default is 600.
    """

    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(
        range(n_components),
        pca.explained_variance_ratio_.cumsum(),
        marker="o",
        linestyle="--",
        color="#0076C2",
    )
    plt.xlabel("Number of components", fontsize=12)
    plt.ylabel("Cumulative explained variance", fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.title("PCA Explained Variance", fontsize=14)
    plt.tight_layout()
    plt.show()


def extract_chosen_events(
    md_reduced_index: List[int],
    extremes_list: List[pd.DataFrame],
    hydrographs_list: List[np.ndarray],
    station_names: List[str],
) -> List[List[Union[str, Any]]]:
    """
    Extract selected storm surge event metadata based on indices from dimensionality reduction.

    Parameters
    ----------
    md_reduced_index : iterable
        Indices of (MDA) selected events.
    extremes_list : list
        List of DataFrames containing extreme event data per station.
    hydrographs_list : list
        List of hydrograph series per station.
    station_names : list
        List of station names corresponding to extremes_list and hydrographs_list.

    Returns
    -------
    list
        List of [station_name, start_date, duration, magnitude] for selected events.
    """

    # Flatten event data
    array_extremes = np.concatenate([df.values for df in extremes_list])
    start = array_extremes[:, 2]
    duration = array_extremes[:, 1]
    magnitude = array_extremes[:, 0]

    # Generate corresponding station names
    stations = []
    for name, hydro in zip(station_names, hydrographs_list):
        stations.extend([name] * hydro.shape[0])
    stations = np.array(stations)

    # Collect selected events
    chosen_events = []
    for idx in md_reduced_index:
        row = [stations[idx], start[idx], duration[idx], magnitude[idx]]
        chosen_events.append(row)

    return chosen_events


def count_events_per_station(
    chosen_events: List[List[Union[str, Any]]],
) -> Dict[str, int]:
    """
    Count the number of selected storm surge events per station.

    Parameters
    ----------
    chosen_events : list
        List of [station_name, start_date, duration, magnitude] for selected events.

    Returns
    -------
    dict
        Dictionary with station names as keys and event counts as values.
    """

    # Extract station names from chosen events
    station_names = [event[0] for event in chosen_events]

    # Count occurrences of each station name
    station_event_counts = Counter(station_names)

    return dict(station_event_counts)


def plot_magnitude_duration_violin(
    chosen_events: List[List[Union[str, Any]]],
    station_names: List[str],
    figsize_mm: Tuple[float, float] = (174, 234 / 3),
    dpi: int = 300,
) -> None:
    """
    Plot violin plots for magnitude and duration (in days) of storm surge events per station.

    Parameters
    ----------
    chosen_events : list of list
        List of [station, start_date, duration (hrs), magnitude] for selected events.
    station_names : list of str
        List of all station names (in desired order).
    figsize_mm : tuple of float, optional
        Figure size in millimeters (width, height). Default is (174, 234/3).
    dpi : int, optional
        Figure resolution in dots per inch. Default is 300.
    """

    # Prepare DataFrame
    df = pd.DataFrame(
        chosen_events, columns=["Station", "Start", "Duration", "Magnitude"]
    )
    df["Duration_Days"] = df["Duration"] / 24
    df["Order"] = df["Station"].map({name: i for i, name in enumerate(station_names)})
    df_sorted = df.sort_values(by="Order")

    # Convert mm to inches
    figsize_in = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)

    # Create figure and grid spec
    fig = plt.figure(figsize=figsize_in, dpi=dpi)
    gs = GridSpec(1, 2)

    # === Magnitude Plot ===
    ax1 = fig.add_subplot(gs[:, 0])
    sns.violinplot(
        data=df_sorted,
        x="Magnitude",
        y="Station",
        order=station_names,
        ax=ax1,
        palette="husl",
        inner="point",
        edgecolor="None",
        alpha=0.8,
        inner_kws=dict(color="0.1", s=1),
        zorder=3,
    )
    ax1.set_xlabel("Magnitude [m]", fontsize=8, fontweight="bold")
    ax1.set_ylabel("")
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.set_yticklabels([])
    ax1.tick_params(axis="y", length=0)
    ax1.set_xlim(-0.5, 4)
    ax1.grid(
        True, which="both", axis="both", color="lightgrey", linestyle="-", linewidth=0.5
    )
    for spine in ax1.spines.values():
        spine.set_color("grey")
        spine.set_linewidth(0.7)
    for label in ax1.get_xticklabels():
        label.set_fontsize(8)
    ax1.text(
        0.02,
        0.98,
        "a)",
        transform=ax1.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="left",
    )

    # === Duration Plot ===
    ax2 = fig.add_subplot(gs[:, 1])
    sns.violinplot(
        data=df_sorted,
        x="Duration_Days",
        y="Station",
        order=station_names,
        ax=ax2,
        palette="husl",
        inner="point",
        edgecolor="None",
        alpha=0.8,
        inner_kws=dict(color="0.1", s=1),
        zorder=3,
    )
    ax2.set_xlabel("Duration [days]", fontsize=8, fontweight="bold")
    ax2.set_ylabel("")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.grid(
        True, which="both", axis="both", color="lightgrey", linestyle="-", linewidth=0.5
    )
    for spine in ax2.spines.values():
        spine.set_color("grey")
        spine.set_linewidth(0.7)
    for label in ax2.get_xticklabels():
        label.set_fontsize(8)
    for label in ax2.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(8)
    ax2.text(
        0.02,
        0.98,
        "b)",
        transform=ax2.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="left",
    )

    # Adjust position of second axis
    pos2 = ax2.get_position()
    ax2.set_position([pos2.x0 - 0.05, pos2.y0, pos2.width, pos2.height])

    plt.tight_layout()
    plt.show()


def sort_cluster_gen_corr_end(
    centers: np.ndarray, dimdim: int
) -> Tuple[np.ndarray, int, int, float]:
    """
    Organize dimdim centroids as in SOM (Self-Organizing Map).

    For dim=10, a matrix 10x10 has 100 centroids. Returns a matrix sc dimxdim
    with the ordered index and a quality index qx, better the lesser.

    Parameters
    ----------
    centers : np.ndarray
        The dimxdim centroids with dimension (dimxdim, d).
    dimdim : int
        Number of clusters (must be rectangular, e.g., 20=4x5, 12=3x4).

    Returns
    -------
    tuple
        sc : np.ndarray
            Ordered cluster matrix.
        dimy : int
            Number of rows in the matrix.
        dimx : int
            Number of columns in the matrix.
        qx : float
            Quality index (lower is better).

    Notes
    -----
    Based on sort_cluster_gen_corr_end.m function created by rancellt@aemet.es,
    Rafael Ancell, 22-10-2010. Modified by antonio.tomas@unican.es, 05-11-2010,
    canovasv@unican.es, 22-12-2010, and aantolinezja@unican.es 03-06-2014.

    Raises
    ------
    ValueError
        If the number of clusters is not rectangular, e.g., 20=4x5, 12=3x4.
    """

    # 2D plots
    if np.sqrt(dimdim) % 1 > 0:
        dimy = int(np.floor(np.sqrt(dimdim)))
        dimx = int(np.ceil(np.sqrt(dimdim)))
    else:
        dimy = int(np.sqrt(dimdim))
        dimx = int(np.sqrt(dimdim))

    print(dimx)
    if dimy * dimx != dimdim:
        raise ValueError("Number of cluster must be rectangular, 20=4x5, 12=3x4.")

    dd = squareform(dist(centers, metric="euclidean"), force="tomatrix")
    qx = 0

    sc = np.reshape(np.random.permutation(dimdim), (dimy, dimx))

    for i in range(dimy):
        for j in range(dimx):
            # FILA F-1
            if (i - 1) >= 0:
                qx = qx + dd[sc[i - 1, j], sc[i, j]]
                if (j - 1) >= 0:
                    qx = qx + dd[sc[i - 1, j - 1], sc[i, j]]
                if (j + 1) < dimx:
                    qx = qx + dd[sc[i - 1, j + 1], sc[i, j]]
            # FILA F
            if (j - 1) >= 0:
                qx = qx + dd[sc[i, j - 1], sc[i, j]]
            if (j + 1) < dimx:
                qx = qx + dd[sc[i, j + 1], sc[i, j]]
            # FILA F+1
            if (i + 1) < dimy:
                qx = qx + dd[sc[i + 1, j], sc[i, j]]
                if (j - 1) >= 0:
                    qx = qx + dd[sc[i + 1, j - 1], sc[i, j]]
                if (j + 1) < dimx:
                    qx = qx + dd[sc[i + 1, j + 1], sc[i, j]]

    q = np.inf
    sigue = 1
    # Test all possible permutations of 3 nodes [i j k]
    for i in range(dimdim):
        if sigue == 0:
            break
        sigue = 0
        for j in range(dimdim):
            for k in range(dimdim):
                if len(np.unique([i, j, k])) == 3:
                    u = sc.copy()
                    u = np.reshape(u.T, (1, dimdim))
                    u[0, i] = sc.T.flat[j]
                    u[0, j] = sc.T.flat[k]
                    u[0, k] = sc.T.flat[i]
                    u = np.reshape(u, (dimy, dimx)).T
                    f = 0
                    for ix in range(dimy):
                        for jx in range(dimx):
                            print(jx)

                            # FILA F-1
                            if (ix - 1) >= 0:
                                f = f + dd[u[ix - 1, jx], u[ix, jx]]
                                if (jx - 1) >= 0:
                                    f = f + dd[u[ix - 1, jx - 1], u[ix, jx]]
                                if (jx + 1) < dimx:
                                    f = f + dd[u[ix - 1, jx + 1], u[ix, jx]]
                            # FILA F
                            if (jx - 1) >= 0:
                                f = f + dd[u[ix, jx - 1], u[ix, jx]]
                            if (jx + 1) < dimx:
                                f = f + dd[u[ix, jx + 1], u[ix, jx]]
                            # FILA F+1
                            if (ix + 1) < dimy:
                                f = f + dd[u[ix + 1, jx], u[ix, jx]]
                                if (jx - 1) >= 0:
                                    f = f + dd[u[ix + 1, jx - 1], u[ix, jx]]
                                if (jx + 1) < dimx:
                                    f = f + dd[u[ix + 1, jx + 1], u[ix, jx]]
                    if f <= q:
                        q = f
                        sc = u
                        if q < qx:
                            qx = q
                            sigue = 1

    return sc, dimy, dimx, qx


def merge_kmedians_with_optics(
    kmed_labels: List[int], optics_labels: List[int]
) -> List[List[int]]:
    """
    Merge K-medians cluster labels based on OPTICS clustering results.

    Parameters
    ----------
    kmed_labels : list
        List of K-medians cluster assignments (e.g., [0, 1, 1, 2, ...]).
    optics_labels : list
        Corresponding OPTICS cluster labels (e.g., [-1, 0, 0, 1, ...]).

    Returns
    -------
    list
        List of cluster indices grouped by OPTICS label
    """

    # Initialize a dictionary to store merged clusters
    merged_clusters = {}
    unchanged_clusters = []

    # Iterate over optics_labels and merge clusters with the same label (excluding -1)
    for i, label in enumerate(optics_labels):
        if label == -1:
            unchanged_clusters.append(kmed_labels[i])
        else:
            if label not in merged_clusters:
                merged_clusters[label] = kmed_labels[i]
            else:
                merged_clusters[label].extend(kmed_labels[i])

    # Remove duplicates in each cluster after merging
    for key in merged_clusters:
        merged_clusters[key] = list(set(merged_clusters[key]))

    # Combine merged clusters and unchanged clusters
    merged_cluster_labels = [merged_clusters[key] for key in sorted(merged_clusters)]
    merged_cluster_labels.extend(unchanged_clusters)

    return merged_cluster_labels


def plot_clusters_NTR_only(
    labels: List[List[int]], hydrographs_list: List[np.ndarray], md_reduced: Any
) -> None:
    """
    Plot clustered NTR hydrographs with centroids.

    Parameters
    ----------
    labels : list
        Cluster labels for each event.
    hydrographs_list : list of np.ndarray
        List of hydrograph arrays per station.
    md_reduced : object
        Dimensionality reduction object with index attribute.
    """

    # Sort clusters by size
    chosen_labels = labels.copy()
    chosen_labels.sort(key=len, reverse=True)

    array_NTR = np.concatenate(hydrographs_list)

    # Calculate centroids
    centroids = []
    for cluster_indices in chosen_labels:
        mapped_indices = [md_reduced.index[index] for index in cluster_indices]
        cluster_data = array_NTR[mapped_indices]
        centroid = np.median(cluster_data, axis=0)
        centroids.append(centroid)

    # Set up figure
    fig, axs = plt.subplots(8, 7, figsize=(174 / 25.4, 234 / 25.4), dpi=600)

    for i, cluster_indices in enumerate(chosen_labels):
        row, col = divmod(i, 7)

        ax = axs[row, col]
        for index in cluster_indices:
            array_index = md_reduced.index[index]
            ax.plot(array_NTR[array_index], alpha=0.7, linewidth=0.5, color="#C49FE7")

        ax.plot(centroids[i], linewidth=1, color="#6F1D77", label="Centroid")

        ax.set_ylim(-2, 4)
        ax.set_yticks([-1, 1, 3])
        ax.tick_params(axis="y", labelsize=8, colors="#6F1D77")
        ax.set_xticks([0, 500, 1000])
        ax.tick_params(axis="x", labelsize=8, colors="black")

        # Add title and likelihood annotation
        likelihood = len(cluster_indices) / 972
        ax.set_title(f"Type {i + 1}", fontsize=10, fontweight="bold", pad=13)
        ax.text(
            0.5,
            1.02,
            f"{likelihood:.2%}",
            transform=ax.transAxes,
            fontsize=8,
            color="black",
            ha="center",
            va="bottom",
        )

        # Hide y-axis labels if not first column
        if col != 0:
            ax.get_yaxis().set_visible(False)
        # Hide x-axis labels if not last row
        if row != 7:
            ax.get_xaxis().set_visible(False)

        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
            spine.set_linewidth(1)
            spine.set_alpha(0.3)

    # Add shared labels
    fig.text(
        0.06,
        0.5,
        "Non-tidal residual [m]",
        va="center",
        rotation="vertical",
        fontsize=10,
        color="#6F1D77",
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.088,
        "Re-sampled time step",
        va="center",
        ha="center",
        fontsize=10,
        color="black",
        fontweight="bold",
    )

    plt.subplots_adjust(hspace=0.7)
    plt.show()
