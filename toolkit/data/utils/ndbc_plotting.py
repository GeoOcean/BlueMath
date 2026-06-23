import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set font sizes
TITLE_SIZE = 20
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 12
TEXT_SIZE = 14


def get_season(month: int) -> str:
    """
    Return season based on month.

    Parameters
    ----------
    month : int
        Month number (1-12)

    Returns
    -------
    str
        Season name (DJF, MAM, JJA, or SON)
    """

    return {
        12: "DJF",
        1: "DJF",
        2: "DJF",
        3: "MAM",
        4: "MAM",
        5: "MAM",
        6: "JJA",
        7: "JJA",
        8: "JJA",
        9: "SON",
        10: "SON",
        11: "SON",
    }[month]


def calculate_directional_spectrum(
    C11: np.ndarray,
    freq: np.ndarray,
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate normalized directional wave spectrum.

    Parameters
    ----------
    C11 : np.ndarray
        Wave energy spectrum
    freq : np.ndarray
        Frequency array
    alpha1 : np.ndarray
        Primary direction array
    alpha2 : np.ndarray
        Secondary direction array
    r1 : np.ndarray
        Primary spreading parameter array
    r2 : np.ndarray
        Secondary spreading parameter array

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing (E, freq_mesh, angle_mesh)
    """

    angles = np.linspace(0, 2 * np.pi, 360)
    angle_mesh, freq_mesh = np.meshgrid(angles, freq)
    alpha1_rad = np.deg2rad(alpha1)
    alpha2_rad = np.deg2rad(alpha2)
    r1 = np.array(r1) / 100
    r2 = np.array(r2) / 100
    E = np.zeros((len(freq), len(angles)))

    for i in range(len(freq)):
        D = (1 / np.pi) * (
            0.5
            + r1[i] * np.cos(angles - alpha1_rad[i])
            + r2[i] * np.cos(2 * (angles - alpha2_rad[i]))
        )
        D = D / np.trapz(D, angles)
        D[D < 0] = 0
        E[i, :] = C11[i] * D

    return E.T, freq_mesh.T, angle_mesh.T


def plot_bulk_timeseries(df: pd.DataFrame) -> plt.Figure:
    """
    Create an enhanced time series plot of wave parameters with three subplots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing wave parameters with the following columns:
        - datetime: datetime index
        - WVHT: Wave height (m)
        - DPD: Dominant wave period (s)
        - APD: Average wave period (s)
        - MWD: Mean wave direction (degrees)

    Returns
    -------
    plt.Figure
        Figure object containing the plot.
    """

    colors = ["plum"]

    # Create datetime column if not exists
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(
            df["YYYY"].astype(str)
            + "-"
            + df["MM"].astype(str).str.zfill(2)
            + "-"
            + df["DD"].astype(str).str.zfill(2)
            + " "
            + df["hh"].astype(str).str.zfill(2)
            + ":"
            + df["mm"].astype(str).str.zfill(2),
            format="%Y-%m-%d %H:%M",
        )

    for col in ["WVHT", "DPD", "APD", "MWD"]:
        # Replace missing value codes with NaN
        df[col] = df[col].replace([99.0, 999.0], np.nan)

        # Wave height and periods should not be 0
        if col in ["WVHT", "DPD", "APD"]:
            df[col] = df[col].where(df[col] > 0, np.nan)

        # Periods should not be greater than 30 seconds
        if col in ["DPD", "APD"]:
            df[col] = df[col].where(df[col] <= 30, np.nan)

    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

    # Plot 1: Wave Height
    valid_wvht = ~pd.isna(df["WVHT"])
    ax1.plot(
        df.loc[valid_wvht, "datetime"],
        df.loc[valid_wvht, "WVHT"],
        color=colors[0],
        label="Wave Height",
    )
    ax1.set_ylabel("Wave Height (m)", fontsize=AXIS_LABEL_SIZE)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    # Plot 2: Wave Periods
    valid_apd = ~pd.isna(df["APD"])
    ax2.plot(
        df.loc[valid_apd, "datetime"],
        df.loc[valid_apd, "APD"],
        color=colors[0],
        markersize=1,
        label="Average Period",
    )
    ax2.set_ylabel("Wave Period (s)", fontsize=AXIS_LABEL_SIZE)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    # Plot 3: Wave Direction
    valid_direction = ~pd.isna(df["MWD"])
    ax3.plot(
        df.loc[valid_direction, "datetime"],
        df.loc[valid_direction, "MWD"],
        ".",
        color=colors[0],
        markersize=1,
        label="Mean Wave Direction",
    )
    ax3.set_ylabel("Wave Direction (°)", fontsize=AXIS_LABEL_SIZE)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    # Set y-axis limits
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax3.set_ylim(0, 360)

    # Align x-axes
    date_min = df["datetime"].min()
    date_max = df["datetime"].max()
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(date_min, date_max)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Add statistics text box
    stats_textHs = (
        f"Mean Hs: {df['WVHT'].mean():.2f} m\n"
        f"Max Hs: {df['WVHT'].max():.2f} m\n"
        f"Min Hs: {df['WVHT'].min():.2f} m\n"
    )
    stats_textTm = (
        f"Mean Tm: {df['APD'].mean():.2f} s\n"
        f"Max Tm: {df['APD'].max():.2f} s\n"
        f"Min Tm: {df['APD'].min():.2f} s\n"
    )
    stats_textDm = f"Mean Dirm: {df['APD'].mean():.2f} °\n"

    ax1.text(
        0.02,
        0.98,
        stats_textHs,
        transform=ax1.transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        verticalalignment="top",
        fontsize=TEXT_SIZE,
    )
    ax2.text(
        0.02,
        0.98,
        stats_textTm,
        transform=ax2.transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        verticalalignment="top",
        fontsize=TEXT_SIZE,
    )
    ax3.text(
        0.02,
        0.98,
        stats_textDm,
        transform=ax3.transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        verticalalignment="top",
        fontsize=TEXT_SIZE,
    )

    plt.tight_layout()
    return fig


def _prepare_wave_spectra_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw NDBC wave spectra data for plotting.

    Converts CSV-style date columns (YYYY/MM/DD/hh or #YY variant) into a
    datetime index and keeps only numeric frequency columns.
    """
    df = df.copy()

    if {"YYYY", "MM", "DD", "hh"}.issubset(df.columns):
        date_cols = ["YYYY", "MM", "DD", "hh"]
        rename_map = {"YYYY": "year", "MM": "month", "DD": "day", "hh": "hour"}
        if "mm" in df.columns:
            date_cols.append("mm")
            rename_map["mm"] = "minute"
        df["datetime"] = pd.to_datetime(df[date_cols].rename(columns=rename_map))
        df = df.drop(columns=date_cols)
    elif {"#YY", "MM", "DD", "hh"}.issubset(df.columns):
        date_cols = ["#YY", "MM", "DD", "hh"]
        rename_map = {"#YY": "year", "MM": "month", "DD": "day", "hh": "hour"}
        if "mm" in df.columns:
            date_cols.append("mm")
            rename_map["mm"] = "minute"
        df["datetime"] = pd.to_datetime(df[date_cols].rename(columns=rename_map))
        df = df.drop(columns=date_cols)

    if "datetime" in df.columns:
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    freq_cols = df.columns[pd.to_numeric(df.columns, errors="coerce").notna()]
    return df[freq_cols]


def plot_yearly_averages(df: pd.DataFrame) -> plt.Figure:
    """
    Plot yearly wave spectra averages with overall average.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing wave spectra data with:
        - datetime index
        - columns: frequency values (Hz)
        - values: spectral density (m²/Hz)

    Returns
    -------
    plt.Figure
        Figure object containing the plot.
    """

    df = _prepare_wave_spectra_df(df)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a common frequency grid (0.01 to 0.5 Hz with 100 points)
    common_freqs = np.linspace(0.01, 0.5, 100)

    # Convert column names to numeric frequencies
    orig_freqs = pd.to_numeric(df.columns)

    # Calculate yearly average
    yearly_avg = df.groupby(df.index.year).mean()

    # Plot each year's average
    for year, avg in yearly_avg.iterrows():
        # Interpolate to common frequency grid
        interp_avg = np.interp(common_freqs, orig_freqs, avg.values, left=0, right=0)
        ax.plot(common_freqs, interp_avg, color="lightblue", alpha=0.4, linewidth=2)

    # Calculate and plot overall average
    overall_avg = df.mean()
    interp_overall = np.interp(
        common_freqs, orig_freqs, overall_avg.values, left=0, right=0
    )
    ax.plot(
        common_freqs,
        interp_overall,
        color="navy",
        linewidth=3,
        label=f"Overall Average ({len(yearly_avg)} years)",
    )

    # Customize plot
    year_range = f"{df.index.year.min()}-{df.index.year.max()}"
    ax.set_title(f"Yearly Wave Spectra ({year_range})", fontsize=TITLE_SIZE)
    ax.set_xlabel("Frequency (Hz)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Spectral Density (m²/Hz)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()
    return fig


def plot_seasonal_averages(df: pd.DataFrame) -> plt.Figure:
    """
    Plot seasonal averages of wave spectra.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing wave spectra data with:
        - datetime index
        - columns: frequency values (Hz)
        - values: spectral density (m²/Hz)

    Returns
    -------
    plt.Figure
        Figure object containing the plot.
    """

    df = _prepare_wave_spectra_df(df)

    # Create subplots for each season
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    seasons = {
        "DJF": {"months": [12, 1, 2], "color": "blue", "ax": axes[0]},
        "MAM": {"months": [3, 4, 5], "color": "green", "ax": axes[1]},
        "JJA": {"months": [6, 7, 8], "color": "red", "ax": axes[2]},
        "SON": {"months": [9, 10, 11], "color": "orange", "ax": axes[3]},
    }

    # Create a common frequency grid
    common_freqs = np.linspace(0.01, 0.5, 100)
    orig_freqs = pd.to_numeric(df.columns)

    # Process each season
    for season, info in seasons.items():
        season_data = df[df.index.month.isin(info["months"])]
        if not season_data.empty:
            # Calculate seasonal average
            avg = season_data.mean()

            # Interpolate to common frequency grid
            interp_avg = np.interp(
                common_freqs, orig_freqs, avg.values, left=0, right=0
            )

            # Plot seasonal average
            info["ax"].plot(
                common_freqs,
                interp_avg,
                color=info["color"],
                linewidth=3,
                label=f"Average ({len(season_data)} points)",
            )

            # Customize plot
            info["ax"].set_title(season, fontsize=TITLE_SIZE)
            info["ax"].set_xlabel("Frequency (Hz)", fontsize=AXIS_LABEL_SIZE)
            info["ax"].set_ylabel("Spectral Density (m²/Hz)", fontsize=AXIS_LABEL_SIZE)
            info["ax"].grid(True, alpha=0.2)
            info["ax"].legend(fontsize=LEGEND_SIZE)
            info["ax"].tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    year_range = f"{df.index.year.min()}-{df.index.year.max()}"
    fig.suptitle(
        f"Seasonal Wave Spectra ({year_range})",
        fontsize=TITLE_SIZE,
        y=1.02,
    )

    plt.tight_layout()
    return fig


def plot_monthly_averages(df: pd.DataFrame) -> plt.Figure:
    """
    Plot monthly wave spectra averages.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing wave spectra data with:
        - datetime index
        - columns: frequency values (Hz)
        - values: spectral density (m²/Hz)

    Returns
    -------
    plt.Figure
        Figure object containing the plot.
    """

    df = _prepare_wave_spectra_df(df)

    # Create a 4x3 grid for all 12 months
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.ravel()

    # Define month names and colors
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    colors = plt.cm.hsv(np.linspace(0, 1, 12))

    # Create a common frequency grid
    common_freqs = np.linspace(0.01, 0.5, 100)
    orig_freqs = pd.to_numeric(df.columns)

    # Process each month
    for month in range(1, 13):
        month_data = df[df.index.month == month]
        if not month_data.empty:
            # Calculate monthly average
            avg = month_data.mean()

            # Interpolate to common frequency grid
            interp_avg = np.interp(
                common_freqs, orig_freqs, avg.values, left=0, right=0
            )

            # Plot monthly average
            axes[month - 1].plot(
                common_freqs,
                interp_avg,
                color=colors[month - 1],
                linewidth=3,
                label=f"Average ({len(month_data)} points)",
            )

            # Customize plot
            axes[month - 1].set_title(month_names[month - 1], fontsize=TITLE_SIZE)
            axes[month - 1].set_xlabel("Frequency (Hz)", fontsize=AXIS_LABEL_SIZE)
            axes[month - 1].set_ylabel(
                "Spectral Density (m²/Hz)", fontsize=AXIS_LABEL_SIZE
            )
            axes[month - 1].grid(True, alpha=0.2)
            axes[month - 1].legend(fontsize=LEGEND_SIZE)
            axes[month - 1].tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    year_range = f"{df.index.year.min()}-{df.index.year.max()}"
    fig.suptitle(
        f"Monthly Wave Spectra Averages ({year_range})",
        y=1.02,
        fontsize=TITLE_SIZE,
    )

    plt.tight_layout()
    return fig


def plot_specific_date_directional_spectrum(
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    c11: np.ndarray,
    freqs: np.ndarray,
    buoy_id: str,
    date_str: str,
) -> None:
    """
    Create a single plot showing directional spectrum for a specific date.

    Parameters
    ----------
    alpha1 : np.ndarray
        Primary direction array
    alpha2 : np.ndarray
        Secondary direction array
    r1 : np.ndarray
        Primary spreading parameter array
    r2 : np.ndarray
        Secondary spreading parameter array
    c11 : np.ndarray
        Wave energy spectrum
    freqs : np.ndarray
        Frequency array
    buoy_id : str
        Buoy identifier for plot title
    date_str : str
        Date string for plot title
    """

    # Calculate directional spectrum
    E, freq_mesh, angle_mesh = calculate_directional_spectrum(
        c11, freqs, alpha1, alpha2, r1, r2
    )

    # Plot
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection="polar")
    pcm = ax.pcolormesh(angle_mesh, freq_mesh, E, cmap="magma")
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    plt.colorbar(pcm, label="Energy Density (m²/Hz/rad)")
    plt.title(f"Directional Spectrum - {buoy_id} at {date_str}")

    return plt.gcf()


def plot_monthly_directional_spectra(
    alpha1_df: pd.DataFrame,
    alpha2_df: pd.DataFrame,
    r1_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    c11_df: pd.DataFrame,
    buoy_id: str,
    start_date: str,
    end_date: str,
) -> None:
    """
    Create a single plot with 12 subplots showing monthly directional spectra.

    Parameters
    ----------
    alpha1_df : pd.DataFrame
        DataFrame containing primary direction data with:
        - datetime index
        - columns: frequency values (Hz)
    alpha2_df : pd.DataFrame
        DataFrame containing secondary direction data with:
        - datetime index
        - columns: frequency values (Hz)
    r1_df : pd.DataFrame
        DataFrame containing primary spreading parameter data with:
        - datetime index
        - columns: frequency values (Hz)
    r2_df : pd.DataFrame
        DataFrame containing secondary spreading parameter data with:
        - datetime index
        - columns: frequency values (Hz)
    c11_df : pd.DataFrame
        DataFrame containing wave energy spectrum data with:
        - datetime index
        - columns: frequency values (Hz)
    buoy_id : str
        Buoy identifier for plot title
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    """

    # Get frequencies from the columns
    freqs = np.array([float(col) for col in r1_df.columns])

    # Process monthly data
    plt.figure(figsize=(20, 15))
    for month in range(1, 13):
        try:
            # Get monthly averages
            r1_monthly = r1_df.groupby(r1_df.index.month).mean().loc[month].values
            r2_monthly = r2_df.groupby(r2_df.index.month).mean().loc[month].values
            alpha1_monthly = (
                alpha1_df.groupby(alpha1_df.index.month).mean().loc[month].values
            )
            alpha2_monthly = (
                alpha2_df.groupby(alpha2_df.index.month).mean().loc[month].values
            )
            c11_monthly = c11_df.groupby(c11_df.index.month).mean().loc[month].values

            # Calculate directional spectrum
            E, freq_mesh, angle_mesh = calculate_directional_spectrum(
                c11_monthly,
                freqs,
                alpha1_monthly,
                alpha2_monthly,
                r1_monthly,
                r2_monthly,
            )

            # Plot
            ax = plt.subplot(4, 3, month, projection="polar")
            _pcm = ax.pcolormesh(angle_mesh, freq_mesh, E, cmap="magma")
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location("N")
            plt.title(f"Month {month}")

        except KeyError:
            logging.warning(f"No data available for month {month}")
            continue

    plt.suptitle(
        f"Monthly Directional Spectra - {buoy_id} ({start_date} to {end_date})"
    )
    plt.tight_layout()

    return plt.gcf()


def plot_seasonal_directional_spectra(
    alpha1_df: pd.DataFrame,
    alpha2_df: pd.DataFrame,
    r1_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    c11_df: pd.DataFrame,
    buoy_id: str,
    start_date: str,
    end_date: str,
) -> None:
    """
    Create a single plot with 4 subplots showing seasonal directional spectra.

    Parameters
    ----------
    alpha1_df : pd.DataFrame
        DataFrame containing primary direction data with:
        - datetime index
        - columns: frequency values (Hz)
    alpha2_df : pd.DataFrame
        DataFrame containing secondary direction data with:
        - datetime index
        - columns: frequency values (Hz)
    r1_df : pd.DataFrame
        DataFrame containing primary spreading parameter data with:
        - datetime index
        - columns: frequency values (Hz)
    r2_df : pd.DataFrame
        DataFrame containing secondary spreading parameter data with:
        - datetime index
        - columns: frequency values (Hz)
    c11_df : pd.DataFrame
        DataFrame containing wave energy spectrum data with:
        - datetime index
        - columns: frequency values (Hz)
    buoy_id : str
        Buoy identifier for plot title
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    """

    # Get frequencies from the columns
    freqs = np.array([float(col) for col in r1_df.columns])

    # Process seasonal data
    plt.figure(figsize=(20, 15))
    seasons = ["DJF", "MAM", "JJA", "SON"]
    for i, season in enumerate(seasons, 1):
        try:
            # Get seasonal averages
            r1_seasonal = (
                r1_df.groupby(r1_df.index.month.map(get_season))
                .mean()
                .loc[season]
                .values
            )
            r2_seasonal = (
                r2_df.groupby(r2_df.index.month.map(get_season))
                .mean()
                .loc[season]
                .values
            )
            alpha1_seasonal = (
                alpha1_df.groupby(alpha1_df.index.month.map(get_season))
                .mean()
                .loc[season]
                .values
            )
            alpha2_seasonal = (
                alpha2_df.groupby(alpha2_df.index.month.map(get_season))
                .mean()
                .loc[season]
                .values
            )
            c11_seasonal = (
                c11_df.groupby(c11_df.index.month.map(get_season))
                .mean()
                .loc[season]
                .values
            )

            # Calculate directional spectrum
            E, freq_mesh, angle_mesh = calculate_directional_spectrum(
                c11_seasonal,
                freqs,
                alpha1_seasonal,
                alpha2_seasonal,
                r1_seasonal,
                r2_seasonal,
            )

            # Plot
            ax = plt.subplot(2, 2, i, projection="polar")
            _pcm = ax.pcolormesh(angle_mesh, freq_mesh, E, cmap="magma")
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location("N")
            plt.title(f"Season {season}")

        except KeyError:
            logging.warning(f"No data available for season {season}")
            continue

    plt.suptitle(
        f"Seasonal Directional Spectra - {buoy_id} ({start_date} to {end_date})"
    )
    plt.tight_layout()

    return plt.gcf()


def plot_average_directional_spectrum(
    alpha1_df: pd.DataFrame,
    alpha2_df: pd.DataFrame,
    r1_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    c11_df: pd.DataFrame,
    buoy_id: str,
    start_date: str,
    end_date: str,
) -> None:
    """
    Create a single plot showing annual directional spectrum.

    Parameters
    ----------
    alpha1_df : pd.DataFrame
        DataFrame containing primary direction data with:
        - datetime index
        - columns: frequency values (Hz)
    alpha2_df : pd.DataFrame
        DataFrame containing secondary direction data with:
        - datetime index
        - columns: frequency values (Hz)
    r1_df : pd.DataFrame
        DataFrame containing primary spreading parameter data with:
        - datetime index
        - columns: frequency values (Hz)
    r2_df : pd.DataFrame
        DataFrame containing secondary spreading parameter data with:
        - datetime index
        - columns: frequency values (Hz)
    c11_df : pd.DataFrame
        DataFrame containing wave energy spectrum data with:
        - datetime index
        - columns: frequency values (Hz)
    buoy_id : str
        Buoy identifier for plot title
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    """

    # Get frequencies from the columns
    freqs = np.array([float(col) for col in r1_df.columns])

    # Process annual data
    r1_annual = r1_df.mean().values
    r2_annual = r2_df.mean().values
    alpha1_annual = alpha1_df.mean().values
    alpha2_annual = alpha2_df.mean().values
    c11_annual = c11_df.mean().values

    E, freq_mesh, angle_mesh = calculate_directional_spectrum(
        c11_annual, freqs, alpha1_annual, alpha2_annual, r1_annual, r2_annual
    )

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection="polar")
    pcm = ax.pcolormesh(angle_mesh, freq_mesh, E, cmap="magma")
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    plt.colorbar(pcm, label="Energy Density (m²/Hz/rad)")
    plt.title(f"Average Directional Spectrum - {buoy_id} ({start_date} to {end_date})")

    return plt.gcf()
