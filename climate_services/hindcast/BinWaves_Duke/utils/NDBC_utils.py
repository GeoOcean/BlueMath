import os
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 10
TEXT_SIZE = 10


def convert_buoy_csv_to_pkl(csv_path: str, output_pkl_path: str) -> None:
    """
    Convert buoy CSV data to PKL format matching the structure of buoy_41025_bulk_parameters.pkl.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file
    output_pkl_path : str
        Path where the PKL file will be saved

    Returns
    -------
    None
        Saves the converted data to a pickle file

    Notes
    -----
    The function processes NDBC buoy data with the following columns:
    - YYYY, MM, DD, hh, mm: Date and time components
    - WVHT: Wave height (m)
    - DPD: Dominant wave period (s)
    - APD: Average wave period (s)
    - MWD: Mean wave direction (degrees)

    The output DataFrame contains:
    - Hs_Buoy: Significant wave height (m)
    - Tm_Buoy: Mean wave period (s)
    - Tp_Buoy: Peak wave period (s)
    - Dir_Buoy: Mean wave direction (degrees)
    - Spr_Buoy: Wave spreading (NaN values)
    """

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create datetime column
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

    # Set datetime as index
    df.set_index("datetime", inplace=True)

    # Drop the individual date/time columns
    df.drop(["YYYY", "MM", "DD", "hh", "mm"], axis=1, inplace=True)

    # Handle missing values in the original columns
    for col in ["WVHT", "DPD", "APD", "MWD"]:
        # Replace all known missing value codes with NaN
        df[col] = df[col].replace([99.0, 999.0, 9999.0, 999], np.nan)

        # Wave height and periods should not be 0 or negative
        if col in ["WVHT", "DPD", "APD"]:
            df[col] = df[col].where(df[col] > 0, np.nan)

        # Periods should not be greater than 25 seconds (based on reference data)
        if col in ["DPD", "APD"]:
            df[col] = df[col].where(df[col] <= 25, np.nan)

        # Direction should be between 0 and 360
        if col == "MWD":
            df[col] = df[col].where((df[col] >= 0) & (df[col] <= 360), np.nan)

    # Sort index before resampling
    df = df.sort_index()

    # Resample to hourly frequency using proper NaN-aware averaging
    df = df.resample("1H").agg(
        {
            "WVHT": lambda x: pd.Series.mean(x, skipna=True),
            "DPD": lambda x: pd.Series.mean(x, skipna=True),
            "APD": lambda x: pd.Series.mean(x, skipna=True),
            "MWD": lambda x: pd.Series.mean(x, skipna=True),
        }
    )

    # Create the final DataFrame with the exact same structure as the reference
    final_df = pd.DataFrame(index=df.index)
    final_df["Hs_Buoy"] = df["WVHT"].astype("float64")
    final_df["Tm_Buoy"] = df["APD"].astype("float64")
    final_df["Tp_Buoy"] = df["DPD"].astype("float64")
    final_df["Dir_Buoy"] = df["MWD"].astype("float64")
    # TODO: Add Spr_Buoy or check if exists before adding nan
    final_df["Spr_Buoy"] = np.nan  # Match reference file which has all NaN values

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)

    # Save to pickle format
    final_df.to_pickle(output_pkl_path)

    print(f"\nSuccessfully converted {csv_path} to {output_pkl_path}")


def plot_bulk_timeseries(df: pd.DataFrame) -> plt.Figure:
    """
    Create an enhanced time series plot of wave parameters with three subplots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing wave parameters with the following columns:
        - datetime: datetime index (optional, will be created if not present)
        - YYYY, MM, DD, hh, mm: Date and time components (if datetime not present)
        - WVHT: Wave height (m)
        - DPD: Dominant wave period (s)
        - APD: Average wave period (s)
        - MWD: Mean wave direction (degrees)

    Returns
    -------
    plt.Figure
        Figure object containing the plot with three subplots:
        - Wave height time series
        - Wave period time series
        - Wave direction time series

    Notes
    -----
    The function automatically handles missing values by replacing common
    NDBC missing value codes (99.0, 999.0) with NaN. It also applies
    physical constraints to the data (positive wave heights/periods,
    periods <= 30s, directions 0-360°).
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


def calculate_directional_spectrum(
    C11: np.ndarray,
    freq: np.ndarray,
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    angles_deg: Optional[np.ndarray] = None,
    freq_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate normalized directional wave spectrum.

    Parameters
    ----------
    C11 : np.ndarray
        Wave energy spectrum (1D array)
    freq : np.ndarray
        Frequency array (1D array)
    alpha1 : np.ndarray
        Primary direction array in degrees (1D array)
    alpha2 : np.ndarray
        Secondary direction array in degrees (1D array)
    r1 : np.ndarray
        Primary spreading parameter array (1D array)
    r2 : np.ndarray
        Secondary spreading parameter array (1D array)
    angles_deg : np.ndarray, optional
        Array of directions in degrees (default: 360 evenly spaced from 0 to 360)
    freq_grid : np.ndarray, optional
        Array of frequencies for meshgrid (default: uses freq parameter)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing:
        - E: Directional spectrum (2D array, shape: n_directions x n_frequencies)
        - freq_mesh: Frequency meshgrid (2D array)
        - angle_mesh: Angle meshgrid in radians (2D array)

    Notes
    -----
    The directional spectrum is calculated using a Fourier series expansion
    with primary and secondary directional components. The spreading parameters
    r1 and r2 are normalized by dividing by 100. The resulting spectrum is
    normalized to ensure proper energy conservation.
    """

    if angles_deg is None:
        angles = np.linspace(0, 2 * np.pi, 360)
    else:
        angles = np.deg2rad(angles_deg)
    if freq_grid is None:
        freq_grid = freq

    angle_mesh, freq_mesh = np.meshgrid(angles, freq_grid)
    alpha1_rad = np.deg2rad(alpha1)
    alpha2_rad = np.deg2rad(alpha2)
    r1 = np.array(r1) / 100
    r2 = np.array(r2) / 100
    E = np.zeros((len(freq_grid), len(angles)))

    for i in range(len(freq_grid)):
        D = (1 / np.pi) * (
            0.5
            + r1[i] * np.cos(angles - alpha1_rad[i])
            + r2[i] * np.cos(2 * (angles - alpha2_rad[i]))
        )
        D = D / np.trapz(D, angles)
        D[D < 0] = 0
        E[i, :] = C11[i] * D

    return E.T, freq_mesh.T, angle_mesh.T


def save_full_spectrum_from_dataframes(
    alpha1_df: pd.DataFrame,
    alpha2_df: pd.DataFrame,
    r1_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    c11_df: pd.DataFrame,
    output_path: str,
    latitude: Union[float, np.ndarray],
    longitude: Union[float, np.ndarray],
    depth: Optional[Union[float, np.ndarray]] = None,
    station: Optional[str] = None,
    directions: Optional[np.ndarray] = None,
    frequencies: Optional[np.ndarray] = None,
) -> xr.Dataset:
    """
    Save full spectrum from dataframes to NetCDF format, with optional custom direction and frequency grid.

    Parameters
    ----------
    alpha1_df : pd.DataFrame
        DataFrame containing primary direction data (time x frequency)
    alpha2_df : pd.DataFrame
        DataFrame containing secondary direction data (time x frequency)
    r1_df : pd.DataFrame
        DataFrame containing primary spreading parameter data (time x frequency)
    r2_df : pd.DataFrame
        DataFrame containing secondary spreading parameter data (time x frequency)
    c11_df : pd.DataFrame
        DataFrame containing wave energy spectrum data (time x frequency)
    output_path : str
        Path where the NetCDF file will be saved
    latitude : float or np.ndarray
        Latitude coordinate(s)
    longitude : float or np.ndarray
        Longitude coordinate(s)
    depth : float or np.ndarray, optional
        Depth coordinate(s)
    station : str, optional
        Station identifier
    directions : np.ndarray, optional
        Custom direction array in degrees (default: 360 evenly spaced from 0 to 360)
    frequencies : np.ndarray, optional
        Custom frequency array (default: uses DataFrame columns)

    Returns
    -------
    xr.Dataset
        Dataset containing the directional wave spectrum with coordinates:
        - time: Time coordinates
        - direction: Direction coordinates in degrees
        - frequency: Frequency coordinates in Hz
        - latitude: Latitude coordinate
        - longitude: Longitude coordinate
        - depth: Depth coordinate (if provided)
        - station: Station identifier (if provided)

    Notes
    -----
    The function processes time series of directional wave spectra and saves
    them in NetCDF format. The spectral energy density (efth) has units of
    m²/Hz/deg. The function automatically creates the output directory if
    it doesn't exist.
    """

    time_coords = alpha1_df.index.to_numpy()
    # Use custom frequencies if provided, otherwise use DataFrame columns
    if frequencies is not None:
        frequency_coords = np.array(frequencies)
    else:
        frequency_coords = np.array([float(col) for col in alpha1_df.columns])
    n_times = len(time_coords)

    # If directions is not provided, use 360 evenly spaced
    if directions is None:
        directions = np.linspace(0, 360, 360, endpoint=False)

    # Use the first time step to determine the number of directions
    c11 = c11_df.iloc[0].values
    alpha1 = alpha1_df.iloc[0].values
    alpha2 = alpha2_df.iloc[0].values
    r1 = r1_df.iloc[0].values
    r2 = r2_df.iloc[0].values

    # Calculate the spectrum for the first time step
    E, freq_mesh, angle_mesh = calculate_directional_spectrum(
        c11, frequency_coords, alpha1, alpha2, r1, r2, angles_deg=directions
    )
    if E.shape[0] == len(frequency_coords):
        E = E.T
    n_dirs = E.shape[0]

    # Prepare efth array
    efth = np.zeros((n_times, n_dirs, len(frequency_coords)))

    # Loop over each time step and compute the spectrum
    for i, t in enumerate(time_coords):
        c11 = c11_df.loc[t].values
        alpha1 = alpha1_df.loc[t].values
        alpha2 = alpha2_df.loc[t].values
        r1 = r1_df.loc[t].values
        r2 = r2_df.loc[t].values

        E, freq_mesh, angle_mesh = calculate_directional_spectrum(
            c11, frequency_coords, alpha1, alpha2, r1, r2, angles_deg=directions
        )
        if E.shape[0] == len(frequency_coords):
            E = E.T
        efth[i, :, :] = E

    coords = {
        "time": time_coords,
        "direction": directions,
        "frequency": frequency_coords,
        "latitude": latitude,
        "longitude": longitude,
    }
    if depth is not None:
        coords["depth"] = depth
    if station is not None:
        coords["station"] = station

    ds = xr.Dataset(
        data_vars={"efth": (("time", "direction", "frequency"), efth)}, coords=coords
    )
    ds.efth.attrs["units"] = "m2/Hz/deg"
    ds.efth.attrs["long_name"] = "Spectral energy density"
    ds.direction.attrs["units"] = "degrees"
    ds.direction.attrs["long_name"] = "Wave direction"
    ds.frequency.attrs["units"] = "Hz"
    ds.frequency.attrs["long_name"] = "Wave frequency"
    ds.latitude.attrs["units"] = "degrees_north"
    ds.longitude.attrs["units"] = "degrees_east"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ds.to_netcdf(output_path)
    print(f"Saved spectrum to {output_path}")

    return ds
