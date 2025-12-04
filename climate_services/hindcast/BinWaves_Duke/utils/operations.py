from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr


def calculate_initial_grid_parameters(
    lonA: float,
    latA: float,
    lonB: float,
    latB: float,
    lonC: float,
    latC: float,
    padding_degrees: float = 0.0,
) -> Tuple[float, float, float, float, float]:
    """
    Calculate initial grid parameters from three points using simple Cartesian approximation.

    Parameters
    ----------
    lonA : float
        Origin point longitude coordinate.
    latA : float
        Origin point latitude coordinate.
    lonB : float
        Point along x-axis longitude coordinate.
    latB : float
        Point along x-axis latitude coordinate.
    lonC : float
        Far corner point longitude coordinate.
    latC : float
        Far corner point latitude coordinate.
    padding_degrees : float, optional
        Padding to add around the grid in all directions (in degrees).
        Default is 0.0 (no padding). Use ~0.1 for ~11km buffer.

    Returns
    -------
    xpc : float
        Grid origin x-coordinate (longitude).
    ypc : float
        Grid origin y-coordinate (latitude).
    alpc : float
        Grid rotation angle in degrees.
    xlenc : float
        Grid length in x-direction (degrees).
    ylenc : float
        Grid length in y-direction (degrees).

    Notes
    -----
    Uses simple Cartesian calculations (no UTM transformation).
    This matches SWAN's approach which treats coordinates as flat.
    """

    # Use simple Cartesian calculations (no UTM transformation)
    # This matches SWAN's approach which treats coordinates as flat

    # Vector AB (treating coordinates as Cartesian)
    dx_AB = lonB - lonA
    dy_AB = latB - latA
    length_AB = np.sqrt(dx_AB**2 + dy_AB**2)
    ux = dx_AB / length_AB
    uy = dy_AB / length_AB

    # Vector AC
    dx_AC = lonC - lonA
    dy_AC = latC - latA

    # Vector BC
    dx_BC = lonC - lonB
    dy_BC = latC - latB

    # Rotation angle using simple atan2 (Cartesian approach)
    alpc = np.degrees(np.arctan2(dy_AB, dx_AB))

    # Perpendicular vector to AB
    vx = -uy
    vy = ux

    xlenc1 = length_AB  # Distance from A to B
    xlenc2 = dx_BC * ux + dy_BC * uy  # Projection of BC onto AB direction
    xlenc = xlenc1 + xlenc2

    # ylenc: perpendicular distance from C to AB line (in degrees)
    ylenc = abs(dx_AC * vx + dy_AC * vy)

    # Apply padding if specified
    if padding_degrees > 0:
        # Convert angle to radians for calculations
        angle_rad = np.radians(alpc)

        # Calculate displacement vector for moving origin backwards by padding amount
        # We need to move back in both the x and y directions of the rotated grid
        dx_back = -padding_degrees * np.cos(angle_rad) - padding_degrees * np.sin(
            angle_rad
        )
        dy_back = padding_degrees * np.sin(angle_rad) - padding_degrees * np.cos(
            angle_rad
        )

        # New origin (moved backwards by padding amount)
        xpc = lonA + dx_back
        ypc = latA + dy_back

        # New grid lengths (add padding to both ends)
        xlenc = xlenc + 2 * padding_degrees
        ylenc = ylenc + 2 * padding_degrees
    else:
        # Set grid origin (same as input)
        xpc = lonA
        ypc = latA

    return xpc, ypc, alpc, xlenc, ylenc


def locations_grid_outputs(
    grid_parameters: Dict[str, Union[float, int]],
    out_dx: Optional[float] = None,
    out_dy: Optional[float] = None,
    outputs_limits: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    buoy_locations: Optional[
        Union[Dict[str, Tuple[float, float]], List[Tuple[float, float]], np.ndarray]
    ] = None,
) -> np.ndarray:
    """
    Generate an output grid accounting for grid rotation.

    Parameters
    ----------
    grid_parameters : dict
        Dictionary containing grid parameters. Must contain keys:
        - xpc : float
            Grid origin x-coordinate (longitude).
        - ypc : float
            Grid origin y-coordinate (latitude).
        - xlenc : float
            Grid length in x-direction (degrees).
        - ylenc : float
            Grid length in y-direction (degrees).
        - alpc : float
            Grid rotation angle in degrees.
        - mxc : int
            Number of grid cells in x-direction.
        - myc : int
            Number of grid cells in y-direction.
    out_dx : float, optional
        Output grid spacing in x-direction (degrees).
        If None, calculated from xlenc / mxc.
    out_dy : float, optional
        Output grid spacing in y-direction (degrees).
        If None, calculated from ylenc / myc.
    outputs_limits : dict, optional
        Dictionary with filtering limits:
        - 'lon' : tuple of (min_lon, max_lon), optional
            Longitude limits for filtering.
        - 'lat' : tuple of (min_lat, max_lat), optional
            Latitude limits for filtering.
    buoy_locations : dict or array-like, optional
        Buoy locations to append (in lon, lat).
        Can be:
        - dict: {buoy_id: (lon, lat)}
        - list: [(lon1, lat1), (lon2, lat2), ...]
        - numpy array: shape (N, 2) with (lon, lat) pairs

    Returns
    -------
    locations : ndarray
        Array of output locations with shape (N, 2) where each row is (lon, lat).

    Notes
    -----
    The function creates a regular grid in computational space, applies rotation
    and translation, then filters by limits and appends buoy locations if provided.
    """

    alpc = grid_parameters.get("alpc")
    xpc = grid_parameters.get("xpc")  # origin
    ypc = grid_parameters.get("ypc")  # origin
    xlenc = grid_parameters.get("xlenc")  # horizontal length
    ylenc = grid_parameters.get("ylenc")  # vertical length
    mxc = grid_parameters.get("mxc")
    myc = grid_parameters.get("myc")

    # Get original grid spacing if out_dx/out_dy not provided
    if out_dx is None:
        out_dx = xlenc / mxc
    if out_dy is None:
        out_dy = ylenc / myc

    # 1. Create output grid in computational space (degrees, Cartesian)
    mxo = int(np.ceil(xlenc / out_dx))
    myo = int(np.ceil(ylenc / out_dy))
    xo = np.linspace(0, xlenc, mxo + 1)
    yo = np.linspace(0, ylenc, myo + 1)
    XO, YO = np.meshgrid(xo, yo)

    # 2. Handle rotation and translation
    angle_rad = np.radians(alpc)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    X_rot = xpc + XO * cos_angle - YO * sin_angle
    Y_rot = ypc + XO * sin_angle + YO * cos_angle

    # 3. Stack coordinates
    locations = np.column_stack((X_rot.ravel(), Y_rot.ravel()))

    # 4. Filter by output limits
    if outputs_limits is not None:
        lon_limits = outputs_limits.get("lon")
        lat_limits = outputs_limits.get("lat")
        mask = np.ones(len(locations), dtype=bool)
        if lon_limits is not None:
            mask &= (locations[:, 0] >= lon_limits[0]) & (
                locations[:, 0] <= lon_limits[1]
            )
        if lat_limits is not None:
            lat_min, lat_max = lat_limits
            if lat_min is not None:
                mask &= locations[:, 1] >= lat_min
            if lat_max is not None:
                mask &= locations[:, 1] <= lat_max
        locations = locations[mask]

    # 5. Append buoy locations if provided (in geographic coordinates)
    if buoy_locations is not None:
        if isinstance(buoy_locations, dict):
            buoy_coords = np.array(list(buoy_locations.values()))
        else:
            buoy_coords = np.array(buoy_locations)
        locations = np.vstack((locations, buoy_coords))

    return locations


def transform_Offshore_spectrum(
    CAWCR_spectrum: xr.Dataset,
    subset_parameters: Dict[str, Any],
    available_case_num: np.ndarray,
    fixed_direction: bool = False,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Transform the wave spectra from ERA5/CAWCAR format to binwaves format.

    Parameters
    ----------
    CAWCR_spectrum : xr.Dataset
        The wave spectra dataset in ERA5/CAWCAR format.
        Must contain 'efth' variable and 'frequency'/'freq' and 'direction'/'dir' dimensions.
    subset_parameters : dict
        A dictionary containing parameters for the subset processing.
        Must contain:
        - 'dir' : array-like
            Direction values for each case.
        - 'freq' : array-like
            Frequency values for each case.
    available_case_num : ndarray
        The available case numbers to process.
    fixed_direction : bool, optional
        If True, skip direction convention transformation.
        Default is False.

    Returns
    -------
    ds : xr.Dataset
        The transformed wave spectra dataset with renamed dimensions.
    ds_case_num : xr.Dataset
        The wave spectra dataset in binwaves format with case_num dimension.

    Notes
    -----
    The function performs two main transformations:
    1. Renames dimensions from 'frequency'/'direction' to 'freq'/'dir' if needed
    2. Converts direction convention (unless fixed_direction=True)
    3. Projects spectra onto available case numbers
    """

    # First, reproject the wave spectra to the binwaves format
    # Only rename if the dimensions don't already have the correct names
    rename_dict = {}
    if "frequency" in CAWCR_spectrum.dims:
        rename_dict["frequency"] = "freq"
    if "direction" in CAWCR_spectrum.dims:
        rename_dict["direction"] = "dir"

    ds = CAWCR_spectrum.rename(rename_dict) if rename_dict else CAWCR_spectrum.copy()

    # direction convention coming from or going to
    if not fixed_direction:
        ds["efth"] = ds["efth"] * np.pi / 180.0
        ds["dir"] = ds["dir"] - 180.0
        ds["dir"] = np.where(ds["dir"] < 0, ds["dir"] + 360, ds["dir"])
        ds = ds.sortby("dir").sortby("freq")

    # Second, reproject into the available case numbers dimension
    case_num_spectra = []
    for case_num, case_dir, case_freq in zip(
        available_case_num,
        np.array(subset_parameters.get("dir"))[available_case_num],
        np.array(subset_parameters.get("freq"))[available_case_num],
    ):
        try:
            closest_case = (
                ds.efth.sel(freq=case_freq, method="nearest", tolerance=0.001)
                .sel(dir=case_dir, method="nearest", tolerance=2)
                .expand_dims({"case_num": [case_num]})
            )
            case_num_spectra.append(closest_case)
        except Exception as _e:
            # Add a zeros array if the case number is not available
            case_num_spectra.append(
                xr.zeros_like(ds.efth.isel(freq=0, dir=0)).expand_dims(
                    {"case_num": [case_num]}
                )
            )
    ds_case_num = (
        xr.concat(case_num_spectra, dim="case_num").drop_vars("dir").drop_vars("freq")
    )

    return ds, ds_case_num
