import os
import os.path as op
import re
from datetime import datetime, timedelta
from itertools import groupby
from typing import List, Union

import numpy as np
import pandas as pd
import scipy.io as sio
import wavespectra
import xarray as xr
from wavespectra.construct import construct_partition

from bluemath_tk.core.operations import nautical_to_mathematical
from bluemath_tk.tcs.tracks import get_vmean, ibtracs_fit_pmin_wmax, wind2rmw
from bluemath_tk.tcs.vortex import vortex_model_grid

from ...core.operations import get_uv_components

# from bluemath_tk.wrappers.swan.swan_wrapper import SwanModelWrapper
from .._base_wrappers import BaseModelWrapper
from .._utils_wrappers import write_array_in_file
from .swan_utils import (
    generate_forcing_file_GreenWaves_additive,
    generate_wind_forcing_dynamic_regular,
    generate_wind_forcing_dynamic_unstructured,
    # generate_forcing_file_GreenWaves_additive_new,
    sbatch_file_greenwaves,
)


class SwanModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SWAN model.
    https://swanmodel.sourceforge.io/online_doc/swanuse/swanuse.html

    Attributes
    ----------
    default_parameters : dict
        The default parameters type for the wrapper.
    available_launchers : dict
        The available launchers for the wrapper.
    output_variables : dict
        The output variables for the wrapper.
    """

    default_parameters = {
        "Hs": {
            "type": float,
            "value": None,
            "description": "Significant wave height.",
        },
        "Tp": {
            "type": float,
            "value": None,
            "description": "Wave peak period.",
        },
        "Dir": {
            "type": float,
            "value": None,
            "description": "Wave direction.",
        },
        "Spr": {
            "type": float,
            "value": None,
            "description": "Directional spread.",
        },
        "dir_dist": {
            "type": str,
            "choices": ["CIRCLE", "SECTOR"],
            "value": "CIRCLE",
            "description": "CIRCLE indicates that the spectral directions cover the full circle. SECTOR indicates that the spectral directions cover a limited sector of the circle.",
        },
        "dir1": {
            "type": float,
            "value": None,
            "description": "Only with SECTOR option. The direction of the right-hand boundary of the sector when looking outward from the sector (in degrees).",
        },
        "dir2": {
            "type": float,
            "value": None,
            "description": "Only with SECTOR option. The direction of the left-hand boundary of the sector when looking outward from the sector (in degrees).",
        },
        "mdc": {
            "type": int,
            "value": 24,
            "description": "Spectral directional discretization.",
        },
        "flow": {
            "type": float,
            "value": 0.03,
            "description": "Low values for frequency.",
        },
        "fhigh": {
            "type": float,
            "value": 0.5,
            "description": "High value for frequency.",
        },
        "Freq_array": {
            "type": np.ndarray,
            "value": None,
            "description": "Array of frequencies for the model.",
        },
        "Dir_array": {
            "type": np.ndarray,
            "value": None,
            "description": "Array of directions for the model.",
        },
    }

    available_launchers = {
        "serial": "swan_serial.exe",
        "docker_serial": "docker run --rm -v .:/case_dir -w /case_dir geoocean/rocky8 swan_serial.exe",
        "geoocean-cluster": "launchSwan.sh",
    }

    output_variables = {
        "Depth": {
            "long_name": "Water depth at the point",
            "units": "m",
        },
        "Hsig": {
            "long_name": "Significant wave height",
            "units": "m",
        },
        "Tm02": {
            "long_name": "Mean wave period",
            "units": "s",
        },
        "Dir": {
            "long_name": "Wave direction",
            "units": "degrees",
        },
        "PkDir": {
            "long_name": "Peak wave direction",
            "units": "degrees",
        },
        "TPsmoo": {
            "long_name": "Peak wave period",
            "units": "s",
        },
        "Dspr": {
            "long_name": "Directional spread",
            "units": "degrees",
        },
    }

    def __init__(
        self,
        templates_dir: str,
        metamodel_parameters: dict,
        fixed_parameters: dict,
        output_dir: str,
        templates_name: dict = "all",
        depth_array: np.ndarray = None,
        locations: np.ndarray = None,
        debug: bool = True,
    ) -> None:
        """
        Initialize the SWAN model wrapper.
        """

        super().__init__(
            templates_dir=templates_dir,
            metamodel_parameters=metamodel_parameters,
            fixed_parameters=fixed_parameters,
            output_dir=output_dir,
            templates_name=templates_name,
            default_parameters=self.default_parameters,
        )
        self.set_logger_name(
            name=self.__class__.__name__, level="DEBUG" if debug else "INFO"
        )

        if depth_array is not None:
            self.depth_array = np.round(depth_array, 5)
        else:
            self.depth_array = None

        if locations is not None:
            self.locations = np.column_stack(locations)
        else:
            self.locations = None

    def list_available_output_variables(self) -> List[str]:
        """
        List available output variables.

        Returns
        -------
        List[str]
            The available output variables.
        """

        return list(self.output_variables.keys())

    def _convert_case_output_files_to_nc(
        self, case_num: int, output_path: str, output_vars: List[str]
    ) -> xr.Dataset:
        """
        Convert mat file to netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        output_path : str
            The output path.
        output_vars : List[str]
            The output variables to use.

        Returns
        -------
        xr.Dataset
            The xarray Dataset.
        """

        # Read mat file
        output_dict = sio.loadmat(output_path)

        # Create Dataset
        ds_output_dict = {var: (("Yp", "Xp"), output_dict[var]) for var in output_vars}
        ds = xr.Dataset(
            ds_output_dict,
            coords={"Xp": output_dict["Xp"][0, :], "Yp": output_dict["Yp"][:, 0]},
        )

        # assign correct coordinate case_num
        ds.coords["case_num"] = case_num

        return ds

    def get_case_percentage_from_file(self, output_log_file: str) -> str:
        """
        Get the case percentage from the output log file.

        Parameters
        ----------
        output_log_file : str
            The output log file.

        Returns
        -------
        str
            The case percentage.
        """

        if not os.path.exists(output_log_file):
            return "0 %"

        progress_pattern = r"OK in\s+(\d+\.\d+)\s*%"
        with open(output_log_file, "r") as f:
            for line in reversed(f.readlines()):
                match = re.search(progress_pattern, line)
                if match:
                    if float(match.group(1)) > 99.5:
                        return "100 %"
                    return f"{match.group(1)} %"

        return "0 %"  # if no progress is found

    def monitor_cases(self, value_counts: str = None) -> Union[pd.DataFrame, dict]:
        """
        Monitor the cases based on the wrapper_out.log file.
        """

        cases_status = {}

        for case_dir in self.cases_dirs:
            output_log_file = os.path.join(case_dir, "wrapper_out.log")
            progress = self.get_case_percentage_from_file(
                output_log_file=output_log_file
            )
            cases_status[os.path.basename(case_dir)] = progress

        return super().monitor_cases(
            cases_status=cases_status, value_counts=value_counts
        )

    def postprocess_case(
        self,
        case_num: int,
        case_dir: str,
        case_context: dict,
        output_vars: List[str] = ["Hsig", "Tm02", "Dir"],
    ) -> xr.Dataset:
        """
        Convert mat ouput files to netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        case_context : dict
            The case context.
        output_vars : list, optional
            The output variables to postprocess. Default is None.

        Returns
        -------
        xr.Dataset
            The postprocessed Dataset.
        """

        if output_vars is None:
            self.logger.info("Postprocessing all available variables.")
            output_vars = list(self.output_variables.keys())

        output_nc_path = os.path.join(case_dir, "output.nc")
        if not os.path.exists(output_nc_path):
            # Convert tab files to netCDF file
            output_path = os.path.join(case_dir, "output.mat")
            output_nc = self._convert_case_output_files_to_nc(
                case_num=case_num,
                output_path=output_path,
                output_vars=output_vars,
            )
            output_nc.to_netcdf(os.path.join(case_dir, "output.nc"))
        else:
            self.logger.info("Reading existing output.nc file.")
            output_nc = xr.open_dataset(output_nc_path)

        return output_nc

    def join_postprocessed_files(
        self, postprocessed_files: List[xr.Dataset]
    ) -> xr.Dataset:
        """
        Join postprocessed files in a single Dataset.

        Parameters
        ----------
        postprocessed_files : list
            The postprocessed files.

        Returns
        -------
        xr.Dataset
            The joined Dataset.
        """

        return xr.concat(postprocessed_files, dim="case_num")


def generate_fixed_parameters(
    grid_parameters: dict,
    freq_array: np.array,
    dir_array: np.array,
) -> dict:
    """
    Generate fixed parameters for the SWAN model based on grid parameters and frequency/direction arrays.
    Parameters
    ----------
    grid_parameters : dict
        Dictionary with grid configuration for SWAN input.
    freq_array : np.ndarray
        Array of frequencies for the SWAN model.
    dir_array : np.ndarray
        Array of directions for the SWAN model.
    Returns
    -------
    dict
        Dictionary with fixed parameters for the SWAN model.
    """

    dirs = np.sort(np.unique(dir_array)) % 360
    step = np.round(np.median(np.diff(np.sort(dirs))), 4)

    # Compute angular gaps between sorted directions (including wrap-around)
    diffs = np.diff(np.concatenate([dirs, [dirs[0] + 360]]))
    max_gap_idx = np.argmax(diffs)

    if np.isclose(diffs[max_gap_idx], step, atol=1e-2):
        dir_dist = "CIRCLE"
        dir1, dir2 = None, None
    else:
        dir_dist = "SECTOR"
        dir1 = float((dirs[(max_gap_idx + 1) % len(dirs)]) % 360)  # right-hand boundary
        dir2 = float((dirs[max_gap_idx]) % 360)  # left-hand boundary
    print("Distribución direccional:", dir_dist)
    if dir_dist == "SECTOR":
        print(f"Direcciones de {dir1}° a {dir2}°")

    return {
        "xpc": grid_parameters["xpc"],  # origin x
        "ypc": grid_parameters["ypc"],  # origin y
        "alpc": grid_parameters["alpc"],  # x-axis direction
        "xlenc": grid_parameters["xlenc"],  # grid length x
        "ylenc": grid_parameters["ylenc"],  # grid length y
        "mxc": grid_parameters["mxc"],  # num mesh x
        "myc": grid_parameters["myc"],  # num mesh y
        "xpinp": grid_parameters["xpinp"],  # origin x for input grid
        "ypinp": grid_parameters["ypinp"],  # origin y for input grid
        "alpinp": grid_parameters["alpinp"],  # x-axis direction
        "mxinp": grid_parameters["mxinp"],  # num mesh x for input grid
        "myinp": grid_parameters["myinp"],  # num mesh y for input grid
        "dxinp": grid_parameters["dxinp"],  # resolution x for input grid
        "dyinp": grid_parameters["dyinp"],  # resolution y for input grid
        "dir_dist": dir_dist,  # direction distribution type
        "dir1": dir1,  # min direction
        "dir2": dir2,  # max direction
        "freq_discretization": len(np.unique(freq_array)),  # frequency discretization
        "dir_discretization": int(
            360 / (np.unique(dir_array)[1] - np.unique(dir_array)[0])
        ),  # direction discretization
        "mdc": int(
            360 / (np.unique(dir_array)[1] - np.unique(dir_array)[0])
        ),  # number of depth cases
        "flow": float(np.min(np.unique(freq_array))),  # low frequency limit
        "fhigh": float(np.max(np.unique(freq_array))),  # high frequency limit
    }


class BinWavesWrapper(SwanModelWrapper):
    """
    Wrapper example for the BinWaves model.
    """

    def build_case(self, case_dir: str, case_context: dict) -> None:
        if self.depth_array is not None:
            write_array_in_file(self.depth_array, f"{case_dir}/depth.dat")
        if self.locations is not None:
            write_array_in_file(self.locations, f"{case_dir}/locations.loc")

        # Construct the input spectrum
        input_spectrum = construct_partition(
            freq_name="jonswap",
            freq_kwargs={
                "freq": np.geomspace(
                    case_context.get("flow", 0.035),
                    case_context.get("fhigh", 0.5),
                    case_context.get("freq_discretization", 29),
                ),
                # "freq": np.linspace(case_context.get("flow", 0.035), case_context.get("fhigh", 0.5), case_context.get("freq_discretization", 29)),
                "fp": 1.0 / case_context.get("tp"),
                "hs": case_context.get("hs"),
            },
            dir_name="cartwright",
            dir_kwargs={
                "dir": np.linspace(0, 360, case_context.get("dir_discretization", 24)),
                "dm": case_context.get("dir"),
                "dspr": case_context.get("spr"),
            },
        )
        argmax_bin = np.argmax(input_spectrum.values)
        mono_spec_array = np.zeros(input_spectrum.freq.size * input_spectrum.dir.size)
        mono_spec_array[argmax_bin] = input_spectrum.sum(dim=["freq", "dir"])
        mono_spec_array = mono_spec_array.reshape(
            input_spectrum.freq.size, input_spectrum.dir.size
        )
        mono_input_spectrum = xr.Dataset(
            {
                "efth": (["freq", "dir"], mono_spec_array),
            },
            coords={
                "freq": input_spectrum.freq,
                "dir": input_spectrum.dir,
            },
        )
        for side in ["N", "S", "E", "W"]:
            wavespectra.SpecDataset(mono_input_spectrum).to_swan(
                os.path.join(case_dir, f"input_spectra_{side}.bnd")
            )


class HyWindSeaWrapper(SwanModelWrapper):
    """
    Wrapper for the HyWindSea metamodel.

    Cases are driven by a single NetCDF holding the high-resolution wind fields
    already selected for simulation: one case per (time, tide level) pair.

    Expected parameters
    -------------------
    metamodel_parameters : dict
        - tide_level : list of float
            Water levels to simulate.
        - time_index : list of int
            Positional indices into the ``time`` dimension of the wind file.
    fixed_parameters : dict
        - wind_file : str
            Path to the NetCDF with the high-resolution winds. Must contain
            ``M`` (wind speed) and ``Dir`` (wind direction) on a (time, lat, lon)
            grid. A ``lev`` dimension, if still present, is reduced using
            ``wind_level``.
        - wind_level : float, optional
            Level to select when the wind file still has a ``lev`` dimension.
            Default is 10.
        - bathy : xr.Dataset
            Bathymetry with a ``depth`` variable, positive downwards.
        - percentile : float
            Percentile of the wind speed over water used for the rescaling.
        - umbral : float
            Wind speed threshold (m/s) that percentile is rescaled to.

    Notes
    -----
    The previous version took a list of daily wind files plus a ``day_hour``
    index and built every combination, so it simulated 24 hours of each file
    whether or not those hours had been selected. Here the wind file already
    holds exactly the times to run, so the case count is
    ``len(tide_level) * len(time_index)`` and nothing is simulated that will not
    be used.

    Examples
    --------
    >>> wind_file = "inputs/wind_hr_10times.nc"
    >>> n_times = xr.open_dataset(wind_file).sizes["time"]
    >>> wrapper = HyWindSeaWrapper(
    ...     templates_dir="templates",
    ...     metamodel_parameters={
    ...         "tide_level": [0.0],
    ...         "time_index": list(range(n_times)),
    ...     },
    ...     fixed_parameters={
    ...         "wind_file": wind_file,
    ...         "bathy": xr.open_dataset("inputs/bati_santander_50m_LONLAT.nc"),
    ...         "percentile": 50,
    ...         "umbral": 9,
    ...     },
    ...     output_dir="outputs/SANTANDER/swan",
    ... )
    """

    def open_case_wind(self, case_context: dict) -> xr.Dataset:
        """
        Open the wind field corresponding to a single case.

        Parameters
        ----------
        case_context : dict
            The case context. Must contain ``wind_file`` and ``time_index``.

        Returns
        -------
        xr.Dataset
            The wind field at the requested time, squeezed to (lat, lon).

        Raises
        ------
        KeyError
            If ``wind_file`` or ``time_index`` is missing from the context.
        """

        for key in ("wind_file", "time_index"):
            if case_context.get(key) is None:
                raise KeyError(f"'{key}' is required in the case context")

        wind = xr.open_dataset(case_context["wind_file"]).isel(
            time=case_context["time_index"]
        )

        # The wind file may already have been reduced to a single level when it
        # was built, in which case 'lev' survives as a scalar coordinate and
        # must not be selected again.
        if "lev" in wind.dims:
            wind = wind.sel(lev=case_context.get("wind_level", 10))

        return wind.squeeze()

    def calculate_alpha_matrix_for_case(
        self, case_dir: str, case_context: dict
    ) -> None:
        """
        Reescale output data for the HyWindSea model.
        """

        # Open wind data and slice for bathymetry adaptation
        wind = self.open_case_wind(case_context)
        wind_edit = wind.sel(
            lon=slice(
                case_context.get("bathy").lon.values.min(),
                case_context.get("bathy").lon.values.max(),
            ),
            lat=slice(
                case_context.get("bathy").lat.values.min(),
                case_context.get("bathy").lat.values.max(),
            ),
        )
        bathy_interp = case_context.get("bathy").interp(
            lon=wind_edit.lon, lat=wind_edit.lat
        )
        wind_edit["bathy"] = bathy_interp.depth

        # Calculate percentiles and modify wind speeds
        perc_dataset = wind_edit.where(wind_edit.bathy > 0).quantile(
            case_context.get("percentile") / 100, dim=["lon", "lat"]
        )
        alpha = case_context.get("umbral") / perc_dataset
        # alpha["M"] = (
        #     "time",
        #     np.where(perc_dataset["M"] > case_context.get("umbral"), 1, alpha["M"]),
        # )

        # Save wind and alpha data in case_context dict
        case_context["alpha"] = np.where(
            perc_dataset["M"] > case_context.get("umbral"), 1, alpha["M"]
        )
        wind_done = wind.copy()
        wind_done["M"] = wind["M"] * case_context["alpha"]
        case_context["wind"] = wind_done

    def transform_write_wind_data(self, case_dir: str, case_context: dict) -> None:
        """
        Transform wind data for the HyWindSea model.
        """

        # Calculate u10 and v10 components
        u10, v10 = get_uv_components(case_context["wind"].Dir)
        case_context["wind"]["u10"] = -u10 * case_context["wind"].M
        case_context["wind"]["v10"] = -v10 * case_context["wind"].M
        w = case_context["wind"].interp(
            lat=case_context.get("bathy").lat.values,
            lon=case_context.get("bathy").lon.values,
            method="linear",
        )

        # extract and save
        u10 = w.u10.values
        v10 = w.v10.values
        arr = np.vstack((u10, v10))

        # Save wind file
        write_array_in_file(arr, f"{case_dir}/wind_file.dat")

    def transform_postprocess_ouput_data(
        self, wave_data: dict, case_context: dict
    ) -> np.ndarray:
        """
        Transform wind data for the HyWindSea model.
        """

        # Use alpha to rescale wave output data

        wave_data["Hsig"] = wave_data["Hsig"] / case_context.get("alpha")
        return wave_data

    def build_case(self, case_dir: str, case_context: dict) -> None:
        if self.depth_array is not None:
            write_array_in_file(self.depth_array, f"{case_dir}/depth_main.dat")
        if self.locations is not None:
            write_array_in_file(self.locations, f"{case_dir}/locations.loc")
        self.calculate_alpha_matrix_for_case(
            case_dir=case_dir, case_context=case_context
        )
        self.transform_write_wind_data(case_dir=case_dir, case_context=case_context)

    def postprocess_case(
        self, case_num, case_dir, case_context, output_vars=["Hsig", "Tm02", "Dir"]
    ):
        """
        Postprocess a single case, rescaling Hsig back with the case alpha.

        Notes
        -----
        ``alpha`` and ``wind`` are normally written into the context by
        ``build_case``. They are absent whenever the cases were not built in
        this session — after ``load_cases()``, or when the runs were submitted
        to a queue and postprocessed later — so they are recomputed here when
        missing. Both come deterministically from the wind file, so recomputing
        gives the same values the build used.
        """

        if case_context.get("alpha") is None or case_context.get("wind") is None:
            self.calculate_alpha_matrix_for_case(
                case_dir=case_dir, case_context=case_context
            )

        wave_data = super().postprocess_case(
            case_num, case_dir, case_context, output_vars
        )
        reescaled_wind = self.transform_postprocess_ouput_data(wave_data, case_context)
        wave_output = reescaled_wind.expand_dims(
            {
                "time": [case_context.get("wind").time.values],
                "tide": [case_context.get("tide_level")],
            }
        )
        return wave_output

    def join_postprocessed_files(
        self, postprocessed_files: List[xr.Dataset]
    ) -> xr.Dataset:
        """
        Join postprocessed files in a single Dataset.

        Parameters
        ----------
        postprocessed_files : list
            The postprocessed files.

        Returns
        -------
        xr.Dataset
            The joined Dataset.
        """
        # combined_time = xr.concat(postprocessed_files, dim="time")
        # expand_dims leaves 'tide' as a length-1 dimension, so its .values is a
        # 1-D array. float() on that raises TypeError from NumPy 2.0 onwards
        # (it was only a DeprecationWarning before), hence the ravel()[0].
        def _tide_value(ds: xr.Dataset) -> float:
            return float(np.ravel(ds.tide.values)[0])

        postprocessed_files_sorted = sorted(postprocessed_files, key=_tide_value)

        # Agrupar por valor de marea
        grouped_by_tide = {
            tide: list(group)
            for tide, group in groupby(postprocessed_files_sorted, key=_tide_value)
        }

        combined_by_tide = []

        # Combinar los datasets de cada marea por tiempo
        for tide_val, ds_list in grouped_by_tide.items():
            ds_tide = xr.concat(
                ds_list, dim="time", combine_attrs="override", join="outer"
            )
            combined_by_tide.append(ds_tide)

        # Combinar todas las mareas
        combined_all = xr.concat(
            combined_by_tide, dim="tide", combine_attrs="override", join="outer"
        )

        return combined_all  # time and tide dimensions


class HyRubyWrapper(SwanModelWrapper):
    """
    Wrapper example for the metamodel of Panama.
    """

    def transform_write_wind_data(self, case_dir: str, case_context: dict) -> None:
        """
        Transform wind data for the swan model.
        """

        # Open wind data and slice for bathymetry adaptation
        wind = (
            xr.open_dataset(case_context.get("wind_file"))
            .drop_vars(["number", "expver"])
            .isel(valid_time=case_context.get("day_hour"))
            .rename({"longitude": "lon", "latitude": "lat"})
            .sortby("lat")
        )

        wind_edit = wind.sel(
            lon=slice(
                case_context.get("bathy").lon.values.min(),
                case_context.get("bathy").lon.values.max(),
            ),
            lat=slice(
                case_context.get("bathy").lat.values.min(),
                case_context.get("bathy").lat.values.max(),
            ),
        )
        bathy_interp = case_context.get("bathy").interp(
            lon=wind_edit.lon, lat=wind_edit.lat
        )
        wind_edit["bathy"] = bathy_interp.elevation

        # Calculate u10 and v10 components
        w = wind.interp(
            lat=case_context.get("bathy").lat.values,
            lon=case_context.get("bathy").lon.values,
            method="linear",
        )
        # extract and save
        u10 = w.u10.values
        v10 = w.v10.values
        arr = np.vstack((u10, v10))

        # Save wind file
        write_array_in_file(arr, f"{case_dir}/wind_file.dat")

    def build_case(self, case_dir: str, case_context: dict) -> None:
        if self.depth_array is not None:
            write_array_in_file(self.depth_array, f"{case_dir}/depth_main.dat")
        if self.locations is not None:
            write_array_in_file(self.locations, f"{case_dir}/locations.loc")

        self.transform_write_wind_data(case_dir=case_dir, case_context=case_context)

    def postprocess_case(
        self, case_num, case_dir, case_context, output_vars=["Hsig", "Tm02", "Dir"]
    ):
        wave_data = super().postprocess_case(
            case_num, case_dir, case_context, output_vars
        )
        return wave_data


class GreenWavesWrapperEti(SwanModelWrapper):
    """
    Wrapper example for the metamodel of GreenWaves.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sbatch_file_example = sbatch_file_greenwaves

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
    ) -> None:
        """
        Build the input files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        generate_forcing_file_GreenWaves_additive(
            case_context=case_context,
            case_dir=case_dir,
            ds_GFD_info=case_context.get("ds_GFD_info"),
        )

class GreenWavesWrapperDynamicUnstrucred(SwanModelWrapper):
    """
    Wrapper example for the Dynamic (Unstructured Wind Forcing) simulation of GreenWaves
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sbatch_file_example = sbatch_file_greenwaves

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
    ) -> None:
        """
        Build the input files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """
        generate_wind_forcing_dynamic_unstructured(
            xds_vortex_GS = case_context.get("xds_vortes_GS"),
            ds_GFD_info = case_context.get("ds_GFD_info"),
            output_file = f"{case_dir}/wind_forcing.wind",
            additional_time= case_context.get("additional_time",0),
        )




class GreenWavesWrapperDynamicRegular(SwanModelWrapper):
    """
    Wrapper example for the Dynamic (Regular Wind Forcing) simulation of GreenWaves.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sbatch_file_example = sbatch_file_greenwaves

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
    ) -> None:
        """
        Build the input files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """
        generate_wind_forcing_dynamic_regular(
            xds_vortex_GS = case_context.get("xds_vortes_GS"),
            fixed_parameters=self.fixed_parameters,
            output_file = f"{case_dir}/wind_forcing.wind",
            additional_time = case_context.get("additional_time")
        )

