import math
import os
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.waves.spectra import spectral_analysis
from bluemath_tk.wrappers._base_wrappers import BaseModelWrapper


class XBeachModelWrapper(BaseModelWrapper):
    """
    Wrapper for the XBeach model.
    https://xbeach.readthedocs.io/en/latest/

    Attributes
    ----------
    default_parameters : dict
        The default parameters type for the wrapper.
    available_launchers : dict
        The available launchers for the wrapper.
    """

    default_parameters = {
        "comptime": {
            "type": int,
            "value": 3600,
            "description": "The computational time.",
        },
        "wbctype": {
            "type": str,
            "value": "off",
            "description": "The time step for the simulation.",
        },
    }

    available_launchers = {
        "geoocean-cluster": "launchXbeach.sh",
    }

    def __init__(
        self,
        templates_dir: str,
        metamodel_parameters: dict,
        fixed_parameters: dict,
        output_dir: str,
        templates_name: dict = "all",
        debug: bool = True,
    ) -> None:
        """
        Initialize the XBeach model wrapper.
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

        if case_context["wbctype"] == "jonstable":
            with open(f"{case_dir}/jonswap.txt", "w") as f:
                for _i in range(math.ceil(case_context["comptime"] / 3600)):
                    f.write(
                        f"{case_context['Hs']} {case_context['Tp']} {case_context['Dir']} 3.300000 30.000000 3600.000000 1.000000 \n"
                    )

    def _get_average_var(self, case_nc: xr.Dataset, var: str) -> np.ndarray:
        """
        Get the average value of a variable except for the first hour of the simulation

        Parameters
        ----------
        case_nc : xr.Dataset
            Simulation .nc file.
        var : str
            Variable of interest.

        Returns
        -------
        np.ndarray
            The average value of the variable.
        """

        if var in case_nc:
            return np.mean(
                case_nc[var]
                .isel(meantime=slice(1, int(case_nc.meantime.values[-1])))
                .values,
                axis=0,
            )

    def _get_max_var(self, case_nc: xr.Dataset, var: str) -> np.ndarray:
        """
        Get the Max value of a variable except for the first hour of the simulation

        Parameters
        ----------
        case_nc : xr.Dataset
            Simulation .nc file.
        var : str
            Variable of interest.

        Returns
        -------
        np.ndarray
            The max value of the variable.
        """

        if var in case_nc:
            return np.max(
                case_nc[var]
                .isel(meantime=slice(1, int(case_nc.meantime.values[-1])))
                .values,
                axis=0,
            )

    def monitor_cases(self, value_counts: str = None) -> Union[pd.DataFrame, dict]:
        """
        Monitor the cases based on different model log files.
        """

        cases_status = {}

        for case_dir in self.cases_dirs:
            case_dir_name = os.path.basename(case_dir)
            if os.path.exists(os.path.join(case_dir, "XBlog.txt")):
                if os.path.exists(os.path.join(case_dir, "XBerror.txt")):
                    if os.path.getsize(os.path.join(case_dir, "XBerror.txt")) != 0:
                        cases_status[case_dir_name] = "XBerror.txt"
                        continue
                else:
                    with open(os.path.join(case_dir, "XBlog.txt"), "r") as f:
                        lines = f.readlines()[-2:]

                    if any("End of program xbeach" in line for line in lines):
                        cases_status[case_dir_name] = "End of run"
                        continue
                    else:
                        cases_status[case_dir_name] = "Running"
                        continue
            else:
                cases_status[case_dir_name] = "No run"
                continue

        return super().monitor_cases(
            cases_status=cases_status, value_counts=value_counts
        )

    def postprocess_case(
        self,
        case_num: int,
        case_dir: str,
        output_vars: List[str] = None,
        overwrite_output: bool = True,
    ) -> xr.Dataset:
        """
        Convert tab output files to netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        output_vars : list, optional
            The output variables to postprocess. Default is None.
        overwrite_output : bool, optional
            Overwrite the output.nc file. Default is True.

        Returns
        -------
        xr.Dataset
            The postprocessed Dataset.
        """

        import warnings

        warnings.filterwarnings("ignore")

        self.logger.info(f"[{case_num}]: Postprocessing case {case_num} in {case_dir}.")

        output_nc_path = os.path.join(case_dir, "xboutput_postprocessed.nc")
        if not os.path.exists(output_nc_path) or overwrite_output:
            output_raw = xr.open_dataset(os.path.join(case_dir, "xboutput.nc"))

            globalx = output_raw.globalx.values
            globaly = output_raw.globaly.values
            zb = output_raw.zb.values[0]
            y = np.arange(globalx.shape[0])
            x = np.arange(globalx.shape[1])

            ds = xr.Dataset(
                {
                    "globalx": (("y", "x"), globalx),
                    "globaly": (("y", "x"), globaly),
                    "zb": (("y", "x"), zb),
                },
                coords={"y": y, "x": x},
            )

            for var in output_vars:
                if var == "zs_max":
                    maxed = self._get_max_var(case_nc=output_raw, var=var)
                    masked = xr.where(ds["zb"] > 0, np.nan, maxed)
                    ds[var] = (("y", "x"), masked.data)
                else:
                    averaged = self._get_average_var(case_nc=output_raw, var=var)
                    masked = xr.where(ds["zb"] > 0, np.nan, averaged)
                    ds[var] = (("y", "x"), masked.data)

            ds = ds.drop_vars("zb")
            ds.to_netcdf(output_nc_path)

            return ds
        else:
            self.logger.info(
                f"[{case_num}]: Reading existing xboutput_postprocessed.nc file."
            )
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
            The joined xarray.Dataset.
        """

        return xr.concat(postprocessed_files, dim="case_num")


class HyXBeachNHProfile(XBeachModelWrapper):
    """
    Wrapper for the HyXBeachNHProfile model.

    Built for Zarautz beach, Basque Country, Spain.
    """

    postprocess_functions = {
        "Msetup": "calculate_setup",
        "Hrms": "calculate_statistical_analysis",
        "Hfreqs": "calculate_spectral_analysis",
    }

    def list_available_postprocess_vars(self) -> List[str]:
        """
        List available postprocess variables.

        Returns
        -------
        List[str]
            The available postprocess variables.
        """

        return list(self.postprocess_functions.keys())

    def postprocess_case(
        self,
        case_num: int,
        case_dir: str,
        output_vars: List[str] = None,
        overwrite_output: bool = True,
        overwrite_output_postprocessed: bool = True,
    ) -> xr.Dataset:
        """
        Postprocess the case.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        output_vars : list, optional
            The output variables to postprocess. Default is None.
        overwrite_output : bool, optional
            Overwrite the output.nc file. Default is True.
        overwrite_output_postprocessed : bool, optional
            Overwrite the output_postprocessed.nc file. Default is True.

        Returns
        -------
        xr.Dataset
            The postprocessed Dataset.
        """

        import warnings

        warnings.filterwarnings("ignore")

        self.logger.info(f"[{case_num}]: Postprocessing case {case_num} in {case_dir}.")

        if output_vars is None:
            self.logger.debug(f"[{case_num}]: Postprocessing all available variables.")
            output_vars = list(self.postprocess_functions.keys())

        output_nc_path = os.path.join(case_dir, "xboutput_profile.nc")
        if not os.path.exists(output_nc_path) or overwrite_output:
            output_raw = xr.open_dataset(os.path.join(case_dir, "xboutput.nc"))

            x = output_raw.globalx.values[1, :]
            time = output_raw.globaltime.values
            zb = output_raw.zb.values[0, 1, :]
            zs = output_raw.zs.values[:, 1, :]

            output_nc = xr.Dataset(
                {
                    "zb": (("Xp"), zb),
                    "Watlev": (("Tsec", "Xp"), zs),
                },
                coords={"Tsec": time, "Xp": x},
            )

            # assign correct coordinate case_num
            output_nc.coords["case_num"] = case_num

            output_nc.to_netcdf(output_nc_path)

        else:
            self.logger.info(
                f"[{case_num}]: Reading existing xboutput_postprocessed.nc file."
            )
            output_nc = xr.open_dataset(output_nc_path)

        processed_nc_path = os.path.join(case_dir, "xboutput_postprocessed.nc")
        if not os.path.exists(processed_nc_path) or overwrite_output_postprocessed:
            var_ds_list = []
            for var in output_vars:
                if var in self.postprocess_functions:
                    self.logger.debug(f"[{case_num}]: Postprocessing variable {var}.")
                    var_ds = getattr(self, self.postprocess_functions[var])(
                        case_num=case_num, case_dir=case_dir, output_nc=output_nc
                    )
                    var_ds_list.append(var_ds)
                else:
                    # If the variable is present in output_raw, extract and squeeze it
                    if var in output_nc:
                        self.logger.debug(f"[{case_num}]: Extracting variable {var}.")
                        var_ds = output_nc[var].squeeze()
                        var_ds_list.append(var_ds)
                    else:
                        self.logger.warning(
                            f"[{case_num}]: Variable {var} is not available for postprocessing."
                        )

            # Merge all variables in one Dataset
            ds = xr.merge(var_ds_list, compat="no_conflicts")
            self.logger.info(
                f"[{case_num}]: Creating new output_postprocessed.nc file from output.nc."
            )
            ds.to_netcdf(processed_nc_path)

        else:
            self.logger.info(
                f"[{case_num}]: Reading existing output_postprocessed.nc file."
            )
            ds = xr.open_dataset(processed_nc_path)

        ## Remove raw files to save space
        # if remove_tab:
        #    os.remove(output_path)
        #    os.remove(run_path)
        # if remove_nc:
        #    os.remove(output_nc_path)

        return ds

    def calculate_setup(
        self, case_num: int, case_dir: str, output_nc: xr.Dataset
    ) -> xr.Dataset:
        """
        Calculates mean setup (Msetup) from the output netCDF file.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        output_nc : xr.Dataset
            The output netCDF file.

        Returns
        -------
        xr.Dataset
            The mean setup (Msetup).
        """

        # create xarray Dataset with mean setup
        ds = output_nc["Watlev"].mean(dim="Tsec")
        ds = ds.to_dataset()

        # eliminate Yp dimension
        ds = ds.squeeze()

        # rename variable
        ds = ds.rename({"Watlev": "Msetup"})

        return ds

    def calculate_statistical_analysis(
        self, case_num: int, case_dir: str, output_nc: xr.Dataset
    ) -> xr.Dataset:
        """
        Calculates zero-upcrossing analysis to obtain individual wave heights (Hi) and wave periods (Ti).

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        output_nc : xr.Dataset
            The output netCDF file.

        Returns
        -------
        xr.Dataset
            The statistical analysis.
        """

        # for every X coordinate in domain
        df_Hrms = pd.DataFrame()

        # for x in output_nc["Xp"].values:
        #     dsw = output_nc.sel(Xp=x)

        #     # obtain series of water level
        #     series_water = dsw["Watlev"].values
        #     time_series = dsw["Tsec"].values

        #     # perform statistical analysis
        #     # _, Hi = upcrossing(time_series, series_water)
        #     _, Hi = upcrossing(np.vstack([time_series, series_water]).T)
        #     Hi = np.std(series_water)
        #     # Calculo de Pablo Zubia
        #     #standard_deviation = np.std(series_water)

        #     # calculate Hrms
        #     Hrms_x = np.sqrt(np.sum(Hi**2)/len(Hi))
        #     df_Hrms.loc[x, "Hrms"] = Hrms_x

        # # convert pd DataFrame to xr Dataset
        # df_Hrms.index.name = "Xp"
        # ds = df_Hrms.to_xarray()

        # # assign coordinate case_num
        # ds = ds.assign_coords({"case_num": [output_nc["case_num"].values]})

        # return ds

        for x in output_nc["Xp"].values:
            dsw = output_nc.sel(Xp=x)

            # obtain series of water level
            series_water = dsw["Watlev"].values
            # time_series = dsw['Tsec'].values

            standard_deviation = np.std(series_water)
            Hrms_x = 2 * np.sqrt(2 * standard_deviation**2)
            df_Hrms.loc[x, "Hrms"] = Hrms_x

        # convert pd DataFrame to xr Dataset
        df_Hrms.index.name = "Xp"
        ds = df_Hrms.to_xarray()

        # assign coordinate case_id
        ds = ds.assign_coords({"case_num": [output_nc["case_num"].values]})
        return ds

    def calculate_spectral_analysis(
        self, case_num: int, case_dir: str, output_nc: xr.Dataset
    ) -> xr.Dataset:
        """
        Makes a water level spectral analysis (scipy.signal.welch)
        then separates incident waves, infragravity waves, very low frequency waves.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        output_nc : xr.Dataset
            The output netCDF file.

        Returns
        -------
        xr.Dataset
            The spectral analysis.
        """

        delttbl = np.diff(output_nc["Tsec"].values)[1]

        df_H_spectral = pd.DataFrame()

        for x in output_nc["Xp"].values:
            dsw = output_nc.sel(Xp=x)
            series_water = dsw["Watlev"].values

            # calculate significant, SS, IG and VLF wave heighs
            Hs, Hss, Hig, Hvlf = spectral_analysis(series_water, delttbl)

            df_H_spectral.loc[x, "Hs"] = Hs
            df_H_spectral.loc[x, "Hss"] = Hss
            df_H_spectral.loc[x, "ig"] = Hig
            df_H_spectral.loc[x, "Hvlf"] = Hvlf

        # convert pd DataFrame to xr Dataset
        df_H_spectral.index.name = "Xp"
        ds = df_H_spectral.to_xarray()

        # assign coordinate case_num
        ds = ds.assign_coords({"case_num": [output_nc["case_num"].values]})

        return ds
