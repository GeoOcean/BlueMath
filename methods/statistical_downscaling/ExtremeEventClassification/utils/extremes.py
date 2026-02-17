from typing import Tuple

import numpy as np
import pandas as pd


def pot(
    var: pd.Series,
    indep: float,
    mindur: float,
    eventopt: str,
    res: float,
    resunit: str,
    qopt: str,
    q: float,
) -> Tuple[pd.DataFrame, pd.Series, float]:
    """
    Peak Over Threshold (POT) method to extract events from time series data.

    This function implements three different methods for detecting extreme events
    in time series data based on threshold exceedances.

    Parameters
    ----------
    var : pd.Series
        Pandas Series with datetime index containing the variable to analyze.
    indep : float
        Independence time in hours, minimum separation between events.
    mindur : float
        Minimum event duration in hours.
    eventopt : str
        Event detection method, one of:
        - 'exc': exceedances separated by more than independence time
        - 'gaps': intervals defined by gaps between exceedances larger than independence time
        - 'cont': continuous intervals of threshold exceedance, no independence enforced
    res : float
        Data resolution (time step) in units defined by resunit.
    resunit : str
        Unit of time for resolution, e.g., 'min', 'h', 'd'.
    qopt : str
        Threshold type specification, either 'quantile' or 'threshold'.
    q : float
        Threshold value or quantile value depending on qopt.

    Returns
    -------
    vare : pd.DataFrame
        DataFrame of detected events with columns ['max', 'duration', 'start', 'end'].
    events : pd.Series
        Series with annual counts of detected events.
    varth : float
        Threshold value used for event detection.

    Raises
    ------
    ValueError
        If qopt is not 'quantile' or 'threshold', or if eventopt is not one of
        the valid options.

    Notes
    -----
    The function supports three different event detection methods:

    - 'exc': Finds exceedances separated by more than the independence time
    - 'gaps': Identifies intervals defined by gaps between exceedances
    - 'cont': Detects continuous intervals of threshold exceedance

    Events with duration less than mindur are filtered out from the results.
    """

    # Determine threshold
    if qopt[:4] == "quan":
        varth = var.quantile(q)
    elif qopt[:4] == "thre":
        varth = q
    else:
        raise ValueError("qopt must start with 'quan' or 'thre'")

    # Select values exceeding threshold
    vare = var[var >= varth]
    print(vare)

    # ----- Event detection using exceedances ('exc') -----
    if eventopt == "exc":
        # Find indices where time gap between exceedances > independence threshold
        istorm = np.insert(
            np.where((vare.index[1:] - vare.index[:-1]) > pd.Timedelta(res, resunit))[
                0
            ],
            0,
            -1,
        )

        indx, vmax, dur, start, end = [], [], [], [], []
        for i, j in zip(istorm[:-1] + 1, istorm[1:]):
            sub_var = vare[i : j + 1]
            indx.append(sub_var.idxmax())
            vmax.append(sub_var.max())
            start.append(sub_var.index[0])
            end.append(sub_var.index[-1])
            dur.append(
                (sub_var.index[-1] - sub_var.index[0]).total_seconds() / 3600
            )  # hours

        indx = pd.Index(indx)

        if len(indx) == 0:
            print("no INDX", flush=True)
            return np.nan, np.nan, np.nan

        # Check event independence based on indep threshold (hours)
        indx2, vmax2, dur2, start2, end2 = (
            [indx[0]],
            [vmax[0]],
            [dur[0]],
            [start[0]],
            [end[0]],
        )
        for i, v, d, st, en in zip(indx[1:], vmax[1:], dur[1:], start[1:], end[1:]):
            if (i - indx2[-1]) > pd.Timedelta(indep, "h"):  # independent event
                indx2.append(i)
                vmax2.append(v)
                dur2.append(d)
                start2.append(st)
                end2.append(en)
            else:
                # Merge events, keep max amplitude and adjust duration
                imx = np.argmax([vmax2[-1], v])
                indx2[-1] = [indx2[-1], i][imx]
                vmax2[-1] = [vmax2[-1], v][imx]
                end2[-1] = en
                dur2[-1] = (en - start2[-1]).total_seconds() / 3600  # hours

        vare = pd.DataFrame(
            index=indx2,
            data=np.vstack((vmax2, dur2, start2, end2)).T,
            columns=["max", "duration", "start", "end"],
        )
        vare = vare[vare["duration"] >= mindur]
        events = vare["max"].groupby(vare.index.year).agg("count")

    # ----- Event detection using gaps between exceedances ('gaps') -----
    elif eventopt == "gaps":
        indx, vmax, dur, start, end = [], [], [], [], []

        if len(vare) == 0:
            print("No exceedances found.")
        elif len(vare) == 1:
            cluster = vare
            indx.append(cluster.idxmax())
            vmax.append(cluster.loc[cluster.idxmax()])
            start.append(cluster.index[0])
            end.append(cluster.index[-1])
            dur.append(
                (cluster.index[-1] - cluster.index[0]).total_seconds() / 3600
            )  # hours
        else:
            # Locate gap indices where gap > independence threshold
            gap_indices = np.argwhere(
                (vare.index[1:] - vare.index[:-1]) > pd.Timedelta(indep, "h")
            ).flatten()

            if len(gap_indices) == 0:
                # All exceedances are in one cluster
                cluster = vare
                indx.append(cluster.idxmax())
                vmax.append(cluster.loc[cluster.idxmax()])
                start.append(cluster.index[0])
                end.append(cluster.index[-1])
                dur.append(
                    (cluster.index[-1] - cluster.index[0]).total_seconds() / 3600
                )  # hours
            else:
                for i, gap_index in enumerate(gap_indices):
                    if i == 0:
                        cluster = vare.iloc[: gap_index + 1]
                    else:
                        cluster = vare.iloc[gap_indices[i - 1] + 1 : gap_index + 1]

                    indx.append(cluster.idxmax())
                    vmax.append(cluster.loc[cluster.idxmax()])
                    start.append(cluster.index[0])
                    end.append(cluster.index[-1])
                    dur.append(
                        (cluster.index[-1] - cluster.index[0]).total_seconds() / 3600
                    )  # hours

        indx = pd.Index(indx)
        vare = pd.DataFrame(
            index=indx,
            data=np.vstack((vmax, dur, start, end)).T,
            columns=["max", "duration", "start", "end"],
        )
        vare = vare[vare["duration"] >= mindur]
        events = vare["max"].groupby(vare.index.year).agg("count")

    # ----- Event detection using continuous threshold exceedance ('cont') -----
    elif eventopt == "cont":
        istorm = np.insert(
            np.where((vare.index[1:] - vare.index[:-1]) > pd.Timedelta(res, resunit))[
                0
            ],
            0,
            -1,
        )

        indx, vmax, dur, start, end = [], [], [], [], []
        for i, j in zip(istorm[:-1] + 1, istorm[1:]):
            sub_var = vare[i : j + 1]
            indx.append(sub_var.idxmax())
            vmax.append(sub_var.max())
            start.append(sub_var.index[0])
            end.append(sub_var.index[-1])
            dur.append(sub_var.count() * res)

        indx = pd.Index(indx)
        vare = pd.DataFrame(
            index=indx,
            data=np.vstack((vmax, dur, start, end)).T,
            columns=["max", "duration", "start", "end"],
        )
        vare = vare[vare["duration"] >= mindur]
        events = vare["max"].groupby(vare.index.year).agg("count")

    else:
        raise ValueError("Invalid eventopt: choose 'exc', 'gaps', or 'cont'")

    return vare, events, varth
