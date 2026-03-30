"""
burst/metrics.py
================
Aggregate ``Burst`` namedtuples into a tidy pandas DataFrame.

The inter-burst interval (IBI) is appended as a column: for burst *i* it
is the gap between the end of burst *i−1* and the start of burst *i*.
The first burst in a train receives ``NaN`` for IBI.

Public API
----------
bursts_to_dataframe(bursts, electrode_id, well_id)
    Convert a list of Burst objects to a per-burst DataFrame row.

aggregate_well_bursts(well_burst_dict, well_id)
    Concatenate per-electrode burst DataFrames for a whole well.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from py_mea_axion.burst.detection import Burst

# Columns emitted by this module (guaranteed order).
BURST_COLUMNS: List[str] = [
    "electrode_id",
    "well_id",
    "start_time",
    "end_time",
    "duration",
    "n_spikes",
    "mean_isi_within",
    "inter_burst_interval",
]


def bursts_to_dataframe(
    bursts: List[Burst],
    electrode_id: str,
    well_id: str,
) -> pd.DataFrame:
    """Convert a list of ``Burst`` namedtuples to a tidy DataFrame.

    Parameters
    ----------
    bursts : list of Burst
        Detected bursts for a single electrode, sorted by ``start_time``
        (as returned by :func:`~py_mea_axion.burst.detection.detect_bursts`).
    electrode_id : str
        Electrode label, e.g. ``'A1_11'``.
    well_id : str
        Well label, e.g. ``'A1'``.

    Returns
    -------
    pd.DataFrame
        One row per burst.  Columns and dtypes:

        =====================  =========  ===================================
        Column                 Dtype      Description
        =====================  =========  ===================================
        electrode_id           object     Electrode label
        well_id                object     Well label
        start_time             float64    First spike timestamp (s)
        end_time               float64    Last spike timestamp (s)
        duration               float64    end_time − start_time (s)
        n_spikes               int64      Spike count in burst
        mean_isi_within        float64    Mean within-burst ISI (s)
        inter_burst_interval   float64    Gap from prev burst end (s); NaN
                                          for the first burst
        =====================  =========  ===================================

        Empty DataFrame (with the correct columns) is returned when
        *bursts* is empty.

    Examples
    --------
    >>> from py_mea_axion.burst.detection import detect_bursts
    >>> import numpy as np
    >>> spikes = np.array([0.01, 0.02, 0.03, 0.04, 0.05,
    ...                    2.01, 2.02, 2.03, 2.04, 2.05])
    >>> bursts = detect_bursts(spikes)
    >>> df = bursts_to_dataframe(bursts, electrode_id="A1_11", well_id="A1")
    >>> len(df)
    2
    >>> df["inter_burst_interval"].iloc[0]
    nan
    >>> round(df["inter_burst_interval"].iloc[1], 3)
    1.96
    """
    if not bursts:
        return pd.DataFrame(columns=BURST_COLUMNS).astype(
            {
                "start_time": "float64",
                "end_time": "float64",
                "duration": "float64",
                "n_spikes": "int64",
                "mean_isi_within": "float64",
                "inter_burst_interval": "float64",
            }
        )

    rows = []
    for i, b in enumerate(bursts):
        ibi = (
            float("nan")
            if i == 0
            else float(b.start_time - bursts[i - 1].end_time)
        )
        rows.append(
            {
                "electrode_id": electrode_id,
                "well_id": well_id,
                "start_time": float(b.start_time),
                "end_time": float(b.end_time),
                "duration": float(b.duration),
                "n_spikes": int(b.n_spikes),
                "mean_isi_within": float(b.mean_isi_within),
                "inter_burst_interval": ibi,
            }
        )

    df = pd.DataFrame(rows, columns=BURST_COLUMNS)
    df["n_spikes"] = df["n_spikes"].astype("int64")
    return df.reset_index(drop=True)


def aggregate_well_bursts(
    well_burst_dict: Dict[str, List[Burst]],
    well_id: str,
) -> pd.DataFrame:
    """Concatenate per-electrode burst DataFrames for an entire well.

    Parameters
    ----------
    well_burst_dict : dict[str, list of Burst]
        Mapping from electrode ID to its burst list, e.g. as produced by
        calling :func:`~py_mea_axion.burst.detection.detect_bursts` for
        every electrode in a well.
    well_id : str
        Well label written into every row.

    Returns
    -------
    pd.DataFrame
        All bursts from all electrodes, with columns as described in
        :func:`bursts_to_dataframe`.  Sorted by ``(electrode_id,
        start_time)``.  Returns an empty DataFrame with correct columns
        if no bursts are found in any electrode.

    Examples
    --------
    >>> import numpy as np
    >>> from py_mea_axion.burst.detection import detect_bursts
    >>> spikes = {"A1_11": detect_bursts(np.array([0.01,0.02,0.03,0.04,0.05])),
    ...           "A1_12": []}
    >>> df = aggregate_well_bursts(spikes, well_id="A1")
    >>> len(df)
    1
    """
    frames = [
        bursts_to_dataframe(burst_list, electrode_id=eid, well_id=well_id)
        for eid, burst_list in well_burst_dict.items()
    ]

    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=BURST_COLUMNS).astype(
            {"n_spikes": "int64"}
        )

    df = pd.concat(non_empty, ignore_index=True)
    return df.sort_values(["electrode_id", "start_time"]).reset_index(drop=True)
