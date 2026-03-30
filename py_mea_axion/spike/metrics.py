"""
spike/metrics.py
================
Spike-level metrics computed from raw timestamp arrays.

All functions operate on a single electrode's spike timestamp array
(1-D ``np.ndarray`` of times in seconds) plus a recording duration.
The ``summarise_well()`` function aggregates across all 16 electrodes
in a well and returns a one-row pandas DataFrame suitable for
concatenation into an experiment-level summary table.

Public API
----------
mean_firing_rate(spike_times, duration_s)
isi_array(spike_times)
isi_stats(spike_times)
is_active(spike_times, duration_s, threshold_hz)
electrode_metrics(spike_times, duration_s, threshold_hz)
summarise_well(well_spike_dict, duration_s, well_id, threshold_hz)
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


# ── Per-electrode functions ───────────────────────────────────────────────────

def mean_firing_rate(
    spike_times: np.ndarray,
    duration_s: float,
) -> float:
    """Compute mean firing rate for one electrode.

    Parameters
    ----------
    spike_times : np.ndarray
        1-D array of spike timestamps in seconds.
    duration_s : float
        Total recording duration in seconds.

    Returns
    -------
    float
        Mean firing rate in Hz (spikes per second).  Returns ``0.0`` if
        *duration_s* is zero or *spike_times* is empty.

    Examples
    --------
    >>> mean_firing_rate(np.array([0.1, 0.2, 0.3]), 1.0)
    3.0
    """
    if duration_s <= 0 or len(spike_times) == 0:
        return 0.0
    return len(spike_times) / duration_s


def isi_array(spike_times: np.ndarray) -> np.ndarray:
    """Compute inter-spike intervals for one electrode.

    Parameters
    ----------
    spike_times : np.ndarray
        1-D array of spike timestamps in seconds.  Need not be sorted,
        but results are most meaningful when it is.

    Returns
    -------
    np.ndarray
        1-D array of ISIs in seconds.  Empty array (``float64``) when
        fewer than two spikes are present.

    Examples
    --------
    >>> isi_array(np.array([0.0, 0.1, 0.3]))
    array([0.1, 0.2])
    """
    if len(spike_times) < 2:
        return np.array([], dtype=np.float64)
    return np.diff(spike_times)


def isi_stats(spike_times: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute ISI summary statistics for one electrode.

    Parameters
    ----------
    spike_times : np.ndarray
        1-D array of spike timestamps in seconds.

    Returns
    -------
    dict
        Keys and values:

        ``mean_isi``
            Mean ISI in seconds; ``None`` if fewer than 2 spikes.
        ``median_isi``
            Median ISI in seconds; ``None`` if fewer than 2 spikes.
        ``cv_isi``
            Coefficient of variation of ISI (std / mean); ``None`` if
            fewer than 2 spikes or mean ISI is zero.  High CV (> 1)
            indicates bursty firing.

    Examples
    --------
    >>> stats = isi_stats(np.array([0.0, 0.1, 0.3, 0.6]))
    >>> round(stats["mean_isi"], 4)
    0.2
    """
    isis = isi_array(spike_times)
    if len(isis) == 0:
        return {"mean_isi": None, "median_isi": None, "cv_isi": None}

    mean = float(np.mean(isis))
    median = float(np.median(isis))
    std = float(np.std(isis, ddof=1)) if len(isis) > 1 else 0.0
    cv = (std / mean) if mean > 0 else None

    return {
        "mean_isi": mean,
        "median_isi": median,
        "cv_isi": cv,
    }


def is_active(
    spike_times: np.ndarray,
    duration_s: float,
    threshold_hz: float = 0.1,
) -> bool:
    """Return whether an electrode exceeds the active firing threshold.

    Parameters
    ----------
    spike_times : np.ndarray
        1-D array of spike timestamps in seconds.
    duration_s : float
        Total recording duration in seconds.
    threshold_hz : float, optional
        Minimum mean firing rate (Hz) to be considered active.
        Default 0.1 Hz matches the AxIS Navigator convention.

    Returns
    -------
    bool
        ``True`` if MFR >= *threshold_hz*.

    Examples
    --------
    >>> is_active(np.array([0.1, 0.2, 0.3]), duration_s=1.0, threshold_hz=0.1)
    True
    >>> is_active(np.array([0.1]), duration_s=100.0, threshold_hz=0.1)
    False
    """
    return mean_firing_rate(spike_times, duration_s) >= threshold_hz


def electrode_metrics(
    spike_times: np.ndarray,
    duration_s: float,
    threshold_hz: float = 0.1,
) -> Dict[str, object]:
    """Compute all spike metrics for a single electrode.

    Combines :func:`mean_firing_rate`, :func:`isi_stats`, and
    :func:`is_active` into one call.

    Parameters
    ----------
    spike_times : np.ndarray
        1-D array of spike timestamps in seconds.
    duration_s : float
        Total recording duration in seconds.
    threshold_hz : float, optional
        Active-electrode MFR threshold in Hz.  Default 0.1 Hz.

    Returns
    -------
    dict
        Keys: ``n_spikes``, ``mfr_hz``, ``mean_isi``, ``median_isi``,
        ``cv_isi``, ``is_active``.

    Examples
    --------
    >>> m = electrode_metrics(np.array([0.0, 0.1, 0.3]), duration_s=1.0)
    >>> m["n_spikes"]
    3
    >>> m["is_active"]
    True
    """
    mfr = mean_firing_rate(spike_times, duration_s)
    stats = isi_stats(spike_times)
    return {
        "n_spikes": int(len(spike_times)),
        "mfr_hz": mfr,
        "mean_isi": stats["mean_isi"],
        "median_isi": stats["median_isi"],
        "cv_isi": stats["cv_isi"],
        "is_active": mfr >= threshold_hz,
    }


# ── Well-level aggregation ────────────────────────────────────────────────────

def summarise_well(
    well_spike_dict: Dict[str, np.ndarray],
    duration_s: float,
    well_id: str,
    threshold_hz: float = 0.1,
) -> pd.DataFrame:
    """Aggregate spike metrics across all electrodes in a well.

    Parameters
    ----------
    well_spike_dict : dict[str, np.ndarray]
        Mapping from electrode ID (e.g. ``'A1_11'``) to a sorted 1-D
        array of spike timestamps in seconds, as returned by
        :func:`~py_mea_axion.io.spk_reader.load_spikes_from_spk`.
    duration_s : float
        Total recording duration in seconds.
    well_id : str
        Well label (e.g. ``'A1'``).  Written into every row of the
        returned DataFrame.
    threshold_hz : float, optional
        Active-electrode MFR threshold in Hz.  Default 0.1 Hz.

    Returns
    -------
    pd.DataFrame
        One row per electrode.  Columns:

        ===============  =========  =======================================
        Column           Dtype      Description
        ===============  =========  =======================================
        well_id          object     Well label
        electrode_id     object     Electrode label (e.g. ``'A1_11'``)
        n_spikes         int64      Total spike count
        mfr_hz           float64    Mean firing rate (Hz)
        mean_isi         float64    Mean ISI (s); NaN if < 2 spikes
        median_isi       float64    Median ISI (s); NaN if < 2 spikes
        cv_isi           float64    CV of ISI; NaN if < 2 spikes
        is_active        bool       MFR >= threshold_hz
        ===============  =========  =======================================

    Examples
    --------
    >>> spikes = {"A1_11": np.array([0.1, 0.2, 0.5]),
    ...           "A1_12": np.array([])}
    >>> df = summarise_well(spikes, duration_s=1.0, well_id="A1")
    >>> len(df)
    2
    >>> df.loc[df["electrode_id"] == "A1_11", "n_spikes"].iloc[0]
    3
    """
    rows = []
    for eid, spike_times in well_spike_dict.items():
        m = electrode_metrics(spike_times, duration_s, threshold_hz)
        rows.append({
            "well_id": well_id,
            "electrode_id": eid,
            "n_spikes": m["n_spikes"],
            "mfr_hz": m["mfr_hz"],
            "mean_isi": m["mean_isi"],
            "median_isi": m["median_isi"],
            "cv_isi": m["cv_isi"],
            "is_active": m["is_active"],
        })

    df = pd.DataFrame(rows, columns=[
        "well_id", "electrode_id", "n_spikes",
        "mfr_hz", "mean_isi", "median_isi", "cv_isi", "is_active",
    ])
    # Replace None with NaN for float columns so arithmetic works cleanly.
    for col in ("mean_isi", "median_isi", "cv_isi"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["n_spikes"] = df["n_spikes"].astype("int64")
    df["is_active"] = df["is_active"].astype(bool)

    return df.reset_index(drop=True)
