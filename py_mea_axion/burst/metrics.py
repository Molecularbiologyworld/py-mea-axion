"""
burst/metrics.py
================
Aggregate ``Burst`` namedtuples into tidy DataFrames and well-level summaries.

The inter-burst interval (IBI) is appended as a column: for burst *i* it
is the gap between the end of burst *i−1* and the start of burst *i*.
The first burst in a train receives ``NaN`` for IBI.

Public API
----------
bursts_to_dataframe(bursts, electrode_id, well_id)
    Convert a list of Burst objects to a per-burst DataFrame.

aggregate_well_bursts(well_burst_dict, well_id)
    Concatenate per-electrode burst DataFrames for a whole well.

well_burst_metrics(well_burst_dict, total_time_s)
    Compute all electrode-burst summary metrics for a well (Category 2).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from py_mea_axion.burst.detection import Burst

_NAN = float("nan")

# Columns emitted by bursts_to_dataframe / aggregate_well_bursts (guaranteed order).
BURST_COLUMNS: List[str] = [
    "electrode_id",
    "well_id",
    "start_time",
    "end_time",
    "duration",
    "n_spikes",
    "mean_isi_within",
    "median_isi_within",
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
        median_isi_within      float64    Median within-burst ISI (s)
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
                "median_isi_within": "float64",
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
                "median_isi_within": float(b.median_isi_within),
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


def well_burst_metrics(
    well_burst_dict: Dict[str, List[Burst]],
    total_time_s: float,
    well_spike_dict: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Compute all electrode-burst summary metrics for a single well.

    Pools burst-level measurements (duration, spike count, within-burst ISI)
    across all electrodes in the well and returns global mean ± std.
    Frequency-based metrics (burst frequency, IBI, burst percentage) are
    computed per electrode and then averaged.

    Parameters
    ----------
    well_burst_dict : dict[str, list of Burst]
        Mapping from electrode ID to its burst list.
    total_time_s : float
        Recording duration in seconds.
    well_spike_dict : dict[str, np.ndarray], optional
        Electrode ID → spike timestamp array for the well.  When provided,
        ``burst_pct`` is computed as the fraction of spikes that fall inside
        bursts (matching NeuralMetric Tools).  When omitted, falls back to
        fraction of recording time spent in bursts.

    Returns
    -------
    dict
        Keys for all 20 Category-2 electrode burst metrics.  All float
        values; counts are int.  NaN is returned for metrics that are
        undefined (e.g. IBI when an electrode has only one burst).
    """
    all_bursts: List[Burst] = []
    bursting_eids: List[str] = []
    electrode_ibis: Dict[str, np.ndarray] = {}

    for eid, bursts in well_burst_dict.items():
        if not bursts:
            continue
        all_bursts.extend(bursts)
        bursting_eids.append(eid)
        n = len(bursts)
        electrode_ibis[eid] = np.array(
            [bursts[i].start_time - bursts[i - 1].end_time for i in range(1, n)],
            dtype=np.float64,
        )

    if not all_bursts:
        return _empty_burst_metrics()

    n_bursts_total = len(all_bursts)
    n_bursting = len(bursting_eids)

    # ── Pool burst-level scalars across all electrodes ────────────────────────
    durations = np.array([b.duration for b in all_bursts])
    n_spk_arr = np.array([b.n_spikes for b in all_bursts], dtype=float)
    mean_isis = np.array([b.mean_isi_within for b in all_bursts])
    median_isis = np.array([b.median_isi_within for b in all_bursts])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = np.where(mean_isis > 0, median_isis / mean_isis, np.nan)

    def _ms(arr: np.ndarray) -> tuple:
        """(mean, std) ignoring NaN; std=0 when only one valid value."""
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return _NAN, _NAN
        m = float(np.mean(valid))
        s = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
        return m, s

    # ── Per-electrode metrics ─────────────────────────────────────────────────
    # Pool all IBIs across all electrodes for global avg/std.
    all_ibis_list: List[float] = []
    bf_vals: List[float] = []
    ibi_cv_vals: List[float] = []
    bpct_vals: List[float] = []

    for eid in bursting_eids:
        bursts = well_burst_dict[eid]
        ibis = electrode_ibis[eid]
        all_ibis_list.extend(ibis.tolist())

        bf_vals.append(len(bursts) / total_time_s if total_time_s > 0 else _NAN)
        spikes_in_bursts = sum(b.n_spikes for b in bursts)
        if well_spike_dict is not None:
            total_spikes = len(well_spike_dict.get(eid, np.array([])))
            bpct_vals.append(
                spikes_in_bursts / total_spikes * 100.0 if total_spikes > 0 else _NAN
            )
        else:
            burst_dur_sum = sum(b.duration for b in bursts)
            bpct_vals.append(
                burst_dur_sum / total_time_s * 100.0 if total_time_s > 0 else _NAN
            )
        if len(ibis) > 1:
            ibi_mean = float(np.mean(ibis))
            ibi_cv_vals.append(
                float(np.std(ibis, ddof=1) / ibi_mean)
                if ibi_mean > 0 else _NAN
            )

    all_ibis = np.array(all_ibis_list, dtype=np.float64)
    bf_arr = np.array(bf_vals)
    bpct_arr = np.array(bpct_vals)
    icv_arr = np.array(ibi_cv_vals) if ibi_cv_vals else np.array([], dtype=np.float64)

    dur_avg, dur_std = _ms(durations)
    nspk_avg, nspk_std = _ms(n_spk_arr)
    misi_avg, misi_std = _ms(mean_isis)
    mdisi_avg, mdisi_std = _ms(median_isis)
    ratio_avg, ratio_std = _ms(ratios)
    ibi_avg, ibi_std = _ms(all_ibis)
    bf_avg, bf_std = _ms(bf_arr)
    icv_avg, icv_std = _ms(icv_arr)
    bpct_avg, bpct_std = _ms(bpct_arr)

    return {
        "n_bursts": n_bursts_total,
        "n_bursting_electrodes": n_bursting,
        "burst_duration_avg": dur_avg,
        "burst_duration_std": dur_std,
        "n_spikes_per_burst_avg": nspk_avg,
        "n_spikes_per_burst_std": nspk_std,
        "mean_isi_within_burst_avg": misi_avg,
        "mean_isi_within_burst_std": misi_std,
        "median_isi_within_burst_avg": mdisi_avg,
        "median_isi_within_burst_std": mdisi_std,
        "median_mean_isi_ratio_burst_avg": ratio_avg,
        "median_mean_isi_ratio_burst_std": ratio_std,
        "ibi_avg": ibi_avg,
        "ibi_std": ibi_std,
        "burst_freq_avg": bf_avg,
        "burst_freq_std": bf_std,
        "ibi_cv_avg": icv_avg,
        "ibi_cv_std": icv_std,
        "burst_pct_avg": bpct_avg,
        "burst_pct_std": bpct_std,
    }


def _empty_burst_metrics() -> Dict[str, Any]:
    """Return all-NaN burst metrics dict for wells with no bursting electrodes."""
    return {
        "n_bursts": 0,
        "n_bursting_electrodes": 0,
        "burst_duration_avg": _NAN,
        "burst_duration_std": _NAN,
        "n_spikes_per_burst_avg": _NAN,
        "n_spikes_per_burst_std": _NAN,
        "mean_isi_within_burst_avg": _NAN,
        "mean_isi_within_burst_std": _NAN,
        "median_isi_within_burst_avg": _NAN,
        "median_isi_within_burst_std": _NAN,
        "median_mean_isi_ratio_burst_avg": _NAN,
        "median_mean_isi_ratio_burst_std": _NAN,
        "ibi_avg": _NAN,
        "ibi_std": _NAN,
        "burst_freq_avg": _NAN,
        "burst_freq_std": _NAN,
        "ibi_cv_avg": _NAN,
        "ibi_cv_std": _NAN,
        "burst_pct_avg": _NAN,
        "burst_pct_std": _NAN,
    }
