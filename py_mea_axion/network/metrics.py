"""
network/metrics.py
==================
Well-level network burst metrics (Categories 3 and 5).

Computes all NeuralMetric-equivalent network burst summary statistics from
a list of :class:`~py_mea_axion.network.detection.NetworkBurst` objects and
the raw per-electrode spike dictionaries.

Public API
----------
network_burst_metrics(network_bursts, well_spike_dict, total_time_s)
    Compute all Category-3 network burst metrics plus Category-5 average
    network burst metrics for a single well.
"""

from typing import Any, Dict, List

import numpy as np

from py_mea_axion.network.detection import NetworkBurst

_NAN = float("nan")


def network_burst_metrics(
    network_bursts: List[NetworkBurst],
    well_spike_dict: Dict[str, np.ndarray],
    total_time_s: float,
) -> Dict[str, Any]:
    """Compute all network burst summary metrics for a single well.

    Covers NeuralMetric Categories 3 (Network Burst Metrics) and 5
    (Average Network Burst Metrics).

    Parameters
    ----------
    network_bursts : list of NetworkBurst
        Detected network bursts sorted by ``start_time``, as returned by
        :func:`~py_mea_axion.network.detection.detect_network_bursts_combined_isi`.
    well_spike_dict : dict[str, np.ndarray]
        Electrode ID → sorted spike timestamp array (s) for the well.
    total_time_s : float
        Recording duration in seconds.

    Returns
    -------
    dict
        12 keys: 11 Category-3 metrics and 1 Category-5 metric.
        ``n_network_bursts`` is always an int (0 when no network bursts).
        All other numeric values are float; NaN when undefined.
        ``nb_start_electrode`` is a string electrode ID or NaN.
    """
    if not network_bursts:
        return _empty_nb_metrics()

    n_nb = len(network_bursts)
    nb_freq = n_nb / total_time_s if total_time_s > 0 else _NAN

    durations = np.array([nb.duration for nb in network_bursts])
    dur_avg = float(np.mean(durations))

    # ── Per-network-burst metrics ─────────────────────────────────────────────
    n_spikes_list: List[float] = []
    mean_isis_list: List[float] = []
    median_isis_list: List[float] = []
    n_elecs_list: List[float] = []
    spk_per_ch_list: List[float] = []

    for nb in network_bursts:
        combined = _collect_spikes_in_window(
            well_spike_dict, nb.start_time, nb.end_time
        )
        n_spk = len(combined)
        n_spikes_list.append(float(n_spk))

        if n_spk > 1:
            isis = np.diff(combined)
            mean_isis_list.append(float(np.mean(isis)))
            median_isis_list.append(float(np.median(isis)))
        else:
            mean_isis_list.append(_NAN)
            median_isis_list.append(_NAN)

        n_elecs = len(nb.participating_electrodes)
        n_elecs_list.append(float(n_elecs))
        spk_per_ch_list.append(n_spk / n_elecs if n_elecs > 0 else _NAN)

    n_spikes_arr = np.array(n_spikes_list)
    mean_isis_arr = np.array(mean_isis_list)
    median_isis_arr = np.array(median_isis_list)
    n_elecs_arr = np.array(n_elecs_list)
    spk_per_ch_arr = np.array(spk_per_ch_list)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = np.where(mean_isis_arr > 0, median_isis_arr / mean_isis_arr, np.nan)

    # ── Network burst percentage (spike-based, matching NeuralMetric) ────────
    total_spikes_all = sum(len(v) for v in well_spike_dict.values())
    if total_spikes_all > 0:
        spikes_in_nbs = sum(
            len(_collect_spikes_in_window(well_spike_dict, nb.start_time, nb.end_time))
            for nb in network_bursts
        )
        nb_pct = float(spikes_in_nbs / total_spikes_all * 100.0)
    else:
        nb_pct = _NAN

    # ── Network IBI CV ────────────────────────────────────────────────────────
    if n_nb > 1:
        ibis = np.array([
            network_bursts[i].start_time - network_bursts[i - 1].end_time
            for i in range(1, n_nb)
        ])
        ibi_mean = float(np.mean(ibis))
        network_ibi_cv = (
            float(np.std(ibis, ddof=1) / ibi_mean)
            if ibi_mean > 0 else _NAN
        )
    else:
        network_ibi_cv = _NAN

    def _agg(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return _NAN
        return float(np.mean(valid))

    nspk_avg = _agg(n_spikes_arr)
    misi_avg = _agg(mean_isis_arr)
    mdisi_avg = _agg(median_isis_arr)
    ratio_avg = _agg(ratios)
    nelec_avg = _agg(n_elecs_arr)
    spkch_avg = _agg(spk_per_ch_arr)

    # ── Category 5: leader electrode ─────────────────────────────────────────
    start_elec = _nb_leader(network_bursts, well_spike_dict)

    return {
        # Category 3
        "n_network_bursts": n_nb,
        "network_burst_freq": nb_freq,
        "network_burst_duration_avg": dur_avg,
        "n_spikes_per_nb_avg": nspk_avg,
        "mean_isi_within_nb_avg": misi_avg,
        "median_isi_within_nb_avg": mdisi_avg,
        "median_mean_isi_ratio_nb_avg": ratio_avg,
        "n_elecs_per_nb_avg": nelec_avg,
        "n_spikes_per_nb_per_channel_avg": spkch_avg,
        "network_burst_pct": nb_pct,
        "network_ibi_cv": network_ibi_cv,
        # Category 5
        "nb_start_electrode": start_elec,
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _collect_spikes_in_window(
    well_spike_dict: Dict[str, np.ndarray],
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """Return a sorted array of all spikes from all electrodes in [t_start, t_end]."""
    parts = [
        ts[(ts >= t_start) & (ts <= t_end)]
        for ts in well_spike_dict.values()
        if len(ts) > 0
    ]
    if not parts:
        return np.array([], dtype=np.float64)
    return np.sort(np.concatenate(parts))


def _nb_leader(
    network_bursts: List[NetworkBurst],
    well_spike_dict: Dict[str, np.ndarray],
) -> Any:
    """Find the electrode that fires first most often across network bursts.

    Returns
    -------
    start_electrode_id : str or float
        Electrode ID string, or ``nan`` when no spikes are found in any NB.
    """
    counts: Dict[str, int] = {}

    for nb in network_bursts:
        first_t = float("inf")
        first_eid = None
        for eid, ts in well_spike_dict.items():
            in_nb = ts[(ts >= nb.start_time) & (ts <= nb.end_time)]
            if len(in_nb) > 0 and in_nb[0] < first_t:
                first_t = in_nb[0]
                first_eid = eid
        if first_eid is not None:
            counts[first_eid] = counts.get(first_eid, 0) + 1

    if not counts:
        return _NAN

    return max(counts, key=counts.__getitem__)


def _empty_nb_metrics() -> Dict[str, Any]:
    """Return all-NaN network burst metrics for wells with no network bursts."""
    return {
        "n_network_bursts": 0,
        "network_burst_freq": _NAN,
        "network_burst_duration_avg": _NAN,
        "n_spikes_per_nb_avg": _NAN,
        "mean_isi_within_nb_avg": _NAN,
        "median_isi_within_nb_avg": _NAN,
        "median_mean_isi_ratio_nb_avg": _NAN,
        "n_elecs_per_nb_avg": _NAN,
        "n_spikes_per_nb_per_channel_avg": _NAN,
        "network_burst_pct": 0.0,
        "network_ibi_cv": _NAN,
        "nb_start_electrode": _NAN,
    }
