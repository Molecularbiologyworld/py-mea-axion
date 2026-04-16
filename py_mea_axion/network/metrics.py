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

from typing import Any, Dict, List, Tuple

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
        Keys for all 19 Category-3 metrics and 4 Category-5 metrics.
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
    dur_std = float(np.std(durations, ddof=1)) if n_nb > 1 else 0.0

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

    # ── Network burst percentage ──────────────────────────────────────────────
    nb_pct = (
        float(np.sum(durations) / total_time_s * 100.0)
        if total_time_s > 0 else _NAN
    )

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

    # ── Network normalised duration IQR ───────────────────────────────────────
    if n_nb > 1:
        mean_dur = float(np.mean(durations))
        if mean_dur > 0:
            q75, q25 = np.percentile(durations / mean_dur, [75, 25])
            network_norm_dur_iqr = float(q75 - q25)
        else:
            network_norm_dur_iqr = _NAN
    else:
        network_norm_dur_iqr = _NAN

    def _agg(arr: np.ndarray) -> Tuple[float, float]:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return _NAN, _NAN
        m = float(np.mean(valid))
        s = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
        return m, s

    nspk_avg, nspk_std = _agg(n_spikes_arr)
    misi_avg, misi_std = _agg(mean_isis_arr)
    mdisi_avg, mdisi_std = _agg(median_isis_arr)
    ratio_avg, ratio_std = _agg(ratios)
    nelec_avg, nelec_std = _agg(n_elecs_arr)
    spkch_avg, spkch_std = _agg(spk_per_ch_arr)

    # ── Category 5: leader electrode and burst peak ───────────────────────────
    start_elec, pct_start = _nb_leader(network_bursts, well_spike_dict)
    peak_rate, time_to_peak = _nb_peak(network_bursts, well_spike_dict)

    return {
        # Category 3
        "n_network_bursts": n_nb,
        "network_burst_freq": nb_freq,
        "network_burst_duration_avg": dur_avg,
        "network_burst_duration_std": dur_std,
        "n_spikes_per_nb_avg": nspk_avg,
        "n_spikes_per_nb_std": nspk_std,
        "mean_isi_within_nb_avg": misi_avg,
        "mean_isi_within_nb_std": misi_std,
        "median_isi_within_nb_avg": mdisi_avg,
        "median_isi_within_nb_std": mdisi_std,
        "median_mean_isi_ratio_nb_avg": ratio_avg,
        "median_mean_isi_ratio_nb_std": ratio_std,
        "n_elecs_per_nb_avg": nelec_avg,
        "n_elecs_per_nb_std": nelec_std,
        "n_spikes_per_nb_per_channel_avg": spkch_avg,
        "n_spikes_per_nb_per_channel_std": spkch_std,
        "network_burst_pct": nb_pct,
        "network_ibi_cv": network_ibi_cv,
        "network_normalized_duration_iqr": network_norm_dur_iqr,
        # Category 5
        "nb_start_electrode": start_elec,
        "nb_pct_bursts_with_start_electrode": pct_start,
        "nb_burst_peak_spikes_per_s": peak_rate,
        "nb_time_to_peak_ms": time_to_peak,
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
) -> Tuple[Any, float]:
    """Find the electrode that fires first most often across network bursts.

    Returns
    -------
    (start_electrode_id, pct_bursts_with_start_electrode)
        ``(nan, nan)`` when no spikes are found in any network burst.
    """
    counts: Dict[str, int] = {}
    n_valid = 0

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
            n_valid += 1

    if not counts:
        return _NAN, _NAN

    start_elec = max(counts, key=counts.__getitem__)
    pct = counts[start_elec] / n_valid * 100.0
    return start_elec, float(pct)


def _nb_peak(
    network_bursts: List[NetworkBurst],
    well_spike_dict: Dict[str, np.ndarray],
    bin_s: float = 0.025,
) -> Tuple[float, float]:
    """Compute mean peak spike rate and mean time-to-peak across network bursts.

    Returns
    -------
    (mean_peak_spikes_per_s, mean_time_to_peak_ms)
        Both ``nan`` when no valid network bursts are found.
    """
    peaks: List[float] = []
    times: List[float] = []

    for nb in network_bursts:
        duration = nb.end_time - nb.start_time
        if duration <= 0:
            continue

        combined = _collect_spikes_in_window(
            well_spike_dict, nb.start_time, nb.end_time
        )
        if len(combined) == 0:
            continue

        n_bins = max(1, int(np.ceil(duration / bin_s)))
        edges = np.linspace(nb.start_time, nb.end_time, n_bins + 1)
        counts, edges_out = np.histogram(combined, bins=edges)
        rates = counts / bin_s   # spikes per second

        peak_idx = int(np.argmax(rates))
        peak_rate = float(rates[peak_idx])
        # Time from NB start to centre of peak bin
        t_peak_ms = float((edges_out[peak_idx] + bin_s / 2.0 - nb.start_time) * 1000.0)

        peaks.append(peak_rate)
        times.append(t_peak_ms)

    if not peaks:
        return _NAN, _NAN

    return float(np.mean(peaks)), float(np.mean(times))


def _empty_nb_metrics() -> Dict[str, Any]:
    """Return all-NaN network burst metrics for wells with no network bursts."""
    return {
        "n_network_bursts": 0,
        "network_burst_freq": _NAN,
        "network_burst_duration_avg": _NAN,
        "network_burst_duration_std": _NAN,
        "n_spikes_per_nb_avg": _NAN,
        "n_spikes_per_nb_std": _NAN,
        "mean_isi_within_nb_avg": _NAN,
        "mean_isi_within_nb_std": _NAN,
        "median_isi_within_nb_avg": _NAN,
        "median_isi_within_nb_std": _NAN,
        "median_mean_isi_ratio_nb_avg": _NAN,
        "median_mean_isi_ratio_nb_std": _NAN,
        "n_elecs_per_nb_avg": _NAN,
        "n_elecs_per_nb_std": _NAN,
        "n_spikes_per_nb_per_channel_avg": _NAN,
        "n_spikes_per_nb_per_channel_std": _NAN,
        "network_burst_pct": 0.0,
        "network_ibi_cv": _NAN,
        "network_normalized_duration_iqr": _NAN,
        "nb_start_electrode": _NAN,
        "nb_pct_bursts_with_start_electrode": _NAN,
        "nb_burst_peak_spikes_per_s": _NAN,
        "nb_time_to_peak_ms": _NAN,
    }
