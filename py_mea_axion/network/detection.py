"""
network/detection.py
====================
Network burst detection across multiple electrodes.

Two algorithms are provided:

``combined_isi`` (matches NeuralMetric Tools)
    All electrode spike trains are merged into a single combined train.
    ISI-threshold burst detection is run on the combined train
    (``max_isi_s``, ``min_spikes`` refer to the combined train).
    Detected bursts are then filtered by electrode participation.

``participation_threshold``
    Works on a discretised time grid:
    1. For each electrode, build a binary burst mask.
    2. Sum masks → participation count per bin.
    3. Threshold at ``ceil(participation_threshold × n_active_electrodes)``.
    4. Identify contiguous runs above the threshold.
    5. Merge runs closer than *min_network_ibi_s*.

Public API
----------
NetworkBurst  (namedtuple)
detect_network_bursts(well_burst_dict, ...)          — participation_threshold algorithm
detect_network_bursts_combined_isi(well_spike_dict, ...)  — combined ISI algorithm
"""

import math
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import numpy as np

from py_mea_axion.burst.detection import Burst

# ── NetworkBurst namedtuple ───────────────────────────────────────────────────

NetworkBurst = namedtuple(
    "NetworkBurst",
    [
        "start_time",               # float  — onset of network burst (s)
        "end_time",                 # float  — offset of network burst (s)
        "duration",                 # float  — end − start (s)
        "participating_electrodes", # list[str] — electrode IDs active during burst
        "participation_fraction",   # float  — fraction of active electrodes
        "peak_participation",       # float  — max instantaneous fraction
    ],
)


# ── Public entry point ────────────────────────────────────────────────────────

def detect_network_bursts(
    well_burst_dict: Dict[str, List[Burst]],
    total_time_s: float,
    participation_threshold: float = 0.25,
    bin_size_s: float = 0.010,
    min_network_ibi_s: float = 1.0,
    min_electrodes: int = 2,
    extend_to_burst_envelope: bool = True,
) -> List[NetworkBurst]:
    """Detect network bursts across electrodes in a single well.

    Parameters
    ----------
    well_burst_dict : dict[str, list of Burst]
        Mapping from electrode ID to its burst list, as returned by
        :func:`~py_mea_axion.burst.detection.detect_bursts` for each
        electrode.  Silent electrodes (empty burst lists) are included
        in the active-electrode count only if they appear in the dict
        *and* have at least one burst.
    total_time_s : float
        Recording duration in seconds.  Sets the length of the time grid.
    participation_threshold : float, optional
        Fraction of active electrodes that must be simultaneously bursting
        for a bin to be considered part of a network burst.  Default 0.25.
    bin_size_s : float, optional
        Time grid resolution in seconds.  Default 0.010 s (10 ms).
    min_network_ibi_s : float, optional
        Minimum gap between network bursts in seconds.  Adjacent detected
        events separated by less than this are merged.  Default 1.0 s.
    min_electrodes : int, optional
        Minimum number of active electrodes required to attempt network
        burst detection.  Returns ``[]`` if fewer are present.
        Default 2.
    extend_to_burst_envelope : bool, optional
        If ``True`` (default), extend each detected network burst's
        start/end times to the earliest start and latest end of all
        electrode bursts that overlap with the participation-threshold
        epoch.  This matches NeuralMetric Tools' convention where
        network burst duration spans the full burst envelope rather than
        just the high-participation core.

    Returns
    -------
    list of NetworkBurst
        Detected network bursts sorted by ``start_time``.  Empty list if
        none are found or the well has too few active electrodes.

    Notes
    -----
    An electrode is *active* in this context if it has at least one burst
    (i.e. its entry in *well_burst_dict* is non-empty).

    Examples
    --------
    >>> from py_mea_axion.burst.detection import detect_bursts
    >>> import numpy as np
    >>> spikes_a = np.array([0.01, 0.02, 0.03, 0.04, 0.05,
    ...                      5.01, 5.02, 5.03, 5.04, 5.05])
    >>> spikes_b = np.array([0.01, 0.02, 0.03, 0.04, 0.05,
    ...                      5.01, 5.02, 5.03, 5.04, 5.05])
    >>> bursts = {"e1": detect_bursts(spikes_a),
    ...           "e2": detect_bursts(spikes_b)}
    >>> nb = detect_network_bursts(bursts, total_time_s=10.0)
    >>> len(nb)
    2
    """
    # Only electrodes with at least one burst count as active.
    active = {eid: bl for eid, bl in well_burst_dict.items() if bl}
    n_active = len(active)

    if n_active < min_electrodes:
        return []

    threshold_count = math.ceil(participation_threshold * n_active)

    # Build time grid.
    n_bins = math.ceil(total_time_s / bin_size_s) + 1
    participation = np.zeros(n_bins, dtype=np.int32)

    # For each electrode, mark bins that fall inside a burst.
    electrode_masks: Dict[str, np.ndarray] = {}
    for eid, burst_list in active.items():
        mask = np.zeros(n_bins, dtype=np.bool_)
        for b in burst_list:
            i_start = int(b.start_time / bin_size_s)
            i_end = min(int(b.end_time / bin_size_s) + 1, n_bins)
            mask[i_start:i_end] = True
        electrode_masks[eid] = mask
        participation += mask.astype(np.int32)

    # Find bins that meet the threshold.
    above = participation >= threshold_count

    # Find contiguous runs of above-threshold bins.
    runs = _find_runs(above)

    if not runs:
        return []

    # Convert bin indices to times.
    run_times = [
        (start * bin_size_s, (end - 1) * bin_size_s)
        for start, end in runs
    ]

    # Merge runs separated by less than min_network_ibi_s.
    run_times = _merge_run_times(run_times, min_network_ibi_s)

    # Build NetworkBurst namedtuples.
    network_bursts: List[NetworkBurst] = []
    for t_start, t_end in run_times:
        # Optionally extend to the full envelope of overlapping electrode bursts.
        if extend_to_burst_envelope:
            t_start, t_end = _extend_to_envelope(t_start, t_end, active)

        participating, peak_frac = _participation_stats(
            t_start, t_end, electrode_masks, n_active, bin_size_s
        )
        overall_frac = len(participating) / n_active
        network_bursts.append(
            NetworkBurst(
                start_time=t_start,
                end_time=t_end,
                duration=t_end - t_start,
                participating_electrodes=participating,
                participation_fraction=overall_frac,
                peak_participation=peak_frac,
            )
        )

    return network_bursts


# ── Private helpers ───────────────────────────────────────────────────────────

def _extend_to_envelope(
    t_start: float,
    t_end: float,
    active: Dict[str, List[Burst]],
) -> Tuple[float, float]:
    """Expand [t_start, t_end] to the full extent of overlapping electrode bursts.

    Any electrode burst whose interval overlaps with [t_start, t_end] is
    included; the returned interval is the union of all such bursts.
    """
    env_start = t_start
    env_end = t_end
    for burst_list in active.values():
        for b in burst_list:
            if b.end_time >= t_start and b.start_time <= t_end:
                if b.start_time < env_start:
                    env_start = b.start_time
                if b.end_time > env_end:
                    env_end = b.end_time
    return env_start, env_end


def _find_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` index pairs for contiguous True runs.

    *end* is exclusive (Python slice convention).
    """
    runs = []
    in_run = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(mask)))
    return runs


def _merge_run_times(
    run_times: List[Tuple[float, float]],
    min_ibi_s: float,
) -> List[Tuple[float, float]]:
    """Merge consecutive (start, end) time pairs closer than *min_ibi_s*."""
    merged = [run_times[0]]
    for t_start, t_end in run_times[1:]:
        prev_start, prev_end = merged[-1]
        if t_start - prev_end < min_ibi_s:
            merged[-1] = (prev_start, max(prev_end, t_end))
        else:
            merged.append((t_start, t_end))
    return merged


def _participation_stats(
    t_start: float,
    t_end: float,
    electrode_masks: Dict[str, np.ndarray],
    n_active: int,
    bin_size_s: float,
) -> Tuple[List[str], float]:
    """Return (participating_electrode_ids, peak_participation_fraction).

    An electrode is considered participating if its burst mask is True
    for *any* bin within [t_start, t_end].
    """
    i_start = int(t_start / bin_size_s)
    i_end = int(t_end / bin_size_s) + 1

    participating = []
    peak_count = 0

    # Sum participation per bin within the window.
    window_counts = np.zeros(max(i_end - i_start, 1), dtype=np.int32)
    for eid, mask in electrode_masks.items():
        segment = mask[i_start:i_end]
        if segment.any():
            participating.append(eid)
        window_counts[:len(segment)] += segment[:len(window_counts)].astype(np.int32)

    peak_count = int(window_counts.max()) if len(window_counts) > 0 else 0
    peak_frac = peak_count / n_active if n_active > 0 else 0.0

    return sorted(participating), peak_frac


# ── Combined-ISI algorithm (matches NeuralMetric Tools) ──────────────────────

def detect_network_bursts_combined_isi(
    well_spike_dict: Dict[str, np.ndarray],
    total_time_s: float,
    max_isi_s: float = 0.1,
    min_spikes: int = 50,
    min_ibi_s: float = 0.0,
    participation_threshold: float = 0.35,
    min_electrodes: int = 2,
) -> List[NetworkBurst]:
    """Detect network bursts using a combined-spike-train ISI threshold.

    This algorithm matches **NeuralMetric Tools** behaviour:

    1. All electrode spike trains are concatenated and sorted into a single
       combined spike train.
    2. :func:`~py_mea_axion.burst.detection.detect_bursts` (ISI-threshold) is
       run on the combined train.  *min_spikes* and *max_isi_s* refer to the
       **combined** train.
    3. Each candidate is post-filtered: it is kept only if at least
       ``ceil(participation_threshold × n_electrodes)`` individual electrodes
       have at least one spike within the candidate window.

    Parameters
    ----------
    well_spike_dict : dict[str, np.ndarray]
        Mapping from electrode ID to sorted spike-time array (seconds).
    total_time_s : float
        Recording duration in seconds (used only for consistency with the
        other algorithm's signature; not consumed directly here).
    max_isi_s : float, optional
        Maximum inter-spike interval in the combined train.  Default 0.1 s.
    min_spikes : int, optional
        Minimum number of spikes (across all electrodes) in a network burst.
        Default 50, matching NeuralMetric Tools.
    min_ibi_s : float, optional
        Minimum inter-network-burst interval.  Default 0.0 s (no merging).
    participation_threshold : float, optional
        Minimum fraction of electrodes that must contribute at least one spike
        within the burst window.  Default 0.35.
    min_electrodes : int, optional
        Minimum number of electrodes required.  Default 2.

    Returns
    -------
    list of NetworkBurst
        Detected network bursts sorted by ``start_time``.
    """
    from py_mea_axion.burst.detection import detect_bursts

    electrodes = {eid: ts for eid, ts in well_spike_dict.items() if len(ts) > 0}
    n_electrodes = len(electrodes)

    if n_electrodes < min_electrodes:
        return []

    threshold_count = math.ceil(participation_threshold * n_electrodes)

    combined = np.sort(np.concatenate(list(electrodes.values())))

    candidates = detect_bursts(
        combined, max_isi_s=max_isi_s, min_spikes=min_spikes, min_ibi_s=min_ibi_s
    )

    network_bursts: List[NetworkBurst] = []
    for burst in candidates:
        t_start, t_end = burst.start_time, burst.end_time

        # Count participating electrodes (at least one spike in [t_start, t_end]).
        participating = [
            eid for eid, ts in electrodes.items()
            if np.any((ts >= t_start) & (ts <= t_end))
        ]
        if len(participating) < threshold_count:
            continue

        peak_frac = len(participating) / n_electrodes
        network_bursts.append(
            NetworkBurst(
                start_time=t_start,
                end_time=t_end,
                duration=t_end - t_start,
                participating_electrodes=sorted(participating),
                participation_fraction=peak_frac,
                peak_participation=peak_frac,
            )
        )

    return network_bursts
