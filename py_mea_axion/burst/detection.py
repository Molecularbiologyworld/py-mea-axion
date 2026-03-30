"""
burst/detection.py
==================
Burst detection from spike timestamp arrays.

Two algorithms are implemented:

``isi_threshold`` (default)
    A burst begins when the inter-spike interval drops below
    *max_isi_s* (default 100 ms, matching the AxIS Navigator default).
    Consecutive bursts separated by less than *min_ibi_s* are merged.
    Burst candidates with fewer than *min_spikes* spikes are discarded.

``poisson_surprise``
    The Poisson Surprise method of Legendy & Salcman (1985).  Each
    candidate burst is scored by S = −ln P, where P is the probability
    of observing ≥ n spikes in the burst duration by chance given the
    electrode's mean firing rate.  Bursts with S < *surprise_threshold*
    are discarded.

Public API
----------
Burst  (namedtuple)
detect_bursts(spike_times, ...)
"""

import math
from collections import namedtuple
from typing import List, Optional

import numpy as np
from scipy.stats import poisson as _poisson_dist

# ── Burst namedtuple ──────────────────────────────────────────────────────────

Burst = namedtuple(
    "Burst",
    [
        "start_time",       # float  — first spike timestamp (s)
        "end_time",         # float  — last spike timestamp (s)
        "spike_times",      # ndarray — timestamps of all spikes in burst (s)
        "n_spikes",         # int    — spike count
        "duration",         # float  — end_time − start_time (s)
        "mean_isi_within",  # float  — mean ISI within the burst (s)
    ],
)


# ── Public entry point ────────────────────────────────────────────────────────

def detect_bursts(
    spike_times: np.ndarray,
    max_isi_s: float = 0.1,
    min_spikes: int = 5,
    min_ibi_s: float = 0.2,
    algorithm: str = "isi_threshold",
    surprise_threshold: float = 2.0,
) -> List[Burst]:
    """Detect bursts in a single electrode's spike train.

    Parameters
    ----------
    spike_times : np.ndarray
        1-D array of spike timestamps in **seconds**, sorted ascending.
    max_isi_s : float, optional
        ISI-threshold algorithm: maximum within-burst inter-spike interval
        in seconds.  Default 0.1 s (100 ms), matching AxIS Navigator.
        Poisson Surprise algorithm: used as the initial candidate ISI
        threshold (defaults to 1 / mean_firing_rate if not overridden).
    min_spikes : int, optional
        Minimum number of spikes required for a valid burst.  Default 5.
    min_ibi_s : float, optional
        Minimum inter-burst interval in seconds.  Pairs of bursts with a
        gap smaller than this are merged.  Default 0.2 s (200 ms).
    algorithm : {'isi_threshold', 'poisson_surprise'}, optional
        Burst detection algorithm to use.  Default ``'isi_threshold'``.
    surprise_threshold : float, optional
        Poisson Surprise only.  Minimum surprise score S = −ln P to
        accept a burst candidate.  Default 2.0 (≈ p < 0.135).

    Returns
    -------
    list of Burst
        Detected bursts sorted by ``start_time``.  Empty list when no
        bursts are found or the spike train is too short.

    Raises
    ------
    ValueError
        If *algorithm* is not one of the supported values.

    Examples
    --------
    >>> spikes = np.array([0.01, 0.02, 0.03, 0.04, 0.05,   # burst 1
    ...                    2.00, 2.01, 2.02, 2.03, 2.04])  # burst 2
    >>> bursts = detect_bursts(spikes, max_isi_s=0.1, min_spikes=5)
    >>> len(bursts)
    2
    >>> round(bursts[0].start_time, 3)
    0.01
    """
    spike_times = np.asarray(spike_times, dtype=np.float64)

    if algorithm == "isi_threshold":
        return _detect_isi_threshold(spike_times, max_isi_s, min_spikes, min_ibi_s)
    elif algorithm == "poisson_surprise":
        return _detect_poisson_surprise(
            spike_times, max_isi_s, min_spikes, min_ibi_s, surprise_threshold
        )
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            "Choose 'isi_threshold' or 'poisson_surprise'."
        )


# ── ISI-threshold algorithm ───────────────────────────────────────────────────

def _detect_isi_threshold(
    spike_times: np.ndarray,
    max_isi_s: float,
    min_spikes: int,
    min_ibi_s: float,
) -> List[Burst]:
    """Core ISI-threshold burst detection."""
    if len(spike_times) < min_spikes:
        return []

    # Step 1 — find runs of consecutive spikes with ISI < max_isi_s.
    groups = _find_linked_groups(spike_times, max_isi_s)

    # Step 2 — discard groups with too few spikes.
    groups = [(s, e) for s, e in groups if (e - s + 1) >= min_spikes]

    if not groups:
        return []

    # Step 3 — merge groups whose gap is less than min_ibi_s.
    groups = _merge_groups(groups, spike_times, min_ibi_s)

    # Step 4 — re-filter after merging (merging can create valid-sized bursts
    # from small fragments, but also discard previously valid ones split
    # across a merge boundary — re-check is cheap).
    groups = [(s, e) for s, e in groups if (e - s + 1) >= min_spikes]

    # Step 5 — build Burst namedtuples.
    return [_make_burst(spike_times, s, e) for s, e in groups]


def _find_linked_groups(
    spike_times: np.ndarray,
    max_isi_s: float,
) -> List[tuple]:
    """Return ``(start_idx, end_idx)`` pairs for runs of short-ISI spikes."""
    isis = np.diff(spike_times)
    links = isis < max_isi_s        # links[i] = True ↔ spikes i & i+1 are paired

    groups = []
    i = 0
    n = len(links)
    while i < n:
        if links[i]:
            j = i + 1
            while j < n and links[j]:
                j += 1
            # Spikes i through j form a burst (j - i + 1 spikes).
            groups.append((i, j))
            i = j
        else:
            i += 1

    return groups


def _merge_groups(
    groups: List[tuple],
    spike_times: np.ndarray,
    min_ibi_s: float,
) -> List[tuple]:
    """Merge adjacent groups whose gap (IBI) is less than *min_ibi_s*."""
    merged = [groups[0]]
    for start, end in groups[1:]:
        prev_start, prev_end = merged[-1]
        gap = spike_times[start] - spike_times[prev_end]
        if gap < min_ibi_s:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _make_burst(spike_times: np.ndarray, start_idx: int, end_idx: int) -> Burst:
    """Construct a Burst namedtuple from a spike index range."""
    burst_spikes = spike_times[start_idx : end_idx + 1]
    isis = np.diff(burst_spikes)
    mean_isi = float(np.mean(isis)) if len(isis) > 0 else float("nan")
    return Burst(
        start_time=float(burst_spikes[0]),
        end_time=float(burst_spikes[-1]),
        spike_times=burst_spikes,
        n_spikes=len(burst_spikes),
        duration=float(burst_spikes[-1] - burst_spikes[0]),
        mean_isi_within=mean_isi,
    )


# ── Poisson Surprise algorithm ────────────────────────────────────────────────

def _detect_poisson_surprise(
    spike_times: np.ndarray,
    max_isi_s: float,
    min_spikes: int,
    min_ibi_s: float,
    surprise_threshold: float,
) -> List[Burst]:
    """Poisson Surprise burst detection (Legendy & Salcman, 1985).

    Strategy
    --------
    1. Estimate the baseline mean firing rate λ from the full spike train.
    2. Use *max_isi_s* (or 1/λ if max_isi_s is larger) to seed initial
       burst candidates — the same linked-group logic as the ISI-threshold
       method.
    3. For each candidate, greedily extend the burst window by one spike
       at a time (both forward and backward) as long as the surprise score
       S = −ln P(X ≥ n | Poisson(λ·T)) increases.
    4. Keep bursts with S ≥ *surprise_threshold* and n ≥ *min_spikes*.
    5. Merge overlapping or adjacent bursts and re-filter.
    """
    if len(spike_times) < min_spikes:
        return []

    total_duration = float(spike_times[-1] - spike_times[0])
    if total_duration <= 0:
        return []

    mean_rate = (len(spike_times) - 1) / total_duration  # spikes per second

    # Seed threshold: use the smaller of max_isi_s and 1/mean_rate.
    if mean_rate > 0:
        seed_isi = min(max_isi_s, 1.0 / mean_rate)
    else:
        seed_isi = max_isi_s

    candidates = _find_linked_groups(spike_times, seed_isi)
    candidates = [(s, e) for s, e in candidates if (e - s + 1) >= min_spikes]

    if not candidates:
        return []

    n_spikes_total = len(spike_times)
    refined: List[tuple] = []

    for seed_start, seed_end in candidates:
        best_s, best_start, best_end = _optimise_surprise(
            spike_times, seed_start, seed_end, mean_rate, n_spikes_total
        )
        if best_s >= surprise_threshold and (best_end - best_start + 1) >= min_spikes:
            refined.append((best_start, best_end))

    if not refined:
        return []

    # Remove overlapping bursts (keep higher-surprise one — greedy).
    refined = _remove_overlaps(refined, spike_times, mean_rate)
    refined = _merge_groups(refined, spike_times, min_ibi_s)
    refined = [(s, e) for s, e in refined if (e - s + 1) >= min_spikes]

    return [_make_burst(spike_times, s, e) for s, e in refined]


def _poisson_surprise_score(n: int, duration: float, mean_rate: float) -> float:
    """Compute S = −ln P(X ≥ n | Poisson(λ·T)).

    Returns ``−inf`` for degenerate inputs.
    """
    if duration <= 0 or mean_rate <= 0 or n < 1:
        return -math.inf
    mu = mean_rate * duration
    # sf(k) = P(X > k) = P(X ≥ k+1); so sf(n-1) = P(X ≥ n).
    p = float(_poisson_dist.sf(n - 1, mu))
    if p <= 0:
        return 700.0  # cap at −ln(~0) ≈ very large
    return -math.log(p)


def _optimise_surprise(
    spike_times: np.ndarray,
    seed_start: int,
    seed_end: int,
    mean_rate: float,
    n_total: int,
) -> tuple:
    """Greedily extend a candidate burst to maximise surprise score.

    Tries extending by one spike forward and one spike backward at each
    step; accepts the extension that most increases S.  Stops when no
    extension improves S.

    Returns
    -------
    (best_s, best_start, best_end) : tuple[float, int, int]
    """
    cur_start, cur_end = seed_start, seed_end

    def score(s, e):
        T = spike_times[e] - spike_times[s]
        n = e - s + 1
        return _poisson_surprise_score(n, T, mean_rate)

    best_s = score(cur_start, cur_end)

    while True:
        improved = False

        # Try extending forward.
        if cur_end + 1 < n_total:
            s_fwd = score(cur_start, cur_end + 1)
            if s_fwd > best_s:
                best_s = s_fwd
                cur_end += 1
                improved = True

        # Try extending backward.
        if cur_start - 1 >= 0:
            s_bwd = score(cur_start - 1, cur_end)
            if s_bwd > best_s:
                best_s = s_bwd
                cur_start -= 1
                improved = True

        if not improved:
            break

    return best_s, cur_start, cur_end


def _remove_overlaps(
    groups: List[tuple],
    spike_times: np.ndarray,
    mean_rate: float,
) -> List[tuple]:
    """Remove overlapping burst groups, keeping the one with higher S."""
    if len(groups) <= 1:
        return groups

    # Sort by start index.
    groups = sorted(groups, key=lambda g: g[0])

    def score(s, e):
        T = spike_times[e] - spike_times[s]
        return _poisson_surprise_score(e - s + 1, T, mean_rate)

    kept = [groups[0]]
    for cur_start, cur_end in groups[1:]:
        prev_start, prev_end = kept[-1]
        if cur_start <= prev_end:  # overlap
            if score(cur_start, cur_end) > score(prev_start, prev_end):
                kept[-1] = (cur_start, cur_end)
        else:
            kept.append((cur_start, cur_end))

    return kept
