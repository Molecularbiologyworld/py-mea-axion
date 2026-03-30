"""
network/synchrony.py
====================
Synchrony metrics between spike trains.

Implements the **Spike Time Tiling Coefficient** (STTC) of
Cutts & Eglen (2014, *J. Neurosci.*), which measures pairwise spike-train
synchrony robustly across a range of firing rates.

Definition (Cutts & Eglen 2014)
--------------------------------
For two spike trains A and B recorded over duration T:

    STTC(A, B) = 0.5 × [ (P_A − T_B) / (1 − P_A·T_B)
                        + (P_B − T_A) / (1 − P_B·T_A) ]

where

* T_X  = fraction of total time within ±Δt of *any* spike in train X
* P_X  = fraction of spikes in train X that fall within ±Δt of *any*
          spike in train Y

Public API
----------
sttc(spike_times_a, spike_times_b, dt_s, total_time_s)
    STTC for a single electrode pair.

sttc_matrix(spike_dict, dt_s, total_time_s)
    Full pairwise STTC matrix for a set of electrodes.

mean_sttc(spike_dict, dt_s, total_time_s)
    Mean STTC across all electrode pairs (well-level synchrony score).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Core STTC calculation ─────────────────────────────────────────────────────

def sttc(
    spike_times_a: np.ndarray,
    spike_times_b: np.ndarray,
    dt_s: float,
    total_time_s: float,
) -> float:
    """Compute the Spike Time Tiling Coefficient between two spike trains.

    Uses the exact definition of Cutts & Eglen (2014).

    Parameters
    ----------
    spike_times_a : np.ndarray
        Sorted spike timestamps for electrode A in seconds.
    spike_times_b : np.ndarray
        Sorted spike timestamps for electrode B in seconds.
    dt_s : float
        Coincidence window half-width in seconds (Δt).  Typical value:
        0.05 s (50 ms).
    total_time_s : float
        Recording duration in seconds.

    Returns
    -------
    float
        STTC in the range [−1, 1].  Returns ``0.0`` when either train is
        empty or *total_time_s* is zero.

    References
    ----------
    Cutts, C. S. & Eglen, S. J. (2014). Detecting pairwise correlations
    in spike trains: an objective comparison of methods and application
    to the study of retinal waves. *J. Neurosci.*, 34(43), 14288-14303.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(0.0, 10.0, 0.1)   # 100 Hz regular
    >>> b = np.arange(0.0, 10.0, 0.1)   # identical train
    >>> round(sttc(a, b, dt_s=0.05, total_time_s=10.0), 4)
    1.0
    """
    if total_time_s <= 0:
        return 0.0
    if len(spike_times_a) == 0 or len(spike_times_b) == 0:
        return 0.0

    t_a = _tile_fraction(spike_times_a, dt_s, total_time_s)
    t_b = _tile_fraction(spike_times_b, dt_s, total_time_s)
    p_a = _proportion_near(spike_times_a, spike_times_b, dt_s)
    p_b = _proportion_near(spike_times_b, spike_times_a, dt_s)

    term1 = _safe_term(p_a, t_b)
    term2 = _safe_term(p_b, t_a)

    return 0.5 * (term1 + term2)


def _tile_fraction(
    spike_times: np.ndarray,
    dt_s: float,
    total_time_s: float,
) -> float:
    """Fraction of total time within ±dt_s of any spike in the train.

    Overlapping windows are counted only once (union of intervals).
    """
    if len(spike_times) == 0:
        return 0.0

    # Build [start, end) intervals clamped to [0, total_time_s].
    starts = np.clip(spike_times - dt_s, 0.0, total_time_s)
    ends = np.clip(spike_times + dt_s, 0.0, total_time_s)

    # Sort by start and merge overlapping intervals.
    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]

    total_covered = 0.0
    cur_start = starts[0]
    cur_end = ends[0]

    for s, e in zip(starts[1:], ends[1:]):
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            total_covered += cur_end - cur_start
            cur_start = s
            cur_end = e
    total_covered += cur_end - cur_start

    return total_covered / total_time_s


def _proportion_near(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dt_s: float,
) -> float:
    """Fraction of spikes in *train_x* that are within dt_s of any spike in *train_y*."""
    if len(train_x) == 0 or len(train_y) == 0:
        return 0.0

    # Use searchsorted for an O(n log n) implementation.
    idx_lo = np.searchsorted(train_y, train_x - dt_s, side="left")
    idx_hi = np.searchsorted(train_y, train_x + dt_s, side="right")
    near = idx_hi > idx_lo          # True when at least one y-spike is close
    return float(near.sum()) / len(train_x)


def _safe_term(p: float, t: float) -> float:
    """Compute (p − t) / (1 − p·t) safely; returns 0 on degenerate input."""
    denom = 1.0 - p * t
    if abs(denom) < 1e-12:
        return 0.0
    return (p - t) / denom


# ── Matrix and well-level score ───────────────────────────────────────────────

def sttc_matrix(
    spike_dict: Dict[str, np.ndarray],
    dt_s: float,
    total_time_s: float,
) -> pd.DataFrame:
    """Compute the full pairwise STTC matrix for a set of electrodes.

    Parameters
    ----------
    spike_dict : dict[str, np.ndarray]
        Mapping from electrode ID to sorted spike timestamp array (s).
    dt_s : float
        Coincidence window half-width in seconds.
    total_time_s : float
        Recording duration in seconds.

    Returns
    -------
    pd.DataFrame
        Symmetric *n × n* DataFrame (index and columns = electrode IDs)
        of pairwise STTC values.  Diagonal is 1.0.  Returns an empty
        DataFrame when *spike_dict* has fewer than two entries.

    Examples
    --------
    >>> spikes = {"e1": np.arange(0, 1, 0.1), "e2": np.arange(0, 1, 0.1)}
    >>> mat = sttc_matrix(spikes, dt_s=0.05, total_time_s=1.0)
    >>> mat.loc["e1", "e2"]
    1.0
    """
    eids = sorted(spike_dict.keys())
    n = len(eids)

    if n < 2:
        return pd.DataFrame()

    matrix = np.ones((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            val = sttc(
                spike_dict[eids[i]],
                spike_dict[eids[j]],
                dt_s,
                total_time_s,
            )
            matrix[i, j] = val
            matrix[j, i] = val

    return pd.DataFrame(matrix, index=eids, columns=eids)


def mean_sttc(
    spike_dict: Dict[str, np.ndarray],
    dt_s: float,
    total_time_s: float,
    active_only: bool = True,
    active_threshold_hz: float = 0.1,
) -> float:
    """Compute the mean STTC across all electrode pairs in a well.

    Parameters
    ----------
    spike_dict : dict[str, np.ndarray]
        Mapping from electrode ID to sorted spike timestamp array (s).
    dt_s : float
        Coincidence window half-width in seconds.
    total_time_s : float
        Recording duration in seconds.
    active_only : bool, optional
        If ``True`` (default), restrict to electrodes whose MFR is at
        least *active_threshold_hz*.  Silent electrodes drag the mean
        towards zero and obscure real synchrony.
    active_threshold_hz : float, optional
        MFR threshold for active-electrode selection.  Default 0.1 Hz.

    Returns
    -------
    float
        Mean pairwise STTC.  Returns ``float('nan')`` when fewer than
        two (active) electrodes are available.

    Examples
    --------
    >>> spikes = {"e1": np.arange(0, 1, 0.1),
    ...           "e2": np.arange(0, 1, 0.1),
    ...           "e3": np.array([])}
    >>> v = mean_sttc(spikes, dt_s=0.05, total_time_s=1.0, active_only=True)
    >>> round(v, 4)
    1.0
    """
    if active_only and total_time_s > 0:
        spike_dict = {
            eid: ts for eid, ts in spike_dict.items()
            if len(ts) / total_time_s >= active_threshold_hz
        }

    eids = sorted(spike_dict.keys())
    n = len(eids)

    if n < 2:
        return float("nan")

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sttc(
                spike_dict[eids[i]],
                spike_dict[eids[j]],
                dt_s,
                total_time_s,
            )
            count += 1

    return total / count if count > 0 else float("nan")
