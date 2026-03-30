"""
viz/burst_charts.py
===================
Burst-level visualisations for a single electrode or well.

Public API
----------
plot_isi_histogram(spike_times, ...)
    ISI distribution with optional log-x axis.

plot_burst_raster(well_spike_dict, well_burst_dict, ...)
    Spike raster + burst overlay for a single well.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from py_mea_axion.burst.detection import Burst


def plot_isi_histogram(
    spike_times: np.ndarray,
    *,
    bins: int = 50,
    log_x: bool = True,
    log_y: bool = False,
    color: str = "#4878CF",
    electrode_id: str = "",
    figsize: Tuple[float, float] = (4.0, 3.0),
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot the inter-spike interval (ISI) distribution for one electrode.

    Parameters
    ----------
    spike_times : np.ndarray
        Sorted spike timestamps in seconds.
    bins : int, optional
        Number of histogram bins.  Default 50.
    log_x : bool, optional
        Use a log-scale x-axis.  Default ``True`` (common for ISI plots).
    log_y : bool, optional
        Use a log-scale y-axis.  Default ``False``.
    color : str, optional
        Bar fill colour.  Default steelblue.
    electrode_id : str, optional
        Electrode label used in the title.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.  Default ``(4, 3)``.
    ax : Axes, optional
        Pre-existing axes.  A new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import numpy as np
    >>> spikes = np.arange(0.0, 10.0, 0.1)
    >>> fig = plot_isi_histogram(spikes, electrode_id="A1_11")
    >>> fig.axes[0].get_xlabel()
    'ISI (s)'
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    isis = np.diff(spike_times) if len(spike_times) > 1 else np.array([])

    if len(isis) == 0:
        ax.text(0.5, 0.5, "No ISI data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888888")
    else:
        if log_x:
            isis_pos = isis[isis > 0]
            if len(isis_pos):
                log_edges = np.linspace(
                    np.log10(isis_pos.min()), np.log10(isis_pos.max()), bins + 1
                )
                edges = 10 ** log_edges
                ax.hist(isis_pos, bins=edges, color=color, edgecolor="none", alpha=0.85)
                ax.set_xscale("log")
            else:
                ax.text(0.5, 0.5, "All ISI = 0", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#888888")
        else:
            ax.hist(isis, bins=bins, color=color, edgecolor="none", alpha=0.85)

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("ISI (s)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    title = f"ISI distribution — {electrode_id}" if electrode_id else "ISI distribution"
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)

    if own_fig:
        fig.tight_layout()

    return fig


def plot_burst_raster(
    well_spike_dict: Dict[str, np.ndarray],
    well_burst_dict: Dict[str, List[Burst]],
    *,
    t_start: float = 0.0,
    t_stop: Optional[float] = None,
    spike_color: str = "#333333",
    burst_color: str = "#e87b14",
    burst_alpha: float = 0.25,
    figsize: Tuple[float, float] = (8.0, 4.0),
    title: str = "",
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot a spike raster with burst-period overlays for a single well.

    Each row of the raster corresponds to one electrode.  Burst periods are
    drawn as filled rectangles behind the spike marks.

    Parameters
    ----------
    well_spike_dict : dict[str, np.ndarray]
        Electrode ID → spike times (s).
    well_burst_dict : dict[str, list[Burst]]
        Electrode ID → list of :class:`~py_mea_axion.burst.detection.Burst`.
        Electrodes absent from this dict are treated as having no bursts.
    t_start, t_stop : float, optional
        Time window to display (seconds).  *t_stop* defaults to the latest
        spike across all electrodes + 0.1 s.
    spike_color : str, optional
        Tick colour for spike marks.  Default dark grey.
    burst_color : str, optional
        Fill colour for burst-period rectangles.  Default orange.
    burst_alpha : float, optional
        Transparency for burst overlays.  Default 0.25.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.  Default ``(8, 4)``.
    title : str, optional
        Figure title.
    ax : Axes, optional
        Pre-existing axes.  A new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import numpy as np
    >>> from py_mea_axion.burst.detection import Burst
    >>> spikes = {"A1_11": np.array([0.1, 0.2, 0.3])}
    >>> bursts = {"A1_11": [Burst(0.1, 0.3, np.array([0.1,0.2,0.3]), 3, 0.2, 0.1)]}
    >>> fig = plot_burst_raster(spikes, bursts)
    >>> len(fig.axes)
    1
    """
    eids = sorted(well_spike_dict.keys())
    n = len(eids)

    # Infer t_stop from the data if not provided.
    if t_stop is None:
        all_times = [t for ts in well_spike_dict.values() if len(ts) for t in [ts.max()]]
        t_stop = (max(all_times) + 0.1) if all_times else 1.0

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for row_idx, eid in enumerate(eids):
        spikes = well_spike_dict[eid]
        in_window = spikes[(spikes >= t_start) & (spikes <= t_stop)]

        # Burst rectangles.
        for burst in well_burst_dict.get(eid, []):
            if burst.end_time < t_start or burst.start_time > t_stop:
                continue
            rect = mpatches.Rectangle(
                (max(burst.start_time, t_start), row_idx - 0.4),
                min(burst.end_time, t_stop) - max(burst.start_time, t_start),
                0.8,
                linewidth=0,
                facecolor=burst_color,
                alpha=burst_alpha,
                zorder=1,
            )
            ax.add_patch(rect)

        # Spike ticks.
        if len(in_window):
            ax.vlines(in_window, row_idx - 0.4, row_idx + 0.4,
                      color=spike_color, linewidth=0.5, zorder=2)

    ax.set_xlim(t_start, t_stop)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(eids, fontsize=6)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Electrode", fontsize=9)
    ax.set_title(title or "Burst raster", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

    if own_fig:
        fig.tight_layout()

    return fig
