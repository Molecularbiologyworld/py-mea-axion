"""
viz/network_plots.py
====================
Network-level visualisations: STTC correlation matrix and network-burst
timeline.

Public API
----------
plot_sttc_matrix(sttc_df, ...)
    Colour-coded pairwise STTC matrix (heatmap).

plot_network_burst_timeline(network_bursts, total_time_s, ...)
    Gantt-style timeline of detected network bursts.
"""

from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from py_mea_axion.network.detection import NetworkBurst


def plot_sttc_matrix(
    sttc_df: pd.DataFrame,
    *,
    cmap: str = "RdYlGn",
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: Tuple[float, float] = (4.5, 4.0),
    title: str = "STTC matrix",
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot the pairwise STTC matrix as a colour-coded heatmap.

    Parameters
    ----------
    sttc_df : pd.DataFrame
        Square symmetric DataFrame returned by
        :func:`~py_mea_axion.network.synchrony.sttc_matrix`.
        Index and columns are electrode IDs.
    cmap : str, optional
        Matplotlib colour map.  Default ``'RdYlGn'`` (red–yellow–green),
        which maps −1 → red, 0 → yellow, +1 → green.
    vmin, vmax : float, optional
        Colour scale limits.  Default [−1, 1].
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.  Default ``(4.5, 4)``.
    title : str, optional
        Figure title.  Default ``'STTC matrix'``.
    ax : Axes, optional
        Pre-existing axes.  A new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame([[1.0, 0.5],[0.5, 1.0]],
    ...                   index=["e1","e2"], columns=["e1","e2"])
    >>> fig = plot_sttc_matrix(df)
    >>> fig.axes[0].get_title()
    'STTC matrix'
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    eids = list(sttc_df.index)
    n = len(eids)

    if n == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888888")
        ax.set_title(title, fontsize=9)
        if own_fig:
            fig.tight_layout()
        return fig

    mat = sttc_df.values.astype(float)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    # Electrode-ID labels on both axes.
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(eids, rotation=90, fontsize=5)
    ax.set_yticklabels(eids, fontsize=5)

    # Colour bar.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("STTC", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=9)

    if own_fig:
        fig.tight_layout()

    return fig


def plot_network_burst_timeline(
    network_bursts: List[NetworkBurst],
    total_time_s: float,
    *,
    bar_height: float = 0.6,
    color: str = "#C44E52",
    alpha: float = 0.75,
    figsize: Tuple[float, float] = (8.0, 1.8),
    title: str = "Network bursts",
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot a Gantt-style timeline of network burst periods.

    Parameters
    ----------
    network_bursts : list[NetworkBurst]
        Detected network bursts (from
        :func:`~py_mea_axion.network.detection.detect_network_bursts`).
    total_time_s : float
        Total recording duration in seconds (sets the x-axis limit).
    bar_height : float, optional
        Vertical height of each burst rectangle.  Default 0.6.
    color : str, optional
        Fill colour for burst bars.  Default muted red.
    alpha : float, optional
        Opacity of burst bars.  Default 0.75.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.  Default ``(8, 1.8)``.
    title : str, optional
        Figure title.  Default ``'Network bursts'``.
    ax : Axes, optional
        Pre-existing axes.  A new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> from py_mea_axion.network.detection import NetworkBurst
    >>> nb = NetworkBurst(1.0, 2.0, 1.0, ["e1","e2"], 0.5, 0.8)
    >>> fig = plot_network_burst_timeline([nb], total_time_s=10.0)
    >>> len(fig.axes)
    1
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_center = 0.5
    for nb in network_bursts:
        rect = mpatches.Rectangle(
            (nb.start_time, y_center - bar_height / 2),
            nb.end_time - nb.start_time,
            bar_height,
            linewidth=0,
            facecolor=color,
            alpha=alpha,
        )
        ax.add_patch(rect)

    ax.set_xlim(0, total_time_s)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

    # Annotate burst count.
    ax.text(
        0.99, 0.95,
        f"n = {len(network_bursts)} bursts",
        ha="right", va="top",
        transform=ax.transAxes,
        fontsize=7, color="#555555",
    )

    if own_fig:
        fig.tight_layout()

    return fig
