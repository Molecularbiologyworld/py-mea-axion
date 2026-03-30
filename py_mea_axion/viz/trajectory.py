"""
viz/trajectory.py
=================
Longitudinal metric trajectory plots (the centrepiece visualisation).

Plots the temporal evolution of a per-well metric across DIV time-points,
grouped by experimental condition, with individual replicate lines and a
group mean ± SEM overlay.

Public API
----------
plot_metric_trajectory(df, metric, time_col, group_col, ...)
"""

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Default colour cycle (colorblind-friendly).
_DEFAULT_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#56B4E9",  # sky-blue
    "#F0E442",  # yellow
]


def plot_metric_trajectory(
    df: pd.DataFrame,
    metric: str,
    time_col: str = "DIV",
    group_col: str = "condition",
    replicate_col: str = "replicate_id",
    *,
    groups: Optional[Sequence[str]] = None,
    palette: Optional[List[str]] = None,
    show_replicates: bool = True,
    replicate_alpha: float = 0.25,
    show_mean: bool = True,
    show_sem: bool = True,
    marker: str = "o",
    figsize: Tuple[float, float] = (6.0, 4.0),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot a longitudinal trajectory of a per-well metric.

    Individual replicate lines are drawn with low opacity; on top, the
    group mean ± SEM is drawn with markers.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with at least *metric*, *time_col*, *group_col*,
        and *replicate_col* columns.
    metric : str
        Numeric column to plot on the y-axis.
    time_col : str, optional
        Column holding the time variable (e.g. ``'DIV'``).  Default
        ``'DIV'``.
    group_col : str, optional
        Column identifying experimental conditions.  Default
        ``'condition'``.
    replicate_col : str, optional
        Column identifying biological replicates / wells.  Default
        ``'replicate_id'``.
    groups : sequence of str, optional
        Subset and ordering of groups to plot.  Defaults to all unique
        values of *group_col* in sorted order.
    palette : list of str, optional
        Hex colour codes, one per group.  Cycles through
        a built-in colorblind-friendly palette when ``None``.
    show_replicates : bool, optional
        Draw individual replicate lines.  Default ``True``.
    replicate_alpha : float, optional
        Opacity for replicate lines.  Default 0.25.
    show_mean : bool, optional
        Draw the group mean line.  Default ``True``.
    show_sem : bool, optional
        Shade ± SEM around the mean.  Default ``True``.
    marker : str, optional
        Matplotlib marker style for mean line.  Default ``'o'``.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.  Default ``(6, 4)``.
    xlabel, ylabel, title : str, optional
        Axis labels and title.  Sensible defaults are inferred from the
        column names.
    ax : Axes, optional
        Pre-existing axes.  A new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If required columns are missing from *df*.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "mfr":          rng.normal(2, 0.3, 12),
    ...     "DIV":          [14, 21, 28] * 4,
    ...     "condition":    ["WT", "WT", "WT", "WT", "WT", "WT",
    ...                      "KD", "KD", "KD", "KD", "KD", "KD"],
    ...     "replicate_id": ["r1","r1","r1","r2","r2","r2",
    ...                      "r3","r3","r3","r4","r4","r4"],
    ... })
    >>> fig = plot_metric_trajectory(df, "mfr")
    >>> len(fig.axes[0].lines) > 0
    True
    """
    required = [metric, time_col, group_col, replicate_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing}. "
            f"Available: {list(df.columns)}"
        )

    data = df[required].dropna(subset=[metric])
    all_groups = sorted(data[group_col].unique())
    groups_to_plot = list(groups) if groups is not None else all_groups

    pal = palette if palette is not None else _DEFAULT_PALETTE
    color_map = {g: pal[i % len(pal)] for i, g in enumerate(groups_to_plot)}

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    time_points = sorted(data[time_col].unique())

    for group in groups_to_plot:
        gdata = data[data[group_col] == group]
        color = color_map[group]

        # Individual replicate lines.
        if show_replicates:
            for rep_id, rep_data in gdata.groupby(replicate_col):
                rep_sorted = rep_data.sort_values(time_col)
                ax.plot(
                    rep_sorted[time_col],
                    rep_sorted[metric],
                    color=color,
                    alpha=replicate_alpha,
                    linewidth=1.0,
                    zorder=1,
                )

        # Group mean ± SEM.
        if show_mean:
            means, sems, ts = [], [], []
            for t in time_points:
                vals = gdata.loc[gdata[time_col] == t, metric].dropna().values
                if len(vals):
                    means.append(vals.mean())
                    sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
                    ts.append(t)

            ts = np.array(ts, dtype=float)
            means = np.array(means)
            sems = np.array(sems)

            ax.plot(ts, means, color=color, linewidth=2, marker=marker,
                    markersize=5, label=group, zorder=3)
            if show_sem and len(sems):
                ax.fill_between(ts, means - sems, means + sems,
                                color=color, alpha=0.2, zorder=2)

    ax.set_xlabel(xlabel or time_col, fontsize=9)
    ax.set_ylabel(ylabel or metric, fontsize=9)
    ax.set_title(title or f"{metric} over {time_col}", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xticks(time_points)

    if groups_to_plot:
        ax.legend(title=group_col, fontsize=8, title_fontsize=8,
                  framealpha=0.7, loc="best")

    if own_fig:
        fig.tight_layout()

    return fig
