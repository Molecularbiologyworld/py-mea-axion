"""
viz/plate.py
============
Whole-plate well-grid heatmaps.

Plots a 4×6 (Axion CytoView 24-well) or arbitrarily-shaped grid of wells
coloured by a per-well metric value, with one panel per plate.

Public API
----------
plot_plate_heatmap(df, metric, ...)
    Multi-panel plate heatmap with shared colour bar.
"""

from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_plate_heatmap(
    df: pd.DataFrame,
    metric: str,
    *,
    plate_col: str = "plate",
    well_col: str = "well_id",
    plate_rows: Sequence[str] = ("A", "B", "C", "D"),
    plate_cols: Sequence[int] = (1, 2, 3, 4, 5, 6),
    cmap: str = "viridis",
    silent_color: str = "#cccccc",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show_plate_labels: bool = True,
    metric_label: Optional[str] = None,
    title: Optional[str] = None,
) -> Figure:
    """Plot a multi-panel plate heatmap, one panel per plate.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format frame with one row per (plate, well) measurement.
        Must contain *metric*, *plate_col*, and *well_col*.
    metric : str
        Numeric column to colour the wells by.
    plate_col, well_col : str, optional
        Column names for the plate identifier and well label
        (e.g. ``'A1'``, ``'B3'``).
    plate_rows : sequence of str, optional
        Row letters in plate-layout order.  Default ``('A','B','C','D')``.
    plate_cols : sequence of int, optional
        Column numbers in plate-layout order.  Default ``(1, 2, …, 6)``.
    cmap : str, optional
        Matplotlib colour map.  Default ``'viridis'``.
    silent_color : str, optional
        Colour for wells absent from the data.  Default light grey.
    vmin, vmax : float, optional
        Colour-scale limits.  Defaults to the data range across all
        plates.
    figsize : tuple, optional
        Figure size in inches.  Defaults to ``(5.5 * n_plates, 3.6)``.
    show_plate_labels : bool, optional
        Label each panel ``"Plate <id>"``.  Default ``True``.
    title : str, optional
        Overall figure title (placed via ``suptitle``).  ``None``
        leaves the figure title-less.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if metric not in df.columns:
        raise ValueError(
            f"Column '{metric}' not in DataFrame.  Available: {list(df.columns)}"
        )
    if plate_col not in df.columns:
        raise ValueError(
            f"Plate column '{plate_col}' not in DataFrame."
        )
    if well_col not in df.columns:
        raise ValueError(
            f"Well column '{well_col}' not in DataFrame."
        )

    plates = sorted(df[plate_col].dropna().unique().tolist())
    if not plates:
        raise ValueError(f"No plates found in '{plate_col}'.")
    n_plates = len(plates)

    if figsize is None:
        figsize = (5.5 * n_plates, 3.6)

    fig, axes = plt.subplots(
        1, n_plates, figsize=figsize,
        gridspec_kw={"wspace": 0.35},
    )
    if n_plates == 1:
        axes = [axes]

    rows = list(plate_rows)
    cols = list(plate_cols)
    n_r, n_c = len(rows), len(cols)

    finite_vals = df[metric].dropna().astype(float).values
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if vmin is None:
        vmin = float(finite_vals.min()) if len(finite_vals) else 0.0
    if vmax is None:
        vmax = float(finite_vals.max()) if len(finite_vals) else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color=silent_color)

    for ax, plate in zip(axes, plates):
        pdf = df[df[plate_col] == plate]
        grid = np.full((n_r, n_c), np.nan)
        for _, row in pdf.iterrows():
            w = row[well_col]
            if not isinstance(w, str) or len(w) < 2:
                continue
            try:
                r = rows.index(w[0])
                c = cols.index(int(w[1:]))
            except (ValueError, IndexError):
                continue
            try:
                grid[r, c] = float(row[metric])
            except (TypeError, ValueError):
                continue

        masked = np.ma.masked_invalid(grid)
        ax.imshow(masked, norm=norm, cmap=cm, aspect="auto")

        # Cell labels.
        for ri in range(n_r):
            for ci in range(n_c):
                val = grid[ri, ci]
                label = f"{val:.2g}" if np.isfinite(val) else "—"
                text_color = "white" if np.isfinite(val) else "#888888"
                ax.text(ci, ri, label, ha="center", va="center",
                        fontsize=7, color=text_color)

        ax.set_xticks(range(n_c))
        ax.set_xticklabels([str(c) for c in cols], fontsize=10)
        ax.set_yticks(range(n_r))
        ax.set_yticklabels(rows, fontsize=10)
        ax.tick_params(length=2, pad=2)
        if show_plate_labels:
            ax.set_title(f"Plate {plate}", fontsize=10, pad=4)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=list(axes), fraction=0.025, pad=0.04)
    cb.set_label(metric_label if metric_label is not None else metric, fontsize=10)
    cb.ax.tick_params(labelsize=9)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    return fig
