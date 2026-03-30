"""
viz/heatmap.py
==============
Spatial electrode activity heatmap for a single well.

Plots the 4×4 electrode grid of an Axion CytoView well coloured by any
pre-computed per-electrode metric (MFR, burst rate, CV(ISI), etc.).

Public API
----------
plot_electrode_heatmap(values, well_id, metric_name, ...)
"""

from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from py_mea_axion.io.spk_reader import ELECTRODES_PER_ROW, ELECTRODES_PER_COL


def plot_electrode_heatmap(
    values: Dict[str, float],
    well_id: str,
    metric_name: str = "value",
    *,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    silent_color: str = "#cccccc",
    figsize: Tuple[float, float] = (3.5, 3.5),
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot a spatial heatmap of per-electrode metric values for one well.

    The Axion 4×4 electrode grid is displayed with row 1 at the top and
    column 1 at the left, matching the AxIS Navigator layout.  Electrodes
    absent from *values* (e.g. grounding electrodes) are rendered in
    *silent_color*.

    Parameters
    ----------
    values : dict[str, float]
        Mapping from electrode ID (e.g. ``'A1_11'``) to a scalar metric
        value.  Electrodes missing from this dict are shown in
        *silent_color*.
    well_id : str
        Well label (e.g. ``'A1'``).  Used only for axis labelling.
    metric_name : str, optional
        Human-readable name for the colour-bar label.  Default ``'value'``.
    cmap : str, optional
        Matplotlib colour map name.  Default ``'viridis'``.
    vmin, vmax : float, optional
        Colour scale limits.  Defaults to the data min/max.
    silent_color : str, optional
        Hex colour for electrodes with no data.  Default ``'#cccccc'``
        (light grey).
    figsize : tuple, optional
        Figure size in inches ``(width, height)``.  Default ``(3.5, 3.5)``.
    title : str, optional
        Figure title.  Defaults to ``'{well_id} — {metric_name}'``.
    ax : Axes, optional
        Pre-existing axes to draw into.  A new figure is created when
        ``None`` (default).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the heatmap.

    Examples
    --------
    >>> import numpy as np
    >>> vals = {f"A1_{r}{c}": float(r + c)
    ...         for r in range(1, 5) for c in range(1, 5)}
    >>> fig = plot_electrode_heatmap(vals, "A1", metric_name="MFR (Hz)")
    >>> fig.axes[0].get_title()
    'A1 — MFR (Hz)'
    """
    nrows = ELECTRODES_PER_ROW
    ncols = ELECTRODES_PER_COL

    # Build the value grid — NaN for absent electrodes.
    grid = np.full((nrows, ncols), np.nan)
    for eid, val in values.items():
        er, ec = _parse_electrode_rc(eid)
        if er is not None:
            grid[er - 1, ec - 1] = val

    # Create masked array so NaN cells get the silent colour.
    masked = np.ma.masked_invalid(grid)

    # Resolve colour limits from non-NaN data.
    finite = grid[np.isfinite(grid)]
    _vmin = vmin if vmin is not None else (float(finite.min()) if len(finite) else 0.0)
    _vmax = vmax if vmax is not None else (float(finite.max()) if len(finite) else 1.0)
    if _vmin == _vmax:
        _vmax = _vmin + 1.0

    # Set up figure / axes.
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Draw silent (NaN) cells first.
    silent_grid = np.where(np.isnan(grid), 1.0, np.nan)
    ax.imshow(
        silent_grid,
        cmap=mcolors.ListedColormap([silent_color]),
        vmin=0, vmax=1,
        aspect="equal",
    )

    # Draw the heatmap.
    cm = plt.get_cmap(cmap)
    im = ax.imshow(masked, cmap=cm, vmin=_vmin, vmax=_vmax, aspect="equal")

    # Electrode labels.
    for r in range(nrows):
        for c in range(ncols):
            val = grid[r, c]
            label = f"{val:.2g}" if np.isfinite(val) else "—"
            text_color = "white" if np.isfinite(val) else "#888888"
            ax.text(
                c, r, label,
                ha="center", va="center",
                fontsize=6, color=text_color,
            )

    # Axes formatting.
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([str(c + 1) for c in range(ncols)], fontsize=8)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([str(r + 1) for r in range(nrows)], fontsize=8)
    ax.set_xlabel("Electrode column", fontsize=8)
    ax.set_ylabel("Electrode row", fontsize=8)

    _title = title if title is not None else f"{well_id} \u2014 {metric_name}"
    ax.set_title(_title, fontsize=9)

    # Colour bar.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name, fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    if own_fig:
        fig.tight_layout()

    return fig


# ── Private helpers ───────────────────────────────────────────────────────────

def _parse_electrode_rc(electrode_id: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract (elec_row, elec_col) from an electrode ID like ``'A1_23'``.

    Returns ``(None, None)`` if the ID cannot be parsed.
    """
    try:
        suffix = electrode_id.split("_")[1]  # e.g. "23"
        er = int(suffix[0])
        ec = int(suffix[1])
        return er, ec
    except (IndexError, ValueError):
        return None, None
