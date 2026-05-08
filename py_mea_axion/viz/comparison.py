"""
viz/comparison.py
=================
Single time-point condition comparison plots.

Public API
----------
plot_condition_violin(df, metric, ...)
    Violin + jitter showing a per-well metric across experimental
    conditions at a single time point.
"""

from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes

_DEFAULT_PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442",
]

# Default marker cycle for --point-shape mappings (matplotlib short codes).
# 'o' circle, 's' square, '^' triangle, 'D' diamond, 'v' down-triangle,
# 'P' filled plus, 'X' filled cross, '*' star.
_DEFAULT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def plot_condition_violin(
    df: pd.DataFrame,
    metric: str,
    *,
    group_col: str = "condition",
    time_col: Optional[str] = "DIV",
    time_value: Optional[float] = None,
    groups: Optional[Sequence[str]] = None,
    palette: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (5.0, 5.0),
    jitter_width: float = 0.08,
    point_size: float = 40.0,
    show_stats: bool = True,
    stat_test: str = "tukey",
    compare_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    point_hue_col: Optional[str] = None,
    point_palette: Optional[dict] = None,
    point_shape_col: Optional[str] = None,
    point_shape_map: Optional[dict] = None,
    show_point_legend: bool = False,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot a violin + jitter comparison of *metric* across conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data containing at least *metric* and *group_col*.
    metric : str
        Numeric column to plot on the y-axis.
    group_col : str, optional
        Column identifying experimental conditions.  Default ``'condition'``.
    time_col : str, optional
        Time column name.  When *time_value* is given, the dataframe is
        filtered to ``df[time_col] == time_value`` before plotting.
        Pass ``None`` (and leave *time_value* ``None``) for data that has
        no time dimension.  Default ``'DIV'``.
    time_value : float, optional
        Specific time point to plot (e.g. ``DIV=26``).  When ``None``
        all rows are pooled.
    groups : sequence of str, optional
        Subset and ordering of groups to plot.  Defaults to sorted
        unique values of *group_col*.
    palette : list of str, optional
        Hex colour codes, one per group.  Cycles through a built-in
        colorblind-friendly palette when ``None``.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.  Default ``(5, 5)``.
    jitter_width : float, optional
        Half-width of the x-jitter applied to individual points.
        Default 0.08.
    title, ylabel : str, optional
        Plot title and y-axis label.  Sensible defaults are inferred.
    ax : Axes, optional
        Pre-existing axes.  A new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "metric":    rng.normal(0, 1, 30),
    ...     "condition": ["A"]*10 + ["B"]*10 + ["C"]*10,
    ...     "DIV":       [21]*30,
    ... })
    >>> fig = plot_condition_violin(df, "metric", time_value=21)
    >>> len(fig.axes[0].get_xticks())
    3
    """
    if metric not in df.columns:
        raise ValueError(
            f"Column '{metric}' not in DataFrame.  Available: {list(df.columns)}"
        )
    if group_col not in df.columns:
        raise ValueError(
            f"Group column '{group_col}' not in DataFrame."
        )

    data = df
    if time_value is not None:
        if time_col is None or time_col not in df.columns:
            raise ValueError(
                f"time_value={time_value} given but time_col '{time_col}' "
                f"is missing from DataFrame."
            )
        data = data[data[time_col] == time_value]

    keep_cols = [metric, group_col]
    if point_hue_col and point_hue_col in data.columns:
        keep_cols.append(point_hue_col)
    if point_shape_col and point_shape_col in data.columns and point_shape_col not in keep_cols:
        keep_cols.append(point_shape_col)
    data = data[keep_cols].dropna(subset=[metric])

    all_groups = sorted(data[group_col].dropna().unique().tolist())
    groups_to_plot = list(groups) if groups is not None else all_groups
    pal = palette if palette is not None else _DEFAULT_PALETTE
    color_map = {g: pal[i % len(pal)] for i, g in enumerate(groups_to_plot)}

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    positions = list(range(len(groups_to_plot)))
    data_by_group: List[np.ndarray] = []
    for g in groups_to_plot:
        vals = data.loc[data[group_col] == g, metric].values
        # Empty groups need a placeholder for matplotlib's violinplot.
        data_by_group.append(vals if len(vals) else np.array([0.0]))

    parts = ax.violinplot(
        data_by_group, positions=positions,
        showmedians=True, showextrema=False, widths=0.7,
    )
    for pc, g in zip(parts["bodies"], groups_to_plot):
        pc.set_facecolor(color_map[g])
        pc.set_alpha(0.7)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

    # Build the per-point colour resolver.
    use_point_hue = bool(point_hue_col) and point_hue_col in data.columns
    if use_point_hue:
        unique_hues = sorted(
            data[point_hue_col].dropna().astype(str).unique().tolist()
        )
        ppal = dict(point_palette) if point_palette else {}
        for i, h in enumerate(unique_hues):
            ppal.setdefault(h, _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)])

    # Build the per-point marker resolver.
    use_point_shape = bool(point_shape_col) and point_shape_col in data.columns
    if use_point_shape:
        unique_shapes = sorted(
            data[point_shape_col].dropna().astype(str).unique().tolist()
        )
        smap = dict(point_shape_map) if point_shape_map else {}
        for i, s in enumerate(unique_shapes):
            smap.setdefault(s, _DEFAULT_MARKERS[i % len(_DEFAULT_MARKERS)])

    rng = np.random.default_rng(42)
    legend_handles: dict = {}
    for xi, g in enumerate(groups_to_plot):
        sub = data[data[group_col] == g]
        if not len(sub):
            continue

        # Compose hue and shape into a single grouping.
        sub_keys: List[str] = []
        if use_point_hue:
            sub_keys.append(point_hue_col)
        if use_point_shape:
            sub_keys.append(point_shape_col)

        if sub_keys:
            grouped = sub.groupby(sub_keys, dropna=False)
            for raw_key, tsub in grouped:
                key_tuple = raw_key if isinstance(raw_key, tuple) else (raw_key,)
                key_dict = dict(zip(sub_keys, key_tuple))
                if use_point_hue:
                    hue_val = key_dict[point_hue_col]
                    hue_key = str(hue_val) if pd.notna(hue_val) else "NaN"
                    color = ppal.get(hue_key, "#888888")
                else:
                    hue_key = None
                    color = color_map[g]
                if use_point_shape:
                    shape_val = key_dict[point_shape_col]
                    shape_key = str(shape_val) if pd.notna(shape_val) else "NaN"
                    marker = smap.get(shape_key, "o")
                else:
                    marker = "o"
                jitter = rng.uniform(-jitter_width, jitter_width, size=len(tsub))
                sc = ax.scatter(
                    xi + jitter, tsub[metric].values,
                    color=color, marker=marker,
                    s=point_size, alpha=0.85 if use_point_hue else 0.7,
                    zorder=3,
                    edgecolors="black", linewidths=0.5,
                )
                if hue_key is not None and hue_key not in legend_handles:
                    legend_handles[hue_key] = sc
        else:
            jitter = rng.uniform(-jitter_width, jitter_width, size=len(sub))
            ax.scatter(
                xi + jitter, sub[metric].values,
                color=color_map[g], s=point_size, alpha=0.7,
                zorder=3,
                edgecolors="black", linewidths=0.5,
            )

    if use_point_hue and legend_handles and show_point_legend:
        ax.legend(
            legend_handles.values(), legend_handles.keys(),
            title=point_hue_col, fontsize=8, title_fontsize=8,
            framealpha=0.7, loc="best",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(groups_to_plot, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel(ylabel or metric, fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_stats:
        _draw_pairwise_brackets(
            ax, data_by_group, groups_to_plot, positions=positions,
            compare_pairs=compare_pairs,
            stat_test=stat_test,
        )

    if title:
        ax.set_title(title, fontsize=11)

    if own_fig:
        fig.tight_layout()
    return fig


def _significance_label(p: float) -> str:
    """Return ``'***' / '**' / '*' / 'ns'`` for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _draw_pairwise_brackets(
    ax: Axes,
    data_by_group: List[np.ndarray],
    group_names: List[str],
    *,
    positions: List[int],
    compare_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    stat_test: str = "tukey",
) -> None:
    """Run a pairwise statistical test and overlay significance brackets.

    Statistical computation is delegated to
    :func:`py_mea_axion.stats.pairwise_test`; this function only handles
    the geometry of drawing brackets above the violins.

    Parameters
    ----------
    stat_test : str, optional
        Which test to run.  One of ``'tukey'`` (default; parametric all-
        pairs Tukey HSD), ``'mannwhitney'`` (pairwise Mann-Whitney U with
        Bonferroni correction), or ``'kruskal'`` (Kruskal-Wallis omnibus
        followed by Dunn's pairwise post-hoc with Bonferroni correction).
        Test choice is the user's responsibility; for n >= ~10 per group
        with continuous metrics, Tukey is the conventional default; for
        skewed distributions or two-condition designs, Mann-Whitney /
        Kruskal-Wallis are usually preferred.
    compare_pairs
        Optional restriction to specific pairs.

    Brackets are silently skipped when fewer than two groups have at
    least 2 observations or when the requested pair references unknown
    groups.
    """
    from py_mea_axion.stats.compare import pairwise_test

    # Build a small long-form frame for the stats helper.
    rows = []
    for name, vals in zip(group_names, data_by_group):
        if vals is None:
            continue
        for v in vals:
            rows.append({"_value": float(v), "_group": name})
    if len(rows) < 2:
        return
    sub_df = pd.DataFrame(rows)

    result = pairwise_test(
        sub_df,
        metric="_value",
        group_col="_group",
        test=stat_test,
        groups=list(group_names),
        pairs=list(compare_pairs) if compare_pairs is not None else None,
    )
    if result.empty:
        return

    name_to_idx = {n: i for i, n in enumerate(group_names)}

    # Determine bracket positions, sorted lowest-to-highest.
    bracket_specs: List[Tuple[int, int, float]] = []
    for _, row in result.iterrows():
        ga, gb = row["group_a"], row["group_b"]
        if ga not in name_to_idx or gb not in name_to_idx:
            continue
        i, j = name_to_idx[ga], name_to_idx[gb]
        if i == j:
            continue
        if i > j:
            i, j = j, i
        bracket_specs.append((i, j, float(row["p_adj"])))
    if not bracket_specs:
        return

    bracket_specs.sort(key=lambda t: (t[0], t[1]))

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin if ymax > ymin else 1.0
    bracket_h = yspan * 0.03
    gap = yspan * 0.10
    y0 = ymax + yspan * 0.05

    for k, (i, j, p) in enumerate(bracket_specs):
        x1 = positions[i]
        x2 = positions[j]
        y = y0 + k * gap
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + bracket_h, y + bracket_h, y],
            color="black", linewidth=1.0,
        )
        ax.text(
            (x1 + x2) / 2, y + bracket_h,
            _significance_label(p),
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylim(ymin, y0 + len(bracket_specs) * gap + bracket_h * 2)
