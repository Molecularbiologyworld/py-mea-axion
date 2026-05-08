"""
viz/pca.py
==========
Principal-component-analysis scatter of well-level metrics.

Public API
----------
compute_pca(df, metric_cols, ...)
    Run PCA on a metric matrix and return the projected coordinates plus
    the variance explained by each component.

compute_pca_loadings(df, metric_cols, ...)
    Per-feature loadings (rows of V, the right singular vectors) for the
    first *n_components* PCs.

plot_pca(df, metric_cols, ...)
    PCA scatter coloured by experimental condition.

plot_pca_loadings(loadings, ...)
    Horizontal bar chart of the top contributors to each PC.
"""

from typing import Dict, List, Optional, Sequence, Tuple

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

_DEFAULT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def compute_pca(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    *,
    n_components: int = 2,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run PCA on the standardised metric matrix.

    Rows with any ``NaN`` in *metric_cols* are dropped.  Each column is
    centered to zero mean and scaled to unit variance before SVD.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with one row per observation (e.g. one well).
    metric_cols : sequence of str
        Numeric columns to use as PCA features.  Constant columns
        (variance == 0) are dropped automatically to keep the SVD stable.
    n_components : int, optional
        Number of principal components to keep.  Default 2.

    Returns
    -------
    scores : pd.DataFrame
        Original *df* (subset to non-NaN rows) with new columns
        ``PC1, PC2, …`` appended.
    var_explained : np.ndarray
        Fraction of variance explained by each retained component
        (length *n_components*).

    Raises
    ------
    ValueError
        If after dropping NaNs and constant columns there are fewer than
        *n_components* usable features, or fewer than 2 rows of data.
    """
    missing = [c for c in metric_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing}"
        )

    sub = df.dropna(subset=list(metric_cols)).copy()
    if len(sub) < 2:
        raise ValueError(
            f"PCA needs >= 2 rows after dropping NaNs (have {len(sub)})."
        )

    X = sub[list(metric_cols)].to_numpy(dtype=float)

    # Drop zero-variance columns (otherwise standardisation divides by 0).
    stds = X.std(axis=0, ddof=0)
    keep = stds > 0
    if keep.sum() < n_components:
        raise ValueError(
            f"PCA needs >= {n_components} non-constant features "
            f"(have {int(keep.sum())})."
        )
    X = X[:, keep]
    means = X.mean(axis=0)
    stds_kept = X.std(axis=0, ddof=0)
    Xs = (X - means) / stds_kept

    # SVD-based PCA.  U @ diag(s) gives the principal-component scores.
    U, s, _ = np.linalg.svd(Xs, full_matrices=False)
    scores = (U * s)[:, :n_components]
    var_explained = (s ** 2) / (s ** 2).sum()
    var_explained = var_explained[:n_components]

    for i in range(n_components):
        sub[f"PC{i + 1}"] = scores[:, i]
    return sub, var_explained


def compute_pca_loadings(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    *,
    n_components: int = 2,
) -> pd.DataFrame:
    """Compute per-feature PC loadings (right singular vectors).

    Loadings are unit-norm columns of the right singular-vector matrix V
    from the centred-and-standardised metric matrix's SVD.  The
    *absolute* loading of a feature on PC *k* tells you how strongly
    that feature contributes to that component; the sign tells you the
    direction.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data, one row per observation.
    metric_cols : sequence of str
        Numeric columns used as PCA features.  Constant columns (zero
        variance) are dropped automatically.
    n_components : int, optional
        Number of PCs to keep.  Default 2.

    Returns
    -------
    pd.DataFrame
        Index = feature name (only non-constant features are kept).
        Columns = ``"PC1"``, ``"PC2"``, …  Values = loading.

    Raises
    ------
    ValueError
        If after dropping NaNs and constant columns there are fewer than
        *n_components* usable features, or fewer than 2 rows of data.
    """
    sub = df.dropna(subset=list(metric_cols)).copy()
    if len(sub) < 2:
        raise ValueError(
            f"PCA loadings need >= 2 rows after dropping NaNs (have {len(sub)})."
        )

    X = sub[list(metric_cols)].to_numpy(dtype=float)
    stds = X.std(axis=0, ddof=0)
    keep = stds > 0
    if keep.sum() < n_components:
        raise ValueError(
            f"PCA loadings need >= {n_components} non-constant features "
            f"(have {int(keep.sum())})."
        )
    kept_cols = [c for c, k in zip(metric_cols, keep) if k]
    X = X[:, keep]
    Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

    _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    components = Vt[:n_components, :]   # (n_components, n_features)

    loadings = pd.DataFrame(
        components.T,
        index=kept_cols,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    loadings.index.name = "feature"
    return loadings


def plot_pca_loadings(
    loadings: pd.DataFrame,
    *,
    n_top: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    pretty_labels: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
) -> Figure:
    """Horizontal bar chart of the top *n_top* contributors to each PC.

    Bars are sorted by signed loading within each PC.  Positive
    contributions are drawn in blue; negative in red.

    Parameters
    ----------
    loadings : pd.DataFrame
        As returned by :func:`compute_pca_loadings`.
    n_top : int, optional
        How many top-|loading| features to draw per PC.  Default 10.
    figsize : tuple, optional
        Figure size in inches.  Default scales with *n_top* and the
        number of PCs.
    pretty_labels : dict, optional
        Mapping from raw column name → human-readable label, used to
        annotate the y-axis.  Falls back to the raw column name.
    title : str, optional
        Figure-level title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pcs = list(loadings.columns)
    n_pcs = len(pcs)
    if figsize is None:
        figsize = (5.5 * n_pcs, max(4.0, 0.35 * n_top + 1.2))

    fig, axes = plt.subplots(1, n_pcs, figsize=figsize)
    if n_pcs == 1:
        axes = [axes]

    for ax, pc in zip(axes, pcs):
        col = loadings[pc]
        top_idx = col.abs().sort_values(ascending=False).head(n_top).index
        sub = col.loc[top_idx].sort_values(ascending=True)

        labels = [
            (pretty_labels.get(f, f) if pretty_labels else f) for f in sub.index
        ]
        colors = ["#3266a8" if v >= 0 else "#a83232" for v in sub.values]
        ax.barh(labels, sub.values, color=colors, edgecolor="none")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel(f"{pc} loading", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_pca(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    *,
    group_col: str = "condition",
    n_components: int = 2,
    groups: Optional[Sequence[str]] = None,
    palette: Optional[List[str]] = None,
    shape_col: Optional[str] = None,
    shape_map: Optional[dict] = None,
    figsize: Tuple[float, float] = (6.0, 5.0),
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Figure:
    """Scatter of the first two principal components, coloured by group.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with one row per observation.  Must contain
        *group_col* and all *metric_cols*.
    metric_cols : sequence of str
        Numeric columns used as PCA features.
    group_col : str, optional
        Column identifying experimental conditions for colouring.
        Default ``'condition'``.
    n_components : int, optional
        Number of PCs to compute.  Only the first two are plotted; the
        rest are returned in the underlying scores DataFrame.  Default 2.
    groups : sequence of str, optional
        Subset and ordering of groups in the legend.  Defaults to sorted
        unique values of *group_col*.
    palette : list of str, optional
        Hex colours, one per group.  Cycles through a colorblind-friendly
        palette when ``None``.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.  Default ``(6, 5)``.
    title : str, optional
        Plot title.  Default ``'PCA'``.
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
    ...     "a":         rng.normal(0, 1, 30),
    ...     "b":         rng.normal(0, 1, 30),
    ...     "c":         rng.normal(0, 1, 30),
    ...     "condition": ["A"]*15 + ["B"]*15,
    ... })
    >>> fig = plot_pca(df, ["a","b","c"])
    >>> fig.axes[0].get_xlabel().startswith("PC1")
    True
    """
    if group_col not in df.columns:
        raise ValueError(
            f"Group column '{group_col}' not in DataFrame."
        )

    scores, var = compute_pca(df, metric_cols, n_components=n_components)

    all_groups = sorted(scores[group_col].dropna().unique().tolist())
    groups_to_plot = list(groups) if groups is not None else all_groups
    pal = palette if palette is not None else _DEFAULT_PALETTE
    color_map = {g: pal[i % len(pal)] for i, g in enumerate(groups_to_plot)}

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    use_shape = bool(shape_col) and shape_col in scores.columns
    if use_shape:
        unique_shapes = sorted(
            scores[shape_col].dropna().astype(str).unique().tolist()
        )
        smap = dict(shape_map) if shape_map else {}
        for i, s in enumerate(unique_shapes):
            smap.setdefault(s, _DEFAULT_MARKERS[i % len(_DEFAULT_MARKERS)])

    for g in groups_to_plot:
        sub = scores[scores[group_col] == g]
        if not len(sub):
            continue
        if use_shape:
            first_label = True
            for shape_val, ssub in sub.groupby(shape_col, dropna=False):
                shape_key = str(shape_val) if pd.notna(shape_val) else "NaN"
                marker = smap.get(shape_key, "o")
                ax.scatter(
                    ssub["PC1"], ssub["PC2"],
                    color=color_map[g], marker=marker,
                    s=30, alpha=0.75,
                    edgecolors="black", linewidths=0.5,
                    label=g if first_label else None,
                    zorder=3,
                )
                first_label = False
        else:
            ax.scatter(
                sub["PC1"], sub["PC2"],
                color=color_map[g], s=30, alpha=0.75,
                edgecolors="white", linewidths=0.5,
                label=g, zorder=3,
            )

    ax.axhline(0, color="#888888", linewidth=0.5, zorder=1)
    ax.axvline(0, color="#888888", linewidth=0.5, zorder=1)
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title=group_col, fontsize=8, title_fontsize=8,
              framealpha=0.7, loc="best")

    if own_fig:
        fig.tight_layout()
    return fig
