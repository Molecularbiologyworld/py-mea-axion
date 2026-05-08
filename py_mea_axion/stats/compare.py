"""
stats/compare.py
================
Statistical comparison tools for MEA experiment data.

Four levels of analysis are provided:

1. **Group comparison** (:func:`compare_conditions`)
   Two conditions, Mann-Whitney U with rank-biserial r.
   Three or more conditions, Kruskal-Wallis H with Dunn's post-hoc
   and Bonferroni correction.

2. **Pairwise Tukey HSD** (:func:`tukey_hsd_pairwise`)
   Parametric all-pairs comparison using SciPy's studentised-range
   implementation.  Drives the significance brackets shown on
   :func:`py_mea_axion.viz.comparison.plot_condition_violin`.

3. **Intraclass correlation** (:func:`compute_icc`)
   ICC(2,k) for assessing electrode-to-electrode or well-to-well
   consistency within a condition, via ``pingouin``.

4. **Longitudinal mixed-effects model** (:func:`longitudinal_model`)
   Linear mixed-effects model with a time x group interaction term
   (``statsmodels`` MixedLM), returning the full coefficient table.

Public API
----------
CompareResult  (namedtuple)
compare_conditions(df, metric, group_col, test='mannwhitney')
tukey_hsd_pairwise(df, metric, group_col, ...)
compute_icc(df, metric, targets_col, raters_col)
longitudinal_model(df, metric, time_col, group_col, subject_col)
"""

from collections import namedtuple
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, norm, rankdata, tukey_hsd

# ── Result container ──────────────────────────────────────────────────────────

CompareResult = namedtuple(
    "CompareResult",
    [
        "test",          # str   — name of the test applied
        "statistic",     # float — test statistic (U for MWU, H for KW)
        "p_value",       # float — p-value
        "effect_size",   # float — rank-biserial r (MWU) or ε² (KW)
        "posthoc",       # DataFrame or None — pairwise Dunn's results (KW only)
    ],
)


# ── Public functions ──────────────────────────────────────────────────────────

def compare_conditions(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    test: str = "mannwhitney",
) -> CompareResult:
    """Compare a metric across experimental conditions.

    Selects the appropriate non-parametric test based on the number of
    unique groups in *group_col*:

    * **Two groups** → Mann-Whitney U + rank-biserial *r* effect size.
    * **Three or more groups** → Kruskal-Wallis *H* + Dunn's pairwise
      post-hoc with Bonferroni correction.

    Parameters
    ----------
    df : pd.DataFrame
        Data table.  Must contain *metric* and *group_col* columns.
        Rows with NaN in *metric* are silently dropped.
    metric : str
        Name of the numeric column to compare.
    group_col : str
        Column identifying condition groups (e.g. ``'condition'``).
    test : str, optional
        Reserved for future use (currently only non-parametric tests are
        implemented).  Default ``'mannwhitney'``.

    Returns
    -------
    CompareResult
        Named tuple with fields:

        ``test``
            ``'mannwhitney'`` or ``'kruskal'``.
        ``statistic``
            U statistic (two groups) or H statistic (≥ 3 groups).
        ``p_value``
            Two-sided p-value.
        ``effect_size``
            Rank-biserial *r* (two groups) or ε² (≥ 3 groups).
        ``posthoc``
            ``None`` for two groups; DataFrame of Dunn's pairwise results
            for ≥ 3 groups with columns
            ``['group1', 'group2', 'z_stat', 'p_value', 'p_adjusted']``.

    Raises
    ------
    ValueError
        If *metric* or *group_col* are not in *df*, or fewer than two
        groups are present.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "mfr_hz":    np.r_[rng.normal(1, 0.2, 20), rng.normal(2, 0.2, 20)],
    ...     "condition": ["WT"]*20 + ["KD"]*20,
    ... })
    >>> res = compare_conditions(df, metric="mfr_hz", group_col="condition")
    >>> res.test
    'mannwhitney'
    >>> res.p_value < 0.05
    True
    """
    _check_columns(df, [metric, group_col])
    data = df[[metric, group_col]].dropna(subset=[metric])
    groups = data[group_col].unique()
    n_groups = len(groups)

    if n_groups < 2:
        raise ValueError(
            f"Need at least 2 groups in '{group_col}'; found {n_groups}."
        )

    group_arrays = {g: data.loc[data[group_col] == g, metric].values for g in groups}

    if n_groups == 2:
        return _mannwhitney(group_arrays)
    else:
        return _kruskal_dunn(group_arrays)


def tukey_hsd_pairwise(
    df: pd.DataFrame,
    metric: str,
    *,
    group_col: str = "condition",
    groups: Optional[Sequence[str]] = None,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Run Tukey HSD pairwise comparisons across conditions.

    Tukey HSD (``scipy.stats.tukey_hsd``) is a parametric all-pairs
    procedure that controls family-wise error rate at alpha = 0.05 via
    the studentised-range distribution.  It assumes approximate
    normality within each group and homogeneity of variance across
    groups; with n >= ~10 per group the central-limit theorem makes the
    test reasonably robust to moderate violations of normality.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with at least *metric* and *group_col* columns.
    metric : str
        Numeric column to compare across groups.
    group_col : str, optional
        Column identifying the grouping variable.  Default
        ``'condition'``.
    groups : sequence of str, optional
        Subset and ordering of groups to test.  Defaults to every value
        present in *group_col* in sorted order.  Groups with fewer than
        two observations are dropped silently.
    pairs : sequence of ``(group_a, group_b)``, optional
        Restrict the output rows to specific pairs.  Pairs referring to
        groups not present in the data are skipped.  Default: every
        pair among *groups*.

    Returns
    -------
    pd.DataFrame
        One row per requested pair, with columns:

        * ``group_a``, ``group_b``: condition labels (alphabetical order
          per row).
        * ``mean_diff``: ``mean(group_a) - mean(group_b)``.
        * ``p_adj``: Tukey-adjusted p-value.

        The returned frame is empty when fewer than two groups have
        sufficient data.

    Raises
    ------
    ValueError
        If *metric* or *group_col* is not in *df*.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "metric":    rng.normal(0, 1, 30),
    ...     "condition": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
    ... })
    >>> res = tukey_hsd_pairwise(df, "metric")
    >>> sorted(res.columns.tolist())
    ['group_a', 'group_b', 'mean_diff', 'p_adj']
    """
    _check_columns(df, [metric, group_col])

    data = df[[metric, group_col]].dropna(subset=[metric])
    all_groups = sorted(data[group_col].dropna().unique().tolist())
    groups_to_use = list(groups) if groups is not None else all_groups

    # Keep only groups with enough data for Tukey HSD.
    valid: List[Tuple[str, np.ndarray]] = []
    for g in groups_to_use:
        vals = data.loc[data[group_col] == g, metric].values
        if len(vals) >= 2:
            valid.append((g, np.asarray(vals, dtype=float)))

    empty = pd.DataFrame(columns=["group_a", "group_b", "mean_diff", "p_adj"])
    if len(valid) < 2:
        return empty

    valid_names = [v[0] for v in valid]
    valid_arrays = [v[1] for v in valid]

    try:
        result = tukey_hsd(*valid_arrays)
    except Exception:  # noqa: BLE001
        return empty

    name_to_idx = {n: i for i, n in enumerate(valid_names)}

    if pairs is None:
        wanted: List[Tuple[int, int]] = [
            (i, j) for i in range(len(valid_names))
                   for j in range(i + 1, len(valid_names))
        ]
    else:
        seen: set = set()
        wanted = []
        for a, b in pairs:
            if a not in name_to_idx or b not in name_to_idx:
                continue
            ia, ib = name_to_idx[a], name_to_idx[b]
            if ia == ib:
                continue
            i, j = (ia, ib) if ia < ib else (ib, ia)
            if (i, j) in seen:
                continue
            seen.add((i, j))
            wanted.append((i, j))

    rows = []
    for i, j in wanted:
        a_name, b_name = valid_names[i], valid_names[j]
        mean_a = float(np.mean(valid_arrays[i]))
        mean_b = float(np.mean(valid_arrays[j]))
        # Sort labels alphabetically per row for deterministic output.
        if a_name <= b_name:
            ga, gb, diff = a_name, b_name, mean_a - mean_b
        else:
            ga, gb, diff = b_name, a_name, mean_b - mean_a
        rows.append({
            "group_a":   ga,
            "group_b":   gb,
            "mean_diff": diff,
            "p_adj":     float(result.pvalue[i, j]),
        })

    return pd.DataFrame(rows, columns=["group_a", "group_b", "mean_diff", "p_adj"])


def pairwise_test(
    df: pd.DataFrame,
    metric: str,
    *,
    group_col: str = "condition",
    test: str = "tukey",
    groups: Optional[Sequence[str]] = None,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Run pairwise comparisons under one of three tests.

    Dispatches to the underlying implementation based on *test* and
    returns a uniformly shaped DataFrame so downstream code (e.g.
    violin-bracket rendering) does not need to know which test ran.

    Parameters
    ----------
    df, metric, group_col, groups, pairs
        Same semantics as :func:`tukey_hsd_pairwise`.
    test : str, optional
        One of ``'tukey'`` (default), ``'mannwhitney'`` (pairwise
        Mann-Whitney U with Bonferroni correction across the requested
        pairs), or ``'kruskal'`` (Kruskal-Wallis omnibus followed by
        Dunn's pairwise post-hoc with Bonferroni correction).

    Returns
    -------
    pd.DataFrame
        Columns ``group_a``, ``group_b``, ``mean_diff``, ``p_adj``.
        ``mean_diff`` is the mean of *group_a* minus *group_b*
        (always present, even for non-parametric tests where it is
        descriptive only).

    Raises
    ------
    ValueError
        If *test* is not one of the three supported values, or if
        *metric* / *group_col* is missing from *df*.
    """
    if test == "tukey":
        return tukey_hsd_pairwise(
            df, metric, group_col=group_col, groups=groups, pairs=pairs,
        )
    if test == "mannwhitney":
        return _pairwise_mannwhitney(
            df, metric, group_col=group_col, groups=groups, pairs=pairs,
        )
    if test == "kruskal":
        return _pairwise_kruskal_dunn(
            df, metric, group_col=group_col, groups=groups, pairs=pairs,
        )
    raise ValueError(
        f"Unknown test '{test}'.  Choose from "
        "'tukey', 'mannwhitney', 'kruskal'."
    )


def compute_icc(
    df: pd.DataFrame,
    metric: str,
    targets_col: str,
    raters_col: str,
) -> pd.DataFrame:
    """Compute the intraclass correlation coefficient (ICC2,k).

    Uses ``pingouin.intraclass_corr`` with a two-way random-effects model
    (ICC2,k — average measures).  This quantifies how consistently a
    metric is expressed across raters (e.g. wells within a condition)
    for the same target units (e.g. electrodes).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data.  Must contain *metric*, *targets_col*, and
        *raters_col* columns.  One row per (target, rater) combination.
    metric : str
        Name of the numeric ratings column.
    targets_col : str
        Column identifying the targets (subjects), e.g. ``'electrode_id'``.
    raters_col : str
        Column identifying the raters, e.g. ``'well_id'``.

    Returns
    -------
    pd.DataFrame
        Full ``pingouin.intraclass_corr`` output filtered to the ICC2
        and ICC2k rows.  Key columns: ``Type``, ``ICC``, ``CI95%``,
        ``F``, ``pval``.

    Raises
    ------
    ImportError
        If ``pingouin`` is not installed.
    ValueError
        If required columns are missing.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame({
    ...     "electrode_id": ["e1","e1","e2","e2"],
    ...     "well_id":      ["w1","w2","w1","w2"],
    ...     "mfr_hz":       rng.normal(2, 0.1, 4),
    ... })
    >>> icc_df = compute_icc(df, "mfr_hz", "electrode_id", "well_id")
    >>> "ICC" in icc_df.columns
    True
    """
    try:
        import pingouin as pg
    except ImportError as exc:
        raise ImportError(
            "pingouin is required for compute_icc(). "
            "Install it with: pip install pingouin"
        ) from exc

    _check_columns(df, [metric, targets_col, raters_col])
    result = pg.intraclass_corr(
        data=df,
        targets=targets_col,
        raters=raters_col,
        ratings=metric,
    )
    # Return the two-way random models (ICC2 single, ICC2k average).
    return result[result["Type"].isin(["ICC2", "ICC2k"])].reset_index(drop=True)


def longitudinal_model(
    df: pd.DataFrame,
    metric: str,
    time_col: str,
    group_col: str,
    subject_col: str = "replicate_id",
) -> pd.DataFrame:
    """Fit a linear mixed-effects model with a time × group interaction.

    Model formula (Wilkinson notation)::

        metric ~ time_col * group_col

    with *subject_col* as the grouping variable for random intercepts.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data, one row per (well, DIV) observation.  Must
        contain *metric*, *time_col*, *group_col*, and *subject_col*.
    metric : str
        Dependent variable (e.g. ``'network_burst_rate'``).
    time_col : str
        Time variable (e.g. ``'DIV'``).  Treated as numeric.
    group_col : str
        Condition column (e.g. ``'condition'``).  Treated as categorical.
    subject_col : str, optional
        Random-intercept grouping variable.  Default ``'replicate_id'``.

    Returns
    -------
    pd.DataFrame
        Coefficient table with columns ``Coef.``, ``Std.Err.``, ``z``,
        ``P>|z|``, ``[0.025``, ``0.975]``.  The first-listed condition
        level is used as the reference category.

    Raises
    ------
    ImportError
        If ``statsmodels`` is not installed.
    ValueError
        If required columns are missing or the model fails to converge.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(2)
    >>> df = pd.DataFrame({
    ...     "mfr":          rng.normal(1, 0.1, 40),
    ...     "DIV":          [14, 21] * 20,
    ...     "condition":    ["WT", "KD"] * 20,
    ...     "replicate_id": [f"r{i}" for i in range(20)] * 2,
    ... })
    >>> coef = longitudinal_model(df, "mfr", "DIV", "condition")
    >>> "Coef." in coef.columns
    True
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for longitudinal_model(). "
            "Install it with: pip install statsmodels"
        ) from exc

    required = [metric, time_col, group_col, subject_col]
    _check_columns(df, required)
    data = df[required].dropna()

    if len(data) < 4:
        raise ValueError(
            "Too few complete observations to fit a mixed-effects model."
        )

    # Sanitise column names for the formula (replace spaces / special chars).
    rename = {
        metric:      "_metric",
        time_col:    "_time",
        group_col:   "_group",
        subject_col: "_subject",
    }
    data = data.rename(columns=rename)
    formula = "_metric ~ _time * C(_group)"

    try:
        model = smf.mixedlm(formula, data, groups=data["_subject"])
        result = model.fit(reml=True, disp=False)
    except Exception as exc:
        raise ValueError(
            f"Mixed-effects model failed to converge: {exc}"
        ) from exc

    coef = result.summary().tables[1]
    # statsmodels returns the table as a SimpleTable; convert to DataFrame.
    if not isinstance(coef, pd.DataFrame):
        coef = pd.read_html(coef.as_html(), header=0)[0]

    return coef


# ── Private helpers ───────────────────────────────────────────────────────────

def _resolve_pairs(
    groups: List[str],
    pairs: Optional[Sequence[Tuple[str, str]]],
) -> List[Tuple[str, str]]:
    """Resolve the user's *pairs* arg into a deduplicated list of pairs.

    Pair labels are normalised to sorted ``(a, b)`` order to make the
    output of pairwise tests deterministic and comparable across tests.
    """
    if pairs is None:
        out: List[Tuple[str, str]] = []
        for i, ga in enumerate(groups):
            for gb in groups[i + 1:]:
                out.append(tuple(sorted((ga, gb))))
        return out
    seen: set = set()
    out: List[Tuple[str, str]] = []
    for a, b in pairs:
        if a not in groups or b not in groups or a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _pairwise_mannwhitney(
    df: pd.DataFrame,
    metric: str,
    *,
    group_col: str,
    groups: Optional[Sequence[str]],
    pairs: Optional[Sequence[Tuple[str, str]]],
) -> pd.DataFrame:
    """Pairwise Mann-Whitney U with Bonferroni correction.

    For each requested pair, runs a two-sided MW U test and applies
    Bonferroni correction across the number of comparisons actually
    tested.  Pairs whose data have fewer than 2 observations on either
    side are silently skipped.
    """
    _check_columns(df, [metric, group_col])
    data = df[[metric, group_col]].dropna(subset=[metric])
    all_groups = sorted(data[group_col].dropna().unique().tolist())
    groups_to_use = list(groups) if groups is not None else all_groups
    candidates = _resolve_pairs(groups_to_use, pairs)

    rows: List[Dict[str, object]] = []
    pre_p: List[Tuple[str, str, float, float]] = []  # (a, b, raw_p, mean_diff)
    for ga, gb in candidates:
        a = data.loc[data[group_col] == ga, metric].values
        b = data.loc[data[group_col] == gb, metric].values
        if len(a) < 2 or len(b) < 2:
            continue
        try:
            _, p = mannwhitneyu(np.asarray(a, dtype=float),
                                np.asarray(b, dtype=float),
                                alternative="two-sided")
        except Exception:  # noqa: BLE001
            continue
        pre_p.append((ga, gb, float(p), float(np.mean(a) - np.mean(b))))

    if not pre_p:
        return pd.DataFrame(columns=["group_a", "group_b", "mean_diff", "p_adj"])

    n = len(pre_p)
    for ga, gb, raw_p, diff in pre_p:
        rows.append({
            "group_a":   ga,
            "group_b":   gb,
            "mean_diff": diff,
            "p_adj":     min(raw_p * n, 1.0),
        })
    return pd.DataFrame(rows, columns=["group_a", "group_b", "mean_diff", "p_adj"])


def _pairwise_kruskal_dunn(
    df: pd.DataFrame,
    metric: str,
    *,
    group_col: str,
    groups: Optional[Sequence[str]],
    pairs: Optional[Sequence[Tuple[str, str]]],
) -> pd.DataFrame:
    """Pairwise post-hoc following a Kruskal-Wallis omnibus test.

    Returns Dunn's pairwise p-values (Bonferroni-corrected).  When the
    omnibus Kruskal-Wallis test is itself not significant, the
    post-hoc p-values are still reported for users who prefer to
    interpret pairwise comparisons directly; the ``mean_diff`` column
    is the descriptive difference of group means.
    """
    _check_columns(df, [metric, group_col])
    data = df[[metric, group_col]].dropna(subset=[metric])
    all_groups = sorted(data[group_col].dropna().unique().tolist())
    groups_to_use = list(groups) if groups is not None else all_groups

    group_arrays = {
        g: np.asarray(data.loc[data[group_col] == g, metric].values, dtype=float)
        for g in groups_to_use
    }
    valid = {g: arr for g, arr in group_arrays.items() if len(arr) >= 2}
    if len(valid) < 2:
        return pd.DataFrame(columns=["group_a", "group_b", "mean_diff", "p_adj"])

    posthoc = _dunn_test(valid)  # already Bonferroni-corrected by default

    candidates = _resolve_pairs(list(valid.keys()), pairs)
    pair_set = {tuple(sorted(p)) for p in candidates}

    rows: List[Dict[str, object]] = []
    for _, prow in posthoc.iterrows():
        ga = prow.get("group1")
        gb = prow.get("group2")
        if ga is None or gb is None:
            continue
        key = tuple(sorted((str(ga), str(gb))))
        if key not in pair_set:
            continue
        p_adj = float(prow.get("p_adjusted", prow.get("p_value", 1.0)))
        a = valid[str(ga)]
        b = valid[str(gb)]
        # Sort labels alphabetically per row for determinism.
        if str(ga) <= str(gb):
            label_a, label_b = str(ga), str(gb)
            diff = float(np.mean(a) - np.mean(b))
        else:
            label_a, label_b = str(gb), str(ga)
            diff = float(np.mean(b) - np.mean(a))
        rows.append({
            "group_a":   label_a,
            "group_b":   label_b,
            "mean_diff": diff,
            "p_adj":     p_adj,
        })

    return pd.DataFrame(rows, columns=["group_a", "group_b", "mean_diff", "p_adj"])


def _check_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing}. "
            f"Available: {list(df.columns)}"
        )


def _mannwhitney(group_arrays: Dict) -> CompareResult:
    """Mann-Whitney U test for exactly two groups."""
    keys = list(group_arrays.keys())
    a, b = group_arrays[keys[0]], group_arrays[keys[1]]
    n1, n2 = len(a), len(b)

    stat, p = mannwhitneyu(a, b, alternative="two-sided")

    # Rank-biserial r: r = 1 − 2U / (n1·n2)
    # Ranges [−1, 1]; |r| > 0.3 medium, > 0.5 large.
    effect_size = 1.0 - (2.0 * stat) / (n1 * n2) if n1 * n2 > 0 else 0.0

    return CompareResult(
        test="mannwhitney",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(effect_size),
        posthoc=None,
    )


def _kruskal_dunn(group_arrays: Dict) -> CompareResult:
    """Kruskal-Wallis H test + Dunn's pairwise post-hoc."""
    keys = list(group_arrays.keys())
    arrays = [group_arrays[k] for k in keys]

    stat, p = kruskal(*arrays)

    # Epsilon-squared effect size: ε² = H / ((N²−1)/(N+1)) ≈ H/(N−1)
    N = sum(len(a) for a in arrays)
    effect_size = float(stat) / (N - 1) if N > 1 else 0.0

    posthoc = _dunn_test(group_arrays)

    return CompareResult(
        test="kruskal",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(effect_size),
        posthoc=posthoc,
    )


def _dunn_test(
    group_arrays: Dict,
    adjust: str = "bonferroni",
) -> pd.DataFrame:
    """Dunn's pairwise post-hoc test following Kruskal-Wallis.

    Parameters
    ----------
    group_arrays : dict[str, np.ndarray]
        Group name → values array.
    adjust : str
        Multiple-comparison correction.  ``'bonferroni'`` (default) or
        ``'none'``.

    Returns
    -------
    pd.DataFrame
        One row per pair with columns:
        ``group1``, ``group2``, ``z_stat``, ``p_value``, ``p_adjusted``.
    """
    keys = list(group_arrays.keys())
    all_data = np.concatenate([group_arrays[k] for k in keys])
    N = len(all_data)

    # Rank all observations together.
    ranks = rankdata(all_data)

    # Compute mean rank and group size per group.
    mean_rank: Dict[str, float] = {}
    ns: Dict[str, int] = {}
    cursor = 0
    for k in keys:
        n = len(group_arrays[k])
        mean_rank[k] = float(ranks[cursor : cursor + n].mean())
        ns[k] = n
        cursor += n

    # Pairwise z-statistics.
    rows = []
    pairs = list(combinations(keys, 2))
    for ki, kj in pairs:
        ni, nj = ns[ki], ns[kj]
        se = np.sqrt(N * (N + 1) / 12.0 * (1.0 / ni + 1.0 / nj))
        z = (mean_rank[ki] - mean_rank[kj]) / se if se > 0 else 0.0
        p = float(2.0 * norm.sf(abs(z)))
        rows.append({"group1": ki, "group2": kj, "z_stat": float(z), "p_value": p})

    posthoc = pd.DataFrame(rows)

    if adjust == "bonferroni":
        posthoc["p_adjusted"] = np.minimum(posthoc["p_value"] * len(pairs), 1.0)
    else:
        posthoc["p_adjusted"] = posthoc["p_value"]

    return posthoc.reset_index(drop=True)
