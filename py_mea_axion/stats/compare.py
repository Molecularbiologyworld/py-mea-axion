"""
stats/compare.py
================
Statistical comparison tools for MEA experiment data.

Three levels of analysis are provided:

1. **Group comparison** (:func:`compare_conditions`)
   Two conditions → Mann-Whitney U + rank-biserial *r*.
   Three or more conditions → Kruskal-Wallis *H* + Dunn's post-hoc
   with Bonferroni correction.

2. **Intraclass correlation** (:func:`compute_icc`)
   ICC(2,k) for assessing electrode-to-electrode or well-to-well
   consistency within a condition, via ``pingouin``.

3. **Longitudinal mixed-effects model** (:func:`longitudinal_model`)
   Linear mixed-effects model with a time × group interaction term
   (``statsmodels`` MixedLM), returning the full coefficient table.

Public API
----------
CompareResult  (namedtuple)
compare_conditions(df, metric, group_col, test='mannwhitney')
compute_icc(df, metric, targets_col, raters_col)
longitudinal_model(df, metric, time_col, group_col, subject_col)
"""

from collections import namedtuple
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal, norm, rankdata

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
