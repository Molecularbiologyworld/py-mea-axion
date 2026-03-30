"""
tests/test_stats.py
====================
Tests for py_mea_axion.stats.compare.

Analytic ground-truth cases
----------------------------
IDENTICAL_GROUPS
    Two groups drawn from the same distribution → p-value should NOT
    be significant at α = 0.05 (most of the time; we use a fixed seed).

SEPARATED_GROUPS
    Two clearly separated groups (no overlap) → U = 0 or n1*n2,
    p ≈ 0, |r| = 1.

THREE_GROUPS
    Three groups: one pair clearly different, one similar pair.
    Kruskal-Wallis should be significant; Dunn's should identify
    which pair differs.
"""

import math

import numpy as np
import pandas as pd
import pytest

from py_mea_axion.stats.compare import (
    CompareResult,
    _dunn_test,
    compare_conditions,
    compute_icc,
    longitudinal_model,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)

# Two identical distributions — should NOT be significant.
SAME = pd.DataFrame({
    "metric": RNG.normal(1.0, 0.3, 40),
    "condition": ["WT"] * 20 + ["KD"] * 20,
    "replicate_id": [f"r{i}" for i in range(20)] * 2,
    "DIV": [14] * 40,
})

# Two completely separated groups — must be significant.
SEP = pd.DataFrame({
    "metric": np.r_[np.ones(20), np.ones(20) * 10],
    "condition": ["WT"] * 20 + ["KD"] * 20,
    "replicate_id": [f"r{i}" for i in range(20)] * 2,
    "DIV": [14] * 40,
})

# Three groups: A≈B, C clearly different.
THREE = pd.DataFrame({
    "metric": np.r_[
        RNG.normal(1.0, 0.2, 15),   # A
        RNG.normal(1.1, 0.2, 15),   # B (similar to A)
        RNG.normal(5.0, 0.2, 15),   # C (very different)
    ],
    "condition": ["A"] * 15 + ["B"] * 15 + ["C"] * 15,
    "replicate_id": [f"r{i}" for i in range(15)] * 3,
    "DIV": [14] * 45,
})


# ── CompareResult namedtuple ──────────────────────────────────────────────────

class TestCompareResultFields:
    def test_fields(self):
        assert CompareResult._fields == (
            "test", "statistic", "p_value", "effect_size", "posthoc"
        )


# ── Mann-Whitney: two groups ──────────────────────────────────────────────────

class TestMannWhitney:
    def test_separated_p_near_zero(self):
        res = compare_conditions(SEP, "metric", "condition")
        assert res.p_value < 1e-6

    def test_separated_effect_size_magnitude_one(self):
        res = compare_conditions(SEP, "metric", "condition")
        assert abs(res.effect_size) == pytest.approx(1.0, abs=1e-9)

    def test_same_dist_not_significant(self):
        res = compare_conditions(SAME, "metric", "condition")
        assert res.p_value > 0.01   # loosely: not p<0.01

    def test_test_name(self):
        res = compare_conditions(SEP, "metric", "condition")
        assert res.test == "mannwhitney"

    def test_posthoc_is_none(self):
        res = compare_conditions(SEP, "metric", "condition")
        assert res.posthoc is None

    def test_statistic_is_float(self):
        res = compare_conditions(SEP, "metric", "condition")
        assert isinstance(res.statistic, float)

    def test_p_value_in_range(self):
        res = compare_conditions(SAME, "metric", "condition")
        assert 0.0 <= res.p_value <= 1.0

    def test_effect_size_in_range(self):
        res = compare_conditions(SAME, "metric", "condition")
        assert -1.0 <= res.effect_size <= 1.0

    def test_symmetry(self):
        # Swapping groups shouldn't change |effect_size| or p_value.
        df_swap = SAME.copy()
        df_swap["condition"] = df_swap["condition"].map({"WT": "KD", "KD": "WT"})
        res1 = compare_conditions(SAME, "metric", "condition")
        res2 = compare_conditions(df_swap, "metric", "condition")
        assert res1.p_value == pytest.approx(res2.p_value, abs=1e-12)
        assert abs(res1.effect_size) == pytest.approx(abs(res2.effect_size), abs=1e-9)

    def test_nan_rows_dropped(self):
        df = SEP.copy()
        df.loc[0, "metric"] = np.nan
        res = compare_conditions(df, "metric", "condition")
        assert res.p_value < 1e-5   # still significant

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="not found"):
            compare_conditions(SEP, "nonexistent", "condition")

    def test_single_group_raises(self):
        df = SAME[SAME["condition"] == "WT"].copy()
        with pytest.raises(ValueError, match="at least 2 groups"):
            compare_conditions(df, "metric", "condition")


# ── Kruskal-Wallis + Dunn: three groups ──────────────────────────────────────

class TestKruskalDunn:
    @pytest.fixture()
    def res(self):
        return compare_conditions(THREE, "metric", "condition")

    def test_test_name(self, res):
        assert res.test == "kruskal"

    def test_significant(self, res):
        assert res.p_value < 0.001

    def test_effect_size_positive(self, res):
        assert res.effect_size > 0

    def test_posthoc_is_dataframe(self, res):
        assert isinstance(res.posthoc, pd.DataFrame)

    def test_posthoc_columns(self, res):
        for col in ("group1", "group2", "z_stat", "p_value", "p_adjusted"):
            assert col in res.posthoc.columns

    def test_posthoc_row_count_three_groups(self, res):
        # C(3,2) = 3 pairs
        assert len(res.posthoc) == 3

    def test_ac_pair_significant(self, res):
        # A vs C should be significant after correction.
        ac = res.posthoc[
            ((res.posthoc["group1"] == "A") & (res.posthoc["group2"] == "C")) |
            ((res.posthoc["group1"] == "C") & (res.posthoc["group2"] == "A"))
        ]
        assert len(ac) == 1
        assert ac["p_adjusted"].iloc[0] < 0.05

    def test_bc_pair_significant(self, res):
        bc = res.posthoc[
            ((res.posthoc["group1"] == "B") & (res.posthoc["group2"] == "C")) |
            ((res.posthoc["group1"] == "C") & (res.posthoc["group2"] == "B"))
        ]
        assert bc["p_adjusted"].iloc[0] < 0.05

    def test_ab_pair_not_significant(self, res):
        # A and B are similar → should NOT be significant.
        ab = res.posthoc[
            ((res.posthoc["group1"] == "A") & (res.posthoc["group2"] == "B")) |
            ((res.posthoc["group1"] == "B") & (res.posthoc["group2"] == "A"))
        ]
        assert ab["p_adjusted"].iloc[0] > 0.05

    def test_p_adjusted_ge_p_value(self, res):
        assert (res.posthoc["p_adjusted"] >= res.posthoc["p_value"] - 1e-12).all()

    def test_p_adjusted_le_one(self, res):
        assert (res.posthoc["p_adjusted"] <= 1.0).all()


# ── Dunn's test internals ─────────────────────────────────────────────────────

class TestDunnTest:
    def test_returns_dataframe(self):
        groups = {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([8.0, 9.0, 10.0])}
        df = _dunn_test(groups)
        assert isinstance(df, pd.DataFrame)

    def test_separated_pair_significant(self):
        groups = {"A": np.ones(20), "B": np.ones(20) * 10}
        df = _dunn_test(groups)
        assert df["p_value"].iloc[0] < 0.001

    def test_bonferroni_inflates_p(self):
        groups = {
            "A": np.array([1.0, 2.0, 3.0]),
            "B": np.array([5.0, 6.0, 7.0]),
            "C": np.array([1.1, 2.1, 3.1]),
        }
        df = _dunn_test(groups, adjust="bonferroni")
        # Bonferroni multiplies by number of pairs (3) → p_adj >= p_val
        assert (df["p_adjusted"] >= df["p_value"] - 1e-12).all()


# ── compute_icc ───────────────────────────────────────────────────────────────

class TestComputeIcc:
    @pytest.fixture()
    def icc_df(self):
        # Reproducible data: 4 electrodes × 3 wells.
        rng = np.random.default_rng(10)
        electrode_ids = ["e1", "e2", "e3", "e4"] * 3
        well_ids = ["w1"] * 4 + ["w2"] * 4 + ["w3"] * 4
        mfr = rng.normal(2.0, 0.2, 12)
        return pd.DataFrame({
            "electrode_id": electrode_ids,
            "well_id": well_ids,
            "mfr_hz": mfr,
        })

    def test_returns_dataframe(self, icc_df):
        result = compute_icc(icc_df, "mfr_hz", "electrode_id", "well_id")
        assert isinstance(result, pd.DataFrame)

    def test_icc_column_present(self, icc_df):
        result = compute_icc(icc_df, "mfr_hz", "electrode_id", "well_id")
        assert "ICC" in result.columns

    def test_type_column_present(self, icc_df):
        result = compute_icc(icc_df, "mfr_hz", "electrode_id", "well_id")
        assert "Type" in result.columns

    def test_returns_icc2_rows(self, icc_df):
        result = compute_icc(icc_df, "mfr_hz", "electrode_id", "well_id")
        assert set(result["Type"]).issubset({"ICC2", "ICC2k"})

    def test_icc_in_range(self, icc_df):
        result = compute_icc(icc_df, "mfr_hz", "electrode_id", "well_id")
        assert (result["ICC"] >= -1.0).all()
        assert (result["ICC"] <= 1.0).all()

    def test_missing_column_raises(self, icc_df):
        with pytest.raises(ValueError, match="not found"):
            compute_icc(icc_df, "nonexistent", "electrode_id", "well_id")


# ── longitudinal_model ────────────────────────────────────────────────────────

class TestLongitudinalModel:
    @pytest.fixture()
    def long_df(self):
        rng = np.random.default_rng(20)
        divs = [14, 21, 28]
        conditions = ["WT", "KD"]
        reps = ["r1", "r2", "r3"]
        rows = []
        for div in divs:
            for cond in conditions:
                for rep in reps:
                    base = (2.0 if cond == "WT" else 1.5) + 0.1 * (div - 14)
                    rows.append({
                        "mfr": base + rng.normal(0, 0.1),
                        "DIV": div,
                        "condition": cond,
                        "replicate_id": f"{rep}_{cond}",
                    })
        return pd.DataFrame(rows)

    def test_returns_dataframe(self, long_df):
        result = longitudinal_model(long_df, "mfr", "DIV", "condition")
        assert isinstance(result, pd.DataFrame)

    def test_coef_column_present(self, long_df):
        result = longitudinal_model(long_df, "mfr", "DIV", "condition")
        assert "Coef." in result.columns

    def test_has_rows(self, long_df):
        result = longitudinal_model(long_df, "mfr", "DIV", "condition")
        assert len(result) > 0

    def test_missing_column_raises(self, long_df):
        with pytest.raises(ValueError, match="not found"):
            longitudinal_model(long_df, "nonexistent", "DIV", "condition")

    def test_too_few_rows_raises(self):
        df = pd.DataFrame({
            "mfr": [1.0, 2.0],
            "DIV": [14, 21],
            "condition": ["WT", "KD"],
            "replicate_id": ["r1", "r2"],
        })
        with pytest.raises(ValueError):
            longitudinal_model(df, "mfr", "DIV", "condition")

    def test_custom_subject_col(self, long_df):
        long_df = long_df.rename(columns={"replicate_id": "subject"})
        result = longitudinal_model(
            long_df, "mfr", "DIV", "condition", subject_col="subject"
        )
        assert isinstance(result, pd.DataFrame)
