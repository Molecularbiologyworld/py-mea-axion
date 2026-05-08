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
    pairwise_test,
    tukey_hsd_pairwise,
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


# ── tukey_hsd_pairwise ────────────────────────────────────────────────────────

class TestTukeyHsdPairwise:
    """Pairwise Tukey HSD across conditions."""

    def _three_group_df(self, separated: bool = True) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        # SCRM around 2.0, KD4 around 4.0 (well separated), KD5 around 4.0 too.
        if separated:
            mus = {"SCRM": 2.0, "KD4": 4.0, "KD5": 4.0}
        else:
            mus = {"SCRM": 2.0, "KD4": 2.0, "KD5": 2.0}
        for cond, mu in mus.items():
            for _ in range(20):
                rows.append({"metric": rng.normal(mu, 0.3), "condition": cond})
        return pd.DataFrame(rows)

    def test_returns_dataframe_with_expected_columns(self):
        df = self._three_group_df()
        res = tukey_hsd_pairwise(df, "metric")
        assert list(res.columns) == ["group_a", "group_b", "mean_diff", "p_adj"]

    def test_three_groups_yield_three_pairs(self):
        df = self._three_group_df()
        res = tukey_hsd_pairwise(df, "metric")
        assert len(res) == 3

    def test_separated_groups_significant(self):
        df = self._three_group_df(separated=True)
        res = tukey_hsd_pairwise(df, "metric")
        # SCRM vs KD4 and SCRM vs KD5 should be significant; KD4 vs KD5 not.
        scrm_kd4 = res[
            ((res["group_a"] == "KD4") & (res["group_b"] == "SCRM")) |
            ((res["group_a"] == "SCRM") & (res["group_b"] == "KD4"))
        ]
        kd4_kd5 = res[
            ((res["group_a"] == "KD4") & (res["group_b"] == "KD5")) |
            ((res["group_a"] == "KD5") & (res["group_b"] == "KD4"))
        ]
        assert float(scrm_kd4["p_adj"].iloc[0]) < 0.001
        assert float(kd4_kd5["p_adj"].iloc[0]) > 0.05

    def test_identical_groups_not_significant(self):
        df = self._three_group_df(separated=False)
        res = tukey_hsd_pairwise(df, "metric")
        # All pairs should be ns at alpha=0.05.
        assert (res["p_adj"] > 0.05).all()

    def test_pairs_filter_restricts_output(self):
        df = self._three_group_df()
        res = tukey_hsd_pairwise(
            df, "metric",
            pairs=[("SCRM", "KD4")],
        )
        assert len(res) == 1
        # Labels are alphabetised within each row.
        assert res.iloc[0]["group_a"] == "KD4"
        assert res.iloc[0]["group_b"] == "SCRM"

    def test_groups_subset(self):
        df = self._three_group_df()
        res = tukey_hsd_pairwise(df, "metric", groups=["SCRM", "KD4"])
        assert len(res) == 1

    def test_unknown_pair_silently_skipped(self):
        df = self._three_group_df()
        res = tukey_hsd_pairwise(
            df, "metric",
            pairs=[("SCRM", "GHOST"), ("SCRM", "KD4")],
        )
        assert len(res) == 1

    def test_self_pair_skipped(self):
        df = self._three_group_df()
        res = tukey_hsd_pairwise(
            df, "metric",
            pairs=[("SCRM", "SCRM")],
        )
        assert res.empty

    def test_singleton_group_dropped(self):
        # Group "KD5" has only one observation; Tukey can't include it.
        df = pd.DataFrame({
            "metric":    [1.0, 1.1, 1.0, 4.0, 4.1, 4.0, 7.0],
            "condition": ["SCRM"] * 3 + ["KD4"] * 3 + ["KD5"],
        })
        res = tukey_hsd_pairwise(df, "metric")
        # Only the SCRM-KD4 pair survives.
        assert len(res) == 1
        labels = {res.iloc[0]["group_a"], res.iloc[0]["group_b"]}
        assert labels == {"SCRM", "KD4"}

    def test_too_few_groups_returns_empty(self):
        df = pd.DataFrame({
            "metric":    [1.0, 1.1, 1.2],
            "condition": ["SCRM", "SCRM", "SCRM"],
        })
        res = tukey_hsd_pairwise(df, "metric")
        assert res.empty

    def test_missing_metric_raises(self):
        df = pd.DataFrame({"condition": ["A", "B"], "metric": [1.0, 2.0]})
        with pytest.raises(ValueError, match="not found"):
            tukey_hsd_pairwise(df, "ghost", group_col="condition")

    def test_missing_group_col_raises(self):
        df = pd.DataFrame({"condition": ["A", "B"], "metric": [1.0, 2.0]})
        with pytest.raises(ValueError, match="not found"):
            tukey_hsd_pairwise(df, "metric", group_col="ghost")

    def test_drops_nan_metric_rows(self):
        df = pd.DataFrame({
            "metric":    [1.0, 1.1, np.nan, 4.0, 4.1, 4.2],
            "condition": ["A", "A", "A", "B", "B", "B"],
        })
        res = tukey_hsd_pairwise(df, "metric")
        # Should still return one pair without crashing.
        assert len(res) == 1


class TestTukeyHsdImportableFromModule:
    def test_importable_from_stats_top_level(self):
        # The function should be exposed at py_mea_axion.stats.tukey_hsd_pairwise.
        from py_mea_axion.stats import tukey_hsd_pairwise as imported
        assert callable(imported)


# ── pairwise_test dispatcher ──────────────────────────────────────────────────

class TestPairwiseTest:
    def _three_group_df(self, separated: bool = True) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        if separated:
            mus = {"SCRM": 2.0, "KD4": 4.0, "KD5": 4.0}
        else:
            mus = {"SCRM": 2.0, "KD4": 2.0, "KD5": 2.0}
        for cond, mu in mus.items():
            for _ in range(20):
                rows.append({"metric": rng.normal(mu, 0.3), "condition": cond})
        return pd.DataFrame(rows)

    def test_tukey_default(self):
        df = self._three_group_df()
        res = pairwise_test(df, "metric")
        assert list(res.columns) == ["group_a", "group_b", "mean_diff", "p_adj"]
        assert len(res) == 3

    def test_mannwhitney_three_groups_returns_three_pairs(self):
        df = self._three_group_df()
        res = pairwise_test(df, "metric", test="mannwhitney")
        assert len(res) == 3
        assert list(res.columns) == ["group_a", "group_b", "mean_diff", "p_adj"]

    def test_mannwhitney_two_groups(self):
        df = self._three_group_df()
        df = df[df["condition"].isin(["SCRM", "KD4"])]
        res = pairwise_test(df, "metric", test="mannwhitney")
        assert len(res) == 1

    def test_kruskal_dunn_three_groups(self):
        df = self._three_group_df()
        res = pairwise_test(df, "metric", test="kruskal")
        assert len(res) == 3
        assert list(res.columns) == ["group_a", "group_b", "mean_diff", "p_adj"]

    def test_separated_groups_significant_across_all_three_tests(self):
        df = self._three_group_df(separated=True)
        for test in ("tukey", "mannwhitney", "kruskal"):
            res = pairwise_test(df, "metric", test=test)
            scrm_kd4 = res[
                ((res["group_a"] == "KD4") & (res["group_b"] == "SCRM")) |
                ((res["group_a"] == "SCRM") & (res["group_b"] == "KD4"))
            ]
            assert float(scrm_kd4["p_adj"].iloc[0]) < 0.05, (
                f"Test '{test}' did not detect a significant difference."
            )

    def test_pairs_filter_works_for_all_tests(self):
        df = self._three_group_df()
        for test in ("tukey", "mannwhitney", "kruskal"):
            res = pairwise_test(
                df, "metric", test=test,
                pairs=[("SCRM", "KD4")],
            )
            assert len(res) == 1, f"Test '{test}' did not honour pairs filter."

    def test_unknown_test_raises(self):
        df = self._three_group_df()
        with pytest.raises(ValueError, match="Unknown test"):
            pairwise_test(df, "metric", test="welch")

    def test_singleton_group_dropped_in_mannwhitney(self):
        df = pd.DataFrame({
            "metric":    [1.0, 1.1, 1.0, 4.0, 4.1, 4.0, 7.0],
            "condition": ["SCRM"] * 3 + ["KD4"] * 3 + ["KD5"],
        })
        res = pairwise_test(df, "metric", test="mannwhitney")
        # Only the SCRM-KD4 pair survives.
        assert len(res) == 1
        labels = {res.iloc[0]["group_a"], res.iloc[0]["group_b"]}
        assert labels == {"SCRM", "KD4"}

    def test_two_groups_with_kruskal_still_works(self):
        # Kruskal-Wallis on 2 groups reduces to a Mann-Whitney-like test;
        # the dispatcher should still return one pair.
        df = self._three_group_df()
        df = df[df["condition"].isin(["SCRM", "KD4"])]
        res = pairwise_test(df, "metric", test="kruskal")
        assert len(res) == 1


class TestPairwiseTestImportableFromModule:
    def test_importable_from_stats_top_level(self):
        from py_mea_axion.stats import pairwise_test as imported
        assert callable(imported)
