"""
tests/test_synchrony.py
========================
Tests for py_mea_axion.network.synchrony (STTC).

Key analytic properties of STTC that we verify:
  1. Identical trains   → STTC = 1.0
  2. Shifted trains (shift >> dt) → STTC ≈ 0
  3. Independent Poisson trains   → STTC ≈ 0 (within statistical tolerance)
  4. Symmetry: STTC(A,B) == STTC(B,A)
  5. Empty train(s)     → STTC = 0.0
  6. STTC always in [−1, 1]
"""

import math

import numpy as np
import pandas as pd
import pytest

from py_mea_axion.network.synchrony import mean_sttc, sttc, sttc_matrix


# ── sttc: basic properties ────────────────────────────────────────────────────

class TestSttcProperties:
    DT = 0.05
    T = 100.0
    REGULAR = np.arange(0.0, 100.0, 0.1)   # 1000 spikes at 10 Hz

    def test_identical_trains_sttc_one(self):
        val = sttc(self.REGULAR, self.REGULAR, self.DT, self.T)
        assert val == pytest.approx(1.0, abs=1e-9)

    def test_symmetry(self):
        rng = np.random.default_rng(0)
        a = np.sort(rng.uniform(0, self.T, 200))
        b = np.sort(rng.uniform(0, self.T, 150))
        assert sttc(a, b, self.DT, self.T) == pytest.approx(
            sttc(b, a, self.DT, self.T), abs=1e-12
        )

    def test_non_overlapping_trains_negative(self):
        # A fires in [0, 50), B fires in [60, 100) → no coincidences.
        # P_A = P_B = 0, T_X > 0  →  STTC = 0.5·(-T_B - T_A) < 0.
        a = np.arange(0.0, 50.0, 0.5)
        b = np.arange(60.0, 100.0, 0.5)
        val = sttc(a, b, dt_s=0.05, total_time_s=self.T)
        assert val < 0

    def test_empty_a_returns_zero(self):
        assert sttc(np.array([]), self.REGULAR, self.DT, self.T) == 0.0

    def test_empty_b_returns_zero(self):
        assert sttc(self.REGULAR, np.array([]), self.DT, self.T) == 0.0

    def test_both_empty_returns_zero(self):
        assert sttc(np.array([]), np.array([]), self.DT, self.T) == 0.0

    def test_zero_duration_returns_zero(self):
        assert sttc(self.REGULAR, self.REGULAR, self.DT, 0.0) == 0.0

    def test_range_minus_one_to_one(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            a = np.sort(rng.uniform(0, self.T, rng.integers(5, 200)))
            b = np.sort(rng.uniform(0, self.T, rng.integers(5, 200)))
            val = sttc(a, b, self.DT, self.T)
            assert -1.0 - 1e-9 <= val <= 1.0 + 1e-9

    def test_large_dt_saturates_near_one(self):
        # dt = T/2 → each spike's window covers almost the whole recording
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([10.0, 20.0, 30.0])
        val = sttc(a, b, dt_s=self.T / 2, total_time_s=self.T)
        assert val == pytest.approx(1.0, abs=0.01)

    def test_independent_poisson_near_zero(self):
        rng = np.random.default_rng(42)
        a = np.sort(rng.uniform(0, self.T, 500))
        b = np.sort(rng.uniform(0, self.T, 500))
        val = sttc(a, b, dt_s=0.01, total_time_s=self.T)
        assert abs(val) < 0.15   # generous bound for random trains


# ── sttc: known analytic cases ────────────────────────────────────────────────

class TestSttcAnalytic:
    def test_single_spike_in_each_coincident(self):
        # Both trains have one spike at t=5.0 → perfect coincidence
        a = np.array([5.0])
        b = np.array([5.0])
        val = sttc(a, b, dt_s=0.1, total_time_s=10.0)
        # p_a = 1, p_b = 1 → each term = (1-T_B)/(1-T_B) = 1 → STTC = 1
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_single_spike_far_apart(self):
        a = np.array([1.0])
        b = np.array([9.0])
        val = sttc(a, b, dt_s=0.05, total_time_s=10.0)
        # No spike in a is near b (gap = 8 s >> 0.05) → p_a=0, p_b=0
        # term1 = (0 − T_B)/(1 − 0) = −T_B, term2 = −T_A → STTC < 0
        assert val < 0


# ── sttc_matrix ───────────────────────────────────────────────────────────────

class TestSttcMatrix:
    @pytest.fixture()
    def three_electrodes(self):
        rng = np.random.default_rng(7)
        base = np.sort(rng.uniform(0, 10, 100))
        return {
            "e1": base,
            "e2": base,                               # identical to e1
            "e3": np.sort(rng.uniform(0, 10, 100)),   # independent
        }

    def test_shape(self, three_electrodes):
        mat = sttc_matrix(three_electrodes, dt_s=0.05, total_time_s=10.0)
        assert mat.shape == (3, 3)

    def test_diagonal_is_one(self, three_electrodes):
        mat = sttc_matrix(three_electrodes, dt_s=0.05, total_time_s=10.0)
        for eid in mat.index:
            assert mat.loc[eid, eid] == pytest.approx(1.0)

    def test_symmetric(self, three_electrodes):
        mat = sttc_matrix(three_electrodes, dt_s=0.05, total_time_s=10.0)
        eids = list(mat.index)
        for i in range(len(eids)):
            for j in range(len(eids)):
                assert mat.iloc[i, j] == pytest.approx(mat.iloc[j, i], abs=1e-12)

    def test_identical_pair_is_one(self, three_electrodes):
        mat = sttc_matrix(three_electrodes, dt_s=0.05, total_time_s=10.0)
        assert mat.loc["e1", "e2"] == pytest.approx(1.0)

    def test_index_and_columns_sorted(self, three_electrodes):
        mat = sttc_matrix(three_electrodes, dt_s=0.05, total_time_s=10.0)
        assert list(mat.index) == sorted(mat.index)
        assert list(mat.columns) == sorted(mat.columns)

    def test_fewer_than_two_electrodes_returns_empty(self):
        mat = sttc_matrix({"e1": np.arange(0, 1, 0.1)}, dt_s=0.05, total_time_s=1.0)
        assert mat.empty

    def test_empty_dict_returns_empty(self):
        mat = sttc_matrix({}, dt_s=0.05, total_time_s=1.0)
        assert mat.empty

    def test_values_in_range(self, three_electrodes):
        mat = sttc_matrix(three_electrodes, dt_s=0.05, total_time_s=10.0)
        vals = mat.values.flatten()
        assert (vals >= -1.0 - 1e-9).all()
        assert (vals <= 1.0 + 1e-9).all()


# ── mean_sttc ─────────────────────────────────────────────────────────────────

class TestMeanSttc:
    def test_identical_trains_mean_one(self):
        a = np.arange(0.0, 10.0, 0.1)
        spikes = {"e1": a, "e2": a}
        val = mean_sttc(spikes, dt_s=0.05, total_time_s=10.0)
        assert val == pytest.approx(1.0, abs=1e-9)

    def test_fewer_than_two_active_returns_nan(self):
        spikes = {"e1": np.arange(0.0, 1.0, 0.1)}
        val = mean_sttc(spikes, dt_s=0.05, total_time_s=1.0)
        assert math.isnan(val)

    def test_empty_dict_returns_nan(self):
        assert math.isnan(mean_sttc({}, dt_s=0.05, total_time_s=10.0))

    def test_active_only_excludes_silent(self):
        a = np.arange(0.0, 10.0, 0.1)
        spikes = {"e1": a, "e2": a, "e3": np.array([])}
        val = mean_sttc(spikes, dt_s=0.05, total_time_s=10.0, active_only=True)
        # Only e1 & e2 are active → mean STTC = STTC(e1,e2) = 1.0
        assert val == pytest.approx(1.0, abs=1e-9)

    def test_active_only_false_includes_silent(self):
        a = np.arange(0.0, 10.0, 0.1)
        spikes = {"e1": a, "e2": a, "e3": np.array([])}
        val = mean_sttc(spikes, dt_s=0.05, total_time_s=10.0, active_only=False)
        # e3 is empty → STTC(e1,e3)=0, STTC(e2,e3)=0, STTC(e1,e2)=1
        assert val == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_mean_is_average_of_pairs(self):
        rng = np.random.default_rng(99)
        trains = {f"e{i}": np.sort(rng.uniform(0, 10, 50)) for i in range(4)}
        val = mean_sttc(trains, dt_s=0.05, total_time_s=10.0, active_only=False)
        mat = sttc_matrix(trains, dt_s=0.05, total_time_s=10.0)
        # Off-diagonal mean
        n = len(mat)
        off_diag = [mat.iloc[i, j] for i in range(n) for j in range(i + 1, n)]
        expected = sum(off_diag) / len(off_diag)
        assert val == pytest.approx(expected, abs=1e-9)


# ── Real-data smoke test ──────────────────────────────────────────────────────

class TestRealDataSmoke:
    def test_sttc_on_real_well(self):
        from pathlib import Path
        spk = Path("LGI2 KD data/20251004_LGI2 KD_Plate 1_D28N(000).spk")
        if not spk.exists():
            pytest.skip("Real .spk file not available")

        from py_mea_axion.io.spk_reader import load_spikes_from_spk
        result, total = load_spikes_from_spk(spk, wells=["A1"])

        val = mean_sttc(result, dt_s=0.05, total_time_s=total, active_only=True)
        # Just check it runs and is in range.
        if not math.isnan(val):
            assert -1.0 <= val <= 1.0
