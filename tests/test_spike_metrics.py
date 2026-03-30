"""
tests/test_spike_metrics.py
===========================
Tests for py_mea_axion.spike.metrics.
"""

import numpy as np
import pandas as pd
import pytest

from py_mea_axion.spike.metrics import (
    electrode_metrics,
    is_active,
    isi_array,
    isi_stats,
    mean_firing_rate,
    summarise_well,
)

# ── Shared spike trains ───────────────────────────────────────────────────────

# Regular train: 10 spikes at 0.1 s intervals → MFR = 1 Hz over 10 s
REGULAR = np.arange(0.0, 1.0, 0.1)          # 10 spikes, ISI = 0.1 s, CV = 0

# Bursty train: two tight bursts separated by a long silence
BURSTY = np.array([
    0.010, 0.020, 0.030, 0.040, 0.050,       # burst 1
    2.010, 2.020, 2.030, 2.040, 2.050,       # burst 2
])

EMPTY = np.array([], dtype=np.float64)
SINGLE = np.array([1.0])


# ── mean_firing_rate ──────────────────────────────────────────────────────────

class TestMeanFiringRate:
    def test_regular(self):
        assert abs(mean_firing_rate(REGULAR, 10.0) - 1.0) < 1e-9

    def test_empty(self):
        assert mean_firing_rate(EMPTY, 10.0) == 0.0

    def test_zero_duration(self):
        assert mean_firing_rate(REGULAR, 0.0) == 0.0

    def test_single_spike(self):
        assert abs(mean_firing_rate(SINGLE, 10.0) - 0.1) < 1e-9

    def test_units(self):
        # 100 spikes over 50 s → 2 Hz
        spikes = np.linspace(0, 50, 100, endpoint=False)
        assert abs(mean_firing_rate(spikes, 50.0) - 2.0) < 1e-9


# ── isi_array ─────────────────────────────────────────────────────────────────

class TestIsiArray:
    def test_regular_intervals(self):
        isis = isi_array(REGULAR)
        assert len(isis) == len(REGULAR) - 1
        np.testing.assert_allclose(isis, 0.1, atol=1e-9)

    def test_empty_input(self):
        result = isi_array(EMPTY)
        assert len(result) == 0
        assert result.dtype == np.float64

    def test_single_spike(self):
        result = isi_array(SINGLE)
        assert len(result) == 0

    def test_two_spikes(self):
        isis = isi_array(np.array([1.0, 1.5]))
        np.testing.assert_allclose(isis, [0.5])

    def test_bursty_has_short_and_long_isis(self):
        isis = isi_array(BURSTY)
        assert isis.min() < 0.05       # within-burst ISI
        assert isis.max() > 1.0        # between-burst ISI


# ── isi_stats ─────────────────────────────────────────────────────────────────

class TestIsiStats:
    def test_regular_mean(self):
        stats = isi_stats(REGULAR)
        assert abs(stats["mean_isi"] - 0.1) < 1e-9

    def test_regular_cv_near_zero(self):
        stats = isi_stats(REGULAR)
        # CV of a perfectly regular train should be ~0
        assert stats["cv_isi"] < 1e-6

    def test_bursty_cv_greater_than_one(self):
        # High CV is the hallmark of bursty firing
        stats = isi_stats(BURSTY)
        assert stats["cv_isi"] > 1.0

    def test_empty_returns_none(self):
        stats = isi_stats(EMPTY)
        assert stats["mean_isi"] is None
        assert stats["median_isi"] is None
        assert stats["cv_isi"] is None

    def test_single_spike_returns_none(self):
        stats = isi_stats(SINGLE)
        assert stats["mean_isi"] is None

    def test_two_spikes_cv_zero(self):
        # Only one ISI → std = 0 (ddof=1 raises; we handle gracefully)
        stats = isi_stats(np.array([0.0, 0.5]))
        assert stats["cv_isi"] == 0.0

    def test_known_values(self):
        # ISIs: 0.1, 0.2, 0.3  → mean=0.2, median=0.2
        spikes = np.array([0.0, 0.1, 0.3, 0.6])
        stats = isi_stats(spikes)
        assert abs(stats["mean_isi"] - 0.2) < 1e-9
        assert abs(stats["median_isi"] - 0.2) < 1e-9


# ── is_active ─────────────────────────────────────────────────────────────────

class TestIsActive:
    def test_active_electrode(self):
        # 10 spikes / 10 s = 1 Hz > 0.1 Hz
        assert is_active(REGULAR, 10.0) is True

    def test_inactive_electrode(self):
        # 1 spike / 100 s = 0.01 Hz < 0.1 Hz
        assert is_active(SINGLE, 100.0) is False

    def test_exactly_at_threshold(self):
        # 1 spike / 10 s = 0.1 Hz == threshold → active
        assert is_active(SINGLE, 10.0) is True

    def test_custom_threshold(self):
        assert is_active(REGULAR, 10.0, threshold_hz=2.0) is False
        assert is_active(REGULAR, 10.0, threshold_hz=0.5) is True

    def test_empty_is_inactive(self):
        assert is_active(EMPTY, 10.0) is False


# ── electrode_metrics ─────────────────────────────────────────────────────────

class TestElectrodeMetrics:
    def test_keys_present(self):
        m = electrode_metrics(REGULAR, 10.0)
        for key in ("n_spikes", "mfr_hz", "mean_isi", "median_isi",
                    "cv_isi", "is_active"):
            assert key in m

    def test_n_spikes(self):
        assert electrode_metrics(REGULAR, 10.0)["n_spikes"] == 10

    def test_mfr(self):
        assert abs(electrode_metrics(REGULAR, 10.0)["mfr_hz"] - 1.0) < 1e-9

    def test_is_active_true(self):
        assert electrode_metrics(REGULAR, 10.0)["is_active"] is True

    def test_empty_electrode(self):
        m = electrode_metrics(EMPTY, 10.0)
        assert m["n_spikes"] == 0
        assert m["mfr_hz"] == 0.0
        assert m["mean_isi"] is None
        assert m["is_active"] is False


# ── summarise_well ────────────────────────────────────────────────────────────

class TestSummariseWell:
    @pytest.fixture()
    def well_dict(self):
        return {
            "A1_11": REGULAR,
            "A1_12": BURSTY,
            "A1_13": EMPTY,
            "A1_14": SINGLE,
        }

    def test_row_count(self, well_dict):
        df = summarise_well(well_dict, 10.0, "A1")
        assert len(df) == 4

    def test_columns(self, well_dict):
        df = summarise_well(well_dict, 10.0, "A1")
        expected = ["well_id", "electrode_id", "n_spikes", "mfr_hz",
                    "mean_isi", "median_isi", "cv_isi", "is_active"]
        assert list(df.columns) == expected

    def test_well_id_column(self, well_dict):
        df = summarise_well(well_dict, 10.0, "A1")
        assert (df["well_id"] == "A1").all()

    def test_n_spikes_correct(self, well_dict):
        df = summarise_well(well_dict, 10.0, "A1")
        row = df.loc[df["electrode_id"] == "A1_11"]
        assert row["n_spikes"].iloc[0] == len(REGULAR)

    def test_empty_electrode_nan_isi(self, well_dict):
        df = summarise_well(well_dict, 10.0, "A1")
        row = df.loc[df["electrode_id"] == "A1_13"]
        assert pd.isna(row["mean_isi"].iloc[0])
        assert pd.isna(row["cv_isi"].iloc[0])

    def test_n_spikes_dtype(self, well_dict):
        df = summarise_well(well_dict, 10.0, "A1")
        assert df["n_spikes"].dtype == "int64"

    def test_is_active_dtype(self, well_dict):
        df = summarise_well(well_dict, 10.0, "A1")
        assert df["is_active"].dtype == bool

    def test_float_columns_numeric(self, well_dict):
        df = summarise_well(well_dict, 10.0, "A1")
        for col in ("mfr_hz", "mean_isi", "median_isi", "cv_isi"):
            assert pd.api.types.is_float_dtype(df[col])

    def test_active_count(self, well_dict):
        df = summarise_well(well_dict, 10.0, "A1")
        # REGULAR (1 Hz), BURSTY (1 Hz), SINGLE (0.1 Hz == threshold) → active
        # EMPTY (0 Hz) → inactive
        assert df["is_active"].sum() == 3

    def test_bursty_high_cv(self, well_dict):
        df = summarise_well(well_dict, 10.0, "A1")
        cv = df.loc[df["electrode_id"] == "A1_12", "cv_isi"].iloc[0]
        assert cv > 1.0

    def test_reset_index(self, well_dict):
        df = summarise_well(well_dict, 10.0, "A1")
        assert list(df.index) == list(range(len(df)))

    def test_empty_well_dict(self):
        df = summarise_well({}, 10.0, "A1")
        assert len(df) == 0
        assert "electrode_id" in df.columns


# ── Real-data smoke test ──────────────────────────────────────────────────────

class TestRealDataSmoke:
    """Load one well from a real .spk file and check metric plausibility."""

    def test_mfr_plausible(self):
        spk_path = (
            "LGI2 KD data/20251004_LGI2 KD_Plate 1_D28N(000).spk"
        )
        from pathlib import Path
        if not Path(spk_path).exists():
            pytest.skip("Real .spk file not available in this environment")

        from py_mea_axion.io.spk_reader import load_spikes_from_spk
        result, total = load_spikes_from_spk(Path(spk_path), wells=["A1"])
        well_dict = {eid: ts for eid, ts in result.items()
                     if eid.startswith("A1")}
        df = summarise_well(well_dict, total, "A1")

        assert len(df) == 16
        # At least some electrodes should be active in a D28 recording
        assert df["is_active"].any()
        # MFR should be in a biologically plausible range (0–200 Hz)
        assert (df["mfr_hz"] >= 0).all()
        assert (df["mfr_hz"] < 200).all()
