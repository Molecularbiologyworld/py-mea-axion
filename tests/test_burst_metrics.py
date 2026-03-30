"""
tests/test_burst_metrics.py
============================
Tests for py_mea_axion.burst.metrics.
"""

import math

import numpy as np
import pandas as pd
import pytest

from py_mea_axion.burst.detection import Burst, detect_bursts
from py_mea_axion.burst.metrics import (
    BURST_COLUMNS,
    aggregate_well_bursts,
    bursts_to_dataframe,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_burst(start, end, n=5, mean_isi=0.01):
    spikes = np.linspace(start, end, n)
    return Burst(
        start_time=start,
        end_time=end,
        spike_times=spikes,
        n_spikes=n,
        duration=end - start,
        mean_isi_within=mean_isi,
    )


BURST_A = _make_burst(0.01, 0.05)   # burst ending at 0.05
BURST_B = _make_burst(2.01, 2.05)   # burst starting at 2.01 → IBI = 2.01−0.05 = 1.96
BURST_C = _make_burst(5.00, 5.04)   # IBI = 5.00 − 2.05 = 2.95

TWO_BURSTS = [BURST_A, BURST_B]
THREE_BURSTS = [BURST_A, BURST_B, BURST_C]


# ── bursts_to_dataframe ───────────────────────────────────────────────────────

class TestBurstsToDataframe:
    def test_row_count(self):
        df = bursts_to_dataframe(TWO_BURSTS, "A1_11", "A1")
        assert len(df) == 2

    def test_columns_exact(self):
        df = bursts_to_dataframe(TWO_BURSTS, "A1_11", "A1")
        assert list(df.columns) == BURST_COLUMNS

    def test_electrode_id_column(self):
        df = bursts_to_dataframe(TWO_BURSTS, "A1_11", "A1")
        assert (df["electrode_id"] == "A1_11").all()

    def test_well_id_column(self):
        df = bursts_to_dataframe(TWO_BURSTS, "A1_11", "A1")
        assert (df["well_id"] == "A1").all()

    def test_start_time_values(self):
        df = bursts_to_dataframe(TWO_BURSTS, "A1_11", "A1")
        assert df["start_time"].iloc[0] == pytest.approx(0.01)
        assert df["start_time"].iloc[1] == pytest.approx(2.01)

    def test_end_time_values(self):
        df = bursts_to_dataframe(TWO_BURSTS, "A1_11", "A1")
        assert df["end_time"].iloc[0] == pytest.approx(0.05)
        assert df["end_time"].iloc[1] == pytest.approx(2.05)

    def test_duration_values(self):
        df = bursts_to_dataframe(TWO_BURSTS, "A1_11", "A1")
        assert df["duration"].iloc[0] == pytest.approx(0.04)
        assert df["duration"].iloc[1] == pytest.approx(0.04)

    def test_n_spikes_values(self):
        df = bursts_to_dataframe(TWO_BURSTS, "A1_11", "A1")
        assert df["n_spikes"].iloc[0] == 5
        assert df["n_spikes"].iloc[1] == 5

    def test_n_spikes_dtype_int64(self):
        df = bursts_to_dataframe(TWO_BURSTS, "A1_11", "A1")
        assert df["n_spikes"].dtype == "int64"

    def test_first_burst_ibi_is_nan(self):
        df = bursts_to_dataframe(TWO_BURSTS, "A1_11", "A1")
        assert math.isnan(df["inter_burst_interval"].iloc[0])

    def test_second_burst_ibi_correct(self):
        df = bursts_to_dataframe(TWO_BURSTS, "A1_11", "A1")
        # IBI = start of burst 2 − end of burst 1 = 2.01 − 0.05 = 1.96
        assert df["inter_burst_interval"].iloc[1] == pytest.approx(1.96)

    def test_three_burst_ibis(self):
        df = bursts_to_dataframe(THREE_BURSTS, "A1_11", "A1")
        assert math.isnan(df["inter_burst_interval"].iloc[0])
        assert df["inter_burst_interval"].iloc[1] == pytest.approx(1.96)
        assert df["inter_burst_interval"].iloc[2] == pytest.approx(2.95)

    def test_empty_bursts_returns_empty_df(self):
        df = bursts_to_dataframe([], "A1_11", "A1")
        assert len(df) == 0
        assert list(df.columns) == BURST_COLUMNS

    def test_empty_df_correct_dtypes(self):
        df = bursts_to_dataframe([], "A1_11", "A1")
        assert df["n_spikes"].dtype == "int64"
        assert df["start_time"].dtype == "float64"

    def test_single_burst_ibi_nan(self):
        df = bursts_to_dataframe([BURST_A], "A1_11", "A1")
        assert len(df) == 1
        assert math.isnan(df["inter_burst_interval"].iloc[0])

    def test_reset_index(self):
        df = bursts_to_dataframe(TWO_BURSTS, "A1_11", "A1")
        assert list(df.index) == [0, 1]

    def test_float_columns_are_float64(self):
        df = bursts_to_dataframe(TWO_BURSTS, "A1_11", "A1")
        for col in ("start_time", "end_time", "duration",
                    "mean_isi_within", "inter_burst_interval"):
            assert df[col].dtype == "float64", f"{col} is not float64"

    def test_mean_isi_within_preserved(self):
        df = bursts_to_dataframe([BURST_A], "A1_11", "A1")
        assert df["mean_isi_within"].iloc[0] == pytest.approx(BURST_A.mean_isi_within)


# ── aggregate_well_bursts ─────────────────────────────────────────────────────

class TestAggregateWellBursts:
    @pytest.fixture()
    def well_dict(self):
        return {
            "A1_11": TWO_BURSTS,
            "A1_12": [BURST_C],
            "A1_13": [],          # silent electrode
        }

    def test_total_row_count(self, well_dict):
        df = aggregate_well_bursts(well_dict, "A1")
        assert len(df) == 3   # 2 + 1 + 0

    def test_columns_exact(self, well_dict):
        df = aggregate_well_bursts(well_dict, "A1")
        assert list(df.columns) == BURST_COLUMNS

    def test_well_id_all_rows(self, well_dict):
        df = aggregate_well_bursts(well_dict, "A1")
        assert (df["well_id"] == "A1").all()

    def test_both_electrodes_present(self, well_dict):
        df = aggregate_well_bursts(well_dict, "A1")
        assert set(df["electrode_id"]) == {"A1_11", "A1_12"}

    def test_sorted_by_electrode_then_start_time(self, well_dict):
        df = aggregate_well_bursts(well_dict, "A1")
        a11 = df[df["electrode_id"] == "A1_11"]["start_time"].tolist()
        assert a11 == sorted(a11)

    def test_empty_well_dict(self):
        df = aggregate_well_bursts({}, "A1")
        assert len(df) == 0
        assert list(df.columns) == BURST_COLUMNS

    def test_all_silent_electrodes(self):
        df = aggregate_well_bursts({"A1_11": [], "A1_12": []}, "A1")
        assert len(df) == 0

    def test_n_spikes_dtype(self, well_dict):
        df = aggregate_well_bursts(well_dict, "A1")
        assert df["n_spikes"].dtype == "int64"

    def test_reset_index(self, well_dict):
        df = aggregate_well_bursts(well_dict, "A1")
        assert list(df.index) == list(range(len(df)))


# ── Integration with detect_bursts ────────────────────────────────────────────

class TestIntegrationWithDetection:
    """Round-trip: detect bursts → convert to DataFrame → check values."""

    TRAIN = np.array([
        0.010, 0.020, 0.030, 0.040, 0.050,   # burst 1
        2.010, 2.020, 2.030, 2.040, 2.050,   # burst 2
    ])

    def test_round_trip_row_count(self):
        bursts = detect_bursts(self.TRAIN, max_isi_s=0.1, min_spikes=5)
        df = bursts_to_dataframe(bursts, "A1_11", "A1")
        assert len(df) == 2

    def test_round_trip_ibi(self):
        bursts = detect_bursts(self.TRAIN, max_isi_s=0.1, min_spikes=5)
        df = bursts_to_dataframe(bursts, "A1_11", "A1")
        # IBI = 2.010 − 0.050 = 1.96 s
        assert df["inter_burst_interval"].iloc[1] == pytest.approx(1.96)

    def test_aggregate_multi_electrode(self):
        trains = {
            "A1_11": detect_bursts(self.TRAIN),
            "A1_12": detect_bursts(self.TRAIN * 1.1),  # slightly shifted
            "A1_13": detect_bursts(np.array([0.1])),   # no bursts
        }
        df = aggregate_well_bursts(trains, "A1")
        assert len(df) == 4   # 2 bursts × 2 active electrodes


# ── Real-data smoke test ──────────────────────────────────────────────────────

class TestRealDataSmoke:
    def test_burst_metrics_on_active_well(self):
        from pathlib import Path
        spk = Path("LGI2 KD data/20251004_LGI2 KD_Plate 1_D28N(000).spk")
        if not spk.exists():
            pytest.skip("Real .spk file not available")

        from py_mea_axion.io.spk_reader import load_spikes_from_spk
        result, total = load_spikes_from_spk(spk, wells=["B1"])

        well_bursts = {
            eid: detect_bursts(ts)
            for eid, ts in result.items()
            if eid.startswith("B1")
        }
        df = aggregate_well_bursts(well_bursts, "B1")

        assert list(df.columns) == BURST_COLUMNS
        if len(df) > 0:
            assert (df["n_spikes"] >= 5).all()
            assert (df["duration"] >= 0).all()
            # IBI should be positive wherever it's not NaN.
            ibi_vals = df["inter_burst_interval"].dropna()
            assert (ibi_vals > 0).all()
