"""
tests/test_pipeline.py
=======================
Tests for py_mea_axion.pipeline.MEAExperiment.

All tests use MEAExperiment.from_spikes() so no .spk file is required.
Two synthetic datasets cover the main scenarios:

SINGLE_WELL
    One well (A1) with 4 active electrodes.  Used for single-well methods.

MULTI_WELL
    Four wells (A1-A2, B1-B2) with metadata, used for stats/trajectory.
"""

import math

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from py_mea_axion.pipeline import MEAExperiment, _group_by_well


# ── Shared fixtures ────────────────────────────────────────────────────────────

RNG = np.random.default_rng(0)
T = 60.0   # 60-second recording

# Four electrodes in A1 — all active (>0.1 Hz).
_A1_SPIKES = {
    "A1_11": np.sort(RNG.uniform(0, T, 120)),   # ~2 Hz
    "A1_12": np.sort(RNG.uniform(0, T, 90)),    # 1.5 Hz
    "A1_21": np.sort(RNG.uniform(0, T, 60)),    # 1.0 Hz
    "A1_22": np.sort(RNG.uniform(0, T, 30)),    # 0.5 Hz
}

# Four wells, two conditions, two replicates each.
def _make_multi_well_spikes():
    rng = np.random.default_rng(7)
    wells = {
        "A1": 4,   # WT rep1
        "A2": 4,   # WT rep2
        "B1": 4,   # KD rep1
        "B2": 4,   # KD rep2
    }
    spikes = {}
    for well, n_elec in wells.items():
        for i in range(1, n_elec + 1):
            eid = f"{well}_{i}{i}"
            spikes[eid] = np.sort(rng.uniform(0, T, 80))
    return spikes


_MULTI_SPIKES = _make_multi_well_spikes()

_METADATA = pd.DataFrame({
    "well_id":      ["A1", "A2", "B1", "B2"],
    "condition":    ["WT", "WT", "KD", "KD"],
    "DIV":          [14,   14,   14,   14],
    "replicate_id": ["r1", "r2", "r3", "r4"],
})


@pytest.fixture(scope="module")
def single_exp():
    exp = MEAExperiment.from_spikes(_A1_SPIKES, total_time_s=T)
    exp.run()
    return exp


@pytest.fixture(scope="module")
def multi_exp():
    exp = MEAExperiment.from_spikes(
        _MULTI_SPIKES, total_time_s=T, metadata=_METADATA
    )
    exp.run()
    return exp


# ── _group_by_well helper ─────────────────────────────────────────────────────

class TestGroupByWell:
    def test_groups_by_prefix(self):
        flat = {"A1_11": np.array([]), "A1_22": np.array([]), "B2_11": np.array([])}
        grouped = _group_by_well(flat)
        assert set(grouped.keys()) == {"A1", "B2"}
        assert set(grouped["A1"].keys()) == {"A1_11", "A1_22"}
        assert set(grouped["B2"].keys()) == {"B2_11"}

    def test_empty_input(self):
        assert _group_by_well({}) == {}


# ── Construction and repr ─────────────────────────────────────────────────────

class TestConstruction:
    def test_from_spikes_not_ran(self):
        exp = MEAExperiment.from_spikes(_A1_SPIKES, total_time_s=T)
        assert not exp._ran

    def test_require_ran_raises(self):
        exp = MEAExperiment.from_spikes(_A1_SPIKES, total_time_s=T)
        with pytest.raises(RuntimeError, match="run()"):
            _ = exp.spike_metrics

    def test_repr_before_run(self):
        exp = MEAExperiment.from_spikes(_A1_SPIKES, total_time_s=T)
        assert "ran=False" in repr(exp)

    def test_repr_after_run(self, single_exp):
        assert "ran=True" in repr(single_exp)

    def test_run_returns_self(self):
        exp = MEAExperiment.from_spikes(_A1_SPIKES, total_time_s=T)
        result = exp.run()
        assert result is exp

    def test_custom_burst_kwargs(self):
        exp = MEAExperiment.from_spikes(
            _A1_SPIKES, total_time_s=T,
            burst_kwargs={"max_isi_s": 0.05, "min_spikes": 3},
        )
        exp.run()
        assert isinstance(exp.burst_table, pd.DataFrame)


# ── wells and total_time_s ────────────────────────────────────────────────────

class TestBasicProperties:
    def test_wells(self, single_exp):
        assert single_exp.wells == ["A1"]

    def test_total_time_s(self, single_exp):
        assert single_exp.total_time_s == pytest.approx(T)

    def test_multi_wells(self, multi_exp):
        assert set(multi_exp.wells) == {"A1", "A2", "B1", "B2"}


# ── spike_metrics ─────────────────────────────────────────────────────────────

class TestSpikeMetrics:
    def test_returns_dataframe(self, single_exp):
        assert isinstance(single_exp.spike_metrics, pd.DataFrame)

    def test_row_count(self, single_exp):
        # 4 electrodes in A1.
        assert len(single_exp.spike_metrics) == 4

    def test_columns_present(self, single_exp):
        for col in ("well_id", "electrode_id", "n_spikes", "mfr_hz",
                    "mean_isi", "median_isi", "cv_isi", "is_active"):
            assert col in single_exp.spike_metrics.columns

    def test_well_id_correct(self, single_exp):
        assert (single_exp.spike_metrics["well_id"] == "A1").all()

    def test_active_electrodes(self, single_exp):
        # All 4 electrodes have MFR > 0.1 Hz → all active.
        assert single_exp.spike_metrics["is_active"].all()

    def test_mfr_positive(self, single_exp):
        assert (single_exp.spike_metrics["mfr_hz"] > 0).all()


# ── burst_table ───────────────────────────────────────────────────────────────

class TestBurstTable:
    def test_returns_dataframe(self, single_exp):
        assert isinstance(single_exp.burst_table, pd.DataFrame)

    def test_columns_present(self, single_exp):
        for col in ("well_id", "electrode_id", "start_time", "end_time",
                    "duration", "n_spikes"):
            assert col in single_exp.burst_table.columns

    def test_well_id_correct(self, single_exp):
        if not single_exp.burst_table.empty:
            assert (single_exp.burst_table["well_id"] == "A1").all()


# ── well_summary ──────────────────────────────────────────────────────────────

class TestWellSummary:
    def test_returns_dataframe(self, single_exp):
        assert isinstance(single_exp.well_summary, pd.DataFrame)

    def test_one_row_per_well(self, single_exp):
        assert len(single_exp.well_summary) == 1

    def test_columns_present(self, single_exp):
        for col in ("well_id", "n_electrodes", "n_active",
                    "mean_mfr_active_hz", "mean_cv_isi",
                    "burst_rate_hz", "mean_burst_duration_s",
                    "n_network_bursts", "mean_sttc"):
            assert col in single_exp.well_summary.columns

    def test_n_electrodes(self, single_exp):
        assert single_exp.well_summary["n_electrodes"].iloc[0] == 4

    def test_n_active(self, single_exp):
        assert single_exp.well_summary["n_active"].iloc[0] == 4

    def test_mean_mfr_positive(self, single_exp):
        assert single_exp.well_summary["mean_mfr_active_hz"].iloc[0] > 0

    def test_multi_well_row_count(self, multi_exp):
        assert len(multi_exp.well_summary) == 4


# ── network_bursts and sttc_matrices ─────────────────────────────────────────

class TestNetworkAndSttc:
    def test_network_bursts_keys(self, single_exp):
        assert "A1" in single_exp.network_bursts

    def test_network_bursts_is_list(self, single_exp):
        assert isinstance(single_exp.network_bursts["A1"], list)

    def test_sttc_matrices_keys(self, single_exp):
        assert "A1" in single_exp.sttc_matrices

    def test_sttc_matrix_is_dataframe(self, single_exp):
        assert isinstance(single_exp.sttc_matrices["A1"], pd.DataFrame)

    def test_sttc_matrix_shape(self, single_exp):
        mat = single_exp.sttc_matrices["A1"]
        n = len(_A1_SPIKES)
        assert mat.shape == (n, n)

    def test_sttc_diagonal_is_one(self, single_exp):
        mat = single_exp.sttc_matrices["A1"]
        for eid in mat.index:
            assert mat.loc[eid, eid] == pytest.approx(1.0)


# ── well_spikes and well_burst_dict ───────────────────────────────────────────

class TestDataAccess:
    def test_well_spikes_returns_dict(self, single_exp):
        ws = single_exp.well_spikes("A1")
        assert isinstance(ws, dict)
        assert set(ws.keys()) == set(_A1_SPIKES.keys())

    def test_well_spikes_bad_well(self, single_exp):
        with pytest.raises(KeyError, match="Z9"):
            single_exp.well_spikes("Z9")

    def test_well_burst_dict_returns_dict(self, single_exp):
        wb = single_exp.well_burst_dict("A1")
        assert isinstance(wb, dict)
        assert set(wb.keys()) == set(_A1_SPIKES.keys())

    def test_well_burst_dict_missing_well(self, single_exp):
        # Should return empty dict, not raise.
        assert single_exp.well_burst_dict("Z9") == {}


# ── joined_summary ────────────────────────────────────────────────────────────

class TestJoinedSummary:
    def test_without_metadata(self, single_exp):
        js = single_exp.joined_summary()
        assert isinstance(js, pd.DataFrame)
        assert "well_id" in js.columns

    def test_with_metadata_has_condition(self, multi_exp):
        js = multi_exp.joined_summary()
        assert "condition" in js.columns

    def test_with_metadata_row_count(self, multi_exp):
        js = multi_exp.joined_summary()
        assert len(js) == 4


# ── compare ───────────────────────────────────────────────────────────────────

class TestCompare:
    def test_returns_compare_result(self, multi_exp):
        from py_mea_axion.stats.compare import CompareResult
        res = multi_exp.compare("mean_mfr_active_hz")
        assert isinstance(res, CompareResult)

    def test_p_value_in_range(self, multi_exp):
        res = multi_exp.compare("mean_mfr_active_hz")
        assert 0.0 <= res.p_value <= 1.0

    def test_no_group_col_raises(self, single_exp):
        with pytest.raises(ValueError, match="condition"):
            single_exp.compare("mean_mfr_active_hz")


# ── Visualisation methods ─────────────────────────────────────────────────────

class TestVizMethods:
    def test_plot_heatmap_returns_figure(self, single_exp):
        fig = single_exp.plot_heatmap("A1")
        assert isinstance(fig, Figure)

    def test_plot_heatmap_custom_metric(self, single_exp):
        fig = single_exp.plot_heatmap("A1", metric="n_spikes")
        assert isinstance(fig, Figure)

    def test_plot_raster_returns_figure(self, single_exp):
        fig = single_exp.plot_raster("A1")
        assert isinstance(fig, Figure)

    def test_plot_isi_returns_figure(self, single_exp):
        fig = single_exp.plot_isi("A1_11")
        assert isinstance(fig, Figure)

    def test_plot_isi_bad_electrode_raises(self, single_exp):
        with pytest.raises(KeyError):
            single_exp.plot_isi("A1_99")

    def test_plot_trajectory_returns_figure(self, multi_exp):
        fig = multi_exp.plot_trajectory("mean_mfr_active_hz")
        assert isinstance(fig, Figure)

    def test_plot_trajectory_no_metadata_raises(self, single_exp):
        # No metadata loaded → missing condition/DIV columns.
        with pytest.raises(ValueError):
            single_exp.plot_trajectory("mean_mfr_active_hz")

    def test_plot_sttc_returns_figure(self, single_exp):
        fig = single_exp.plot_sttc("A1")
        assert isinstance(fig, Figure)

    def test_plot_network_timeline_returns_figure(self, single_exp):
        fig = single_exp.plot_network_timeline("A1")
        assert isinstance(fig, Figure)


# ── Real .spk file smoke test ─────────────────────────────────────────────────

class TestRealSpkSmoke:
    def test_pipeline_on_real_file(self):
        from pathlib import Path
        spk = Path("LGI2 KD data/20251004_LGI2 KD_Plate 1_D28N(000).spk")
        if not spk.exists():
            pytest.skip("Real .spk file not available")

        exp = MEAExperiment(spk, wells=["A1"], fs_override=12500).run()
        assert "A1" in exp.wells
        assert len(exp.spike_metrics) > 0
        assert isinstance(exp.well_summary, pd.DataFrame)
