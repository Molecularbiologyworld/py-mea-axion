"""
tests/test_viz.py
==================
Tests for the four viz modules:
  - py_mea_axion.viz.heatmap
  - py_mea_axion.viz.burst_charts
  - py_mea_axion.viz.trajectory
  - py_mea_axion.viz.network_plots

All tests use the Agg backend (headless) and only check figure/axes
properties — no pixel-level rendering is required.
"""

import math

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from py_mea_axion.burst.detection import Burst
from py_mea_axion.network.detection import NetworkBurst
from py_mea_axion.viz.heatmap import plot_electrode_heatmap, _parse_electrode_rc
from py_mea_axion.viz.burst_charts import plot_isi_histogram, plot_burst_raster
from py_mea_axion.viz.trajectory import plot_metric_trajectory
from py_mea_axion.viz.network_plots import plot_sttc_matrix, plot_network_burst_timeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def full_well_values():
    """16-electrode well, A1, each value = row+col."""
    return {f"A1_{r}{c}": float(r + c)
            for r in range(1, 5) for c in range(1, 5)}


@pytest.fixture()
def sparse_values():
    """Only 4 electrodes present."""
    return {"A1_11": 1.0, "A1_22": 2.0, "A1_33": 3.0, "A1_44": 4.0}


@pytest.fixture()
def spike_train():
    return np.arange(0.0, 10.0, 0.1)


@pytest.fixture()
def well_spikes():
    rng = np.random.default_rng(5)
    return {
        f"A1_{r}{c}": np.sort(rng.uniform(0, 10, 50))
        for r in range(1, 3) for c in range(1, 3)
    }


@pytest.fixture()
def well_bursts(well_spikes):
    """Minimal single-burst per electrode."""
    bursts = {}
    for eid, ts in well_spikes.items():
        b = Burst(
            start_time=float(ts[0]),
            end_time=float(ts[-1]),
            spike_times=ts,
            n_spikes=len(ts),
            duration=float(ts[-1] - ts[0]),
            mean_isi_within=float(np.diff(ts).mean()),
        )
        bursts[eid] = [b]
    return bursts


@pytest.fixture()
def long_df():
    rng = np.random.default_rng(10)
    divs = [14, 21, 28]
    conditions = ["WT", "KD"]
    reps = ["r1", "r2", "r3"]
    rows = []
    for div in divs:
        for cond in conditions:
            for rep in reps:
                rows.append({
                    "mfr": rng.normal(2.0 if cond == "WT" else 1.5, 0.2),
                    "DIV": div,
                    "condition": cond,
                    "replicate_id": f"{rep}_{cond}",
                })
    return pd.DataFrame(rows)


@pytest.fixture()
def sttc_df():
    mat = np.array([[1.0, 0.7, 0.2],
                    [0.7, 1.0, 0.3],
                    [0.2, 0.3, 1.0]])
    eids = ["e1", "e2", "e3"]
    return pd.DataFrame(mat, index=eids, columns=eids)


@pytest.fixture()
def network_bursts():
    return [
        NetworkBurst(1.0, 2.0, 1.0, ["e1", "e2"], 0.5, 0.8),
        NetworkBurst(5.0, 6.5, 1.5, ["e1", "e2", "e3"], 0.75, 1.0),
    ]


# ── viz/heatmap.py ────────────────────────────────────────────────────────────

class TestParseElectrodeRc:
    def test_valid_id(self):
        assert _parse_electrode_rc("A1_23") == (2, 3)

    def test_valid_id_first(self):
        assert _parse_electrode_rc("B3_11") == (1, 1)

    def test_no_underscore(self):
        assert _parse_electrode_rc("A1") == (None, None)

    def test_non_numeric_suffix(self):
        assert _parse_electrode_rc("A1_ab") == (None, None)

    def test_empty_string(self):
        assert _parse_electrode_rc("") == (None, None)


class TestPlotElectrodeHeatmap:
    def test_returns_figure(self, full_well_values):
        fig = plot_electrode_heatmap(full_well_values, "A1", metric_name="MFR (Hz)")
        assert isinstance(fig, Figure)

    def test_title_default(self, full_well_values):
        fig = plot_electrode_heatmap(full_well_values, "A1", metric_name="MFR (Hz)")
        assert fig.axes[0].get_title() == "A1 \u2014 MFR (Hz)"

    def test_custom_title(self, full_well_values):
        fig = plot_electrode_heatmap(full_well_values, "A1", title="My title")
        assert fig.axes[0].get_title() == "My title"

    def test_colorbar_present(self, full_well_values):
        fig = plot_electrode_heatmap(full_well_values, "A1")
        # A colorbar adds an extra axes.
        assert len(fig.axes) == 2

    def test_sparse_values_no_crash(self, sparse_values):
        fig = plot_electrode_heatmap(sparse_values, "A1")
        assert isinstance(fig, Figure)

    def test_empty_values(self):
        fig = plot_electrode_heatmap({}, "A1")
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, full_well_values):
        fig = plot_electrode_heatmap(full_well_values, "A1", figsize=(5.0, 5.0))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(5.0) and h == pytest.approx(5.0)

    def test_vmin_vmax_respected(self, full_well_values):
        # Should not raise even when vmin==vmax (guard: vmax += 1).
        fig = plot_electrode_heatmap(full_well_values, "A1", vmin=0.0, vmax=0.0)
        assert isinstance(fig, Figure)

    def test_use_existing_axes(self, full_well_values):
        fig_pre, ax_pre = plt.subplots()
        fig_out = plot_electrode_heatmap(full_well_values, "A1", ax=ax_pre)
        assert fig_out is fig_pre
        plt.close(fig_pre)

    def test_xlabel_ylabel(self, full_well_values):
        fig = plot_electrode_heatmap(full_well_values, "A1")
        ax = fig.axes[0]
        assert "column" in ax.get_xlabel().lower()
        assert "row" in ax.get_ylabel().lower()

    def test_xtick_count(self, full_well_values):
        fig = plot_electrode_heatmap(full_well_values, "A1")
        assert len(fig.axes[0].get_xticks()) == 4

    def test_ytick_count(self, full_well_values):
        fig = plot_electrode_heatmap(full_well_values, "A1")
        assert len(fig.axes[0].get_yticks()) == 4


# ── viz/burst_charts.py ───────────────────────────────────────────────────────

class TestPlotIsiHistogram:
    def test_returns_figure(self, spike_train):
        fig = plot_isi_histogram(spike_train, electrode_id="A1_11")
        assert isinstance(fig, Figure)

    def test_xlabel(self, spike_train):
        fig = plot_isi_histogram(spike_train)
        assert fig.axes[0].get_xlabel() == "ISI (s)"

    def test_ylabel(self, spike_train):
        fig = plot_isi_histogram(spike_train)
        assert fig.axes[0].get_ylabel() == "Count"

    def test_title_with_eid(self, spike_train):
        fig = plot_isi_histogram(spike_train, electrode_id="A1_11")
        assert "A1_11" in fig.axes[0].get_title()

    def test_empty_spike_train(self):
        fig = plot_isi_histogram(np.array([]))
        assert isinstance(fig, Figure)

    def test_single_spike(self):
        fig = plot_isi_histogram(np.array([5.0]))
        assert isinstance(fig, Figure)

    def test_log_x(self, spike_train):
        fig = plot_isi_histogram(spike_train, log_x=True)
        assert fig.axes[0].get_xscale() == "log"

    def test_linear_x(self, spike_train):
        fig = plot_isi_histogram(spike_train, log_x=False)
        assert fig.axes[0].get_xscale() == "linear"

    def test_log_y(self, spike_train):
        fig = plot_isi_histogram(spike_train, log_y=True)
        assert fig.axes[0].get_yscale() == "log"

    def test_use_existing_axes(self, spike_train):
        fig_pre, ax_pre = plt.subplots()
        fig_out = plot_isi_histogram(spike_train, ax=ax_pre)
        assert fig_out is fig_pre
        plt.close(fig_pre)


class TestPlotBurstRaster:
    def test_returns_figure(self, well_spikes, well_bursts):
        fig = plot_burst_raster(well_spikes, well_bursts)
        assert isinstance(fig, Figure)

    def test_two_axes(self, well_spikes, well_bursts):
        # Own-figure call produces ASDR (axes[0]) + raster (axes[1]).
        fig = plot_burst_raster(well_spikes, well_bursts)
        assert len(fig.axes) == 2

    def test_ytick_count_matches_electrodes(self, well_spikes, well_bursts):
        fig = plot_burst_raster(well_spikes, well_bursts)
        assert len(fig.axes[1].get_yticks()) == len(well_spikes)

    def test_xlabel(self, well_spikes, well_bursts):
        fig = plot_burst_raster(well_spikes, well_bursts)
        assert "time" in fig.axes[1].get_xlabel().lower()

    def test_custom_title(self, well_spikes, well_bursts):
        fig = plot_burst_raster(well_spikes, well_bursts, title="Test well")
        assert fig.axes[0].get_title() == "Test well"

    def test_empty_bursts_dict(self, well_spikes):
        fig = plot_burst_raster(well_spikes, {})
        assert isinstance(fig, Figure)

    def test_use_existing_axes(self, well_spikes, well_bursts):
        fig_pre, ax_pre = plt.subplots()
        fig_out = plot_burst_raster(well_spikes, well_bursts, ax=ax_pre)
        assert fig_out is fig_pre
        plt.close(fig_pre)

    def test_t_start_t_stop(self, well_spikes, well_bursts):
        fig = plot_burst_raster(well_spikes, well_bursts, t_start=2.0, t_stop=5.0)
        xlim = fig.axes[0].get_xlim()
        assert xlim[0] == pytest.approx(2.0)
        assert xlim[1] == pytest.approx(5.0)


# ── viz/trajectory.py ─────────────────────────────────────────────────────────

class TestPlotMetricTrajectory:
    def test_returns_figure(self, long_df):
        fig = plot_metric_trajectory(long_df, "mfr")
        assert isinstance(fig, Figure)

    def test_one_axes(self, long_df):
        fig = plot_metric_trajectory(long_df, "mfr")
        assert len(fig.axes) == 1

    def test_has_legend(self, long_df):
        fig = plot_metric_trajectory(long_df, "mfr")
        legend = fig.axes[0].get_legend()
        assert legend is not None

    def test_xlabel_default(self, long_df):
        fig = plot_metric_trajectory(long_df, "mfr")
        assert fig.axes[0].get_xlabel() == "DIV"

    def test_ylabel_default(self, long_df):
        fig = plot_metric_trajectory(long_df, "mfr")
        assert fig.axes[0].get_ylabel() == "mfr"

    def test_custom_labels(self, long_df):
        fig = plot_metric_trajectory(long_df, "mfr",
                                     xlabel="Day in vitro", ylabel="MFR (Hz)")
        assert fig.axes[0].get_xlabel() == "Day in vitro"
        assert fig.axes[0].get_ylabel() == "MFR (Hz)"

    def test_custom_title(self, long_df):
        fig = plot_metric_trajectory(long_df, "mfr", title="Network activity")
        assert fig.axes[0].get_title() == "Network activity"

    def test_lines_drawn(self, long_df):
        fig = plot_metric_trajectory(long_df, "mfr", show_replicates=True)
        assert len(fig.axes[0].lines) > 0

    def test_no_replicates_still_has_mean(self, long_df):
        fig = plot_metric_trajectory(long_df, "mfr",
                                     show_replicates=False, show_mean=True)
        assert len(fig.axes[0].lines) > 0

    def test_groups_subset(self, long_df):
        fig = plot_metric_trajectory(long_df, "mfr", groups=["WT"])
        legend = fig.axes[0].get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert "KD" not in labels

    def test_missing_column_raises(self, long_df):
        with pytest.raises(ValueError, match="not found"):
            plot_metric_trajectory(long_df, "nonexistent")

    def test_custom_palette(self, long_df):
        fig = plot_metric_trajectory(long_df, "mfr",
                                     palette=["#ff0000", "#0000ff"])
        assert isinstance(fig, Figure)

    def test_use_existing_axes(self, long_df):
        fig_pre, ax_pre = plt.subplots()
        fig_out = plot_metric_trajectory(long_df, "mfr", ax=ax_pre)
        assert fig_out is fig_pre
        plt.close(fig_pre)


# ── viz/network_plots.py ──────────────────────────────────────────────────────

class TestPlotSttcMatrix:
    def test_returns_figure(self, sttc_df):
        fig = plot_sttc_matrix(sttc_df)
        assert isinstance(fig, Figure)

    def test_title(self, sttc_df):
        fig = plot_sttc_matrix(sttc_df, title="Synchrony")
        assert fig.axes[0].get_title() == "Synchrony"

    def test_colorbar_present(self, sttc_df):
        fig = plot_sttc_matrix(sttc_df)
        assert len(fig.axes) == 2

    def test_xtick_count(self, sttc_df):
        fig = plot_sttc_matrix(sttc_df)
        assert len(fig.axes[0].get_xticks()) == 3

    def test_ytick_count(self, sttc_df):
        fig = plot_sttc_matrix(sttc_df)
        assert len(fig.axes[0].get_yticks()) == 3

    def test_empty_dataframe(self):
        fig = plot_sttc_matrix(pd.DataFrame())
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, sttc_df):
        fig = plot_sttc_matrix(sttc_df, figsize=(6.0, 6.0))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(6.0) and h == pytest.approx(6.0)

    def test_use_existing_axes(self, sttc_df):
        fig_pre, ax_pre = plt.subplots()
        fig_out = plot_sttc_matrix(sttc_df, ax=ax_pre)
        assert fig_out is fig_pre
        plt.close(fig_pre)


class TestPlotNetworkBurstTimeline:
    def test_returns_figure(self, network_bursts):
        fig = plot_network_burst_timeline(network_bursts, total_time_s=10.0)
        assert isinstance(fig, Figure)

    def test_one_axes(self, network_bursts):
        fig = plot_network_burst_timeline(network_bursts, total_time_s=10.0)
        assert len(fig.axes) == 1

    def test_title(self, network_bursts):
        fig = plot_network_burst_timeline(network_bursts, total_time_s=10.0,
                                          title="NB timeline")
        assert fig.axes[0].get_title() == "NB timeline"

    def test_xlim(self, network_bursts):
        fig = plot_network_burst_timeline(network_bursts, total_time_s=10.0)
        assert fig.axes[0].get_xlim() == pytest.approx((0.0, 10.0))

    def test_xlabel(self, network_bursts):
        fig = plot_network_burst_timeline(network_bursts, total_time_s=10.0)
        assert "time" in fig.axes[0].get_xlabel().lower()

    def test_empty_burst_list(self):
        fig = plot_network_burst_timeline([], total_time_s=10.0)
        assert isinstance(fig, Figure)

    def test_burst_count_annotation(self, network_bursts):
        fig = plot_network_burst_timeline(network_bursts, total_time_s=10.0)
        # The annotation text should mention "2 bursts".
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("2" in t for t in texts)

    def test_use_existing_axes(self, network_bursts):
        fig_pre, ax_pre = plt.subplots()
        fig_out = plot_network_burst_timeline(network_bursts, total_time_s=10.0,
                                              ax=ax_pre)
        assert fig_out is fig_pre
        plt.close(fig_pre)
