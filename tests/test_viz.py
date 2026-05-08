"""
tests/test_viz.py
==================
Tests for the six viz modules:
  - py_mea_axion.viz.heatmap
  - py_mea_axion.viz.burst_charts
  - py_mea_axion.viz.trajectory
  - py_mea_axion.viz.network_plots
  - py_mea_axion.viz.comparison
  - py_mea_axion.viz.pca

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
from py_mea_axion.viz.comparison import plot_condition_violin
from py_mea_axion.viz.pca import (
    compute_pca,
    compute_pca_loadings,
    plot_pca,
    plot_pca_loadings,
)
from py_mea_axion.viz.plate import plot_plate_heatmap


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
            median_isi_within=float(np.median(np.diff(ts))),
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


@pytest.fixture()
def violin_df():
    """Long-format frame with three conditions across two DIVs."""
    rng = np.random.default_rng(42)
    rows = []
    for div in (21, 26):
        for cond, mu in (("SCRM", 2.0), ("KD4", 1.5), ("KD5", 1.7)):
            for _ in range(8):
                rows.append({"metric": rng.normal(mu, 0.2),
                             "condition": cond, "DIV": div})
    return pd.DataFrame(rows)


@pytest.fixture()
def pca_df():
    """Long-format frame with five metric columns and two conditions."""
    rng = np.random.default_rng(7)
    n = 30
    cond = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    return pd.DataFrame({
        "m1":        rng.normal(0, 1, n),
        "m2":        rng.normal(0, 2, n),
        "m3":        rng.normal(0, 1, n) + np.arange(n) * 0.05,
        "m4":        rng.normal(0, 0.5, n),
        "m5":        rng.normal(0, 1.5, n),
        "condition": cond,
        "DIV":       [21] * n,
    })


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

    def test_title_default_is_empty(self, full_well_values):
        fig = plot_electrode_heatmap(full_well_values, "A1", metric_name="MFR (Hz)")
        assert fig.axes[0].get_title() == ""

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

    def test_title_default_is_empty(self, spike_train):
        fig = plot_isi_histogram(spike_train, electrode_id="A1_11")
        assert fig.axes[0].get_title() == ""

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

    def test_x_offset_shifts_xlim(self, well_spikes, well_bursts):
        # well_spikes covers ~[0, 10).  With x_offset=100 and no t_start/t_stop,
        # the displayed window should be ~[100, 110].
        fig = plot_burst_raster(well_spikes, well_bursts, x_offset=100.0)
        xlim_lo, xlim_hi = fig.axes[0].get_xlim()
        assert xlim_lo == pytest.approx(100.0)
        assert xlim_hi > 100.0  # data max + offset + 0.1

    def test_x_offset_with_explicit_window(self, well_spikes, well_bursts):
        # User passes the absolute window (100-105) and the matching offset.
        fig = plot_burst_raster(
            well_spikes, well_bursts,
            x_offset=100.0, t_start=100.0, t_stop=105.0,
        )
        xlim = fig.axes[0].get_xlim()
        assert xlim[0] == pytest.approx(100.0)
        assert xlim[1] == pytest.approx(105.0)

    def test_default_asdr_color_is_black(self, well_spikes, well_bursts):
        from inspect import signature
        params = signature(plot_burst_raster).parameters
        assert params["asdr_color"].default == "#000000"

    def test_density_color_off_by_default_single_color(self, well_spikes, well_bursts):
        fig = plot_burst_raster(well_spikes, well_bursts)
        # No density colourbar axes added.
        assert len(fig.axes) == 2

    def test_density_color_adds_colorbar(self, well_spikes, well_bursts):
        fig = plot_burst_raster(well_spikes, well_bursts, density_color=True)
        # ASDR + raster + colourbar = 3 axes.
        assert len(fig.axes) == 3

    def test_density_color_uses_per_spike_colors(self, well_spikes, well_bursts):
        fig = plot_burst_raster(well_spikes, well_bursts, density_color=True)
        # Each LineCollection from vlines should now carry one colour per
        # spike (an array of N colours), not a single shared colour.
        from matplotlib.collections import LineCollection
        line_collections = [
            c for c in fig.axes[1].collections if isinstance(c, LineCollection)
        ]
        # At least one collection should have per-spike colours (>1 unique
        # RGBA when there's spread in density).
        if line_collections:
            colors = line_collections[0].get_colors()
            # Either an array of per-segment colours, or a single shared one.
            assert colors.ndim == 2  # (N, 4) RGBA per segment

    def test_density_cmap_param_accepted(self, well_spikes, well_bursts):
        fig = plot_burst_raster(
            well_spikes, well_bursts,
            density_color=True, density_cmap="plasma",
        )
        assert isinstance(fig, Figure)


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

    def test_default_does_not_draw_replicate_lines(self):
        """By default the new style is mean+errorbar only — no replicate lines."""
        rng = np.random.default_rng(0)
        rows = []
        for div in (12, 14):
            for cond in ("A", "B"):
                for rep in ("r1", "r2", "r3"):
                    rows.append({"mfr": rng.normal(2.0, 0.2),
                                 "DIV": div, "condition": cond,
                                 "replicate_id": f"{rep}_{cond}"})
        df = pd.DataFrame(rows)
        fig = plot_metric_trajectory(df, "mfr")
        # Each errorbar() emits one Line2D for the central line plus caps and
        # vertical bars; we only need to confirm no per-replicate lines are
        # drawn.  With 2 conditions and show_replicates=False, we expect
        # exactly 2 main coloured lines (one per group).
        coloured_lines = [
            ln for ln in fig.axes[0].lines
            if ln.get_marker() == "o"
        ]
        assert len(coloured_lines) == 2

    def test_replicate_col_not_required_when_show_replicates_false(self):
        df = pd.DataFrame({
            "mfr":  [1.0, 2.0, 3.0, 4.0],
            "DIV":  [12, 14, 12, 14],
            "condition": ["A", "A", "B", "B"],
        })  # NB: no replicate_id column
        fig = plot_metric_trajectory(df, "mfr")
        assert isinstance(fig, Figure)

    def test_replicate_col_required_when_show_replicates_true(self):
        df = pd.DataFrame({
            "mfr":  [1.0, 2.0, 3.0, 4.0],
            "DIV":  [12, 14, 12, 14],
            "condition": ["A", "A", "B", "B"],
        })
        with pytest.raises(ValueError, match="replicate_id"):
            plot_metric_trajectory(df, "mfr", show_replicates=True)

    def test_xticks_span_full_integer_range(self):
        """When DIVs are sparse, the x-axis should still tick every integer."""
        rng = np.random.default_rng(0)
        # Skip DIVs 13, 15-20, 22-25 — only show 12, 14, 21, 26.
        rows = []
        for div in (12, 14, 21, 26):
            for cond in ("WT", "KD"):
                for rep in ("r1", "r2"):
                    rows.append({"mfr":   rng.normal(2.0, 0.2),
                                 "DIV":   div,
                                 "condition":    cond,
                                 "replicate_id": f"{rep}_{cond}"})
        df = pd.DataFrame(rows)
        fig = plot_metric_trajectory(df, "mfr")
        ticks = list(fig.axes[0].get_xticks())
        # Expect ticks at every integer 12..26.
        assert ticks == list(range(12, 27))


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


# ── viz/comparison.py ─────────────────────────────────────────────────────────

class TestPlotConditionViolin:
    def test_returns_figure(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", time_value=21)
        assert isinstance(fig, Figure)

    def test_one_axes(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", time_value=21)
        assert len(fig.axes) == 1

    def test_xtick_count_matches_groups(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", time_value=21)
        # violin_df has 3 conditions: SCRM, KD4, KD5.
        assert len(fig.axes[0].get_xticks()) == 3

    def test_xtick_labels_are_groups(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", time_value=21)
        labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        assert set(labels) == {"SCRM", "KD4", "KD5"}

    def test_default_title_is_empty_with_time(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric",
                                    time_col="DIV", time_value=21)
        assert fig.axes[0].get_title() == ""

    def test_default_title_is_empty_no_time_value(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric")
        assert fig.axes[0].get_title() == ""

    def test_custom_title(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", title="My violin")
        assert fig.axes[0].get_title() == "My violin"

    def test_default_ylabel_is_metric(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", time_value=21)
        assert fig.axes[0].get_ylabel() == "metric"

    def test_custom_ylabel(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", time_value=21,
                                    ylabel="MFR (Hz)")
        assert fig.axes[0].get_ylabel() == "MFR (Hz)"

    def test_groups_subset_changes_xticks(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", time_value=21,
                                    groups=["SCRM", "KD4"])
        assert len(fig.axes[0].get_xticks()) == 2

    def test_groups_order_preserved(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", time_value=21,
                                    groups=["KD5", "SCRM", "KD4"])
        labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        assert labels == ["KD5", "SCRM", "KD4"]

    def test_jitter_points_drawn(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", time_value=21)
        # Each group's jittered scatter is one PathCollection.
        from matplotlib.collections import PathCollection
        scatters = [c for c in fig.axes[0].collections
                    if isinstance(c, PathCollection)]
        assert len(scatters) == 3

    def test_time_value_filters_data(self, violin_df):
        # DIV 21 has 8 rows per condition; total scatter points = 24.
        fig = plot_condition_violin(violin_df, "metric", time_value=21)
        from matplotlib.collections import PathCollection
        scatters = [c for c in fig.axes[0].collections
                    if isinstance(c, PathCollection)]
        n_points = sum(len(c.get_offsets()) for c in scatters)
        assert n_points == 24

    def test_pool_all_when_no_time_value(self, violin_df):
        # Two DIVs × 8 rows × 3 conditions = 48.
        fig = plot_condition_violin(violin_df, "metric")
        from matplotlib.collections import PathCollection
        scatters = [c for c in fig.axes[0].collections
                    if isinstance(c, PathCollection)]
        n_points = sum(len(c.get_offsets()) for c in scatters)
        assert n_points == 48

    def test_missing_metric_raises(self, violin_df):
        with pytest.raises(ValueError, match="not in DataFrame"):
            plot_condition_violin(violin_df, "nonexistent", time_value=21)

    def test_missing_group_col_raises(self, violin_df):
        with pytest.raises(ValueError, match="Group column"):
            plot_condition_violin(violin_df, "metric",
                                  group_col="nope", time_value=21)

    def test_time_value_without_time_col_raises(self, violin_df):
        with pytest.raises(ValueError, match="time_col"):
            plot_condition_violin(violin_df, "metric",
                                  time_col=None, time_value=21)

    def test_empty_group_does_not_crash(self, violin_df):
        # 'GHOST' has no rows; should still render without raising.
        fig = plot_condition_violin(violin_df, "metric", time_value=21,
                                    groups=["SCRM", "GHOST", "KD4"])
        assert isinstance(fig, Figure)

    def test_custom_palette(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", time_value=21,
                                    palette=["#ff0000", "#00ff00", "#0000ff"])
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", time_value=21,
                                    figsize=(7.0, 4.0))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(7.0) and h == pytest.approx(4.0)

    def test_use_existing_axes(self, violin_df):
        fig_pre, ax_pre = plt.subplots()
        fig_out = plot_condition_violin(violin_df, "metric", time_value=21,
                                        ax=ax_pre)
        assert fig_out is fig_pre
        plt.close(fig_pre)

    def test_stats_brackets_drawn_by_default(self, violin_df):
        # 3 groups with 8 samples each at DIV 21 → 3 pairs → 3 brackets.
        fig = plot_condition_violin(violin_df, "metric", time_value=21)
        from matplotlib.lines import Line2D
        # Each bracket is one Line2D (4-point polyline).  Filter out the
        # axis spines and any other auxiliary lines by selecting black.
        bracket_lines = [
            ln for ln in fig.axes[0].lines
            if ln.get_color() == "black" and len(ln.get_xdata()) == 4
        ]
        assert len(bracket_lines) == 3

    def test_stats_label_text_present(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", time_value=21)
        labels = [t.get_text() for t in fig.axes[0].texts]
        # Expect each bracket to be annotated with one of */**/***/ns.
        valid = {"*", "**", "***", "ns"}
        bracket_labels = [t for t in labels if t in valid]
        assert len(bracket_labels) == 3

    def test_show_stats_false_skips_brackets(self, violin_df):
        fig = plot_condition_violin(violin_df, "metric", time_value=21,
                                    show_stats=False)
        bracket_labels = [
            t.get_text() for t in fig.axes[0].texts
            if t.get_text() in {"*", "**", "***", "ns"}
        ]
        assert bracket_labels == []

    def test_stats_skipped_for_single_group(self, violin_df):
        # Only one group plotted -> no brackets possible.
        fig = plot_condition_violin(violin_df, "metric", time_value=21,
                                    groups=["SCRM"])
        bracket_labels = [
            t.get_text() for t in fig.axes[0].texts
            if t.get_text() in {"*", "**", "***", "ns"}
        ]
        assert bracket_labels == []

    def test_point_hue_assigns_distinct_colors_per_batch(self, violin_df):
        df = violin_df.copy()
        # Tag rows alternately as batch1 / batch2 so each condition has both.
        df["batch"] = ["batch1", "batch2"] * (len(df) // 2)
        fig = plot_condition_violin(
            df, "metric", time_value=21,
            point_hue_col="batch",
        )
        from matplotlib.collections import PathCollection
        scatters = [c for c in fig.axes[0].collections
                    if isinstance(c, PathCollection)]
        # 3 conditions x 2 batches = 6 scatter collections.
        assert len(scatters) == 6
        # Distinct colours used across batches.
        colors_per_collection = [tuple(c.get_facecolor()[0][:3]) for c in scatters]
        assert len(set(colors_per_collection)) >= 2

    def test_point_hue_legend_hidden_by_default(self, violin_df):
        df = violin_df.copy()
        df["batch"] = ["batch1", "batch2"] * (len(df) // 2)
        fig = plot_condition_violin(
            df, "metric", time_value=21,
            point_hue_col="batch",
        )
        assert fig.axes[0].get_legend() is None

    def test_point_hue_legend_shown_when_requested(self, violin_df):
        df = violin_df.copy()
        df["batch"] = ["batch1", "batch2"] * (len(df) // 2)
        fig = plot_condition_violin(
            df, "metric", time_value=21,
            point_hue_col="batch", show_point_legend=True,
        )
        leg = fig.axes[0].get_legend()
        assert leg is not None
        labels = [t.get_text() for t in leg.get_texts()]
        assert set(labels) == {"batch1", "batch2"}

    def test_point_palette_overrides_default(self, violin_df):
        df = violin_df.copy()
        df["batch"] = ["batch1", "batch2"] * (len(df) // 2)
        fig = plot_condition_violin(
            df, "metric", time_value=21,
            point_hue_col="batch",
            point_palette={"batch1": "#ff0000", "batch2": "#0000ff"},
        )
        from matplotlib.collections import PathCollection
        scatters = [c for c in fig.axes[0].collections
                    if isinstance(c, PathCollection)]
        # Convert colours to a hashable form for set membership.
        colors_seen = set()
        for c in scatters:
            rgba = tuple(round(x, 3) for x in c.get_facecolor()[0])
            colors_seen.add(rgba)
        red = (1.0, 0.0, 0.0, 0.85)
        blue = (0.0, 0.0, 1.0, 0.85)
        assert red in colors_seen
        assert blue in colors_seen

    def test_point_shape_assigns_default_markers(self, violin_df):
        df = violin_df.copy()
        # Cycle batches across rows so each condition has all three.
        df["batch"] = (["B1", "B2", "B3"] * ((len(df) + 2) // 3))[: len(df)]
        fig = plot_condition_violin(
            df, "metric", time_value=21,
            point_shape_col="batch",
        )
        from matplotlib.collections import PathCollection
        scatters = [c for c in fig.axes[0].collections
                    if isinstance(c, PathCollection)]
        # 3 conditions × 3 batches = 9 scatter collections.
        assert len(scatters) == 9
        # Distinct marker paths drawn.
        marker_paths = {
            tuple(p.vertices.tobytes() for p in c.get_paths()) for c in scatters
        }
        assert len(marker_paths) >= 2  # at least two different shapes drawn

    def test_point_shape_map_overrides_default(self, violin_df):
        df = violin_df.copy()
        df["batch"] = (["B1", "B2", "B3"] * ((len(df) + 2) // 3))[: len(df)]
        # Map B1, B2, B3 explicitly to circle, square, triangle.
        fig = plot_condition_violin(
            df, "metric", time_value=21,
            point_shape_col="batch",
            point_shape_map={"B1": "o", "B2": "s", "B3": "^"},
        )
        assert isinstance(fig, Figure)

    def test_point_hue_and_shape_combined(self, violin_df):
        df = violin_df.copy()
        # Two batches × 3 conditions × 8 rows ÷ 2 = 4 rows each.
        df["batch"] = ["B1", "B2"] * (len(df) // 2)
        fig = plot_condition_violin(
            df, "metric", time_value=21,
            point_hue_col="batch",
            point_shape_col="batch",
        )
        from matplotlib.collections import PathCollection
        scatters = [c for c in fig.axes[0].collections
                    if isinstance(c, PathCollection)]
        # 3 conditions × 2 batches = 6 collections (one per (cond, batch)).
        assert len(scatters) == 6

    def test_point_hue_unknown_column_silently_falls_back(self, violin_df):
        # Specifying a non-existent column should fall back to per-condition colours
        # rather than crash.
        fig = plot_condition_violin(
            violin_df, "metric", time_value=21, point_hue_col="ghost_col",
        )
        assert isinstance(fig, Figure)

    def test_compare_pairs_restricts_brackets(self, violin_df):
        # 3 groups → 3 pairs by default; restrict to 1 pair → 1 bracket.
        fig = plot_condition_violin(
            violin_df, "metric", time_value=21,
            compare_pairs=[("SCRM", "KD4")],
        )
        bracket_labels = [
            t.get_text() for t in fig.axes[0].texts
            if t.get_text() in {"*", "**", "***", "ns"}
        ]
        assert len(bracket_labels) == 1

    def test_compare_pairs_two_pairs(self, violin_df):
        fig = plot_condition_violin(
            violin_df, "metric", time_value=21,
            compare_pairs=[("SCRM", "KD4"), ("SCRM", "KD5")],
        )
        bracket_labels = [
            t.get_text() for t in fig.axes[0].texts
            if t.get_text() in {"*", "**", "***", "ns"}
        ]
        assert len(bracket_labels) == 2

    def test_compare_pairs_unknown_group_silently_skipped(self, violin_df):
        # GHOST isn't a group in the data → that pair is skipped, not an error.
        fig = plot_condition_violin(
            violin_df, "metric", time_value=21,
            compare_pairs=[("SCRM", "GHOST"), ("SCRM", "KD4")],
        )
        bracket_labels = [
            t.get_text() for t in fig.axes[0].texts
            if t.get_text() in {"*", "**", "***", "ns"}
        ]
        assert len(bracket_labels) == 1

    def test_compare_pairs_self_comparison_skipped(self, violin_df):
        fig = plot_condition_violin(
            violin_df, "metric", time_value=21,
            compare_pairs=[("SCRM", "SCRM")],
        )
        bracket_labels = [
            t.get_text() for t in fig.axes[0].texts
            if t.get_text() in {"*", "**", "***", "ns"}
        ]
        assert bracket_labels == []

    def test_stat_test_mannwhitney_renders_brackets(self, violin_df):
        fig = plot_condition_violin(
            violin_df, "metric", time_value=21,
            stat_test="mannwhitney",
        )
        bracket_labels = [
            t.get_text() for t in fig.axes[0].texts
            if t.get_text() in {"*", "**", "***", "ns"}
        ]
        # 3 conditions with separated means -> 3 brackets.
        assert len(bracket_labels) == 3

    def test_stat_test_kruskal_renders_brackets(self, violin_df):
        fig = plot_condition_violin(
            violin_df, "metric", time_value=21,
            stat_test="kruskal",
        )
        bracket_labels = [
            t.get_text() for t in fig.axes[0].texts
            if t.get_text() in {"*", "**", "***", "ns"}
        ]
        assert len(bracket_labels) == 3

    def test_stat_test_unknown_raises(self, violin_df):
        with pytest.raises(ValueError, match="Unknown test"):
            plot_condition_violin(
                violin_df, "metric", time_value=21, stat_test="welch",
            )

    def test_stats_robust_to_empty_group(self, violin_df):
        # 'GHOST' has no rows, but the other two should still produce one bracket.
        fig = plot_condition_violin(
            violin_df, "metric", time_value=21,
            groups=["SCRM", "GHOST", "KD4"],
        )
        bracket_labels = [
            t.get_text() for t in fig.axes[0].texts
            if t.get_text() in {"*", "**", "***", "ns"}
        ]
        assert len(bracket_labels) == 1


# ── viz/pca.py — compute_pca ──────────────────────────────────────────────────

class TestComputePca:
    def test_returns_dataframe_and_var(self, pca_df):
        scores, var = compute_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        assert isinstance(scores, pd.DataFrame)
        assert isinstance(var, np.ndarray)

    def test_pc_columns_added(self, pca_df):
        scores, _ = compute_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        assert "PC1" in scores.columns and "PC2" in scores.columns

    def test_var_explained_length(self, pca_df):
        _, var = compute_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"],
                             n_components=2)
        assert len(var) == 2

    def test_var_explained_sums_le_one(self, pca_df):
        _, var = compute_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"],
                             n_components=2)
        assert var.sum() <= 1.0 + 1e-9
        assert all(v >= 0.0 for v in var)

    def test_var_explained_descending(self, pca_df):
        _, var = compute_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"],
                             n_components=3)
        assert var[0] >= var[1] >= var[2]

    def test_n_components_3_adds_pc3(self, pca_df):
        scores, _ = compute_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"],
                                n_components=3)
        assert {"PC1", "PC2", "PC3"}.issubset(scores.columns)

    def test_drops_nan_rows(self, pca_df):
        df = pca_df.copy()
        df.loc[0, "m1"] = np.nan
        scores, _ = compute_pca(df, ["m1", "m2", "m3", "m4", "m5"])
        assert len(scores) == len(df) - 1

    def test_preserves_group_column(self, pca_df):
        scores, _ = compute_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        assert "condition" in scores.columns

    def test_zero_variance_column_dropped(self, pca_df):
        df = pca_df.copy()
        df["constant"] = 5.0
        # Should not raise even though one column has zero variance.
        scores, var = compute_pca(
            df, ["m1", "m2", "m3", "m4", "m5", "constant"], n_components=2,
        )
        assert "PC1" in scores.columns
        assert len(var) == 2

    def test_missing_column_raises(self, pca_df):
        with pytest.raises(ValueError, match="not found"):
            compute_pca(pca_df, ["m1", "ghost"])

    def test_too_few_rows_raises(self, pca_df):
        with pytest.raises(ValueError, match=">= 2 rows"):
            compute_pca(pca_df.iloc[:1], ["m1", "m2", "m3"])

    def test_too_few_features_raises(self, pca_df):
        df = pca_df.copy()
        df["c1"] = 1.0
        df["c2"] = 2.0
        with pytest.raises(ValueError, match="non-constant features"):
            compute_pca(df, ["c1", "c2"], n_components=2)


# ── viz/pca.py — plot_pca ─────────────────────────────────────────────────────

class TestPlotPca:
    def test_returns_figure(self, pca_df):
        fig = plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        assert isinstance(fig, Figure)

    def test_one_main_axes(self, pca_df):
        fig = plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        assert len(fig.axes) == 1

    def test_has_legend(self, pca_df):
        fig = plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        assert fig.axes[0].get_legend() is not None

    def test_xlabel_mentions_pc1(self, pca_df):
        fig = plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        assert fig.axes[0].get_xlabel().startswith("PC1")

    def test_xlabel_includes_variance_pct(self, pca_df):
        fig = plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        assert "%" in fig.axes[0].get_xlabel()

    def test_ylabel_mentions_pc2(self, pca_df):
        fig = plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        assert fig.axes[0].get_ylabel().startswith("PC2")

    def test_default_title_is_empty(self, pca_df):
        fig = plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        assert fig.axes[0].get_title() == ""

    def test_custom_title(self, pca_df):
        fig = plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"], title="My PCA")
        assert fig.axes[0].get_title() == "My PCA"

    def test_groups_subset_in_legend(self, pca_df):
        fig = plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"], groups=["A"])
        legend = fig.axes[0].get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert "B" not in labels and "A" in labels

    def test_one_scatter_per_group(self, pca_df):
        fig = plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        from matplotlib.collections import PathCollection
        scatters = [c for c in fig.axes[0].collections
                    if isinstance(c, PathCollection)]
        # Two groups (A and B) -> two scatter collections.
        assert len(scatters) == 2

    def test_shape_col_splits_by_value(self, pca_df):
        df = pca_df.copy()
        df["batch"] = ["B1", "B2"] * (len(df) // 2)
        fig = plot_pca(
            df, ["m1", "m2", "m3", "m4", "m5"], shape_col="batch",
        )
        from matplotlib.collections import PathCollection
        scatters = [c for c in fig.axes[0].collections
                    if isinstance(c, PathCollection)]
        # 2 conditions × 2 batches = 4 collections.
        assert len(scatters) == 4

    def test_shape_map_accepted(self, pca_df):
        df = pca_df.copy()
        df["batch"] = ["B1", "B2"] * (len(df) // 2)
        fig = plot_pca(
            df, ["m1", "m2", "m3", "m4", "m5"],
            shape_col="batch", shape_map={"B1": "o", "B2": "s"},
        )
        assert isinstance(fig, Figure)

    def test_missing_group_col_raises(self, pca_df):
        with pytest.raises(ValueError, match="Group column"):
            plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"],
                     group_col="nope")

    def test_custom_palette(self, pca_df):
        fig = plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"],
                       palette=["#aa0000", "#00aa00"])
        assert isinstance(fig, Figure)

    def test_custom_figsize(self, pca_df):
        fig = plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"],
                       figsize=(8.0, 6.0))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(8.0) and h == pytest.approx(6.0)

    def test_use_existing_axes(self, pca_df):
        fig_pre, ax_pre = plt.subplots()
        fig_out = plot_pca(pca_df, ["m1", "m2", "m3", "m4", "m5"], ax=ax_pre)
        assert fig_out is fig_pre
        plt.close(fig_pre)


# ── viz/pca.py — compute_pca_loadings & plot_pca_loadings ─────────────────────

class TestComputePcaLoadings:
    def test_returns_dataframe_with_pc_columns(self, pca_df):
        loadings = compute_pca_loadings(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        assert isinstance(loadings, pd.DataFrame)
        assert list(loadings.columns) == ["PC1", "PC2"]

    def test_index_is_features(self, pca_df):
        loadings = compute_pca_loadings(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        assert set(loadings.index) == {"m1", "m2", "m3", "m4", "m5"}
        assert loadings.index.name == "feature"

    def test_n_components_three(self, pca_df):
        loadings = compute_pca_loadings(
            pca_df, ["m1", "m2", "m3", "m4", "m5"], n_components=3,
        )
        assert list(loadings.columns) == ["PC1", "PC2", "PC3"]

    def test_components_unit_norm(self, pca_df):
        loadings = compute_pca_loadings(pca_df, ["m1", "m2", "m3", "m4", "m5"])
        # Each PC is a unit-norm vector across features.
        for pc in loadings.columns:
            assert loadings[pc].pow(2).sum() == pytest.approx(1.0, abs=1e-9)

    def test_drops_constant_columns(self, pca_df):
        df = pca_df.copy()
        df["constant"] = 5.0
        loadings = compute_pca_loadings(
            df, ["m1", "m2", "m3", "m4", "m5", "constant"],
        )
        assert "constant" not in loadings.index

    def test_too_few_rows_raises(self, pca_df):
        with pytest.raises(ValueError, match=">= 2 rows"):
            compute_pca_loadings(pca_df.iloc[:1], ["m1", "m2", "m3"])


class TestPlotPcaLoadings:
    def _loadings(self):
        return pd.DataFrame(
            {
                "PC1": [0.7, -0.5, 0.4, 0.1, -0.05, 0.02, 0.01, -0.01,
                        0.005, -0.003, 0.001, -0.0005],
                "PC2": [0.1, 0.6, -0.4, 0.5, -0.3, 0.2, 0.05, -0.05,
                        0.04, -0.04, 0.02, 0.01],
            },
            index=[f"m{i}" for i in range(1, 13)],
        )

    def test_returns_figure(self):
        fig = plot_pca_loadings(self._loadings())
        assert isinstance(fig, Figure)

    def test_two_panels(self):
        fig = plot_pca_loadings(self._loadings())
        assert len(fig.axes) == 2

    def test_top_n_bars_per_panel(self):
        fig = plot_pca_loadings(self._loadings(), n_top=5)
        for ax in fig.axes:
            # One Rectangle per bar.
            bar_count = sum(1 for p in ax.patches)
            assert bar_count == 5

    def test_xlabels_say_loading(self):
        fig = plot_pca_loadings(self._loadings())
        labels = [ax.get_xlabel() for ax in fig.axes]
        assert all("loading" in lbl for lbl in labels)

    def test_pretty_labels_applied(self):
        fig = plot_pca_loadings(
            self._loadings(),
            n_top=3,
            pretty_labels={"m1": "Mean firing rate (Hz)"},
        )
        # The y-axis tick labels should include the pretty label for m1.
        ytick_texts = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert "Mean firing rate (Hz)" in ytick_texts


# ── viz/plate.py ──────────────────────────────────────────────────────────────

@pytest.fixture()
def plate_df():
    """Long-format frame: 2 plates x 24 wells (A1..D6) x 2 metrics."""
    rng = np.random.default_rng(11)
    rows = []
    for plate in (1, 2):
        for r in "ABCD":
            for c in range(1, 7):
                rows.append({
                    "well_id": f"{r}{c}",
                    "plate":   plate,
                    "DIV":     21,
                    "metric":  float(rng.normal(2.0, 0.3)),
                })
    return pd.DataFrame(rows)


class TestPlotPlateHeatmap:
    def test_returns_figure(self, plate_df):
        fig = plot_plate_heatmap(plate_df, "metric")
        assert isinstance(fig, Figure)

    def test_one_axes_per_plate_plus_colorbar(self, plate_df):
        fig = plot_plate_heatmap(plate_df, "metric")
        # 2 plates -> 2 panel axes + 1 colourbar axes = 3 total.
        assert len(fig.axes) == 3

    def test_single_plate_input(self, plate_df):
        fig = plot_plate_heatmap(plate_df[plate_df["plate"] == 1], "metric")
        assert len(fig.axes) == 2  # 1 panel + colourbar

    def test_default_no_suptitle(self, plate_df):
        fig = plot_plate_heatmap(plate_df, "metric")
        assert fig._suptitle is None or fig._suptitle.get_text() == ""

    def test_panel_labels_when_show_plate_labels_true(self, plate_df):
        fig = plot_plate_heatmap(plate_df, "metric", show_plate_labels=True)
        titles = [ax.get_title() for ax in fig.axes[:2]]
        assert "Plate 1" in titles[0]
        assert "Plate 2" in titles[1]

    def test_panel_labels_disabled(self, plate_df):
        fig = plot_plate_heatmap(plate_df, "metric", show_plate_labels=False)
        for ax in fig.axes[:2]:
            assert ax.get_title() == ""

    def test_xtick_count_matches_plate_cols(self, plate_df):
        fig = plot_plate_heatmap(plate_df, "metric")
        # Default plate is 4 rows x 6 cols.
        assert len(fig.axes[0].get_xticks()) == 6
        assert len(fig.axes[0].get_yticks()) == 4

    def test_custom_plate_layout(self, plate_df):
        fig = plot_plate_heatmap(
            plate_df, "metric",
            plate_rows=("A", "B"), plate_cols=(1, 2, 3),
        )
        assert len(fig.axes[0].get_xticks()) == 3
        assert len(fig.axes[0].get_yticks()) == 2

    def test_missing_well_renders_silent(self, plate_df):
        # Drop everything except A1 on plate 1 -> all other cells silent.
        sparse = plate_df[(plate_df["plate"] == 1) & (plate_df["well_id"] == "A1")]
        fig = plot_plate_heatmap(sparse, "metric")
        assert isinstance(fig, Figure)

    def test_missing_metric_raises(self, plate_df):
        with pytest.raises(ValueError, match="not in DataFrame"):
            plot_plate_heatmap(plate_df, "ghost")

    def test_missing_plate_col_raises(self, plate_df):
        with pytest.raises(ValueError, match="Plate column"):
            plot_plate_heatmap(plate_df.drop(columns=["plate"]), "metric")

    def test_missing_well_col_raises(self, plate_df):
        with pytest.raises(ValueError, match="Well column"):
            plot_plate_heatmap(plate_df.drop(columns=["well_id"]), "metric")

    def test_no_plates_raises(self):
        empty = pd.DataFrame({"plate": [], "well_id": [], "metric": []})
        with pytest.raises(ValueError, match="No plates"):
            plot_plate_heatmap(empty, "metric")
