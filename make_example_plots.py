"""
make_example_plots.py
=====================
Generate example plots and CSV outputs from the Plate 2 D28 recording
using the py-mea-axion pipeline.

Usage
-----
    cd <repo-root>
    python make_example_plots.py

Outputs (written to 'example plots/')
--------------------------------------
  01_heatmap_grid_MFR.png    — 4×6 well grid coloured by mean firing rate
  02_raster_<well>.png       — spike raster + burst overlay, one per well
  03_sttc_<well>.png         — STTC matrix, one per well
  04_network_timelines.png   — network-burst timelines, all wells
  05_condition_comparison.png — violin plots comparing conditions
  06_isi_histograms.png      — ISI histograms for one electrode per condition
  burst_table.csv            — all detected single-electrode bursts
  spike_metrics.csv          — per-electrode spike metrics
  well_summary.csv           — per-well summary merged with plate map
"""

from __future__ import annotations

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from py_mea_axion import MEAExperiment
from py_mea_axion.viz.heatmap import plot_electrode_heatmap

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT    = pathlib.Path(__file__).parent
SPK     = ROOT / "LGI2 KD data" / "20251004_LGI2 KD_Plate 2_D28N(000).spk"
OUT_DIR = ROOT / "example plots"
OUT_DIR.mkdir(exist_ok=True)

# ── Full plate layout (all 24 wells) ──────────────────────────────────────────
# Columns 4-6 are structurally empty for Plate 2 but are still recorded.
# min_active_electrodes=0 ensures the pipeline returns data for every well
# so users can inspect any well they choose.

PLATE_MAP = pd.DataFrame([
    ("A1", "SCRM",     28, "rep1"),
    ("A2", "LGI2_KD4", 28, "rep1"),
    ("A3", "LGI2_KD5", 28, "rep1"),
    ("B1", "SCRM",     28, "rep2"),
    ("B2", "LGI2_KD4", 28, "rep2"),
    ("B3", "LGI2_KD5", 28, "rep2"),
    ("C1", "SCRM",     28, "rep3"),
    ("C2", "LGI2_KD4", 28, "rep3"),
    ("C3", "LGI2_KD5", 28, "rep3"),
    ("D1", "SCRM",     28, "rep4"),
    ("D2", "LGI2_KD4", 28, "rep4"),
    ("D3", "LGI2_KD5", 28, "rep4"),
], columns=["well_id", "condition", "DIV", "replicate_id"])

ROWS    = ["A", "B", "C", "D"]
COLS    = [1, 2, 3, 4, 5, 6]
ALL_WELLS = [f"{r}{c}" for r in ROWS for c in COLS]

# ── Condition colours (colorblind-friendly) ───────────────────────────────────

COND_COLOR = {
    "SCRM":     "#4477AA",
    "LGI2_KD4": "#EE6677",
    "LGI2_KD5": "#CCBB44",
}
COND_ORDER = ["SCRM", "LGI2_KD4", "LGI2_KD5"]

# ── Run pipeline ──────────────────────────────────────────────────────────────

print("Running pipeline …")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    exp = MEAExperiment(
        SPK,
        metadata=PLATE_MAP,
        fs_override=12500,
        active_threshold_hz=5 / 60,
        min_active_electrodes=0,          # keep all wells regardless of activity
        burst_kwargs=dict(
            algorithm="isi_threshold",
            min_spikes=5,
            max_isi_s=0.1,
            min_ibi_s=0.0,
        ),
        network_kwargs=dict(
            algorithm="combined_isi",
            min_spikes=50,
            max_isi_s=0.1,
            participation_threshold=0.35,
        ),
    ).run()

ws  = exp.joined_summary()
sm  = exp.spike_metrics
print(f"  {len(exp.wells)} wells loaded")

cond_of = PLATE_MAP.set_index("well_id")["condition"].to_dict()

# ── Save CSVs ─────────────────────────────────────────────────────────────────

exp.spike_metrics.to_csv(OUT_DIR / "spike_metrics.csv", index=False, float_format="%.6f")
exp.burst_table.to_csv(OUT_DIR / "burst_table.csv", index=False, float_format="%.6f")
exp.to_csv(OUT_DIR / "well_summary.csv")
print("  saved CSVs")

# ── 01: MFR heatmap grid (all 24 wells, 4×6) ─────────────────────────────────

print("  01 heatmap grid …")

# Shared colour scale from active electrodes only
active_mfr = sm.loc[sm["is_active"], "mfr_hz"]
vmin = 0.0
vmax = float(np.nanpercentile(active_mfr, 97)) if len(active_mfr) else 1.0

fig, axes = plt.subplots(
    len(ROWS), len(COLS),
    figsize=(len(COLS) * 3.6, len(ROWS) * 3.4),
)

for ri, row_letter in enumerate(ROWS):
    for ci, col_num in enumerate(COLS):
        well = f"{row_letter}{col_num}"
        ax   = axes[ri, ci]
        well_rows = sm[sm["well_id"] == well]
        values = dict(zip(well_rows["electrode_id"], well_rows["mfr_hz"]))
        cond   = cond_of.get(well, "—")
        plot_electrode_heatmap(
            values, well_id=well, metric_name="MFR (Hz)",
            vmin=vmin, vmax=vmax, ax=ax,
            title=f"{well}  {cond}",
        )

fig.suptitle("Mean Firing Rate per electrode  |  Plate 2, DIV 28", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "01_heatmap_grid_MFR.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  saved 01_heatmap_grid_MFR.png")

# ── 02: Spike rasters — one file per well ────────────────────────────────────

print("  02 rasters …")
for well in ALL_WELLS:
    cond  = cond_of.get(well, "empty")
    fname = f"02_raster_{well}_{cond}.png"
    fig   = exp.plot_raster(well)
    fig.axes[0].set_title(f"Spike raster — {well} ({cond})", fontsize=10)
    fig.savefig(OUT_DIR / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
print(f"  saved {len(ALL_WELLS)} raster files")

# ── 03: STTC matrices — one file per well ────────────────────────────────────

print("  03 STTC …")
for well in ALL_WELLS:
    cond  = cond_of.get(well, "empty")
    fname = f"03_sttc_{well}_{cond}.png"
    fig   = exp.plot_sttc(well, title=f"STTC — {well} ({cond})")
    fig.savefig(OUT_DIR / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
print(f"  saved {len(ALL_WELLS)} STTC files")

# ── 04: Network-burst timelines (all wells, stacked) ─────────────────────────

print("  04 network timelines …")
fig, axes = plt.subplots(
    len(ALL_WELLS), 1,
    figsize=(12, 1.6 * len(ALL_WELLS)),
    sharex=True,
)
total = exp.total_time_s

for ax, well in zip(axes, ALL_WELLS):
    cond  = cond_of.get(well, None)
    color = COND_COLOR.get(cond, "#AAAAAA")
    nbs   = exp.network_bursts.get(well, [])
    if nbs:
        ax.barh(
            0,
            [nb.duration for nb in nbs],
            left=[nb.start_time for nb in nbs],
            height=0.6,
            color=color, alpha=0.85, edgecolor="none",
        )
    ax.set_xlim(0, total)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([0])
    label = f"{well}  {cond if cond else '—'}  (n={len(nbs)})"
    ax.set_yticklabels([label], fontsize=7)

axes[-1].set_xlabel("Time (s)", fontsize=9)
axes[0].set_title("Network bursts  |  Plate 2, DIV 28", fontsize=11)

handles = [mpatches.Patch(color=COND_COLOR[c], label=c) for c in COND_ORDER]
fig.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.7)
fig.tight_layout()
fig.savefig(OUT_DIR / "04_network_timelines.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  saved 04_network_timelines.png")

# ── 05: Condition comparison — violin plots ───────────────────────────────────

print("  05 condition comparison …")
METRICS = [
    ("mean_mfr_active_hz",    "MFR (Hz)"),
    ("n_active",              "N active electrodes"),
    ("n_network_bursts",      "N network bursts"),
    ("burst_duration_avg",    "Mean burst duration (s)"),
]

# Only wells with a known condition
ws_cond = ws[ws["condition"].notna()].copy()

fig, axes = plt.subplots(1, len(METRICS), figsize=(3.8 * len(METRICS), 5))

rng = np.random.default_rng(42)

for ax, (col, label) in zip(axes, METRICS):
    parts = ax.violinplot(
        [ws_cond.loc[ws_cond["condition"] == c, col].dropna().values.astype(float)
         for c in COND_ORDER],
        positions=range(len(COND_ORDER)),
        showmedians=True,
        showextrema=True,
    )
    # Colour each violin body
    for body, cond in zip(parts["bodies"], COND_ORDER):
        body.set_facecolor(COND_COLOR[cond])
        body.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    parts["cbars"].set_color("black")
    parts["cmins"].set_color("black")
    parts["cmaxes"].set_color("black")

    # Overlay individual data points
    for xi, cond in enumerate(COND_ORDER):
        vals = ws_cond.loc[ws_cond["condition"] == cond, col].dropna().values.astype(float)
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(xi + jitter, vals, color="black", s=22, zorder=4, alpha=0.8)

    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels([c.replace("_", "\n") for c in COND_ORDER], fontsize=8)
    ax.set_ylabel(label, fontsize=8)
    ax.set_title(label, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles = [mpatches.Patch(color=COND_COLOR[c], label=c) for c in COND_ORDER]
fig.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.7)
fig.suptitle("Condition comparison  |  Plate 2, DIV 28  (n=4 replicates per condition)",
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "05_condition_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  saved 05_condition_comparison.png")

# ── 06: ISI histograms (most active electrode per condition) ──────────────────

print("  06 ISI histograms …")
EXAMPLE_WELLS = [("A1", "SCRM"), ("A2", "LGI2_KD4"), ("A3", "LGI2_KD5")]

fig, axes = plt.subplots(1, len(EXAMPLE_WELLS), figsize=(4.5 * len(EXAMPLE_WELLS), 3.5))

for ax, (well, cond) in zip(axes, EXAMPLE_WELLS):
    well_sm = sm[(sm["well_id"] == well) & sm["is_active"]]
    if well_sm.empty:
        ax.text(0.5, 0.5, "no active\nelectrodes", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="grey")
        ax.set_title(f"{well}  ({cond})", fontsize=9)
        continue
    top_eid = well_sm.sort_values("mfr_hz", ascending=False).iloc[0]["electrode_id"]
    isis = np.diff(exp.well_spikes(well)[top_eid])
    isis = isis[isis > 0]
    color = COND_COLOR.get(cond, "#888888")
    bins  = np.logspace(np.log10(isis.min()), np.log10(isis.max()), 50)
    ax.hist(isis, bins=bins, color=color, alpha=0.8, edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel("ISI (s)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(f"{top_eid}  ({cond})", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("ISI histograms — most active electrode per condition  |  Plate 2, DIV 28",
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "06_isi_histograms.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  saved 06_isi_histograms.png")

# ── Done ──────────────────────────────────────────────────────────────────────

print(f"\nAll outputs written to:  {OUT_DIR}")
print(f"\nWell summary (condition × metric):")
print(ws[["well_id", "condition", "n_active", "mean_mfr_active_hz",
          "n_network_bursts", "burst_duration_avg"]].to_string(index=False))
