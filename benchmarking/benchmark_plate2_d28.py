"""
Benchmark py-mea-axion against NeuralMetric Tools.

Recording : Plate 2, DIV 28
    20251004_LGI2 KD_Plate 2_D28N(000).spk
NeuralMetric export:
    20251004_LGI2 KD_Plate 2_D28N(000)_neuralMetrics.csv

Parameters matched to the NeuralMetric export settings:
    Active electrode   : >= 5 spks/min  (= 5/60 Hz)
    Electrode burst    : ISI threshold, min_spikes=5, max_isi=0.1 s
    Network burst      : participation=35%, min_network_ibi=1.0 s
"""

from __future__ import annotations

import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent.parent
SPK_FILE = ROOT / "LGI2 KD data" / "20251004_LGI2 KD_Plate 2_D28N(000).spk"
NM_CSV = pathlib.Path(__file__).parent / "20251004_LGI2 KD_Plate 2_D28N(000)_neuralMetrics.csv"
OUT_DIR = pathlib.Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Parse NeuralMetric well-average CSV
# ---------------------------------------------------------------------------

def parse_neuralmetric_csv(path: pathlib.Path) -> pd.DataFrame:
    """Return a DataFrame (rows=metrics, columns=well_id) from the wide export."""
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()

    # Find the "Well Averages" header line
    header_idx = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("Well Averages")
    )
    header_line = lines[header_idx]
    well_ids = [c.strip() for c in header_line.split(",")[1:] if c.strip()]

    # Collect data rows until the electrode-level section begins
    # (electrode section starts with a line like "Measurement,A1_11,...")
    records: dict[str, list[float | None]] = {}
    section = ""
    for ln in lines[header_idx + 1 :]:
        if re.match(r"Measurement,\w+_\d+", ln):
            break  # electrode-level section starts
        # Section headers (e.g. "Activity Metrics", "Burst Metrics") — no comma prefix
        if not ln.startswith(" ") and not ln.startswith(","):
            section = ln.strip().rstrip(",")
            continue
        parts = ln.split(",")
        metric_raw = parts[0].strip()
        if not metric_raw:
            continue
        metric = f"{section} / {metric_raw}" if section else metric_raw
        # Parse values (empty string → None)
        values: list[float | None] = []
        for v in parts[1 : len(well_ids) + 1]:
            v = v.strip()
            values.append(float(v) if v else None)
        # Pad to length if row is shorter than header
        while len(values) < len(well_ids):
            values.append(None)
        records[metric] = values

    df = pd.DataFrame(records, index=well_ids).T
    df.index.name = "metric"
    df.columns.name = "well_id"
    return df


nm_df = parse_neuralmetric_csv(NM_CSV)

# Extract the metrics we want to compare (well columns A1-D6)
ACTIVE_WELLS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2", "D3"]

def nm_metric(name_fragment: str) -> pd.Series:
    """Return values for active wells for a NeuralMetric metric (partial match)."""
    matches = [m for m in nm_df.index if name_fragment.lower() in m.lower()]
    if not matches:
        raise KeyError(f"No metric matching {name_fragment!r}. Available:\n{list(nm_df.index)}")
    row = nm_df.loc[matches[0], ACTIVE_WELLS].astype(float)
    row.name = matches[0]
    return row

nm_mfr         = nm_metric("Weighted Mean Firing Rate")
nm_n_active    = nm_metric("Number of Active Electrodes")
nm_n_bursts    = nm_metric("Electrode Burst Metrics / Number of Bursts")
nm_burst_dur   = nm_metric("Burst Duration - Avg (sec)")
nm_n_netbursts = nm_metric("Number of Network Bursts")
nm_nb_dur      = nm_metric("Network Burst Duration - Avg (sec)")

# ---------------------------------------------------------------------------
# 2. Run py-mea-axion with matching parameters
# ---------------------------------------------------------------------------

from py_mea_axion.pipeline import MEAExperiment  # noqa: E402

plate_map = pd.DataFrame(
    [
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
    ],
    columns=["well_id", "condition", "DIV", "replicate_id"],
)

print("Running py-mea-axion pipeline …")
exp = MEAExperiment(
    SPK_FILE,
    metadata=plate_map,
    fs_override=12500,
    active_threshold_hz=5 / 60,          # match NeuralMetric: 5 spks/min
    burst_kwargs=dict(
        algorithm="isi_threshold",
        min_spikes=5,                     # match NeuralMetric electrode burst
        max_isi_s=0.1,
        min_ibi_s=0.0,                    # no IBI merging, matches NeuralMetric
    ),
    network_kwargs=dict(
        algorithm="combined_isi",         # combined spike-train ISI, matches NeuralMetric
        min_spikes=50,                    # match NeuralMetric: 50 spikes min
        max_isi_s=0.1,                    # match NeuralMetric: 100 ms max ISI
        participation_threshold=0.35,     # match NeuralMetric: 35 %
    ),
).run()

ws = exp.well_summary.set_index("well_id")

# Total burst count per well (summed across all electrodes)
burst_counts = (
    exp.burst_table.groupby("well_id").size()
    if not exp.burst_table.empty
    else pd.Series(dtype=float)
)
# Mean network burst duration per well
nb_dur_dict = {
    well: float(np.mean([nb.duration for nb in nbs]))
    if nbs else float("nan")
    for well, nbs in exp.network_bursts.items()
}

# Build py-mea-axion series aligned to ACTIVE_WELLS
pma_mfr         = ws.loc[ACTIVE_WELLS, "mean_mfr_active_hz"]
pma_n_active    = ws.loc[ACTIVE_WELLS, "n_active"]
pma_n_bursts    = pd.Series(
    [burst_counts.get(w, 0) for w in ACTIVE_WELLS],
    index=ACTIVE_WELLS, name="n_bursts",
).astype(float)
pma_burst_dur   = ws.loc[ACTIVE_WELLS, "mean_burst_duration_s"]
pma_n_netbursts = ws.loc[ACTIVE_WELLS, "n_network_bursts"]
pma_nb_dur      = pd.Series(
    [nb_dur_dict.get(w, float("nan")) for w in ACTIVE_WELLS],
    index=ACTIVE_WELLS, name="mean_nb_dur_s",
)

# ---------------------------------------------------------------------------
# 3. Scatter-plot helper
# ---------------------------------------------------------------------------

CONDITION_COLOR = {
    "SCRM":     "#4477AA",
    "LGI2_KD4": "#EE6677",
    "LGI2_KD5": "#CCBB44",
}

def condition_of(well: str) -> str:
    row = plate_map[plate_map.well_id == well]
    return row.iloc[0]["condition"] if len(row) else "unknown"

def scatter_benchmark(
    nm_vals: pd.Series,
    pma_vals: pd.Series,
    xlabel: str,
    ylabel: str,
    title: str,
    fname: str,
) -> None:
    x = nm_vals.values.astype(float)
    y = pma_vals.values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    wells = [w for w, m in zip(ACTIVE_WELLS, mask) if m]

    fig, ax = plt.subplots(figsize=(5, 5))

    # y = x reference line
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, zorder=0)

    for well, xi, yi in zip(wells, x, y):
        cond = condition_of(well)
        color = CONDITION_COLOR.get(cond, "grey")
        ax.scatter(xi, yi, color=color, s=60, zorder=3, edgecolors="white", linewidths=0.4)
        ax.annotate(well, (xi, yi), fontsize=6, textcoords="offset points", xytext=(4, 2))

    if len(x) >= 2:
        r, p = pearsonr(x, y)
        ax.text(
            0.05, 0.93, f"r = {r:.3f}  (p = {p:.3f})",
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top",
        )

    # legend
    for cond, color in CONDITION_COLOR.items():
        ax.scatter([], [], color=color, label=cond, s=40)
    ax.legend(fontsize=7, framealpha=0.6)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=600)
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# 4. Generate figures
# ---------------------------------------------------------------------------

print("Generating comparison figures …")

scatter_benchmark(
    nm_mfr, pma_mfr,
    xlabel="NeuralMetric Tools — Weighted MFR (Hz)",
    ylabel="py-mea-axion — Mean MFR active (Hz)",
    title="Mean firing rate",
    fname="bench_mfr.png",
)

scatter_benchmark(
    nm_n_active, pma_n_active,
    xlabel="NeuralMetric Tools — N active electrodes",
    ylabel="py-mea-axion — N active electrodes",
    title="Active electrode count",
    fname="bench_n_active.png",
)

scatter_benchmark(
    nm_n_bursts, pma_n_bursts,
    xlabel="NeuralMetric Tools — N bursts (electrode total)",
    ylabel="py-mea-axion — N bursts (electrode total)",
    title="Electrode burst count",
    fname="bench_n_bursts.png",
)

scatter_benchmark(
    nm_burst_dur, pma_burst_dur,
    xlabel="NeuralMetric Tools — Mean burst duration (s)",
    ylabel="py-mea-axion — Mean burst duration (s)",
    title="Mean burst duration",
    fname="bench_burst_dur.png",
)

scatter_benchmark(
    nm_n_netbursts, pma_n_netbursts,
    xlabel="NeuralMetric Tools — N network bursts",
    ylabel="py-mea-axion — N network bursts",
    title="Network burst count",
    fname="bench_n_network_bursts.png",
)

scatter_benchmark(
    nm_nb_dur, pma_nb_dur,
    xlabel="NeuralMetric Tools — Mean network burst duration (s)",
    ylabel="py-mea-axion — Mean network burst duration (s)",
    title="Mean network burst duration",
    fname="bench_network_burst_dur.png",
)

# ---------------------------------------------------------------------------
# 5. Numeric summary table
# ---------------------------------------------------------------------------

summary_rows = []
for metric_name, nm_vals, pma_vals in [
    ("MFR (Hz)",                  nm_mfr,         pma_mfr),
    ("N active electrodes",       nm_n_active,     pma_n_active),
    ("N bursts",                  nm_n_bursts,     pma_n_bursts),
    ("Mean burst duration (s)",   nm_burst_dur,    pma_burst_dur),
    ("N network bursts",          nm_n_netbursts,  pma_n_netbursts),
    ("Mean NB duration (s)",      nm_nb_dur,       pma_nb_dur),
]:
    x = nm_vals.values.astype(float)
    y = pma_vals.values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 2:
        r, p = pearsonr(x[mask], y[mask])
        mae = np.mean(np.abs(x[mask] - y[mask]))
        pct_err = np.mean(np.abs(x[mask] - y[mask]) / (np.abs(x[mask]) + 1e-12)) * 100
    else:
        r, p, mae, pct_err = np.nan, np.nan, np.nan, np.nan
    summary_rows.append(
        dict(metric=metric_name, n=mask.sum(), pearson_r=r, p_value=p,
             mean_abs_error=mae, mean_pct_error=pct_err)
    )

summary = pd.DataFrame(summary_rows)
summary_path = pathlib.Path(__file__).parent / "benchmark_summary.csv"
summary.to_csv(summary_path, index=False, float_format="%.4f")
print(f"\nSummary written to {summary_path.name}")
print(summary.to_string(index=False))
