"""
benchmark_all.py
================
Run py-mea-axion on every Plate 2 recording that has a matching
NeuralMetric Tools CSV export, then pool the per-well results and
report overall agreement statistics + scatter plots.

Usage
-----
    cd <repo-root>
    python benchmarking/benchmark_all.py
"""

from __future__ import annotations

import pathlib
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = pathlib.Path(__file__).parent.parent
SPK_DIR     = ROOT / "LGI2 KD data"
BENCH_DIR   = pathlib.Path(__file__).parent
OUT_DIR     = BENCH_DIR / "figures_all"
OUT_DIR.mkdir(exist_ok=True)

# ── Plate-2 map (fixed across all DIVs) ──────────────────────────────────────

PLATE2_MAP = pd.DataFrame([
    ("A1", "SCRM",     "rep1"),
    ("A2", "LGI2_KD4", "rep1"),
    ("A3", "LGI2_KD5", "rep1"),
    ("B1", "SCRM",     "rep2"),
    ("B2", "LGI2_KD4", "rep2"),
    ("B3", "LGI2_KD5", "rep2"),
    ("C1", "SCRM",     "rep3"),
    ("C2", "LGI2_KD4", "rep3"),
    ("C3", "LGI2_KD5", "rep3"),
    ("D1", "SCRM",     "rep4"),
    ("D2", "LGI2_KD4", "rep4"),
    ("D3", "LGI2_KD5", "rep4"),
], columns=["well_id", "condition", "replicate_id"])

ALL_WELLS = list(PLATE2_MAP["well_id"])

# ── NeuralMetric CSV parser (same as benchmark_plate2_d28.py) ────────────────

def parse_nm_csv(path: pathlib.Path) -> pd.DataFrame:
    """Return wide DataFrame (rows=metrics, columns=well_ids A1-D6)."""
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()

    header_idx = next(
        i for i, ln in enumerate(lines) if ln.startswith("Well Averages")
    )
    well_ids = [c.strip() for c in lines[header_idx].split(",")[1:] if c.strip()]

    records: dict[str, list] = {}
    section = ""
    for ln in lines[header_idx + 1:]:
        if re.match(r"Measurement,\w+_\d+", ln):
            break
        if not ln.startswith(" ") and not ln.startswith(","):
            section = ln.strip().rstrip(",")
            continue
        parts = ln.split(",")
        metric_raw = parts[0].strip()
        if not metric_raw:
            continue
        metric = f"{section} / {metric_raw}" if section else metric_raw
        values = []
        for v in parts[1: len(well_ids) + 1]:
            v = v.strip()
            values.append(float(v) if v else None)
        while len(values) < len(well_ids):
            values.append(None)
        records[metric] = values

    df = pd.DataFrame(records, index=well_ids).T
    df.index.name = "metric"
    df.columns.name = "well_id"
    return df


def nm_row(df: pd.DataFrame, fragment: str, wells: list[str]) -> pd.Series:
    """Extract one metric row for the requested wells."""
    matches = [m for m in df.index if fragment.lower() in m.lower()]
    if not matches:
        return pd.Series([np.nan] * len(wells), index=wells)
    return df.loc[matches[0], wells].astype(float)


# ── Find matching .spk for each CSV ──────────────────────────────────────────

def spk_for_csv(csv_path: pathlib.Path) -> pathlib.Path | None:
    """Return the .spk file matching a NeuralMetric CSV, or None."""
    stem = csv_path.stem.replace("_neuralMetrics", "")
    spk = SPK_DIR / f"{stem}.spk"
    return spk if spk.exists() else None


# ── Main loop ─────────────────────────────────────────────────────────────────

from py_mea_axion.pipeline import MEAExperiment  # noqa: E402

csv_files = sorted(BENCH_DIR.glob("*_neuralMetrics.csv"))
print(f"Found {len(csv_files)} NeuralMetric CSV files.")

rows = []   # one row per (recording, well)

for csv_path in csv_files:
    spk_path = spk_for_csv(csv_path)
    if spk_path is None:
        print(f"  [SKIP] no .spk for {csv_path.name}")
        continue

    # Extract DIV from filename  e.g. "D28N" → 28
    m = re.search(r"_D(\d+)N", csv_path.stem)
    div = int(m.group(1)) if m else -1

    print(f"  {csv_path.stem[:40]}  DIV={div} … ", end="", flush=True)

    # Parse NeuralMetric CSV
    try:
        nm_df = parse_nm_csv(csv_path)
    except Exception as e:
        print(f"NM parse error: {e}")
        continue

    nm_mfr       = nm_row(nm_df, "Weighted Mean Firing Rate", ALL_WELLS)
    nm_n_active  = nm_row(nm_df, "Number of Active Electrodes", ALL_WELLS)
    nm_n_bursts  = nm_row(nm_df, "Electrode Burst Metrics / Number of Bursts", ALL_WELLS)
    nm_burst_dur = nm_row(nm_df, "Burst Duration - Avg (sec)", ALL_WELLS)
    nm_n_nb      = nm_row(nm_df, "Number of Network Bursts", ALL_WELLS)
    nm_nb_dur    = nm_row(nm_df, "Network Burst Duration - Avg (sec)", ALL_WELLS)

    # Run py-mea-axion
    plate_map = PLATE2_MAP.copy()
    plate_map["DIV"] = div

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = MEAExperiment(
                spk_path,
                metadata=plate_map,
                fs_override=12500,
                active_threshold_hz=5 / 60,
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
    except Exception as e:
        print(f"pipeline error: {e}")
        continue

    ws  = exp.well_summary.set_index("well_id")
    bt  = exp.burst_table
    nbs = exp.network_bursts

    bc  = bt.groupby("well_id").size()
    bdc = bt.groupby("well_id")["duration"].mean()
    nb_dur = {
        w: float(np.mean([nb.duration for nb in nbs.get(w, [])]))
        if nbs.get(w) else np.nan
        for w in ALL_WELLS
    }

    for well in ALL_WELLS:
        # Skip wells that are inactive in BOTH tools (NM n_active == 0 or NaN)
        nm_active = nm_n_active.get(well, np.nan)
        if pd.isna(nm_active) or nm_active == 0:
            continue

        rows.append({
            "recording": csv_path.stem,
            "div":       div,
            "well":      well,
            "condition": PLATE2_MAP.set_index("well_id").loc[well, "condition"],
            # NeuralMetric values
            "nm_mfr":       nm_mfr.get(well, np.nan),
            "nm_n_active":  nm_n_active.get(well, np.nan),
            "nm_n_bursts":  nm_n_bursts.get(well, np.nan),
            "nm_burst_dur": nm_burst_dur.get(well, np.nan),
            "nm_n_nb":      nm_n_nb.get(well, np.nan),
            "nm_nb_dur":    nm_nb_dur.get(well, np.nan),
            # py-mea-axion values
            "pma_mfr":       ws.loc[well, "mean_mfr_active_hz"] if well in ws.index else np.nan,
            "pma_n_active":  ws.loc[well, "n_active"]           if well in ws.index else np.nan,
            "pma_n_bursts":  float(bc.get(well, 0)),
            "pma_burst_dur": float(bdc.get(well, np.nan)),
            "pma_n_nb":      ws.loc[well, "n_network_bursts"]   if well in ws.index else np.nan,
            "pma_nb_dur":    nb_dur[well],
        })

    print("done")

df = pd.DataFrame(rows)
print(f"\nCollected {len(df)} active well-observations across {df['recording'].nunique()} recordings.\n")

# Save raw comparison table
df.to_csv(BENCH_DIR / "all_benchmark_raw.csv", index=False, float_format="%.6f")

# ── Summary statistics ────────────────────────────────────────────────────────

METRIC_PAIRS = [
    ("MFR (Hz)",                "nm_mfr",       "pma_mfr"),
    ("N active electrodes",     "nm_n_active",  "pma_n_active"),
    ("N bursts",                "nm_n_bursts",  "pma_n_bursts"),
    ("Mean burst duration (s)", "nm_burst_dur", "pma_burst_dur"),
    ("N network bursts",        "nm_n_nb",      "pma_n_nb"),
    ("Mean NB duration (s)",    "nm_nb_dur",    "pma_nb_dur"),
]

summary_rows = []
for label, nm_col, pma_col in METRIC_PAIRS:
    sub = df[[nm_col, pma_col]].dropna()
    x, y = sub[nm_col].values, sub[pma_col].values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n >= 2 and x.std() > 0 and y.std() > 0:
        r, p = pearsonr(x, y)
    else:
        r, p = np.nan, np.nan
    mae  = float(np.mean(np.abs(x - y))) if n > 0 else np.nan
    mpct = float(np.mean(np.abs(x - y) / (np.abs(x) + 1e-12)) * 100) if n > 0 else np.nan
    bias = float(np.mean(y - x)) if n > 0 else np.nan
    summary_rows.append(dict(
        metric=label, n=n,
        pearson_r=r, p_value=p,
        bias=bias,
        mean_abs_error=mae,
        mean_pct_error=mpct,
    ))

summary = pd.DataFrame(summary_rows)
summary.to_csv(BENCH_DIR / "all_benchmark_summary.csv", index=False, float_format="%.4f")

print("=" * 72)
print(f"{'Metric':<26} {'n':>5} {'r':>7} {'bias':>10} {'MAE':>10} {'%err':>7}")
print("-" * 72)
for _, row in summary.iterrows():
    print(
        f"{row['metric']:<26} {int(row['n']):>5} "
        f"{row['pearson_r']:>7.4f} "
        f"{row['bias']:>+10.4f} "
        f"{row['mean_abs_error']:>10.4f} "
        f"{row['mean_pct_error']:>6.1f}%"
    )
print("=" * 72)

# ── Scatter plots ─────────────────────────────────────────────────────────────

COND_COLOR = {
    "SCRM":     "#4477AA",
    "LGI2_KD4": "#EE6677",
    "LGI2_KD5": "#CCBB44",
}


def scatter_all(nm_col, pma_col, xlabel, ylabel, title, fname):
    sub = df[[nm_col, pma_col, "condition"]].dropna()
    x = sub[nm_col].values.astype(float)
    y = sub[pma_col].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, conds = x[mask], y[mask], sub["condition"].values[mask]

    fig, ax = plt.subplots(figsize=(5, 5))
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, zorder=0)

    for cond in COND_COLOR:
        idx = conds == cond
        ax.scatter(x[idx], y[idx], color=COND_COLOR[cond], s=20,
                   alpha=0.7, zorder=3, edgecolors="none", label=cond)

    if len(x) >= 2 and x.std() > 0 and y.std() > 0:
        r, p = pearsonr(x, y)
        ax.text(0.05, 0.93, f"r = {r:.4f}  n = {len(x)}",
                transform=ax.transAxes, fontsize=8, verticalalignment="top")

    ax.legend(fontsize=7, framealpha=0.6, markerscale=1.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=600)
    plt.close(fig)
    print(f"  saved {fname}")


print("\nGenerating figures …")
scatter_all("nm_mfr",       "pma_mfr",
            "NeuralMetric — Weighted MFR (Hz)",
            "py-mea-axion — Mean MFR active (Hz)",
            "Mean firing rate", "all_mfr.png")

scatter_all("nm_n_active",  "pma_n_active",
            "NeuralMetric — N active electrodes",
            "py-mea-axion — N active electrodes",
            "Active electrode count", "all_n_active.png")

scatter_all("nm_n_bursts",  "pma_n_bursts",
            "NeuralMetric — N bursts (electrode total)",
            "py-mea-axion — N bursts (electrode total)",
            "Electrode burst count", "all_n_bursts.png")

scatter_all("nm_burst_dur", "pma_burst_dur",
            "NeuralMetric — Mean burst duration (s)",
            "py-mea-axion — Mean burst duration (s)",
            "Mean burst duration", "all_burst_dur.png")

scatter_all("nm_n_nb",      "pma_n_nb",
            "NeuralMetric — N network bursts",
            "py-mea-axion — N network bursts",
            "Network burst count", "all_n_nb.png")

scatter_all("nm_nb_dur",    "pma_nb_dur",
            "NeuralMetric — Mean NB duration (s)",
            "py-mea-axion — Mean NB duration (s)",
            "Mean network burst duration", "all_nb_dur.png")

print("\nDone.  Results in benchmarking/figures_all/ and benchmarking/all_benchmark_summary.csv")
