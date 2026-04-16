"""
benchmark_all.py
================
Compare py-mea-axion output against NeuralMetric Tools exports across
all recordings (both plates, all DIVs).

Covers all metric categories implemented in the pipeline:
  - Activity          : MFR, weighted MFR, ISI CV, active electrode count
  - Electrode burst   : duration, spike count, ISI (mean/median/ratio),
                        IBI, frequency, burst %, IBI CV
  - Network burst     : count, frequency, duration, spikes, ISI, participation,
                        burst %, IBI CV, normalised duration IQR
  - Average NB        : peak rate, time to peak, leader-electrode %

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

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT      = pathlib.Path(__file__).parent.parent
SPK_DIR   = ROOT / "LGI2 KD data"
BENCH_DIR = pathlib.Path(__file__).parent
NM_DIR    = BENCH_DIR / "neuralmetrics"
OUT_DIR   = BENCH_DIR / "figures_all"
OUT_DIR.mkdir(exist_ok=True)

# ── Plate maps ─────────────────────────────────────────────────────────────────

# Plate 1 — 24 wells (6 per row × 4 rows).
# Columns 1-3 = Batch 1, columns 4-6 = Batch 2.
_p1_rows = []
for row in "ABCD":
    for col, (cond, batch) in enumerate(
        [("SCRM","B1"),("LGI2_KD4","B1"),("LGI2_KD5","B1"),
         ("SCRM","B2"),("LGI2_KD4","B2"),("LGI2_KD5","B2")],
        start=1,
    ):
        _p1_rows.append((f"{row}{col}", cond, batch))

PLATE1_MAP = pd.DataFrame(_p1_rows, columns=["well_id", "condition", "batch"])
PLATE1_WELLS = list(PLATE1_MAP["well_id"])

# Plate 2 — 12 wells (3 per row × 4 rows).
PLATE2_MAP = pd.DataFrame([
    ("A1","SCRM","B1"), ("A2","LGI2_KD4","B1"), ("A3","LGI2_KD5","B1"),
    ("B1","SCRM","B1"), ("B2","LGI2_KD4","B1"), ("B3","LGI2_KD5","B1"),
    ("C1","SCRM","B1"), ("C2","LGI2_KD4","B1"), ("C3","LGI2_KD5","B1"),
    ("D1","SCRM","B1"), ("D2","LGI2_KD4","B1"), ("D3","LGI2_KD5","B1"),
], columns=["well_id", "condition", "batch"])
PLATE2_WELLS = list(PLATE2_MAP["well_id"])

PLATE_INFO = {
    "Plate 1": (PLATE1_MAP, PLATE1_WELLS),
    "Plate 2": (PLATE2_MAP, PLATE2_WELLS),
}

# ── Metric mapping: (label, nm_fragment, pma_column) ──────────────────────────
#
# nm_fragment : case-insensitive substring matched against NeuralMetric row names
# pma_column  : column name in exp.well_summary

METRIC_PAIRS = [
    # ── Activity ──────────────────────────────────────────────────────────────
    ("Mean MFR (Hz)",              "Mean Firing Rate (Hz)",               "mean_mfr_active_hz"),
    ("Weighted MFR (Hz)",          "Weighted Mean Firing Rate",           "weighted_mean_mfr_hz"),
    ("N active electrodes",        "Number of Active Electrodes",         "n_active"),
    ("ISI CV",                     "ISI Coefficient of Variation - Avg",  "isi_cv_avg"),
    # ── Electrode burst ───────────────────────────────────────────────────────
    ("N bursts (total)",           "Electrode Burst Metrics / Number of Bursts",             "n_bursts"),
    ("N bursting electrodes",      "Number of Bursting Electrodes",                          "n_bursting_electrodes"),
    ("Burst duration avg (s)",     "Burst Duration - Avg (sec)",                             "burst_duration_avg"),
    ("Burst duration std (s)",     "Burst Duration - Std (sec)",                             "burst_duration_std"),
    ("Spikes/burst avg",           "Number of Spikes per Burst - Avg",                       "n_spikes_per_burst_avg"),
    ("Spikes/burst std",           "Number of Spikes per Burst - Std",                       "n_spikes_per_burst_std"),
    ("Mean ISI burst avg (s)",     "Mean ISI within Burst - Avg",                            "mean_isi_within_burst_avg"),
    ("Mean ISI burst std (s)",     "Mean ISI within Burst - Std",                            "mean_isi_within_burst_std"),
    ("Median ISI burst avg (s)",   "Median ISI within Burst - Avg",                          "median_isi_within_burst_avg"),
    ("Median ISI burst std (s)",   "Median ISI within Burst - Std",                          "median_isi_within_burst_std"),
    ("Median/Mean ISI burst avg",  "Median/Mean ISI within Burst - Avg",                     "median_mean_isi_ratio_burst_avg"),
    ("Median/Mean ISI burst std",  "Median/Mean ISI within Burst - Std",                     "median_mean_isi_ratio_burst_std"),
    ("IBI avg (s)",                "Inter-Burst Interval - Avg",                             "ibi_avg"),
    ("IBI std (s)",                "Inter-Burst Interval - Std",                             "ibi_std"),
    ("Burst freq avg (Hz)",        "Burst Frequency - Avg",                                  "burst_freq_avg"),
    ("Burst freq std (Hz)",        "Burst Frequency - Std",                                  "burst_freq_std"),
    ("IBI CV avg",                 "IBI Coefficient of Variation - Avg",                     "ibi_cv_avg"),
    ("IBI CV std",                 "IBI Coefficient of Variation - Std",                     "ibi_cv_std"),
    ("Burst % avg",                "Burst Percentage - Avg",                                 "burst_pct_avg"),
    ("Burst % std",                "Burst Percentage - Std",                                 "burst_pct_std"),
    # ── Network burst ─────────────────────────────────────────────────────────
    ("N network bursts",           "Number of Network Bursts",                               "n_network_bursts"),
    ("NB frequency (Hz)",          "Network Burst Frequency",                                "network_burst_freq"),
    ("NB duration avg (s)",        "Network Burst Duration - Avg",                           "network_burst_duration_avg"),
    ("NB duration std (s)",        "Network Burst Duration - Std",                           "network_burst_duration_std"),
    ("Spikes/NB avg",              "Number of Spikes per Network Burst - Avg",               "n_spikes_per_nb_avg"),
    ("Spikes/NB std",              "Number of Spikes per Network Burst - Std",               "n_spikes_per_nb_std"),
    ("Mean ISI NB avg (s)",        "Mean ISI within Network Burst - Avg",                    "mean_isi_within_nb_avg"),
    ("Mean ISI NB std (s)",        "Mean ISI within Network Burst - Std",                    "mean_isi_within_nb_std"),
    ("Median ISI NB avg (s)",      "Median ISI within Network Burst - Avg",                  "median_isi_within_nb_avg"),
    ("Median ISI NB std (s)",      "Median ISI within Network Burst - Std",                  "median_isi_within_nb_std"),
    ("Median/Mean ISI NB avg",     "Median/Mean ISI within Network Burst - Avg",             "median_mean_isi_ratio_nb_avg"),
    ("Median/Mean ISI NB std",     "Median/Mean ISI within Network Burst - Std",             "median_mean_isi_ratio_nb_std"),
    ("Elecs/NB avg",               "Number of Elecs Participating in Burst - Avg",           "n_elecs_per_nb_avg"),
    ("Elecs/NB std",               "Number of Elecs Participating in Burst - Std",           "n_elecs_per_nb_std"),
    ("Spikes/NB/ch avg",           "Number of Spikes per Network Burst per Channel - Avg",   "n_spikes_per_nb_per_channel_avg"),
    ("Spikes/NB/ch std",           "Number of Spikes per Network Burst per Channel - Std",   "n_spikes_per_nb_per_channel_std"),
    ("NB % time",                  "Network Burst Percentage",                               "network_burst_pct"),
    ("Network IBI CV",             "Network IBI Coefficient of Variation",                   "network_ibi_cv"),
    ("Network norm. dur. IQR",     "Network Normalized Duration IQR",                        "network_normalized_duration_iqr"),
    # ── Average NB ────────────────────────────────────────────────────────────
    ("NB peak rate (sp/s)",        "Burst Peak (Max Spikes per sec)",                        "nb_burst_peak_spikes_per_s"),
    ("Time to NB peak (ms)",       "Time to Burst Peak",                                     "nb_time_to_peak_ms"),
    ("% bursts w/ lead elec",      "Percent Bursts with Start Electrode",                    "nb_pct_bursts_with_start_electrode"),
]

# Figure groups: (filename_stem, list of (label, nm_fragment, pma_col))
FIGURE_GROUPS = [
    ("fig1_activity", [p for p in METRIC_PAIRS if p[0] in {
        "Mean MFR (Hz)", "Weighted MFR (Hz)", "N active electrodes", "ISI CV",
    }]),
    ("fig2_burst_core", [p for p in METRIC_PAIRS if p[0] in {
        "N bursts (total)", "N bursting electrodes",
        "Burst duration avg (s)", "Spikes/burst avg",
        "Burst freq avg (Hz)", "Burst % avg",
    }]),
    ("fig3_burst_isi_ibi", [p for p in METRIC_PAIRS if p[0] in {
        "Mean ISI burst avg (s)", "Median ISI burst avg (s)",
        "Median/Mean ISI burst avg", "IBI avg (s)", "IBI CV avg",
        "Burst duration std (s)",
    }]),
    ("fig4_nb_core", [p for p in METRIC_PAIRS if p[0] in {
        "N network bursts", "NB frequency (Hz)",
        "NB duration avg (s)", "Spikes/NB avg",
        "NB % time", "Network IBI CV",
    }]),
    ("fig5_nb_isi_participation", [p for p in METRIC_PAIRS if p[0] in {
        "Mean ISI NB avg (s)", "Median ISI NB avg (s)",
        "Median/Mean ISI NB avg", "Elecs/NB avg",
        "Spikes/NB/ch avg", "Network norm. dur. IQR",
    }]),
    ("fig6_avg_nb", [p for p in METRIC_PAIRS if p[0] in {
        "NB peak rate (sp/s)", "Time to NB peak (ms)", "% bursts w/ lead elec",
    }]),
]

# ── NeuralMetric CSV parser ────────────────────────────────────────────────────

def parse_nm_csv(path: pathlib.Path) -> pd.DataFrame:
    """Return wide DataFrame indexed by 'Section / Metric' with well IDs as columns."""
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()

    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Well Averages"))
    well_ids = [c.strip() for c in lines[header_idx].split(",")[1:] if c.strip()]

    records: dict[str, list] = {}
    section = ""
    for ln in lines[header_idx + 1:]:
        # Stop at the per-electrode section
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
        for v in parts[1 : len(well_ids) + 1]:
            v = v.strip()
            values.append(float(v) if v else np.nan)
        while len(values) < len(well_ids):
            values.append(np.nan)
        records[metric] = values

    df = pd.DataFrame(records, index=well_ids).T
    df.index.name = "metric"
    df.columns.name = "well_id"
    return df


def nm_row(df: pd.DataFrame, fragment: str, wells: list[str]) -> pd.Series:
    """Extract one metric row for the requested wells (first substring match)."""
    matches = [m for m in df.index if fragment.lower() in m.lower()]
    if not matches:
        return pd.Series([np.nan] * len(wells), index=wells)
    return df.loc[matches[0], wells].astype(float)


def spk_for_csv(csv_path: pathlib.Path) -> pathlib.Path | None:
    stem = csv_path.stem.replace("_neuralMetrics", "")
    spk = SPK_DIR / f"{stem}.spk"
    return spk if spk.exists() else None


# ── Pipeline import ────────────────────────────────────────────────────────────

from py_mea_axion.pipeline import MEAExperiment  # noqa: E402

# ── Main loop ─────────────────────────────────────────────────────────────────

rows = []  # one row per (recording × well)

for plate_label, (plate_map, plate_wells) in PLATE_INFO.items():
    pattern = f"*{plate_label}*_neuralMetrics.csv"
    csv_files = sorted(NM_DIR.glob(pattern))
    print(f"\n{'='*72}")
    print(f"{plate_label}: {len(csv_files)} NeuralMetric CSV files found.")
    print(f"{'='*72}")

    cond_lookup = plate_map.set_index("well_id")["condition"].to_dict()

    for csv_path in csv_files:
        spk_path = spk_for_csv(csv_path)
        if spk_path is None:
            print(f"  [SKIP] no .spk for {csv_path.name}")
            continue

        m = re.search(r"_D(\d+)N", csv_path.stem)
        div = int(m.group(1)) if m else -1
        print(f"  {csv_path.stem[:50]}  DIV={div} … ", end="", flush=True)

        try:
            nm_df = parse_nm_csv(csv_path)
        except Exception as e:
            print(f"NM parse error: {e}")
            continue

        # Pre-extract all NM metric series
        nm_vals: dict[str, pd.Series] = {}
        for label, nm_frag, _ in METRIC_PAIRS:
            nm_vals[label] = nm_row(nm_df, nm_frag, plate_wells)

        # Run py-mea-axion
        meta = plate_map.copy()
        meta["DIV"] = div
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exp = MEAExperiment(
                    spk_path,
                    metadata=meta,
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

        ws = exp.well_summary.set_index("well_id")
        nm_n_active = nm_vals["N active electrodes"]

        for well in plate_wells:
            # Skip wells inactive in NM (no reference data)
            nm_active = nm_n_active.get(well, np.nan)
            if pd.isna(nm_active) or nm_active == 0:
                continue

            row: dict = {
                "recording": csv_path.stem,
                "plate":     plate_label,
                "div":       div,
                "well":      well,
                "condition": cond_lookup.get(well, ""),
            }

            for label, _, pma_col in METRIC_PAIRS:
                row[f"nm_{label}"]  = nm_vals[label].get(well, np.nan)
                row[f"pma_{label}"] = ws.loc[well, pma_col] if well in ws.index else np.nan

            rows.append(row)

        print("done")

df = pd.DataFrame(rows)
print(f"\nCollected {len(df)} active well-observations across "
      f"{df['recording'].nunique()} recordings.\n")

df.to_csv(BENCH_DIR / "all_benchmark_raw.csv", index=False, float_format="%.6f")
print("Raw table saved → all_benchmark_raw.csv")

# ── Summary statistics ─────────────────────────────────────────────────────────

summary_rows = []
for label, _, _ in METRIC_PAIRS:
    nm_col  = f"nm_{label}"
    pma_col = f"pma_{label}"
    if nm_col not in df.columns:
        continue
    sub = df[[nm_col, pma_col]].dropna()
    x, y = sub[nm_col].values.astype(float), sub[pma_col].values.astype(float)
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
        pearson_r=round(r, 4) if not np.isnan(r) else np.nan,
        p_value=round(p, 4) if not np.isnan(p) else np.nan,
        bias=round(bias, 4) if not np.isnan(bias) else np.nan,
        mean_abs_error=round(mae, 4) if not np.isnan(mae) else np.nan,
        mean_pct_error=round(mpct, 2) if not np.isnan(mpct) else np.nan,
    ))

summary = pd.DataFrame(summary_rows)
summary.to_csv(BENCH_DIR / "all_benchmark_summary.csv", index=False)
print("Summary saved → all_benchmark_summary.csv\n")

# Print to console
print(f"{'Metric':<32} {'n':>5} {'r':>7} {'bias':>10} {'MAE':>10} {'%err':>7}")
print("-" * 72)
for _, row in summary.iterrows():
    r_str    = f"{row['pearson_r']:7.4f}"    if pd.notna(row['pearson_r'])    else "    n/a"
    bias_str = f"{row['bias']:+10.4f}"       if pd.notna(row['bias'])         else "       n/a"
    mae_str  = f"{row['mean_abs_error']:10.4f}" if pd.notna(row['mean_abs_error']) else "       n/a"
    pct_str  = f"{row['mean_pct_error']:6.1f}%" if pd.notna(row['mean_pct_error']) else "   n/a"
    print(f"{row['metric']:<32} {int(row['n']):>5} {r_str} {bias_str} {mae_str} {pct_str}")

# ── Scatter plots ──────────────────────────────────────────────────────────────

COND_COLOR = {
    "SCRM":     "#4477AA",
    "LGI2_KD4": "#EE6677",
    "LGI2_KD5": "#CCBB44",
}


def _scatter_panel(ax, x, y, conds, label):
    """Draw one scatter panel on *ax*."""
    fin = np.isfinite(x) & np.isfinite(y)
    x, y, conds = x[fin], y[fin], conds[fin]
    if len(x) == 0:
        ax.set_visible(False)
        return

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = (hi - lo) * 0.05 or 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, zorder=0)

    for cond, col in COND_COLOR.items():
        idx = conds == cond
        ax.scatter(x[idx], y[idx], color=col, s=10, alpha=0.65,
                   zorder=3, edgecolors="none", label=cond)

    if len(x) >= 2 and x.std() > 0 and y.std() > 0:
        r, _ = pearsonr(x, y)
        ax.text(0.05, 0.95, f"r={r:.3f}  n={len(x)}",
                transform=ax.transAxes, fontsize=6, va="top")

    ax.set_title(label, fontsize=7, pad=2)
    ax.tick_params(labelsize=6)


print("\nGenerating figures …")
for fname_stem, pairs in FIGURE_GROUPS:
    n_panels = len(pairs)
    if n_panels == 0:
        continue
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)

    # One shared legend handle per condition
    legend_handles = []
    for cond, col in COND_COLOR.items():
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=col, markersize=5, label=cond)
        )

    for idx, (label, _, _) in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        nm_col  = f"nm_{label}"
        pma_col = f"pma_{label}"
        if nm_col not in df.columns:
            ax.set_visible(False)
            continue
        sub = df[[nm_col, pma_col, "condition"]].copy()
        _scatter_panel(
            ax,
            sub[nm_col].values.astype(float),
            sub[pma_col].values.astype(float),
            sub["condition"].values,
            label,
        )
        ax.set_xlabel("NeuralMetric Tools", fontsize=6)
        ax.set_ylabel("py-mea-axion", fontsize=6)

    # Hide unused panels
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.legend(handles=legend_handles, loc="lower right",
               fontsize=7, framealpha=0.7, ncol=1,
               bbox_to_anchor=(0.98, 0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = OUT_DIR / f"{fname_stem}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  saved {out_path.name}")

print(f"\nDone. Results in benchmarking/figures_all/ and benchmarking/all_benchmark_summary.csv")
