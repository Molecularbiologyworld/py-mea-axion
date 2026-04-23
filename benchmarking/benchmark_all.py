"""
benchmark_all.py
================
Compare py-mea-axion output against NeuralMetric Tools exports across
all recordings (both plates, all DIVs).

Covers all metric categories retained in the pipeline:
  - Activity          : MFR, ISI CV, active electrode count
  - Electrode burst   : duration, spike count, ISI (mean/median/ratio),
                        IBI, frequency, burst %, IBI CV
  - Network burst     : count, frequency, duration, spikes, ISI, participation,
                        burst %, IBI CV

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

# ── Plate maps ─────────────────────────────────────────────────────────────────

# Plate 1 — 24 wells (6 per row x 4 rows).
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

# Plate 2 — 12 wells (3 per row x 4 rows).
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
    ("Mean MFR (Hz)",          "Mean Firing Rate (Hz)",               "mean_mfr_active_hz"),
    ("N active electrodes",    "Number of Active Electrodes",         "n_active"),
    ("ISI CV",                 "ISI Coefficient of Variation - Avg",  "isi_cv_avg"),
    # ── Network burst ─────────────────────────────────────────────────────────
    ("N network bursts",       "Number of Network Bursts",                               "n_network_bursts"),
    ("NB frequency (Hz)",      "Network Burst Frequency",                                "network_burst_freq"),
    ("NB duration avg (s)",    "Network Burst Duration - Avg",                           "network_burst_duration_avg"),
    ("Spikes/NB avg",          "Number of Spikes per Network Burst - Avg",               "n_spikes_per_nb_avg"),
    ("Mean ISI NB avg (s)",    "Mean ISI within Network Burst - Avg",                    "mean_isi_within_nb_avg"),
    ("Median ISI NB avg (s)",  "Median ISI within Network Burst - Avg",                  "median_isi_within_nb_avg"),
    ("Median/Mean ISI NB avg", "Median/Mean ISI within Network Burst - Avg",             "median_mean_isi_ratio_nb_avg"),
    ("Elecs/NB avg",           "Number of Elecs Participating in Burst - Avg",           "n_elecs_per_nb_avg"),
    ("Spikes/NB/ch avg",       "Number of Spikes per Network Burst per Channel - Avg",   "n_spikes_per_nb_per_channel_avg"),
    ("NB %",                   "Network Burst Percentage",                               "network_burst_pct"),
    ("Network IBI CV",         "Network IBI Coefficient of Variation",                   "network_ibi_cv"),
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

rows = []  # one row per (recording x well)

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
        print(f"  {csv_path.stem[:50]}  DIV={div} ... ", end="", flush=True)

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
print("Raw table saved -> all_benchmark_raw.csv")

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
print("Summary saved -> all_benchmark_summary.csv\n")

# Print to console
print(f"{'Metric':<32} {'n':>5} {'r':>7} {'bias':>10} {'MAE':>10} {'%err':>7}")
print("-" * 72)
for _, row in summary.iterrows():
    r_str    = f"{row['pearson_r']:7.4f}"       if pd.notna(row['pearson_r'])       else "    n/a"
    bias_str = f"{row['bias']:+10.4f}"           if pd.notna(row['bias'])            else "       n/a"
    mae_str  = f"{row['mean_abs_error']:10.4f}"  if pd.notna(row['mean_abs_error'])  else "       n/a"
    pct_str  = f"{row['mean_pct_error']:6.1f}%"  if pd.notna(row['mean_pct_error'])  else "   n/a"
    print(f"{row['metric']:<32} {int(row['n']):>5} {r_str} {bias_str} {mae_str} {pct_str}")

# ── Individual scatter plots (one PNG per metric) ──────────────────────────────

COND_COLOR = {
    "SCRM":     "#4477AA",
    "LGI2_KD4": "#EE6677",
    "LGI2_KD5": "#CCBB44",
}

# Legend handles (shared across all figures)
_legend_handles = [
    plt.Line2D([0], [0], marker="o", color="none",
               markerfacecolor=col, markersize=6, label=cond)
    for cond, col in COND_COLOR.items()
]

OUT_DIR.mkdir(parents=True, exist_ok=True)
print("\nGenerating individual figures ...")
for label, _, _ in METRIC_PAIRS:
    nm_col  = f"nm_{label}"
    pma_col = f"pma_{label}"
    if nm_col not in df.columns:
        continue

    sub = df[[nm_col, pma_col, "condition"]].copy()
    x = sub[nm_col].values.astype(float)
    y = sub[pma_col].values.astype(float)
    conds = sub["condition"].values

    fin = np.isfinite(x) & np.isfinite(y)
    x_f, y_f, conds_f = x[fin], y[fin], conds[fin]

    fig, ax = plt.subplots(figsize=(4, 4))

    if len(x_f) > 0:
        lo = min(x_f.min(), y_f.min())
        hi = max(x_f.max(), y_f.max())
        pad = (hi - lo) * 0.05 or 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, zorder=0)

        for cond, col in COND_COLOR.items():
            idx = conds_f == cond
            ax.scatter(x_f[idx], y_f[idx], color=col, s=12, alpha=0.7,
                       zorder=3, edgecolors="none", label=cond)

        if len(x_f) >= 2 and x_f.std() > 0 and y_f.std() > 0:
            r, _ = pearsonr(x_f, y_f)
            ax.text(0.05, 0.95, f"r={r:.3f}  n={len(x_f)}",
                    transform=ax.transAxes, fontsize=8, va="top")
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray")

    ax.set_title(label, fontsize=9)
    ax.set_xlabel("NeuralMetric Tools", fontsize=8)
    ax.set_ylabel("py-mea-axion", fontsize=8)
    ax.tick_params(labelsize=7)

    ax.legend(handles=_legend_handles, loc="lower right",
              fontsize=7, framealpha=0.7, ncol=1)

    fig.tight_layout()
    # Sanitise label for use as filename
    safe_name = re.sub(r"[^\w\-]", "_", label).strip("_")
    out_path = OUT_DIR / f"{safe_name}.png"
    fig.savefig(out_path, dpi=1200)
    plt.close(fig)
    print(f"  saved {out_path.name}")

print(f"\nDone. {len(METRIC_PAIRS)} figures in benchmarking/figures_all/")

# ── Plate heatmap figures (one per metric per DIV) ────────────────────────────
#
# One figure per (metric, DIV).  Each figure has two subplots side-by-side:
#   left  = Plate 1 (4 × 6 well grid)
#   right = Plate 2 (4 × 3 well grid)
# Cell colour = py-mea-axion metric value.  Wells with no data (excluded /
# inactive) are shown in grey.  Condition abbreviation is overlaid in each
# cell.  A patch legend for conditions sits below the axes.

HEATMAP_DIR = BENCH_DIR / "figures_plate_heatmap"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

_PLATE_ROW_LABELS = list("ABCD")
_PLATE_COLS = list(range(1, 7))   # both plates shown as full 4×6 grid

_all_divs = sorted(df["div"].unique())

print("\nGenerating plate heatmap figures ...")
for label, _, _ in METRIC_PAIRS:
    pma_col = f"pma_{label}"
    if pma_col not in df.columns:
        continue

    _safe = re.sub(r"[^\w\-]", "_", label).strip("_")
    _metric_dir = HEATMAP_DIR / _safe
    _metric_dir.mkdir(parents=True, exist_ok=True)

    # Global colour scale — consistent across all DIVs for this metric.
    _vals = df[pma_col].dropna().values.astype(float)
    _vals = _vals[np.isfinite(_vals)]
    if len(_vals) == 0:
        continue
    _vmin, _vmax = float(_vals.min()), float(_vals.max())
    if _vmin == _vmax:
        _vmin -= 0.5; _vmax += 0.5

    _norm = plt.Normalize(vmin=_vmin, vmax=_vmax)
    _cmap = plt.cm.viridis.copy()
    _cmap.set_bad(color="#cccccc")   # grey for empty / no-data wells

    _n_r = len(_PLATE_ROW_LABELS)
    _n_c = len(_PLATE_COLS)

    for div in _all_divs:
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(13, 3.6),
            gridspec_kw={"wspace": 0.35},
        )

        for ax, plate in [(ax1, "Plate 1"), (ax2, "Plate 2")]:
            _pdf = df[(df["plate"] == plate) & (df["div"] == div)]

            # Full 4×6 grid — wells absent from df stay NaN (shown grey).
            _grid = np.full((_n_r, _n_c), np.nan)
            for _, _wr in _pdf.iterrows():
                _w = _wr["well"]
                try:
                    _r = _PLATE_ROW_LABELS.index(_w[0])
                    _c = _PLATE_COLS.index(int(_w[1:]))
                    _grid[_r, _c] = _wr[pma_col]
                except (ValueError, IndexError):
                    pass

            ax.imshow(_grid, norm=_norm, cmap=_cmap, aspect="auto")
            ax.set_xticks(range(_n_c))
            ax.set_xticklabels(_PLATE_COLS, fontsize=8)
            ax.set_yticks(range(_n_r))
            ax.set_yticklabels(_PLATE_ROW_LABELS, fontsize=8)
            ax.tick_params(length=2, pad=2)
            ax.set_title(plate, fontsize=9, pad=4)

        # Colorbar.
        _sm = plt.cm.ScalarMappable(norm=_norm, cmap=_cmap)
        _sm.set_array([])
        _cb = fig.colorbar(_sm, ax=[ax1, ax2], fraction=0.025, pad=0.04)
        _cb.set_label(label, fontsize=8)
        _cb.ax.tick_params(labelsize=7)

        fig.suptitle(f"{label} — DIV {div}", fontsize=10, y=1.02)
        fig.tight_layout()

        _out = _metric_dir / f"DIV_{div:02d}.png"
        fig.savefig(_out, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"  {label}: {len(_all_divs)} figures -> {_metric_dir.name}/")

print(f"\nDone. Plate heatmap figures in benchmarking/figures_plate_heatmap/")
