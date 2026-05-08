# py-mea-axion

[![PyPI version](https://img.shields.io/pypi/v/py-mea-axion.svg)](https://pypi.org/project/py-mea-axion/)
[![Python](https://img.shields.io/pypi/pyversions/py-mea-axion.svg)](https://pypi.org/project/py-mea-axion/)
[![GitHub](https://img.shields.io/badge/GitHub-py--mea--axion-blue?logo=github)](https://github.com/Molecularbiologyworld/py-mea-axion)

End-to-end analysis of multi-electrode array (MEA) recordings from
**Axion Biosystems** instruments, in Python.

`py-mea-axion` reads raw `.spk` binary files and carries the analysis through spike metrics, burst detection, network-burst detection, and synchrony measurement — all the way to statistical comparisons and publication-ready figures, **driven entirely from the command line**.

---

## Installation

```bash
conda create -n mea python=3.11
conda activate mea
pip install py-mea-axion
```

For development against a local clone:

```bash
git clone https://github.com/Molecularbiologyworld/py-mea-axion.git
cd py-mea-axion
pip install -e .
```

**Requirements:** Python ≥ 3.10, numpy, scipy, pandas, matplotlib, pingouin, statsmodels.

---

## Workflow at a glance

The CLI is organised around three phases:

1. **`mea-axion build`** — process a folder of `.spk` files into a single labelled master CSV.
2. **`mea-axion plot <subcommand>`** — generate figures from the master CSV.
3. **`mea-axion show-layout`** — print the well-to-condition map saved in any master CSV.

There are also `mea-axion run` (single recording → CSVs + figures) and `mea-axion summary` (single recording → stdout table) for one-off analyses.

---

## Filename convention

`build` requires every `.spk` file to carry **plate** and **DIV** in its name:

```
<anything>_Plate<N>_DIV<N>.spk
```

Detection regex (case-insensitive): `Plate(\d+)_DIV(\d+)`. Anything before/after that block is ignored. Files in the input folder that don't match are skipped with a warning.

| Filename | Plate | DIV |
|---|---|---|
| `Plate1_DIV14.spk` | 1 | 14 |
| `20251002_Plate2_DIV26.spk` | 2 | 26 |
| `LGI2KD_Plate3_DIV7_recording2.spk` | 3 | 7 |

---

## Build a master CSV

You specify the well-to-condition layout once, on the command line:

```bash
mea-axion build "data/" --fs-override 12500 \
  --well 1_A1=SCRM,B1,T1,Batch1 \
  --well 1_A2=KD,B1,T1,Batch1 \
  --well 1_B1=SCRM,B1,T2,Batch1 \
  --well 1_B2=KD,B1,T2,Batch1 \
  ... \
  --out master.csv
```

Each `--well` entry: `<plate>_<well>=<condition>[,B<bio>,T<tech>,<batch>]`.

- `B<n>` (biological replicate) and `T<n>` (technical replicate) are optional.
- The 4th comma-separated value is the batch label (free-form, e.g. `Batch1`, `Plate2`).
- Bio rep auto-derives from `(plate, well)` if not specified.

**Output**: `master.csv` (one row per well per DIV, with metric columns + condition / bio_rep / tech_rep / batch columns) and a sibling `master.build.sh` containing the command for re-runs.

### Re-using a layout

Three ways to avoid retyping the layout for a follow-up build:

```bash
# Option 1: re-run the auto-saved script.
bash master.build.sh

# Option 2: pull the layout from an existing master CSV.
mea-axion build "data_v2/" --fs-override 12500 --layout-from master.csv --out master_v2.csv

# Option 3: print the equivalent --well flags to stdout.
mea-axion show-layout master.csv
```

---

## Plot subcommands

```bash
mea-axion plot heatmap     master.csv --metric mean_mfr_active_hz --div 21 --out figs/
mea-axion plot trajectory  master.csv --metric mean_mfr_active_hz --out figs/
mea-axion plot timepoint   master.csv --metric mean_mfr_active_hz --div 21 26 33 --out figs/
mea-axion plot pca         master.csv --out figs/
mea-axion plot raster      "recording.spk" --fs-override 12500 --wells A1 --out figs/
```

| Subcommand | Input | What it produces |
|---|---|---|
| `heatmap` | master CSV | 4×6 plate grid, one panel per plate, coloured by metric. One PNG per (metric, DIV). |
| `trajectory` | master CSV | Mean ± SEM error-bar per condition vs DIV. One PNG per metric. |
| `timepoint` | master CSV | Violin + jitter per condition at one DIV, with Tukey HSD significance brackets. One PNG per (metric, DIV). |
| `pca` | master CSV | PCA scatter coloured by condition. One pooled PNG, or one per `--div`. |
| `raster` | `.spk` file | Burst raster + ASDR histogram for each well. |

### Common plot flags

| Flag | Effect |
|---|---|
| `--metric COL [COL ...]` | One or more metric columns. Omit to plot all numeric columns. |
| `--div N [N ...]` | Specific time points (timepoint, pca, heatmap). |
| `--div-min N --div-max N` | Filter the time range (trajectory). |
| `--group-order GROUP [GROUP ...]` | Explicit condition order (default: alphabetical). |
| `--color GROUP=#HEX` | Colour per condition. Repeatable. |
| `--pool` | Flatten bio/tech replicate hierarchy; treat every well as independent. Default is hierarchical (collapse tech reps within bio rep before stats). |
| `--compare GROUP_A GROUP_B` | Restrict Tukey HSD brackets to specific pairs (timepoint). Repeatable. |
| `--point-hue COL` | Colour individual jitter points by another column (e.g. `batch`). |
| `--point-color VALUE=#HEX` | Per-value colour palette for `--point-hue`. |
| `--point-size N` | Marker area for jitter points (default 40). |
| `--out DIR` | Output directory (auto-created). |

Run `mea-axion plot <subcommand> --help` for the full list.

### Example: a complete violin command

```bash
mea-axion plot timepoint master.csv \
  --metric mean_mfr_active_hz burst_freq_avg network_burst_freq mean_sttc \
  --div 21 26 33 \
  --group-order SCRM LGI2_KD4 LGI2_KD5 \
  --color SCRM=green --color LGI2_KD4=lightcoral --color LGI2_KD5=red \
  --point-hue batch --point-color Batch1=#222222 --point-color Batch2=#888888 \
  --compare LGI2_KD4 SCRM --compare LGI2_KD5 SCRM \
  --pool \
  --out figs/timepoint
```

### Pretty axis labels

Axis labels and colour-bar titles render human-readable names (`Mean firing rate (Hz)` rather than `mean_mfr_active_hz`). The CLI flag values still use raw column names — easier to type.

---

## Single-recording commands

For ad-hoc inspection of one `.spk` file without going through `build`:

```bash
# Quick per-well summary printed to the terminal
mea-axion summary recording.spk --fs-override 12500

# Full pipeline — writes CSVs + figures to results/
mea-axion run recording.spk --out results/ --fs-override 12500

# Restrict to specific wells
mea-axion summary recording.spk --wells A1 B1 C1 --fs-override 12500

# Analyse only a time window
mea-axion run recording.spk --time-start 300 --time-end 600 --out results/ --fs-override 12500
```

---

## Python API

For programmatic use:

```python
from py_mea_axion import MEAExperiment

exp = MEAExperiment(
    "recording.spk",
    metadata="plate_map.csv",   # optional CSV with well_id, condition, DIV, replicate_id
    fs_override=12500,
).run()

exp.spike_metrics      # per-electrode metrics
exp.burst_table        # one row per detected burst
exp.well_summary       # per-well aggregate (27 metrics)
exp.to_csv("results.csv")

exp.plot_heatmap("A1")
exp.plot_raster("A1")
exp.plot_trajectory("mean_mfr_active_hz")
```

---

## Note on Axion `.spk` files

Some recordings lack a `BlockVectorHeader` (firmware-dependent). When that happens, sampling-rate detection may guess wrong; pass `--fs-override` explicitly:

```bash
mea-axion run recording.spk --fs-override 12500
```

12500 Hz is the standard rate for Axion CytoView 24-well plates.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

~575 tests covering all modules.

---

## Citing

If you use `py-mea-axion` in your research, please cite:

> [manuscript in preparation]

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19350375.svg)](https://doi.org/10.5281/zenodo.19350375)

---

## License

MIT — see [LICENSE](LICENSE).
