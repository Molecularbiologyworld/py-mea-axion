# py-mea-axion

[![PyPI version](https://img.shields.io/pypi/v/py-mea-axion.svg)](https://pypi.org/project/py-mea-axion/)
[![Python](https://img.shields.io/pypi/pyversions/py-mea-axion.svg)](https://pypi.org/project/py-mea-axion/)
[![GitHub](https://img.shields.io/badge/GitHub-py--mea--axion-blue?logo=github)](https://github.com/Molecularbiologyworld/py-mea-axion)

End-to-end analysis of multi-electrode array (MEA) recordings from **Axion Biosystems** instruments, in Python.

`py-mea-axion` reads raw `.spk` binary files and carries the analysis through spike metrics, burst detection, network-burst detection, and synchrony measurement, all the way to statistical comparisons and publication-ready figures, **driven entirely from the command line**.

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

To install the latest commit on `master` directly from GitHub:

```bash
pip install git+https://github.com/Molecularbiologyworld/py-mea-axion.git
```

**Requirements:** Python ≥ 3.10, numpy, scipy, pandas, matplotlib, pingouin, statsmodels.

---

## Workflow at a glance

The CLI is organised around four phases:

1. `mea-axion build` processes a folder of `.spk` files into a single labelled master CSV.
2. `mea-axion plot <subcommand>` generates figures from the master CSV.
3. `mea-axion stats` writes a long-format CSV of pairwise condition-comparison results, using the same statistical dispatcher that drives the violin brackets.
4. `mea-axion show-layout` prints the well-to-condition map encoded in any master CSV.

There are also `mea-axion run` (single recording, CSVs and figures) and `mea-axion summary` (single recording, stdout table) for one-off analyses.

---

## Filename convention

`build` requires every `.spk` file to carry **plate** and **DIV** in its name:

```
<anything>_Plate<N>_DIV<N>.spk
```

Detection regex (case-insensitive): `Plate(\d+)_DIV(\d+)`. Anything before or after that block is ignored. Files in the input folder that do not match are skipped with a warning.

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

Each `--well` entry follows the form `<plate>_<well>=<condition>[,B<bio>,T<tech>,<batch>]`.

- `B<n>` (biological replicate) and `T<n>` (technical replicate) are optional.
- The 4th comma-separated value is the batch label (free-form, e.g. `Batch1`, `Plate2`).
- Bio rep auto-derives from `(plate, well)` if not specified.

**Output**: `master.csv` (one row per well per DIV, with metric columns plus condition / bio_rep / tech_rep / batch columns) and a sibling `master.build.sh` containing the command for re-runs.

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
| `trajectory` | master CSV | Mean ± SEM error-bars per condition across DIVs. One PNG per metric. |
| `timepoint` | master CSV | Violin and jitter per condition at one DIV, with user-selectable significance brackets. One PNG per (metric, DIV). |
| `pca` | master CSV | PCA scatter coloured by condition. One pooled PNG, or one per `--div`. Optional loadings figure. |
| `raster` | `.spk` file | Burst raster and ASDR histogram for each well. |

### Common plot flags

| Flag | Effect |
|---|---|
| `--metric COL [COL ...]` | One or more metric columns. Omit to plot all numeric columns. |
| `--div N [N ...]` | Specific time points (timepoint, pca, heatmap). |
| `--div-min N --div-max N` | Filter the time range (trajectory, pca). |
| `--filter COL=VAL[,VAL...]` | Restrict the input to specific column values. Repeatable; multiple flags AND together. Example: `--filter condition=SCRM --filter DIV=14,28`. |
| `--group-order GROUP [GROUP ...]` | Explicit condition order (default: alphabetical). Listing a subset excludes other groups from the plot. |
| `--color GROUP=#HEX` | Colour per condition. Repeatable. Hex codes or matplotlib named colours both work. |
| `--pool` | Flatten bio/tech replicate hierarchy and treat every well as independent. Default is hierarchical (collapse tech reps within bio rep before stats). |
| `--figsize W H` | Figure size in inches. Default depends on the plot type. |
| `--out DIR` | Output directory (auto-created). |

### Flags specific to `plot timepoint` (violins)

| Flag | Effect |
|---|---|
| `--test {tukey,mannwhitney,kruskal}` | Statistical test for the significance brackets. Default `tukey` (parametric all-pairs Tukey HSD). `mannwhitney` runs pairwise Mann-Whitney U with Bonferroni correction. `kruskal` runs Kruskal-Wallis omnibus followed by Dunn's pairwise post-hoc. The choice is yours; the package does not auto-select. |
| `--compare GROUP_A GROUP_B` | Restrict significance brackets to specific pairs. Repeatable. |
| `--point-hue COL` | Colour individual jitter points by another column (e.g. `batch`). |
| `--point-color VALUE=#HEX` | Per-value colour palette for `--point-hue`. Repeatable. |
| `--point-shape COL` | Use marker shape (in addition to or instead of colour) for individual jitter points. Default markers cycle circle, square, triangle, diamond, etc. |
| `--point-shape-map VALUE=MARKER` | Per-value marker for `--point-shape`. Repeatable. Markers can be matplotlib codes (`o`, `s`, `^`, ...) or aliases (`circle`, `square`, `triangle`, ...). |
| `--point-size N` | Marker area for jitter points. Default 40. |
| `--show-point-legend` | Show a legend mapping point colours to `--point-hue` values. Default hidden. |

### Flags specific to `plot raster`

| Flag | Effect |
|---|---|
| `--asdr-bin S` | ASDR histogram bin width in seconds. Default 0.2. |
| `--time-start S --time-end S` | Restrict to a time window in seconds within the recording. Displayed times keep the original timestamps. |
| `--density-color` | Colour each spike tick by local spike density (binned at `--asdr-bin`). Adds a colour bar underneath the raster. |
| `--density-cmap NAME` | Matplotlib colour map for `--density-color`. Default `viridis`. |
| `--max-isi S --min-spikes N` | Burst-detection thresholds for the orange overlay rectangles. |

### Flags specific to `plot pca`

| Flag | Effect |
|---|---|
| `--loadings` | Also save a per-feature PC loadings CSV and a horizontal bar chart of the top contributors to PC1 and PC2. |
| `--top N` | Number of top features to show in the loadings figure. Default 10. |
| `--point-shape COL --point-shape-map VALUE=MARKER` | Same as for `timepoint`, useful for indicating biological replicates by marker shape. |

Run `mea-axion plot <subcommand> --help` for the full per-subcommand option list.

### Example: a complete violin command

```bash
mea-axion plot timepoint master.csv \
  --metric mean_mfr_active_hz burst_freq_avg network_burst_freq mean_sttc \
  --div 21 26 33 \
  --group-order SCRM LGI2_KD4 LGI2_KD5 \
  --color SCRM=green --color LGI2_KD4=lightcoral --color LGI2_KD5=red \
  --point-shape batch --point-shape-map Batch1=circle --point-shape-map Batch2=square --point-shape-map Batch3=triangle \
  --compare LGI2_KD4 SCRM --compare LGI2_KD5 SCRM \
  --test tukey \
  --pool \
  --out figs/timepoint
```

### Pretty axis labels

Axis labels and colour-bar titles render human-readable names (`Mean firing rate (Hz)` rather than `mean_mfr_active_hz`). The CLI flag values still use raw column names, which are easier to type.

---

## Statistical analyses (`mea-axion stats`)

Beyond the brackets shown on violin plots, `mea-axion stats` writes a long-format CSV of pairwise condition-comparison results. Same data, same dispatcher as the violin brackets, just persisted as a tidy table for supplementary materials.

```bash
mea-axion stats master.csv \
  --metric mean_mfr_active_hz burst_freq_avg network_burst_freq mean_sttc \
  --div 28 41 \
  --compare LGI2_KD4 SCRM --compare LGI2_KD5 SCRM \
  --test tukey \
  --pool \
  --out stats_summary.csv
```

The output CSV has one row per (metric, time-point, pair) combination, with columns `metric`, `DIV`, `test`, `group_a`, `group_b`, `mean_diff`, `p_adj`, `significance`, `n_a`, `n_b`. Sort by metric and DIV in Excel for a clean supplementary table.

The same `--test` choice (`tukey`, `mannwhitney`, `kruskal`) applies. Significance labels (`*`, `**`, `***`, `ns`) match those used in the figures.

---

## Single-recording commands

For ad-hoc inspection of one `.spk` file without going through `build`:

```bash
# Quick per-well summary printed to the terminal (no files written)
mea-axion summary recording.spk --fs-override 12500

# Full pipeline: writes CSVs and figures to results/
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

The `py_mea_axion.stats` module exposes the same statistical functions used by the CLI:

```python
from py_mea_axion.stats import (
    pairwise_test,            # unified dispatcher: tukey / mannwhitney / kruskal
    tukey_hsd_pairwise,       # Tukey HSD only
    compare_conditions,       # auto-dispatch MW (2 groups) or KW (3+ groups)
    longitudinal_model,       # linear mixed-effects with time x group interaction
    compute_icc,              # intraclass correlation
)
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

649 tests covering every module: binary parser, burst detection, network-burst detection, synchrony, statistical tests, every viz function, and every CLI subcommand.

---

## Citing

If you use `py-mea-axion` in your research, please cite:

> [manuscript in preparation]

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19350375.svg)](https://doi.org/10.5281/zenodo.19350375)

---

## License

MIT, see [LICENSE](LICENSE).
