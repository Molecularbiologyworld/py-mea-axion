# py-mea-axion

[![PyPI version](https://img.shields.io/pypi/v/py-mea-axion.svg)](https://pypi.org/project/py-mea-axion/)
[![Python](https://img.shields.io/pypi/pyversions/py-mea-axion.svg)](https://pypi.org/project/py-mea-axion/)
[![GitHub](https://img.shields.io/badge/GitHub-py--mea--axion-blue?logo=github)](https://github.com/Molecularbiologyworld/py-mea-axion)

End-to-end analysis of multi-electrode array (MEA) recordings from
**Axion Biosystems** instruments, in Python.

`py-mea-axion` reads raw `.spk` binary files and carries the analysis
through spike metrics, burst detection, network burst detection, and
synchrony measurement — all the way to statistical comparisons and
publication-ready figures.

---

## Installation

### Recommended: new conda environment

```bash
conda create -n mea python=3.11
conda activate mea
pip install py-mea-axion
```

### Or into an existing environment

```bash
pip install py-mea-axion
```

### For development (editable install from a local clone)

```bash
git clone https://github.com/Molecularbiologyworld/py-mea-axion.git
cd py-mea-axion
pip install -e .
```

**Requirements:** Python ≥ 3.10, numpy, scipy, pandas, matplotlib, pingouin, statsmodels.

---

## Quickstart

### Command-line interface

The fastest way to get results without writing any Python:

```bash
# Quick per-well summary printed to the terminal
mea-axion summary recording.spk --fs-override 12500

# Full pipeline — writes CSVs + figures to results/
mea-axion run recording.spk --out results/ --fs-override 12500

# With a plate map (adds condition labels to outputs)
mea-axion run recording.spk --metadata plate_map.csv --out results/ --fs-override 12500

# Only CSVs, skip figure generation
mea-axion run recording.spk --out results/ --no-figures --fs-override 12500

# Restrict to specific wells
mea-axion summary recording.spk --wells A1 B1 C1 --fs-override 12500

# Analyse only a time window (300-600 s)
mea-axion run recording.spk --time-start 300 --time-end 600 --out results/ --fs-override 12500
```

Run `mea-axion --help` or `mea-axion run --help` for the full option list.

---

### Python API

```python
from py_mea_axion import MEAExperiment

exp = MEAExperiment(
    "recording.spk",
    metadata="plate_map.csv",   # optional: well_id, condition, DIV, replicate_id
    fs_override=12500,
    time_start_s=300.0,         # optional: crop to a time window before analysis
    time_end_s=600.0,
    active_threshold_hz=5/60,   # 5 spikes/min, matching NeuralMetric Tools default
).run()

# Tabular results
exp.spike_metrics      # per-electrode: MFR, ISI stats, active flag
exp.burst_table        # per-burst: start/end time, duration, spike count
exp.well_summary       # per-well: n_active, mean MFR, burst rate, STTC
exp.excluded_wells     # wells dropped for having too few active electrodes

# One-liner export
exp.to_csv("results.csv")        # well_summary merged with metadata

# Statistics
res  = exp.compare("mean_mfr_active_hz")        # Mann-Whitney / Kruskal-Wallis
coef = exp.longitudinal("mean_mfr_active_hz")   # mixed-effects model

# Figures
exp.plot_heatmap("A1")                       # electrode MFR heatmap
exp.plot_raster("A1")                        # spike raster + burst overlays
exp.plot_sttc("A1")                          # pairwise STTC matrix
exp.plot_network_timeline("A1")              # network burst Gantt chart
exp.plot_trajectory("mean_mfr_active_hz")    # longitudinal mean ± SEM
```

### Plate-map CSV format

```
well_id,condition,DIV,replicate_id
A1,WT,14,rep1
A2,KD,14,rep1
B1,WT,14,rep2
B2,KD,14,rep2
```

---

## Key parameters

### Burst detection (`burst_kwargs`)

| Parameter | Default | Description |
|---|---|---|
| `algorithm` | `isi_threshold` | Detection algorithm |
| `max_isi_s` | 0.1 s | Maximum within-burst ISI |
| `min_spikes` | 5 | Minimum spikes per burst |
| `min_ibi_s` | 0.0 s | Minimum inter-burst interval (0 = no merging) |

### Network burst detection (`network_kwargs`)

| Parameter | Default | Description |
|---|---|---|
| `algorithm` | `combined_isi` | Merges all electrode spike trains, matches NeuralMetric Tools |
| `max_isi_s` | 0.1 s | Maximum within-network-burst ISI |
| `min_spikes` | 50 | Minimum spikes in combined train |
| `participation_threshold` | 0.35 | Minimum fraction of electrodes that must fire |

Pass these via `burst_kwargs` / `network_kwargs` in `MEAExperiment`.

### Analysis window

Use `time_start_s` / `time_end_s` in the Python API or
`--time-start` / `--time-end` in the CLI to analyse only a sub-window of
the recording. Spikes outside the window are excluded before metrics,
burst detection, network detection, and STTC are computed. Internally,
the retained timestamps are shifted so that the analysis window starts at
`t=0`.

---

## Note on Axion .spk files

Some recordings lack a `BlockVectorHeader` (common with certain firmware versions).
In this case the sampling frequency is inferred automatically; a warning is printed.
If the inferred frequency is wrong, override it explicitly:

```python
exp = MEAExperiment("recording.spk", fs_override=12500)
```

```bash
mea-axion run recording.spk --fs-override 12500
```

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

397 tests covering all modules.

---

## Citing

If you use `py-mea-axion` in your research, please cite:

> [manuscript in preparation]

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19350375.svg)](https://doi.org/10.5281/zenodo.19350375)

---

## License

MIT — see [LICENSE](LICENSE).
