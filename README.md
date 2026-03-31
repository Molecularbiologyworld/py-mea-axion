# py-mea-axion

End-to-end analysis of multi-electrode array (MEA) recordings from
**Axion Biosystems** instruments, in Python.

`py-mea-axion` reads raw `.spk` binary files and carries the analysis
through spike metrics, burst detection, network burst detection, and
synchrony measurement — all the way to statistical comparisons and
publication-ready figures.

---

## Features

- **Binary I/O** — direct parsing of Axion `.spk` files; no export step required
- **Spike metrics** — mean firing rate, ISI statistics, active electrode classification
- **Burst detection** — ISI-threshold and Poisson Surprise algorithms, configurable parameters
- **Network burst detection** — participation-threshold method on a 10 ms time grid
- **Synchrony** — Spike Time Tiling Coefficient (STTC; Cutts & Eglen 2014), pairwise and well-level
- **Statistics** — Mann-Whitney U, Kruskal-Wallis + Dunn post-hoc, ICC(2,k), longitudinal mixed-effects model
- **Visualisation** — electrode heatmaps, burst rasters, ISI histograms, STTC matrices, network-burst timelines, longitudinal trajectory plots
- **CLI** — `mea-axion run` and `mea-axion summary` for batch processing without Python

---

## Installation

```bash
pip install py-mea-axion
```

Or install from source:

```bash
git clone https://github.com/Molecularbiologyworld/py-mea-axion.git
cd py-mea-axion
pip install -e .
```

**Requirements:** Python ≥ 3.10, numpy, scipy, pandas, matplotlib, pingouin, statsmodels.

---

## Quickstart

### Python API

```python
from py_mea_axion.pipeline import MEAExperiment

exp = MEAExperiment(
    "recording.spk",
    metadata="plate_map.csv",   # well_id, condition, DIV, replicate_id
).run()

# Tabular results
exp.spike_metrics      # per-electrode: MFR, ISI stats, active flag
exp.burst_table        # per-burst: start/end time, duration, spike count
exp.well_summary       # per-well: n_active, mean MFR, burst rate, STTC

# Statistics
res = exp.compare("mean_mfr_active_hz")          # Mann-Whitney / Kruskal-Wallis
coef = exp.longitudinal("mean_mfr_active_hz")    # mixed-effects model

# Figures
exp.plot_heatmap("A1")             # electrode MFR grid
exp.plot_raster("A1")              # spike raster + burst overlays
exp.plot_trajectory("mean_mfr_active_hz")  # longitudinal mean ± SEM
exp.plot_sttc("A1")                # pairwise STTC matrix
exp.plot_network_timeline("A1")    # network burst Gantt chart
```

### Plate-map CSV format

```
well_id,condition,DIV,replicate_id
A1,WT,14,rep1
A2,KD,14,rep1
B1,WT,14,rep2
B2,KD,14,rep2
```

### Command-line interface

```bash
# Full pipeline — writes CSVs + figures to results/
mea-axion run recording.spk --metadata plate_map.csv --out results/

# Adjust burst parameters
mea-axion run recording.spk --max-isi 0.05 --min-spikes 50 --out results/

# Quick well summary to stdout
mea-axion summary recording.spk
```

---

## Burst detection parameters

| Parameter | Default | Description |
|---|---|---|
| `max_isi_s` | 0.1 s | Maximum within-burst ISI |
| `min_spikes` | 5 | Minimum spikes per burst |
| `min_ibi_s` | 0.2 s | Minimum inter-burst interval |
| `algorithm` | `isi_threshold` | Also supports `poisson_surprise` |

## Network burst parameters

| Parameter | Default | Description |
|---|---|---|
| `participation_threshold` | 0.25 | Minimum fraction of active electrodes |
| `bin_size_s` | 0.010 s | Time-bin resolution |
| `min_network_ibi_s` | 1.0 s | Minimum inter-network-burst interval |

Pass these via `burst_kwargs` / `network_kwargs` in `MEAExperiment`, or
`--max-isi` / `--min-spikes` on the command line.

---

## Note on Axion .spk files

Some recordings lack a `BlockVectorHeader` (common in older firmware).
In this case the sampling frequency is inferred from the record size; a
warning is printed. If inference is incorrect, override it explicitly:

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

---

## License

MIT — see [LICENSE](LICENSE).
