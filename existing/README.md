# mea-raster-generator

Python tool for analysing multielectrode array (MEA) recordings from the **Axion Maestro** system.

Generates per-well figures matching the output of the MATLAB pipeline (`Axion_TVAloopv3` / `v4`):

- **Top panel** — ASDR histogram (spike counts per 200 ms bin, summed across all 16 electrodes)
- **Lower rows** — Raster plot (one row per electrode, vertical tick per spike)
- **Standalone ASDR PNG** and **statistics CSV** also saved

---

## Supported input files

| Extension | Description |
|-----------|-------------|
| `.spk` | Axion spike file — recommended, no extra parameters needed |
| `.raw` | Axion Maestro continuous raw recording (spike detection on-the-fly) |
| `.npz` | Pre-parsed spike cache |

---

## Installation

### From PyPI

```bash
pip install mea-raster-generator
```

### Local install (from cloned repo)

```bash
pip install .
```

### Inside a conda environment

```bash
conda create -n mea python=3.10
conda activate mea
pip install mea-raster-generator
```

Python 3.8+ required.

---

## Usage

```
mea-raster-generator INPUT_FILE [options]
```

### Positional argument

| Argument | Description |
|----------|-------------|
| `INPUT_FILE` | Path to `.spk`, `.raw`, or `.npz` recording file |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir DIR` | auto | Output directory (default: `mea_raster_generator_output/` next to input file) |
| `--wells WELL [WELL ...]` | `ALL` | Wells to analyse, e.g. `A1 B2 C3`, or `ALL` |
| `--time-start FLOAT` | `0` | Start of time window in seconds |
| `--time-end FLOAT` | `0` | End of time window in seconds; `0` = full recording |
| `--asdr-thresh INT` | `50` | ASDR threshold spike count (red dashed line) |
| `--combined` / `--no-combined` | on | Save combined raster+histogram figure |
| `--asdr-y-max FLOAT` | `0` | Y-axis maximum for ASDR panels; `0` = autoscale |
| `--dpi INT` | `300` | Figure resolution in DPI |
| `--rec-seconds FLOAT` | — | Recording duration in seconds (required for `.raw` and `.npz`) |
| `--bin-ms INT` | `200` | ASDR bin width in ms |
| `--thresh-k FLOAT` | `5.0` | Spike threshold multiplier K for MAD thresholding (`.raw` only) |

### Examples

```bash
# Analyse well D1 from an .spk file, time window 0–420 s
mea-raster-generator recording.spk --wells D1 --time-end 420 --asdr-thresh 50

# Analyse all wells, autoscale Y axis
mea-raster-generator recording.spk --wells ALL --time-end 600

# Raw file (rec-seconds required), save to custom output directory
mea-raster-generator recording.raw --rec-seconds 360 --wells A1 B1 --output-dir ./results

# ASDR histogram only (no raster)
mea-raster-generator recording.spk --wells D1 --no-combined
```

---

## Output

All figures are saved to `mea_raster_generator_output/` next to the input file (or to `--output-dir` if set):

| File | Description |
|------|-------------|
| `<stem>_<well>_raster_histogram.png` | Combined raster + ASDR figure |
| `<stem>_<well>_asdr_histogram.png` | Standalone ASDR histogram |
| `<stem>_stats.csv` | Per-well statistics (spike counts, rates, ISI, bursts) |

---

## Background

This tool is a Python port of the MATLAB MEA analysis pipeline developed in the **Deane Lab**, University of Cambridge. It reads the Axion binary `.spk` container format based on the official AxIS MATLAB source (`AxisFile.m`, `BlockVectorData.m`, etc.) and replicates the spike detection (MAD threshold, identical to MATLAB) and ASDR histogram used in the original pipeline.
