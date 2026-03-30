# py-mea-axion — project context for Claude Code

## What already exists

The `existing/` folder contains a complete, working Python package
called `mea_raster_generator`. Read `existing/mea_raster_generator.py`
in full before doing anything else. Here is a summary of what it does:

### .spk binary parser — `load_spikes_from_spk()`
This is the most important function in the existing code. It parses
the Axion BioSystems .spk binary format from first principles, based
on the AxisFile.m MATLAB source. Key implementation details to preserve:

- Validates the "AxionBio" magic header bytes
- Parses the primary file header (version, entry slots as uint64)
- Walks entry records sequentially: ChannelArray (0x02),
  BlockVectorHeader (0x03), BlockVectorData (0x04)
- Handles sub-headers (multi-header files) by detecting the magic
  bytes at the Terminate entry position
- Falls back gracefully when BlockVectorHeader is missing (tries
  known record sizes)
- Returns: dict[well_label] -> dict[electrode_1to16] -> np.ndarray
  of spike timestamps in seconds
- Also returns total_time_s (inferred from last spike timestamp)

### Plate geometry constants
- 24-well CytoView plate: rows A-D, columns 1-6
- 16 electrodes per well (4x4 grid)
- 384 total channels
- well_label() maps (row_1based, col_1based) -> e.g. "A1", "D6"

### Other loaders
- `load_spikes_from_raw()` — spike detection from .raw continuous
  files using MAD thresholding (thresh_k * sigma_MAD), chunked
  processing with mmap
- `load_spikes_from_npz()` — loads pre-parsed .npz spike cache

### ASDR computation — `compute_asdr()`
Bins spikes across all 16 electrodes into 200ms bins (configurable).
Returns (bin_left_edges_ms, counts). Used for both plotting and
the statistics CSV.

### Plotting
- `plot_combined()` — ASDR histogram (top) + 16-electrode raster
  (bottom), saved as PNG. Uses matplotlib GridSpec with shared x-axis.
- `plot_asdr_standalone()` — ASDR histogram only

### Statistics — `compute_stats()`
Per-well stats including: total spikes, average spike rate, max
electrode rate, average ISI, total detected bursts, ASDR peaks.
Exports to CSV.

### Public API
- `run()` — single entry point, accepts input_file, output_dir,
  wells, time_start, time_end, and all plot/analysis parameters
- `main()` — argparse CLI wrapper around run()

### Dependencies (existing)
numpy>=1.22, matplotlib>=3.5, scipy>=1.8

---

## What we are building

We are expanding this into a full analysis package called
`py_mea_axion`. The goals are:

1. Preserve and extend the existing .spk parser and raster/ASDR
   plotting — do NOT rewrite them from scratch. Refactor and clean
   them into the new module structure.

2. Add spike-level metrics (MFR, ISI distribution, CV of ISI)

3. Add proper burst detection from spike timestamps using the
   ISI-threshold algorithm (and optionally Poisson Surprise),
   producing reproducible results independent of AxIS Navigator's
   internal burst detection

4. Add network burst detection across electrodes

5. Add synchrony metrics (STTC — Spike Time Tiling Coefficient)

6. Add statistical comparison tools (Mann-Whitney, mixed-effects
   models for longitudinal data, ICC)

7. Add new visualisations: spatial electrode heatmaps, longitudinal
   trajectory plots (metric vs DIV), synchrony matrices

8. Provide a high-level MEAExperiment class as the primary user-
   facing API

9. Provide a CLI entry point: mea-axion

The scientific use case is a longitudinal LGI2 knockdown experiment
in cortical neurons: multiple DIV timepoints, WT vs KD condition.
The centrepiece output is a trajectory plot of network burst rate
vs DIV, WT vs KD, with the first-burst-onset DIV annotated.

---

## Key rules for Claude Code

- Always read this file and all files in existing/ at the start
  of each session before writing any code
- The .spk parser in existing/mea_raster_generator.py is known
  to work correctly on real Axion data — preserve its logic
  exactly when refactoring into io/spk_reader.py
- Build and test one module at a time; do not proceed to the next
  module until the current one has a passing pytest test
- All public functions must have NumPy-style docstrings
- Use type hints on all function signatures
- Return DataFrames with c