# py-mea-axion — project context

## What this package does

`py_mea_axion` is a Python package for end-to-end analysis of Axion Biosystems `.spk`
multielectrode array (MEA) recordings. It parses binary spike data, computes a
comprehensive set of activity and network metrics matched to NeuralMetric Tools, and
provides statistical analysis and visualisation tools for longitudinal experiments.

---

## Package structure

```
py_mea_axion/
  io/           .spk binary parser (preserves original AxisFile.m logic)
  burst/        ISI-threshold burst detection and electrode burst metrics
  network/      Combined-ISI network burst detection and network metrics
  sync/         STTC (Spike Time Tiling Coefficient)
  viz/          Raster, ASDR, and burst chart visualisations
  pipeline.py   MEAExperiment — main user-facing API
  cli.py        mea-axion CLI (run / summary subcommands)
```

---

## Pipeline parameters (matched to NeuralMetric Tools defaults)

| Parameter | Value |
|---|---|
| `fs_override` | 12500 Hz (files lack BlockVectorHeader) |
| `active_threshold_hz` | 5/60 Hz (5 spikes/min) |
| Burst algorithm | `isi_threshold` |
| Burst `min_spikes` | 5 |
| Burst `max_isi_s` | 0.1 s |
| Burst `min_ibi_s` | 0.0 s |
| Network burst algorithm | `combined_isi` |
| Network burst `min_spikes` | 50 |
| Network burst `max_isi_s` | 0.1 s |
| Network burst `participation_threshold` | 0.35 |

---

## Metrics produced (`well_summary`)

### Activity (4)
| Column | Description |
|---|---|
| `mean_mfr_active_hz` | Mean firing rate across active electrodes (Hz) |
| `n_active` | Number of active electrodes |
| `n_spikes` | Total spike count |
| `isi_cv_avg` | Mean ISI coefficient of variation across active electrodes |

### Electrode burst (11)
| Column | Description |
|---|---|
| `n_bursts` | Total number of bursts across all electrodes |
| `n_bursting_electrodes` | Number of electrodes with ≥1 burst |
| `burst_duration_avg` | Mean burst duration (s), per-electrode-first |
| `n_spikes_per_burst_avg` | Mean spikes per burst, per-electrode-first |
| `mean_isi_within_burst_avg` | Mean ISI within bursts (s), per-electrode-first |
| `median_isi_within_burst_avg` | Median ISI within bursts (s), per-electrode-first |
| `median_mean_isi_ratio_burst_avg` | Median/Mean ISI ratio within bursts |
| `ibi_avg` | Mean inter-burst interval (s) |
| `burst_freq_avg` | Mean burst frequency (Hz) |
| `ibi_cv_avg` | Mean IBI coefficient of variation |
| `burst_pct_avg` | Mean burst percentage (% of recording time in bursts) |

### Network burst (11)
| Column | Description |
|---|---|
| `n_network_bursts` | Number of network bursts |
| `network_burst_freq` | Network burst frequency (Hz) |
| `network_burst_duration_avg` | Mean network burst duration (s) |
| `n_spikes_per_nb_avg` | Mean spikes per network burst |
| `mean_isi_within_nb_avg` | Mean ISI within network bursts (s) |
| `median_isi_within_nb_avg` | Median ISI within network bursts (s) |
| `median_mean_isi_ratio_nb_avg` | Median/Mean ISI ratio within network bursts |
| `n_elecs_per_nb_avg` | Mean electrodes participating per network burst |
| `n_spikes_per_nb_per_channel_avg` | Mean spikes per network burst per channel |
| `network_burst_pct` | Network burst percentage (% of recording time) |
| `network_ibi_cv` | Network IBI coefficient of variation |

### Synchrony (1)
| Column | Description |
|---|---|
| `mean_sttc` | Mean pairwise Spike Time Tiling Coefficient |

**NaN convention**: NaN values represent true absence of activity (e.g. no bursts detected).
For downstream analysis and plotting, NaN is treated as 0.

---

## Aggregation convention

Electrode burst metrics use **per-electrode-first averaging** (mean-of-means), matching
NeuralMetric Tools: for each electrode, compute the mean across its bursts; then average
those per-electrode means across all bursting electrodes. This gives equal weight to each
electrode regardless of burst count. Pooling all bursts before averaging produces biased
results when burst counts vary across electrodes.

---

## Benchmarking results

946 well-observations across both plates and all DIVs, compared against NeuralMetric Tools
exports. All 27 metrics benchmarked. Agreement reported as r²:

| Category | Result |
|---|---|
| Activity (MFR, N active, ISI CV, N bursts, N bursting elecs) | r² = 1.000, ~0% error |
| Electrode burst metrics (9/11) | r² = 1.000, ~0% error |
| Burst frequency avg | r² = 1.000, 0.1% error |
| Burst % avg | r² = 0.998, 5.8% error (known; denominator convention difference) |
| Network burst metrics (10/11) | r² = 1.000, ~0% error |
| Network IBI CV | r² = 0.980, 4% error |

Summary CSV: `benchmarking/all_benchmark_summary.csv`  
Scatter plots (one per metric, open circles with condition colour): `benchmarking/figures_all/`  
Plate layout heatmaps (per metric per DIV): `benchmarking/figures_plate_heatmap/`

---

## Experimental data — LGI2 knockdown

- **Conditions**: SCRM (scramble control), LGI2_KD4, LGI2_KD5
- **Plates**: Plate 1 (24 wells, Batch 1 + Batch 2), Plate 2 (12 wells, columns 4–6 empty)
- **Replicates**: 12 per condition (rep01–rep12)
- **DIVs**: 12 to 44
- **Data location**: `LGI2 KD data/` (not version-controlled; unpublished)

---

## Scientific analysis

Script: `analysis/run_analysis.py`  
Output: `analysis/figures/`, `analysis/results_raw.csv`, `analysis/anova_summary.csv`,
`analysis/posthoc_summary.csv`

**Figures generated per metric (27 total)**:
- `trajectory_<metric>.png` — mean ± SEM per condition across DIV
- `violin_<metric>.png` — per-DIV violin + jitter, one panel per DIV
- `plate_heatmaps/<metric>/DIV_XX.png` — per-DIV plate layout heatmap (Plate 1 + Plate 2 side by side, well colour = metric value, grey = absent well)
- `anova_pvalue_heatmap.png` — −log10(p_adj) across all metrics × DIVs

**Condition colours**:
- SCRM: `#4477AA` (blue)
- LGI2 KD4: `#CC3311` (dark brick red)
- LGI2 KD5: `#EE9988` (light salmon)

---

## Statistical approach

### 1. One-way ANOVA (pingouin)
Run per metric × DIV across the three conditions. Justification: standard parametric test
for continuous metrics; robust to non-normality at n=12 per group by CLT. Identifies DIVs
where any condition difference exists before post-hoc testing.

### 2. Dunnett's post-hoc test (scipy.stats.dunnett)
Pairwise comparisons: KD4 vs SCRM and KD5 vs SCRM only. Justification: purpose-built for
comparing multiple treatment groups against a single control — more powerful than Tukey's
or Bonferroni for this design because it does not test all pairwise combinations.

### 3. Benjamini-Hochberg FDR correction (pingouin.multicomp)
Applied across all post-hoc tests (~1,296 total). Justification: with 27 metrics × 27 DIVs
× 2 comparisons, FWER correction (Bonferroni) is too conservative for a multi-endpoint
biological study. FDR-BH is the standard approach and maintains acceptable false discovery
control while preserving power to detect true effects.

**Known limitation**: at early DIVs where all groups have zero variance (e.g. no network
bursts at D12–D17), ANOVA cannot compute an F-statistic. These cases are silently skipped
via `try/except`; they are biologically uninformative regardless of test choice.

---

## Key bugs fixed (history)

| Bug | Fix |
|---|---|
| D26 `.spk` silent failure | Wrong record size (68 vs 106 bytes) selected because it divided file length evenly; fixed by validating last-record timestamp must be 0–86400 s |
| `min_ibi_s` default 0.2 s | Was merging adjacent bursts; changed to 0.0 to match NeuralMetric |
| Burst metric aggregation bias | Pooling all bursts weighted high-burst electrodes; fixed by per-electrode-first averaging |
