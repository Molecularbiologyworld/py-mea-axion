"""
cli.py
======
``mea-axion`` command-line interface.

Subcommands
-----------
run
    Full pipeline: load spikes → compute metrics → detect bursts →
    network bursts → STTC → save CSVs and optional PNG figures.

summary
    Quick well-level summary table printed to stdout.

Usage examples
--------------
::

    # Full analysis, save results to ./results/
    mea-axion run recording.spk --metadata plate_map.csv --out results/

    # Restrict to two wells, skip figures
    mea-axion run recording.spk --wells A1 B2 --no-figures --out results/

    # Quick console summary
    mea-axion summary recording.spk --wells A1 A2

    # Override sampling frequency for files without a BlockVectorHeader
    mea-axion run recording.spk --fs-override 12500 --out results/

    # Analyse only a 5-minute window from 300 s to 600 s
    mea-axion run recording.spk --time-start 300 --time-end 600 --out results/
"""

import argparse
import logging
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from py_mea_axion.pipeline import MEAExperiment

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[mea-axion] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ── Argument parser ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="mea-axion",
        description="End-to-end MEA analysis for Axion Biosystems recordings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    _add_run_parser(sub)
    _add_summary_parser(sub)
    _add_plot_parser(sub)
    _add_build_parser(sub)
    _add_show_layout_parser(sub)
    _add_stats_parser(sub)

    return parser


def _common_args(p: argparse.ArgumentParser) -> None:
    """Add arguments shared by both subcommands."""
    p.add_argument(
        "spk_file",
        metavar="SPK_FILE",
        help="Path to the Axion *.spk recording file.",
    )
    p.add_argument(
        "--wells", "-w",
        nargs="+",
        metavar="WELL",
        default=None,
        help="Subset of wells to analyse (e.g. A1 B2).  Default: all.",
    )
    p.add_argument(
        "--fs-override",
        type=float,
        default=None,
        metavar="HZ",
        help=(
            "Force a specific sampling frequency in Hz.  Use when the .spk "
            "file lacks a BlockVectorHeader and the automatic detection is "
            "wrong (common value: 12500)."
        ),
    )
    p.add_argument(
        "--active-threshold",
        type=float,
        default=0.1,
        metavar="HZ",
        help="MFR threshold (Hz) for classifying an electrode as active.  Default: 0.1.",
    )
    p.add_argument(
        "--time-start",
        type=float,
        default=None,
        metavar="S",
        help=(
            "Analysis-window start time in seconds.  Spikes before this "
            "time are excluded, and retained timestamps are shifted so the "
            "window starts at 0 s."
        ),
    )
    p.add_argument(
        "--time-end",
        type=float,
        default=None,
        metavar="S",
        help=(
            "Analysis-window end time in seconds.  Spikes after this time "
            "are excluded.  Default: use the full recording."
        ),
    )


def _add_run_parser(sub) -> None:
    p = sub.add_parser(
        "run",
        help="Run the full analysis pipeline and save results.",
        description=(
            "Load a .spk file, run the complete analysis pipeline "
            "(spike metrics, burst detection, network bursts, STTC), "
            "and write CSVs + PNG figures to the output directory."
        ),
    )
    _common_args(p)
    p.add_argument(
        "--metadata", "-m",
        default=None,
        metavar="CSV",
        help=(
            "Path to plate-map CSV with columns: well_id, condition, DIV, "
            "replicate_id.  Required for trajectory plots and stats."
        ),
    )
    p.add_argument(
        "--out", "-o",
        default=None,
        metavar="DIR",
        help=(
            "Output directory.  Created if it does not exist.  "
            "Defaults to a folder named after the recording file."
        ),
    )
    p.add_argument(
        "--max-isi",
        type=float,
        default=0.1,
        metavar="S",
        help="Max within-burst ISI (s) for burst detection.  Default: 0.1.",
    )
    p.add_argument(
        "--min-spikes",
        type=int,
        default=5,
        metavar="N",
        help="Minimum spikes per burst.  Default: 5.",
    )
    p.add_argument(
        "--sttc-dt",
        type=float,
        default=0.05,
        metavar="S",
        help="STTC coincidence window half-width (s).  Default: 0.05.",
    )
    p.add_argument(
        "--asdr-bin",
        type=float,
        default=0.2,
        metavar="S",
        help=(
            "ASDR histogram bin width (s) for the burst-raster figure.  "
            "Default: 0.2."
        ),
    )
    p.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip PNG figure export (CSVs are always written).",
    )


def _add_summary_parser(sub) -> None:
    p = sub.add_parser(
        "summary",
        help="Print a quick per-well summary table to stdout.",
        description=(
            "Run the analysis pipeline and print the well-level summary "
            "table to standard output.  No files are written."
        ),
    )
    _common_args(p)


# ── Plot subcommands ──────────────────────────────────────────────────────────

# Identifier columns that must be excluded when iterating over "all metrics".
_NON_METRIC_COLUMNS = frozenset({
    "well_id", "electrode_id", "condition", "DIV", "replicate_id",
    "batch", "plate", "n_electrodes", "bio_rep", "tech_rep",
})


# Human-readable labels for the 27 metric columns produced by the
# pipeline.  Used for plot axis labels / colour-bar titles.
_METRIC_LABELS: Dict[str, str] = {
    # Activity
    "mean_mfr_active_hz":              "Mean firing rate (Hz)",
    "n_active":                        "N active electrodes",
    "n_spikes":                        "N spikes (total)",
    "isi_cv_avg":                      "ISI CV (avg)",
    # Electrode burst
    "n_bursts":                        "N bursts",
    "n_bursting_electrodes":           "N bursting electrodes",
    "burst_duration_avg":              "Burst duration avg (s)",
    "n_spikes_per_burst_avg":          "Spikes per burst avg",
    "mean_isi_within_burst_avg":       "Mean ISI within burst avg (s)",
    "median_isi_within_burst_avg":     "Median ISI within burst avg (s)",
    "median_mean_isi_ratio_burst_avg": "Median/Mean ISI within burst",
    "ibi_avg":                         "IBI avg (s)",
    "burst_freq_avg":                  "Burst frequency avg (Hz)",
    "ibi_cv_avg":                      "IBI CV avg",
    "burst_pct_avg":                   "Burst % avg",
    # Network burst
    "n_network_bursts":                "N network bursts",
    "network_burst_freq":              "Network burst freq (Hz)",
    "network_burst_duration_avg":      "Network burst duration avg (s)",
    "n_spikes_per_nb_avg":             "Spikes per NB avg",
    "mean_isi_within_nb_avg":          "Mean ISI within NB avg (s)",
    "median_isi_within_nb_avg":        "Median ISI within NB avg (s)",
    "median_mean_isi_ratio_nb_avg":    "Median/Mean ISI within NB",
    "n_elecs_per_nb_avg":              "Electrodes per NB avg",
    "n_spikes_per_nb_per_channel_avg": "Spikes/NB/channel avg",
    "network_burst_pct":               "Network burst %",
    "network_ibi_cv":                  "Network IBI CV",
    # Synchrony
    "mean_sttc":                       "Mean STTC",
}


def _label_for(metric: str) -> str:
    """Pretty axis label for a metric column.  Falls back to the column name."""
    return _METRIC_LABELS.get(metric, metric)


def _fill_metric_nans(df, *, group_col: str, time_col: str):
    """Replace NaN with 0 in numeric metric columns.

    Absence of bursts / network bursts on early DIVs is a true zero —
    not missing data — so plotting and aggregation should treat it as 0
    rather than dropping the row.  Identifier columns and the time/group
    axes are left untouched.
    """
    import pandas as pd

    excluded = set(_NON_METRIC_COLUMNS) | {group_col, time_col}
    cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if cols:
        df[cols] = df[cols].fillna(0)
    return df


def _add_plot_parser(sub) -> None:
    p = sub.add_parser(
        "plot",
        help="Render a specific figure type into its own folder.",
        description=(
            "Render a specific figure type into its own folder.  Per-recording "
            "plots (heatmap, raster) take a .spk file; cross-DIV plots "
            "(trajectory, timepoint, pca) take a pre-computed well-summary "
            "CSV (e.g. analysis/results_raw.csv)."
        ),
    )
    pp = p.add_subparsers(dest="plot_type", metavar="<plot-type>")
    pp.required = True

    _add_plot_heatmap_parser(pp)
    _add_plot_raster_parser(pp)
    _add_plot_trajectory_parser(pp)
    _add_plot_timepoint_parser(pp)
    _add_plot_pca_parser(pp)


def _add_plot_heatmap_parser(pp) -> None:
    p = pp.add_parser(
        "heatmap",
        help=(
            "Plate-layout heatmap (4x6 well grid, one panel per plate) "
            "of a per-well metric, one figure per metric per DIV."
        ),
    )
    _common_csv_args(p)
    _add_plate_map_args(p)
    p.add_argument(
        "--metric", "-M",
        nargs="+",
        default=None,
        metavar="COL",
        help=(
            "Well-level metric column(s) to plot.  Default: every numeric "
            "column other than identifier columns."
        ),
    )
    p.add_argument(
        "--div",
        type=float,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "One or more time-point values to plot.  Default: every unique "
            "value of the time column."
        ),
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        metavar="NAME",
        help="Matplotlib colour map for the heatmap.  Default: viridis.",
    )
    p.add_argument(
        "--out", "-o",
        default="figures/heatmaps",
        metavar="DIR",
        help="Output directory.  Default: figures/heatmaps/.",
    )


def _add_plot_raster_parser(pp) -> None:
    p = pp.add_parser(
        "raster",
        help="Burst raster + ASDR figures from a single .spk recording.",
    )
    _common_args(p)
    p.add_argument(
        "--asdr-bin",
        type=float,
        default=0.2,
        metavar="S",
        help="ASDR histogram bin width (s).  Default: 0.2.",
    )
    p.add_argument(
        "--max-isi",
        type=float,
        default=0.1,
        metavar="S",
        help="Max within-burst ISI (s) for burst detection.  Default: 0.1.",
    )
    p.add_argument(
        "--min-spikes",
        type=int,
        default=5,
        metavar="N",
        help="Minimum spikes per burst.  Default: 5.",
    )
    p.add_argument(
        "--density-color",
        action="store_true",
        help=(
            "Colour each spike tick by local spike density (binned at "
            "--asdr-bin width, normalised to the busiest bin in the "
            "well).  Brighter colour = more dense.  Adds a colour bar."
        ),
    )
    p.add_argument(
        "--density-cmap",
        default="viridis",
        metavar="NAME",
        help=(
            "Matplotlib colour map for --density-color.  Default: "
            "viridis (dark blue → yellow), matching the plate heatmap."
        ),
    )
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help=(
            "Figure size in inches (width height).  Default: 8 5.5."
        ),
    )
    p.add_argument(
        "--out", "-o",
        default="figures/rasters",
        metavar="DIR",
        help="Output directory.  Default: figures/rasters/.",
    )


def _common_csv_args(p: argparse.ArgumentParser) -> None:
    """Arguments shared by trajectory / timepoint / pca subcommands."""
    p.add_argument(
        "csv_file",
        metavar="CSV_FILE",
        help=(
            "Pre-computed well-summary CSV with metric columns plus "
            "condition, DIV, and replicate_id columns "
            "(e.g. analysis/results_raw.csv)."
        ),
    )
    p.add_argument(
        "--group-col",
        default="condition",
        metavar="COL",
        help="Column identifying experimental conditions.  Default: condition.",
    )
    p.add_argument(
        "--time-col",
        default="DIV",
        metavar="COL",
        help="Column holding the time variable.  Default: DIV.",
    )
    p.add_argument(
        "--pool",
        action="store_true",
        help=(
            "Treat every well as one independent observation, ignoring "
            "the bio_rep / tech_rep columns.  Default: aggregate technical "
            "replicates within each biological replicate before computing "
            "statistics (hierarchical)."
        ),
    )
    p.add_argument(
        "--group-order",
        nargs="+",
        default=None,
        metavar="GROUP",
        help=(
            "Order of conditions on the x-axis (violin) or in the legend "
            "(trajectory / pca).  Example: --group-order SCRM LGI2_KD4 "
            "LGI2_KD5.  Default: alphabetical.  Listing a subset excludes "
            "the unlisted groups from the plot."
        ),
    )
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help=(
            "Figure size in inches (width height).  Example: "
            "--figsize 8 4.  Default depends on the plot type."
        ),
    )
    p.add_argument(
        "--filter",
        action="append",
        default=None,
        metavar="COL=VAL[,VAL...]",
        help=(
            "Restrict the input to rows whose COL is one of the listed "
            "VALs.  Repeatable; multiple --filter flags are AND-ed.  "
            "Example: --filter condition=SCRM --filter DIV=14,28."
        ),
    )


def _add_color_arg(p: argparse.ArgumentParser) -> None:
    """Inline per-condition colour mapping shared by CSV plot subcommands."""
    p.add_argument(
        "--color",
        action="append",
        default=None,
        metavar="GROUP=#HEX",
        help=(
            "Inline colour for one condition, repeatable.  "
            "Example: --color SCRM=#4477AA --color LGI2_KD4=#CC3311.  "
            "Hex codes (with the leading '#') or named colours both work."
        ),
    )


def _add_plate_map_args(p: argparse.ArgumentParser) -> None:
    """Inline plate-mapping flags shared by trajectory and timepoint."""
    p.add_argument(
        "--condition",
        action="append",
        default=None,
        metavar="GROUP=WELL[,WELL...]",
        help=(
            "Inline well-to-condition assignment.  Repeat for each condition.  "
            "Example: --condition SCRM=A1,A2,A3 --condition KD=B1,B2,B3.  "
            "Overrides the condition column in the input CSV."
        ),
    )
    p.add_argument(
        "--plate-replicate",
        action="append",
        default=None,
        metavar="PLATE=REP_ID",
        help=(
            "Inline plate-to-replicate assignment (one replicate ID per "
            "plate).  Repeat for each plate.  Example: "
            "--plate-replicate 1=rep01 --plate-replicate 2=rep02.  "
            "Use --replicate instead when the same plate contains "
            "multiple biological replicates."
        ),
    )
    p.add_argument(
        "--replicate",
        action="append",
        default=None,
        metavar="REP_ID=WELL[,WELL...]",
        help=(
            "Inline well-to-replicate assignment.  Repeat for each "
            "replicate.  Example: --replicate rep01=A1,A2 "
            "--replicate rep02=B1,B2.  Use this when several replicates "
            "share a plate (rows or batches as separate replicates).  "
            "Mutually exclusive with --plate-replicate."
        ),
    )
    p.add_argument(
        "--plate",
        default=None,
        metavar="N",
        help=(
            "Plate identifier to assign to every row.  Use when the CSV "
            "lacks a 'plate' column and you want to combine it with "
            "--plate-replicate."
        ),
    )


def _add_plot_trajectory_parser(pp) -> None:
    p = pp.add_parser(
        "trajectory",
        help="Time-course trajectory plots (mean +/- SEM per condition).",
    )
    _common_csv_args(p)
    _add_plate_map_args(p)
    _add_color_arg(p)
    p.add_argument(
        "--metric", "-M",
        nargs="+",
        default=None,
        metavar="COL",
        help=(
            "Well-level metric column(s) to plot.  Default: every numeric "
            "column other than identifier columns."
        ),
    )
    p.add_argument(
        "--div-min",
        type=float,
        default=None,
        metavar="N",
        help="Minimum value of the time column to include.  Default: no lower bound.",
    )
    p.add_argument(
        "--div-max",
        type=float,
        default=None,
        metavar="N",
        help="Maximum value of the time column to include.  Default: no upper bound.",
    )
    p.add_argument(
        "--out", "-o",
        default="figures/trajectory",
        metavar="DIR",
        help="Output directory.  Default: figures/trajectory/.",
    )


def _add_plot_timepoint_parser(pp) -> None:
    p = pp.add_parser(
        "timepoint",
        help="Single time-point condition violins (one figure per metric per DIV).",
    )
    _common_csv_args(p)
    _add_plate_map_args(p)
    _add_color_arg(p)
    p.add_argument(
        "--point-hue",
        default=None,
        metavar="COL",
        help=(
            "Colour individual jitter points by this column (e.g. 'batch').  "
            "Default: points coloured by condition (same as the violin body)."
        ),
    )
    p.add_argument(
        "--point-color",
        action="append",
        default=None,
        metavar="VALUE=#HEX",
        help=(
            "Per-value colour for points when --point-hue is set.  "
            "Repeat once per unique value of the hue column.  "
            "Example: --point-color batch1=#222222 --point-color batch2=#888888."
        ),
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=40.0,
        metavar="N",
        help=(
            "Marker area for the jittered data points (matplotlib `s` "
            "parameter).  Default: 40."
        ),
    )
    p.add_argument(
        "--show-point-legend",
        action="store_true",
        help=(
            "Show a legend mapping point colours to --point-hue values.  "
            "Default: hidden."
        ),
    )
    p.add_argument(
        "--point-shape",
        default=None,
        metavar="COL",
        help=(
            "Use marker shape (instead of, or in addition to, colour) to "
            "indicate the value of this column for individual jitter "
            "points.  Example: --point-shape batch."
        ),
    )
    p.add_argument(
        "--point-shape-map",
        action="append",
        default=None,
        metavar="VALUE=MARKER",
        help=(
            "Per-value marker for --point-shape.  Repeatable.  Example: "
            "--point-shape-map Batch1=o --point-shape-map Batch2=s "
            "--point-shape-map Batch3=^.  Markers follow matplotlib "
            "codes (o circle, s square, ^ triangle, D diamond, v down-"
            "triangle, P plus, X cross, * star)."
        ),
    )
    p.add_argument(
        "--compare",
        action="append",
        nargs=2,
        default=None,
        metavar=("GROUP_A", "GROUP_B"),
        help=(
            "Restrict significance brackets to specific pairs.  "
            "Repeat for each comparison.  Example: "
            "--compare LGI2_KD4 SCRM --compare LGI2_KD5 SCRM.  "
            "Default: every pair."
        ),
    )
    p.add_argument(
        "--test",
        choices=("tukey", "mannwhitney", "kruskal"),
        default="tukey",
        help=(
            "Statistical test used for the violin significance brackets.  "
            "tukey (default): parametric all-pairs Tukey HSD with built-in "
            "family-wise correction.  mannwhitney: pairwise Mann-Whitney U "
            "with Bonferroni correction.  kruskal: Kruskal-Wallis omnibus "
            "followed by Dunn's pairwise post-hoc with Bonferroni "
            "correction.  The choice is the user's responsibility; the "
            "package does not auto-select."
        ),
    )
    p.add_argument(
        "--metric", "-M",
        nargs="+",
        default=None,
        metavar="COL",
        help=(
            "Well-level metric column(s) to plot.  Default: every numeric "
            "column other than identifier columns."
        ),
    )
    p.add_argument(
        "--div",
        type=float,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "One or more time-point values to plot.  Default: every unique "
            "value of the time column."
        ),
    )
    p.add_argument(
        "--out", "-o",
        default="figures/timepoint",
        metavar="DIR",
        help="Output directory.  Default: figures/timepoint/.",
    )


def _add_plot_pca_parser(pp) -> None:
    p = pp.add_parser(
        "pca",
        help="PCA scatter coloured by condition.",
    )
    _common_csv_args(p)
    _add_color_arg(p)
    p.add_argument(
        "--point-shape",
        default=None,
        metavar="COL",
        help=(
            "Use marker shape (in addition to colour) to indicate the "
            "value of this column for individual points.  Example: "
            "--point-shape batch."
        ),
    )
    p.add_argument(
        "--point-shape-map",
        action="append",
        default=None,
        metavar="VALUE=MARKER",
        help=(
            "Per-value marker for --point-shape.  Repeatable.  Example: "
            "--point-shape-map Batch1=o --point-shape-map Batch2=s."
        ),
    )
    p.add_argument(
        "--loadings",
        action="store_true",
        help=(
            "Also save a per-feature PC loadings CSV plus a horizontal "
            "bar chart of the top --top contributors to PC1 and PC2."
        ),
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Number of top-|loading| features to draw per PC in the "
            "loadings figure.  Default: 10.  Ignored without --loadings."
        ),
    )
    p.add_argument(
        "--metric", "-M",
        nargs="+",
        default=None,
        metavar="COL",
        help=(
            "Subset of metric columns to use as PCA features.  Default: "
            "every numeric column other than identifier columns."
        ),
    )
    p.add_argument(
        "--div",
        type=float,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Restrict to specific time-point values (one figure per value).  "
            "Default: pool all time points into a single figure."
        ),
    )
    p.add_argument(
        "--div-min",
        type=float,
        default=None,
        metavar="N",
        help="Minimum value of the time column to include.  Default: no lower bound.",
    )
    p.add_argument(
        "--div-max",
        type=float,
        default=None,
        metavar="N",
        help="Maximum value of the time column to include.  Default: no upper bound.",
    )
    p.add_argument(
        "--out", "-o",
        default="figures/pca",
        metavar="DIR",
        help="Output directory.  Default: figures/pca/.",
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list of str, optional
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return _cmd_run(args)
        elif args.command == "summary":
            return _cmd_summary(args)
        elif args.command == "plot":
            return _cmd_plot(args)
        elif args.command == "build":
            return _cmd_build(args)
        elif args.command == "show-layout":
            return _cmd_show_layout(args)
        elif args.command == "stats":
            return _cmd_stats(args)
    except KeyboardInterrupt:
        log.info("Interrupted.")
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("%s", exc)
        return 1

    return 0


# ── Subcommand implementations ─────────────────────────────────────────────────

def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand."""
    spk = Path(args.spk_file)
    if not spk.exists():
        log.error("File not found: %s", spk)
        return 1

    out_dir = Path(args.out) if args.out else spk.parent / spk.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    burst_kwargs = {
        "max_isi_s": args.max_isi,
        "min_spikes": args.min_spikes,
    }

    exp = MEAExperiment(
        spk,
        metadata=args.metadata,
        wells=args.wells,
        fs_override=args.fs_override,
        time_start_s=args.time_start,
        time_end_s=args.time_end,
        active_threshold_hz=args.active_threshold,
        burst_kwargs=burst_kwargs,
        sttc_dt_s=args.sttc_dt,
    )

    log.info("Running pipeline …")
    exp.run()

    _save_csvs(exp, out_dir)

    if not args.no_figures:
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        _save_figures(exp, fig_dir, asdr_bin_s=args.asdr_bin)

    log.info("Done.  Results saved to %s", out_dir)
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    """Execute the 'summary' subcommand."""
    spk = Path(args.spk_file)
    if not spk.exists():
        log.error("File not found: %s", spk)
        return 1

    exp = MEAExperiment(
        spk,
        wells=args.wells,
        fs_override=args.fs_override,
        time_start_s=args.time_start,
        time_end_s=args.time_end,
        active_threshold_hz=args.active_threshold,
    )

    log.info("Running pipeline …")
    exp.run()

    _print_summary(exp)
    return 0


# ── Output helpers ─────────────────────────────────────────────────────────────

def _save_csvs(exp, out_dir: Path) -> None:
    """Write the three result CSVs."""
    exp.spike_metrics.to_csv(out_dir / "spike_metrics.csv", index=False)
    log.info("Saved spike_metrics.csv (%d rows)", len(exp.spike_metrics))

    exp.burst_table.to_csv(out_dir / "burst_table.csv", index=False)
    log.info("Saved burst_table.csv (%d rows)", len(exp.burst_table))

    exp.well_summary.to_csv(out_dir / "well_summary.csv", index=False)
    log.info("Saved well_summary.csv (%d rows)", len(exp.well_summary))


def _save_figures(exp, fig_dir: Path, asdr_bin_s: float = 0.2) -> None:
    """Export standard PNG figures for every well."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for well_id in exp.wells:
        # Electrode MFR heatmap.
        fig = exp.plot_heatmap(well_id, metric="mfr_hz")
        fig.savefig(fig_dir / f"{well_id}_heatmap_mfr.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        # Burst raster.
        fig = exp.plot_raster(well_id, asdr_bin_s=asdr_bin_s)
        fig.savefig(fig_dir / f"{well_id}_raster.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        # STTC matrix.
        fig = exp.plot_sttc(well_id)
        fig.savefig(fig_dir / f"{well_id}_sttc.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        # Network-burst timeline.
        fig = exp.plot_network_timeline(well_id)
        fig.savefig(fig_dir / f"{well_id}_network_timeline.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    log.info("Saved %d figures to %s", len(exp.wells) * 4, fig_dir)

    # Trajectory plot (only when metadata provides condition + DIV).
    if exp.metadata is not None:
        js = exp.joined_summary()
        if "condition" in js.columns and "DIV" in js.columns:
            for metric in ("mean_mfr_active_hz", "mean_sttc", "network_burst_freq"):
                try:
                    fig = exp.plot_trajectory(metric)
                    fig.savefig(
                        fig_dir / f"trajectory_{metric}.png",
                        dpi=600, bbox_inches="tight",
                    )
                    plt.close(fig)
                except Exception:  # noqa: BLE001
                    pass


def _print_summary(exp) -> None:
    """Print well_summary as a formatted table to stdout."""
    ws = exp.well_summary.copy()

    # Round float columns for readability.
    float_cols = ws.select_dtypes("float64").columns
    ws[float_cols] = ws[float_cols].round(3)

    # Replace NaN with '—' for display.
    ws = ws.fillna("—")

    try:
        print(ws.to_string(index=False))
    except Exception:  # noqa: BLE001
        print(ws.to_csv(index=False))


# ── Plot subcommand implementations ──────────────────────────────────────────

def _cmd_plot(args: argparse.Namespace) -> int:
    dispatch = {
        "heatmap":    _cmd_plot_heatmap,
        "raster":     _cmd_plot_raster,
        "trajectory": _cmd_plot_trajectory,
        "timepoint":  _cmd_plot_timepoint,
        "pca":        _cmd_plot_pca,
    }
    return dispatch[args.plot_type](args)


def _build_recording_experiment(args: argparse.Namespace, *, with_burst: bool):
    """Construct an MEAExperiment for the recording-level plot subcommands."""
    spk = Path(args.spk_file)
    if not spk.exists():
        log.error("File not found: %s", spk)
        return None

    burst_kwargs = None
    if with_burst:
        burst_kwargs = {
            "max_isi_s": args.max_isi,
            "min_spikes": args.min_spikes,
        }

    exp = MEAExperiment(
        spk,
        wells=args.wells,
        fs_override=args.fs_override,
        time_start_s=args.time_start,
        time_end_s=args.time_end,
        active_threshold_hz=args.active_threshold,
        # Plot subcommands always render every requested well so the user
        # can decide which are biologically interesting; do not silently
        # exclude wells with no active electrodes.
        min_active_electrodes=0,
        burst_kwargs=burst_kwargs,
    )
    log.info("Running pipeline …")
    exp.run()
    return exp


def _cmd_plot_heatmap(args: argparse.Namespace) -> int:
    df = _load_summary_csv(args)
    if df is None:
        return 1

    try:
        df = _apply_inline_plate_map(df, args)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    if "plate" not in df.columns:
        log.error(
            "Plate-layout heatmap requires a 'plate' column.  Pass "
            "--plate N to tag a single-plate CSV."
        )
        return 1
    if "well_id" not in df.columns:
        log.error("Plate-layout heatmap requires a 'well_id' column.")
        return 1

    metrics = _resolve_metric_columns(
        df, args.metric, group_col=args.group_col, time_col=args.time_col,
    )
    if metrics is None:
        return 1

    if args.time_col in df.columns:
        divs = args.div if args.div is not None else sorted(df[args.time_col].unique())
    else:
        divs = [None]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from py_mea_axion.viz.plate import plot_plate_heatmap

    n_saved = 0
    for metric in metrics:
        metric_dir = out_dir / metric
        metric_dir.mkdir(exist_ok=True)
        for div in divs:
            sub = df if div is None else df[df[args.time_col] == div]
            if sub.empty:
                log.warning("Heatmap skipped for %s=%s (no rows).", args.time_col, div)
                continue
            try:
                fig = plot_plate_heatmap(
                    sub, metric=metric, cmap=args.cmap,
                    metric_label=_label_for(metric),
                    figsize=tuple(args.figsize) if getattr(args, "figsize", None) else None,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Heatmap failed for %s @ %s=%s: %s",
                            metric, args.time_col, div, exc)
                continue

            if div is None:
                fname = "all.png"
            else:
                try:
                    tag = f"{int(div):02d}" if float(div).is_integer() else f"{div}"
                except (TypeError, ValueError):
                    tag = str(div)
                fname = f"{args.time_col}_{tag}.png"
            fig.savefig(metric_dir / fname, dpi=600, bbox_inches="tight")
            plt.close(fig)
            n_saved += 1

    log.info("Saved %d heatmap figures to %s", n_saved, out_dir)
    return 0


def _cmd_plot_raster(args: argparse.Namespace) -> int:
    exp = _build_recording_experiment(args, with_burst=True)
    if exp is None:
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The pipeline shifts spikes to a 0-based timeline when --time-start
    # is given.  For display we want the original absolute timestamps
    # back, so the raster's x-axis reads e.g. 100-200 s rather than 0-100.
    x_offset = args.time_start if args.time_start is not None else 0.0
    t_start = args.time_start if args.time_start is not None else None
    t_stop = args.time_end if args.time_end is not None else None

    figsize_r = tuple(args.figsize) if getattr(args, "figsize", None) else (8.0, 5.5)
    for well_id in exp.wells:
        fig = exp.plot_raster(
            well_id,
            asdr_bin_s=args.asdr_bin,
            x_offset=x_offset,
            t_start=t_start,
            t_stop=t_stop,
            figsize=figsize_r,
            density_color=getattr(args, "density_color", False),
            density_cmap=getattr(args, "density_cmap", "viridis"),
        )
        fig.savefig(
            out_dir / f"{well_id}_raster.png",
            dpi=600, bbox_inches="tight",
        )
        plt.close(fig)

    log.info("Saved %d raster figures to %s", len(exp.wells), out_dir)
    return 0


def _load_summary_csv(args: argparse.Namespace):
    """Load the well-summary CSV and apply --div-min/--div-max if present."""
    import pandas as pd

    csv = Path(args.csv_file)
    if not csv.exists():
        log.error("File not found: %s", csv)
        return None

    df = pd.read_csv(csv)
    if args.time_col not in df.columns:
        log.error(
            "Time column '%s' not found in %s.  Available: %s",
            args.time_col, csv, list(df.columns),
        )
        return None

    div_min = getattr(args, "div_min", None)
    div_max = getattr(args, "div_max", None)
    if div_min is not None:
        df = df[df[args.time_col] >= div_min]
    if div_max is not None:
        df = df[df[args.time_col] <= div_max]

    if df.empty:
        log.error("No rows left after applying --div-min/--div-max.")
        return None

    # --filter COL=VAL[,VAL...] (repeatable, AND across flags)
    import pandas as pd  # local import (used below)
    filter_specs = getattr(args, "filter", None) or []
    for spec in filter_specs:
        try:
            col, vals = _parse_inline_assignment(spec, flag="--filter")
        except ValueError as exc:
            log.error("%s", exc)
            return None
        if col not in df.columns:
            log.error("--filter column '%s' not in CSV.  Available: %s",
                      col, list(df.columns))
            return None
        # Match numeric values when the column is numeric.
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                cast_vals = [float(v) for v in vals]
            except ValueError:
                cast_vals = vals
        else:
            cast_vals = vals
        df = df[df[col].isin(cast_vals)]
        if df.empty:
            log.error("No rows left after filter '%s'.", spec)
            return None

    # Fill NaN metric values with 0 — biologically, no bursts means a
    # true zero rate, not missing data.
    df = _fill_metric_nans(df, group_col=args.group_col, time_col=args.time_col)
    return df


def _parse_inline_assignment(spec: str, *, flag: str) -> Tuple[str, List[str]]:
    """Parse 'KEY=V1[,V2,...]' into ``(key, [v1, v2, ...])``."""
    if "=" not in spec:
        raise ValueError(
            f"Invalid {flag} value '{spec}': expected KEY=VALUE[,VALUE...]."
        )
    key, _, values = spec.partition("=")
    key = key.strip()
    items = [v.strip() for v in values.split(",") if v.strip()]
    if not key or not items:
        raise ValueError(
            f"Invalid {flag} value '{spec}': missing key or values."
        )
    return key, items


def _parse_scoped_assignment(
    spec: str, *, flag: str,
) -> Tuple[Optional[str], str, List[str]]:
    """Parse '[PLATE:]KEY=V1[,V2,...]' into ``(plate, key, values)``.

    Returns ``plate=None`` when no ``PLATE:`` prefix is provided, in
    which case the assignment applies to every row regardless of plate.
    """
    eq_idx = spec.find("=")
    if eq_idx < 0:
        raise ValueError(
            f"Invalid {flag} value '{spec}': expected [PLATE:]KEY=VALUE[,VALUE...]."
        )
    head = spec[:eq_idx]
    if ":" in head:
        plate_part, _, key_part = head.partition(":")
        plate = plate_part.strip()
        if not plate:
            raise ValueError(
                f"Invalid {flag} value '{spec}': empty plate prefix before ':'."
            )
        rest = f"{key_part}{spec[eq_idx:]}"
    else:
        plate = None
        rest = spec
    key, items = _parse_inline_assignment(rest, flag=flag)
    return plate, key, items


def _apply_inline_plate_map(df, args: argparse.Namespace):
    """Apply the inline ``--plate`` / ``--condition`` / ``--plate-replicate`` map.

    Mutates and returns a copy of *df* whose ``plate``,
    ``args.group_col`` and ``replicate_id`` columns reflect the
    command-line assignments.  Rows whose well or plate is not mentioned
    on the command line keep ``NaN`` in the affected column and will be
    dropped by downstream plotting functions.
    """
    df = df.copy()

    plate_value = getattr(args, "plate", None)
    if plate_value is not None:
        df["plate"] = plate_value

    cond_specs = getattr(args, "condition", None) or []
    if cond_specs:
        if "well_id" not in df.columns:
            raise ValueError(
                "--condition requires a 'well_id' column in the input CSV."
            )
        well_to_cond: dict = {}
        for spec in cond_specs:
            cond, wells = _parse_inline_assignment(spec, flag="--condition")
            for w in wells:
                if w in well_to_cond and well_to_cond[w] != cond:
                    raise ValueError(
                        f"Well '{w}' assigned to multiple conditions: "
                        f"'{well_to_cond[w]}' and '{cond}'."
                    )
                well_to_cond[w] = cond
        df[args.group_col] = df["well_id"].astype(str).map(well_to_cond)
        n_unmapped = int(df[args.group_col].isna().sum())
        if n_unmapped:
            log.warning(
                "--condition: %d row(s) had wells not in any flag; "
                "they will be excluded from grouped plots.",
                n_unmapped,
            )

    plate_rep_specs = getattr(args, "plate_replicate", None) or []
    well_rep_specs = getattr(args, "replicate", None) or []
    if plate_rep_specs and well_rep_specs:
        raise ValueError(
            "--plate-replicate and --replicate are mutually exclusive."
        )

    if plate_rep_specs:
        if "plate" not in df.columns:
            raise ValueError(
                "--plate-replicate requires a 'plate' column in the input "
                "CSV.  Pass --plate N to tag a single-plate CSV."
            )
        plate_to_rep: dict = {}
        for spec in plate_rep_specs:
            plate, reps = _parse_inline_assignment(spec, flag="--plate-replicate")
            if len(reps) != 1:
                raise ValueError(
                    f"--plate-replicate expects exactly one value, got '{spec}'."
                )
            if plate in plate_to_rep:
                raise ValueError(
                    f"Plate '{plate}' specified more than once."
                )
            plate_to_rep[plate] = reps[0]
        df["replicate_id"] = df["plate"].astype(str).map(plate_to_rep)
        n_unmapped = int(df["replicate_id"].isna().sum())
        if n_unmapped:
            log.warning(
                "--plate-replicate: %d row(s) had plates not in any flag; "
                "they will be excluded.",
                n_unmapped,
            )

    if well_rep_specs:
        if "well_id" not in df.columns:
            raise ValueError(
                "--replicate requires a 'well_id' column in the input CSV."
            )
        well_to_rep: dict = {}
        for spec in well_rep_specs:
            rep, wells = _parse_inline_assignment(spec, flag="--replicate")
            for w in wells:
                if w in well_to_rep and well_to_rep[w] != rep:
                    raise ValueError(
                        f"Well '{w}' assigned to multiple replicates: "
                        f"'{well_to_rep[w]}' and '{rep}'."
                    )
                well_to_rep[w] = rep
        df["replicate_id"] = df["well_id"].astype(str).map(well_to_rep)
        n_unmapped = int(df["replicate_id"].isna().sum())
        if n_unmapped:
            log.warning(
                "--replicate: %d row(s) had wells not in any flag; "
                "they will be excluded from per-replicate views.",
                n_unmapped,
            )

    # Default: derive a unique replicate id from (plate, well_id) if no
    # replicate column was supplied either in the CSV or via flags.
    if "replicate_id" not in df.columns and "well_id" in df.columns:
        if "plate" in df.columns:
            df["replicate_id"] = (
                df["plate"].astype(str) + "_" + df["well_id"].astype(str)
            )
        else:
            df["replicate_id"] = df["well_id"].astype(str)

    return df


# Friendly aliases so cmd users can avoid escaping '^' etc.
_MARKER_ALIASES = {
    "circle":        "o",
    "square":        "s",
    "triangle":      "^",
    "triangle_up":   "^",
    "triangle_down": "v",
    "diamond":       "D",
    "plus":          "P",
    "cross":         "X",
    "star":          "*",
}


def _resolve_marker(value: str) -> str:
    """Map alias names ('triangle') to matplotlib codes; else pass through."""
    return _MARKER_ALIASES.get(value.lower(), value)


def _resolve_shape_map(args):
    """Parse --point-shape-map specs into a {value: marker} dict.

    Accepts both matplotlib short codes (``o``, ``s``, ``^``, ...) and
    friendly aliases (``circle``, ``square``, ``triangle``, ...).
    Returns ``None`` when no flag is provided.
    """
    specs = getattr(args, "point_shape_map", None) or []
    if not specs:
        return None
    out: dict = {}
    for spec in specs:
        try:
            key, vals = _parse_inline_assignment(spec, flag="--point-shape-map")
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        if len(vals) != 1:
            raise ValueError(
                f"--point-shape-map expects exactly one marker, got '{spec}'."
            )
        if key in out:
            raise ValueError(f"Value '{key}' specified more than once in --point-shape-map.")
        out[key] = _resolve_marker(vals[0])
    return out


def _maybe_aggregate_replicates(df, *, pool: bool, group_col: str, time_col: str):
    """Hierarchical replicate aggregation, unless --pool is set.

    For ``pool=False`` (default), collapse each unique
    ``(group_col, time_col, bio_rep)`` triple into a single row whose
    metric values are the mean across that replicate's wells (which
    averages over technical replicates).  All other identifier columns
    are dropped to avoid ambiguity.

    For ``pool=True`` (or when no ``bio_rep`` column exists), return the
    DataFrame unchanged.
    """
    import pandas as pd

    if pool:
        return df
    if "bio_rep" not in df.columns:
        return df

    keys = [c for c in (group_col, time_col, "bio_rep") if c in df.columns]
    if not keys:
        return df

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in keys
    ]
    if not numeric_cols:
        return df

    return df.groupby(keys, dropna=False)[numeric_cols].mean().reset_index()


def _resolve_palette(args, groups):
    """Build a list of colours aligned to *groups* from --color flags.

    Returns ``None`` (matplotlib default palette) when no --color flag is
    provided.  Groups not mentioned by any flag fall back to the default
    cycle.
    """
    color_specs = getattr(args, "color", None) or []
    if not color_specs:
        return None
    color_map: dict = {}
    for spec in color_specs:
        group, vals = _parse_inline_assignment(spec, flag="--color")
        if len(vals) != 1:
            raise ValueError(
                f"--color expects exactly one value, got '{spec}'."
            )
        if group in color_map:
            raise ValueError(f"Group '{group}' specified more than once in --color.")
        color_map[group] = vals[0]

    # Build per-group palette aligned to *groups* order.  Groups missing
    # from --color fall back to the default cycle.
    default_cycle = [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7",
        "#E69F00", "#56B4E9", "#F0E442",
    ]
    palette = []
    cycle_idx = 0
    for g in groups:
        # Try several forms of the key so numeric DIVs (14, 14.0) match
        # the user's typed string ("14").
        candidates = [g, str(g)]
        try:
            if isinstance(g, float) and g.is_integer():
                candidates.append(str(int(g)))
        except (TypeError, AttributeError):
            pass
        match = next((color_map[c] for c in candidates if c in color_map), None)
        if match is not None:
            palette.append(match)
        else:
            palette.append(default_cycle[cycle_idx % len(default_cycle)])
            cycle_idx += 1
    return palette


def _resolve_metric_columns(
    df, requested, *, group_col: str, time_col: str,
):
    """Return the metric columns to plot, defaulting to all numeric columns."""
    import pandas as pd

    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            log.error("Metric column(s) not in CSV: %s", missing)
            return None
        return list(requested)

    excluded = set(_NON_METRIC_COLUMNS) | {group_col, time_col}
    metrics = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not metrics:
        log.error("No numeric metric columns inferred.  Pass --metric explicitly.")
        return None
    return metrics


def _cmd_plot_trajectory(args: argparse.Namespace) -> int:
    df = _load_summary_csv(args)
    if df is None:
        return 1

    try:
        df = _apply_inline_plate_map(df, args)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    metrics = _resolve_metric_columns(
        df, args.metric, group_col=args.group_col, time_col=args.time_col,
    )
    if metrics is None:
        return 1

    if getattr(args, "group_order", None):
        groups = list(args.group_order)
    else:
        groups = sorted(df[args.group_col].dropna().unique().tolist())
    try:
        palette = _resolve_palette(args, groups)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    df = _maybe_aggregate_replicates(
        df, pool=getattr(args, "pool", False),
        group_col=args.group_col, time_col=args.time_col,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from py_mea_axion.viz.trajectory import plot_metric_trajectory

    figsize = tuple(args.figsize) if getattr(args, "figsize", None) else (6.0, 4.0)
    for metric in metrics:
        try:
            fig = plot_metric_trajectory(
                df, metric=metric,
                time_col=args.time_col, group_col=args.group_col,
                groups=groups, palette=palette,
                figsize=figsize,
                ylabel=_label_for(metric),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Trajectory failed for %s: %s", metric, exc)
            continue
        fig.savefig(
            out_dir / f"trajectory_{metric}.png",
            dpi=600, bbox_inches="tight",
        )
        plt.close(fig)

    log.info("Saved %d trajectory figures to %s", len(metrics), out_dir)
    return 0


def _cmd_plot_timepoint(args: argparse.Namespace) -> int:
    df = _load_summary_csv(args)
    if df is None:
        return 1

    try:
        df = _apply_inline_plate_map(df, args)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    metrics = _resolve_metric_columns(
        df, args.metric, group_col=args.group_col, time_col=args.time_col,
    )
    if metrics is None:
        return 1

    divs = args.div if args.div is not None else sorted(df[args.time_col].unique())

    if getattr(args, "group_order", None):
        groups = list(args.group_order)
    else:
        groups = sorted(df[args.group_col].dropna().unique().tolist())
    try:
        palette = _resolve_palette(args, groups)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    df = _maybe_aggregate_replicates(
        df, pool=getattr(args, "pool", False),
        group_col=args.group_col, time_col=args.time_col,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from py_mea_axion.viz.comparison import plot_condition_violin

    point_hue_col = getattr(args, "point_hue", None)
    point_palette = None
    point_color_specs = getattr(args, "point_color", None) or []
    if point_color_specs:
        if not point_hue_col:
            log.error("--point-color requires --point-hue.")
            return 1
        point_palette = {}
        for spec in point_color_specs:
            try:
                key, vals = _parse_inline_assignment(spec, flag="--point-color")
            except ValueError as exc:
                log.error("%s", exc)
                return 1
            if len(vals) != 1:
                log.error("--point-color expects exactly one value, got '%s'.", spec)
                return 1
            point_palette[key] = vals[0]

    point_shape_col = getattr(args, "point_shape", None)
    try:
        point_shape_map = _resolve_shape_map(args)
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    if point_shape_map and not point_shape_col:
        log.error("--point-shape-map requires --point-shape.")
        return 1

    n_saved = 0
    for metric in metrics:
        metric_dir = out_dir / metric
        metric_dir.mkdir(exist_ok=True)
        for div in divs:
            figsize_t = tuple(args.figsize) if getattr(args, "figsize", None) else (5.0, 5.0)
            try:
                fig = plot_condition_violin(
                    df, metric=metric,
                    group_col=args.group_col,
                    time_col=args.time_col,
                    time_value=div,
                    groups=groups, palette=palette,
                    point_hue_col=point_hue_col,
                    point_palette=point_palette,
                    point_shape_col=point_shape_col,
                    point_shape_map=point_shape_map,
                    point_size=getattr(args, "point_size", 40.0),
                    show_point_legend=getattr(args, "show_point_legend", False),
                    compare_pairs=getattr(args, "compare", None),
                    stat_test=getattr(args, "test", "tukey"),
                    figsize=figsize_t,
                    ylabel=_label_for(metric),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Violin failed for %s @ %s=%s: %s",
                            metric, args.time_col, div, exc)
                continue
            try:
                tag = f"{int(div):02d}" if float(div).is_integer() else f"{div}"
            except (TypeError, ValueError):
                tag = str(div)
            fig.savefig(
                metric_dir / f"{args.time_col}_{tag}.png",
                dpi=600, bbox_inches="tight",
            )
            plt.close(fig)
            n_saved += 1

    log.info("Saved %d timepoint figures to %s", n_saved, out_dir)
    return 0


def _cmd_plot_pca(args: argparse.Namespace) -> int:
    df = _load_summary_csv(args)
    if df is None:
        return 1

    metrics = _resolve_metric_columns(
        df, args.metric, group_col=args.group_col, time_col=args.time_col,
    )
    if metrics is None:
        return 1

    if getattr(args, "group_order", None):
        groups = list(args.group_order)
    elif args.group_col in df.columns:
        groups = sorted(df[args.group_col].dropna().unique().tolist())
    else:
        groups = []
    try:
        palette = _resolve_palette(args, groups)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    df = _maybe_aggregate_replicates(
        df, pool=getattr(args, "pool", False),
        group_col=args.group_col, time_col=args.time_col,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from py_mea_axion.viz.pca import plot_pca

    figsize_p = tuple(args.figsize) if getattr(args, "figsize", None) else (6.0, 5.0)

    shape_col = getattr(args, "point_shape", None)
    try:
        shape_map = _resolve_shape_map(args)
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    if shape_map and not shape_col:
        log.error("--point-shape-map requires --point-shape.")
        return 1

    def _save_loadings(sub_df, *, suffix: str = "") -> None:
        from py_mea_axion.viz.pca import compute_pca_loadings, plot_pca_loadings
        try:
            loadings = compute_pca_loadings(sub_df, metric_cols=metrics, n_components=2)
        except Exception as exc:  # noqa: BLE001
            log.warning("Loadings computation failed%s: %s", suffix, exc)
            return
        csv_name = f"pca_loadings{suffix}.csv"
        fig_name = f"pca_loadings{suffix}.png"
        loadings.to_csv(out_dir / csv_name, float_format="%.6f")
        pretty = {col: _label_for(col) for col in metrics}
        fig_l = plot_pca_loadings(loadings, n_top=args.top, pretty_labels=pretty)
        fig_l.savefig(out_dir / fig_name, dpi=600, bbox_inches="tight")
        plt.close(fig_l)
        log.info("Saved loadings: %s, %s", csv_name, fig_name)
    if args.div is not None:
        divs = args.div
        n_saved = 0
        for div in divs:
            sub = df[df[args.time_col] == div]
            if len(sub) < 2:
                log.warning("PCA skipped for %s=%s (only %d rows).",
                            args.time_col, div, len(sub))
                continue
            try:
                fig = plot_pca(
                    sub, metric_cols=metrics,
                    group_col=args.group_col,
                    groups=groups or None, palette=palette,
                    shape_col=shape_col, shape_map=shape_map,
                    figsize=figsize_p,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("PCA failed for %s=%s: %s", args.time_col, div, exc)
                continue
            try:
                tag = f"{int(div):02d}" if float(div).is_integer() else f"{div}"
            except (TypeError, ValueError):
                tag = str(div)
            fig.savefig(
                out_dir / f"pca_{args.time_col}_{tag}.png",
                dpi=600, bbox_inches="tight",
            )
            plt.close(fig)
            n_saved += 1

            if args.loadings:
                _save_loadings(sub, suffix=f"_{args.time_col}_{tag}")
        log.info("Saved %d PCA figures to %s", n_saved, out_dir)
    else:
        try:
            fig = plot_pca(
                df, metric_cols=metrics,
                group_col=args.group_col,
                groups=groups or None, palette=palette,
                shape_col=shape_col, shape_map=shape_map,
                figsize=figsize_p,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("PCA failed: %s", exc)
            return 1
        fig.savefig(out_dir / "pca.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved PCA figure to %s", out_dir)

        if args.loadings:
            _save_loadings(df)

    return 0


# ── build subcommand ─────────────────────────────────────────────────────────

# Filename pattern: must contain Plate<N>_DIV<N> somewhere (case-insensitive).
_FILENAME_PATTERN = re.compile(r"Plate(\d+)_DIV(\d+)", re.IGNORECASE)

# Per-well spec: optional Plate/P prefix, then digits, '_', well, '=', body.
_WELL_KEY_PATTERN = re.compile(
    r"^(?:Plate|P)?(\d+)_([A-Z]\d+)$", re.IGNORECASE,
)


def _parse_well_spec(spec: str, *, flag: str = "--well") -> Dict[str, object]:
    """Parse '<plate>_<well>=COND[,Bn[,Tn[,batch]]]' into a row dict."""
    if "=" not in spec:
        raise ValueError(
            f"Invalid {flag} value '{spec}': expected PLATE_WELL=COND[,Bn,Tn,BATCH]."
        )
    head, _, tail = spec.partition("=")
    head = head.strip()

    m = _WELL_KEY_PATTERN.match(head)
    if not m:
        raise ValueError(
            f"Invalid {flag} key '{head}': expected like '1_A1' or 'Plate1_A1'."
        )
    plate = int(m.group(1))
    well = m.group(2).upper()

    parts = [p.strip() for p in tail.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError(
            f"Invalid {flag} value '{spec}': missing condition (after '=')."
        )

    condition = parts[0]
    bio_rep: Optional[str] = parts[1] if len(parts) >= 2 else None
    tech_rep: Optional[str] = parts[2] if len(parts) >= 3 else None
    batch: Optional[str] = parts[3] if len(parts) >= 4 else None

    if bio_rep is not None and not re.match(r"^B\d+$", bio_rep, re.IGNORECASE):
        raise ValueError(
            f"Invalid {flag} biological-replicate '{bio_rep}' for {head}: "
            f"must be 'B<digits>' (e.g. B1)."
        )
    if tech_rep is not None and not re.match(r"^T\d+$", tech_rep, re.IGNORECASE):
        raise ValueError(
            f"Invalid {flag} technical-replicate '{tech_rep}' for {head}: "
            f"must be 'T<digits>' (e.g. T1)."
        )

    return {
        "plate":     plate,
        "well_id":   well,
        "condition": condition,
        "bio_rep":   bio_rep,
        "tech_rep":  tech_rep,
        "batch":     batch,
    }


def _build_layout_from_wells(specs: List[str]) -> "pd.DataFrame":
    """Build a (plate, well_id, condition, bio_rep, tech_rep, batch) frame."""
    import pandas as pd

    rows = []
    seen_keys: set = set()
    for spec in specs:
        row = _parse_well_spec(spec)
        key = (row["plate"], row["well_id"])
        if key in seen_keys:
            raise ValueError(
                f"Duplicate --well entry for Plate{row['plate']}_{row['well_id']}."
            )
        seen_keys.add(key)
        rows.append(row)
    if not rows:
        raise ValueError("No --well entries provided.")
    return pd.DataFrame(rows)


def _load_layout_from_csv(path: Path) -> "pd.DataFrame":
    """Read a previously-built master.csv and extract its layout columns."""
    import pandas as pd

    df = pd.read_csv(path)
    required = ["plate", "well_id", "condition"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"--layout-from CSV is missing required column(s): {missing}"
        )
    keep = ["plate", "well_id", "condition"]
    for c in ("bio_rep", "tech_rep", "batch"):
        if c in df.columns:
            keep.append(c)
    layout = df[keep].drop_duplicates(subset=["plate", "well_id"]).reset_index(drop=True)
    return layout


def _extract_plate_div(filename: str) -> Optional[Tuple[int, int]]:
    """Return ``(plate, div)`` extracted from the filename, or None."""
    m = _FILENAME_PATTERN.search(filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _add_build_parser(sub) -> None:
    p = sub.add_parser(
        "build",
        help=(
            "Process a folder of .spk files into a single master CSV "
            "labelled with condition / replicate / batch."
        ),
        description=(
            "Build a master CSV by running the analysis pipeline on every "
            ".spk file in a folder, applying a per-well layout you supply "
            "on the command line.  Plate and DIV are extracted from each "
            "filename via the 'Plate<N>_DIV<N>' convention."
        ),
    )
    p.add_argument(
        "spk_path",
        metavar="SPK_FOLDER",
        help=(
            "Folder containing .spk files (or a glob pattern).  "
            "Files must include 'Plate<N>_DIV<N>' in their name."
        ),
    )
    p.add_argument(
        "--well",
        action="append",
        default=None,
        metavar="PLATE_WELL=COND[,Bn,Tn,BATCH]",
        help=(
            "Per-well layout entry, repeatable.  Example: "
            "--well 1_A1=SCRM,B1,T1 --well 1_A2=LGI2_KD4,B1,T2.  "
            "Bio rep 'B<n>' and tech rep 'T<n>' are optional; batch "
            "label is the optional 4th value."
        ),
    )
    p.add_argument(
        "--layout-from",
        default=None,
        metavar="CSV",
        help=(
            "Reuse the layout (condition + replicate + batch columns) "
            "from a previously-built master CSV instead of typing --well "
            "flags.  Mutually exclusive with --well."
        ),
    )
    p.add_argument(
        "--fs-override",
        type=float,
        default=None,
        metavar="HZ",
        help="Force a specific sampling frequency in Hz (e.g. 12500).",
    )
    p.add_argument(
        "--active-threshold",
        type=float,
        default=0.1,
        metavar="HZ",
        help="MFR threshold for classifying an electrode as active.  Default: 0.1.",
    )
    p.add_argument(
        "--max-isi",
        type=float,
        default=0.1,
        metavar="S",
        help="Max within-burst ISI (s) for burst detection.  Default: 0.1.",
    )
    p.add_argument(
        "--min-spikes",
        type=int,
        default=5,
        metavar="N",
        help="Minimum spikes per burst.  Default: 5.",
    )
    p.add_argument(
        "--sttc-dt",
        type=float,
        default=0.05,
        metavar="S",
        help="STTC coincidence window half-width (s).  Default: 0.05.",
    )
    p.add_argument(
        "--out", "-o",
        required=True,
        metavar="CSV",
        help="Output master CSV path.",
    )
    p.add_argument(
        "--no-script",
        action="store_true",
        help="Don't auto-save the equivalent .build.sh next to the CSV.",
    )


def _cmd_build(args: argparse.Namespace) -> int:
    import pandas as pd

    if args.well and args.layout_from:
        log.error("--well and --layout-from are mutually exclusive.")
        return 1
    if not args.well and not args.layout_from:
        log.error("Need --well flags or --layout-from to define the layout.")
        return 1

    try:
        if args.layout_from:
            layout_path = Path(args.layout_from)
            if not layout_path.exists():
                log.error("--layout-from file not found: %s", layout_path)
                return 1
            layout_df = _load_layout_from_csv(layout_path)
        else:
            layout_df = _build_layout_from_wells(args.well)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    spk_in = Path(args.spk_path)
    if "*" in args.spk_path or "?" in args.spk_path:
        spk_paths = sorted(Path().glob(args.spk_path))
    elif spk_in.is_dir():
        spk_paths = sorted(spk_in.glob("*.spk"))
    elif spk_in.is_file() and spk_in.suffix == ".spk":
        spk_paths = [spk_in]
    else:
        log.error("Input is not a folder, glob, or .spk file: %s", spk_in)
        return 1

    if not spk_paths:
        log.error("No .spk files found at: %s", args.spk_path)
        return 1

    log.info("Processing %d .spk file(s).", len(spk_paths))

    burst_kwargs = {
        "max_isi_s": args.max_isi,
        "min_spikes": args.min_spikes,
    }

    all_rows = []
    n_skipped_pattern = 0
    n_skipped_layout = 0
    for spk in spk_paths:
        info = _extract_plate_div(spk.name)
        if info is None:
            log.warning("Skipping (no Plate<N>_DIV<N> in name): %s", spk.name)
            n_skipped_pattern += 1
            continue
        plate, div = info
        plate_layout = layout_df[layout_df["plate"] == plate].drop(columns=["plate"])
        if plate_layout.empty:
            log.warning("Skipping %s: no layout entries for Plate %d.", spk.name, plate)
            n_skipped_layout += 1
            continue

        log.info("  Plate %d  DIV %3d  %s", plate, div, spk.name)
        try:
            exp = MEAExperiment(
                spk,
                metadata=plate_layout,
                fs_override=args.fs_override,
                active_threshold_hz=args.active_threshold,
                min_active_electrodes=0,
                burst_kwargs=burst_kwargs,
                sttc_dt_s=args.sttc_dt,
            ).run()
        except Exception as exc:  # noqa: BLE001
            log.error("    Failed: %s", exc)
            continue

        js = exp.joined_summary()
        # Keep only wells covered by the layout (drop spurious empty wells).
        if "condition" in js.columns:
            js = js[js["condition"].notna()].copy()
        js["plate"] = plate
        js["DIV"] = div
        all_rows.append(js)

    if not all_rows:
        log.error("No usable recordings.  Skipped: %d (pattern), %d (layout).",
                  n_skipped_pattern, n_skipped_layout)
        return 1

    df = pd.concat(all_rows, ignore_index=True)

    # Auto-derive bio_rep / tech_rep when absent from the layout.
    if "bio_rep" not in df.columns or df["bio_rep"].isna().all():
        df["bio_rep"] = df["plate"].astype(str) + "_" + df["well_id"].astype(str)
    else:
        # Fill any missing bio_rep entries (mixed layout).
        mask = df["bio_rep"].isna()
        df.loc[mask, "bio_rep"] = (
            df.loc[mask, "plate"].astype(str) + "_" + df.loc[mask, "well_id"].astype(str)
        )
    if "tech_rep" not in df.columns or df["tech_rep"].isna().all():
        df["tech_rep"] = "T1"
    else:
        df["tech_rep"] = df["tech_rep"].fillna("T1")

    # Fill NaN metric values with 0 — absence of bursts / network bursts
    # is a true zero, not missing data.  Use 'condition' / 'DIV' as the
    # group/time axes (those are what the plot subcommands default to).
    df = _fill_metric_nans(df, group_col="condition", time_col="DIV")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    log.info("Saved master CSV: %s  (%d rows)", out_path, len(df))

    if not args.no_script:
        _save_build_script(out_path)

    return 0


def _save_build_script(out_csv: Path) -> None:
    """Save the original argv as a runnable shell script next to the CSV.

    Flags and their values are grouped on the same line for readability.
    """
    script = out_csv.with_suffix(".build.sh")
    # Always emit the canonical CLI name for the script — the original
    # argv[0] is often a python interpreter path that wouldn't replay.
    argv = ["mea-axion"] + list(sys.argv[1:])

    # Group: subcommand + positional on the first line; each --flag with its
    # value on its own line.
    lines: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            lines.append(f"{shlex.quote(token)} {shlex.quote(argv[i + 1])}")
            i += 2
        else:
            lines.append(shlex.quote(token))
            i += 1

    # First two tokens (executable + subcommand + positional) on one line.
    head, tail = [], []
    consumed = 0
    for ln in lines:
        if consumed < 3 and not ln.startswith("'--'") and not ln.startswith("--"):
            head.append(ln)
            consumed += 1
        else:
            tail.append(ln)

    body = " ".join(head)
    if tail:
        body += " \\\n  " + " \\\n  ".join(tail)

    text = (
        "#!/usr/bin/env bash\n"
        "# Auto-generated by mea-axion build.\n"
        f"# Re-run to rebuild {out_csv.name}.\n"
        "set -euo pipefail\n\n"
        f"{body}\n"
    )
    script.write_text(text, encoding="utf-8")
    try:
        script.chmod(0o755)
    except OSError:
        pass
    log.info("Saved build script:  %s", script)


# ── show-layout subcommand ───────────────────────────────────────────────────

def _add_show_layout_parser(sub) -> None:
    p = sub.add_parser(
        "show-layout",
        help="Print --well flags reconstructed from a master CSV.",
        description=(
            "Read a master CSV and print one --well flag per (plate, well) "
            "row to stdout, ready to copy-paste into a future build command."
        ),
    )
    p.add_argument(
        "csv_file",
        metavar="CSV_FILE",
        help="Path to a previously-built master CSV.",
    )


def _cmd_show_layout(args: argparse.Namespace) -> int:
    import pandas as pd

    csv = Path(args.csv_file)
    if not csv.exists():
        log.error("File not found: %s", csv)
        return 1
    try:
        layout = _load_layout_from_csv(csv)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    layout = layout.sort_values(["plate", "well_id"])
    for _, row in layout.iterrows():
        plate = int(row["plate"])
        well = str(row["well_id"])
        cond = str(row["condition"])
        parts = [cond]
        for c in ("bio_rep", "tech_rep", "batch"):
            if c in layout.columns:
                v = row.get(c)
                if pd.notna(v) and str(v) != "":
                    parts.append(str(v))
                else:
                    # Stop at the first missing value so positional shape stays valid.
                    break
        body = ",".join(parts)
        print(f"--well {plate}_{well}={body}")
    return 0


# ── stats subcommand ─────────────────────────────────────────────────────────

def _add_stats_parser(sub) -> None:
    p = sub.add_parser(
        "stats",
        help=(
            "Run pairwise statistical comparisons across conditions and "
            "write a tidy CSV of results."
        ),
        description=(
            "Run Tukey HSD (default), Mann-Whitney U, or Kruskal-Wallis H "
            "tests on a master CSV.  Output is a long-format CSV with one "
            "row per (metric, time-point, pair) combination, suitable for "
            "supplementary tables."
        ),
    )
    _common_csv_args(p)
    _add_plate_map_args(p)
    p.add_argument(
        "--metric", "-M",
        nargs="+",
        default=None,
        metavar="COL",
        help=(
            "Well-level metric column(s) to test.  Default: every numeric "
            "column other than identifier columns."
        ),
    )
    p.add_argument(
        "--div",
        type=float,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "One or more time-point values to test.  Default: every "
            "unique value of the time column."
        ),
    )
    p.add_argument(
        "--div-min",
        type=float,
        default=None,
        metavar="N",
        help="Minimum value of the time column to include.",
    )
    p.add_argument(
        "--div-max",
        type=float,
        default=None,
        metavar="N",
        help="Maximum value of the time column to include.",
    )
    p.add_argument(
        "--test",
        choices=("tukey", "mannwhitney", "kruskal"),
        default="tukey",
        help=(
            "Which statistical test to run.  Default: tukey (Tukey HSD "
            "pairwise, parametric, all-pairs).  mannwhitney auto-selects "
            "for 2 groups; kruskal for 3+ groups."
        ),
    )
    p.add_argument(
        "--compare",
        action="append",
        nargs=2,
        default=None,
        metavar=("GROUP_A", "GROUP_B"),
        help=(
            "Restrict pairwise comparisons to specific pairs.  Repeatable. "
            "Only used by --test=tukey.  Example: --compare LGI2_KD4 SCRM "
            "--compare LGI2_KD5 SCRM."
        ),
    )
    p.add_argument(
        "--out", "-o",
        required=True,
        metavar="CSV",
        help="Output CSV path (long-format, one row per pair).",
    )


def _significance_label_text(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _cmd_stats(args: argparse.Namespace) -> int:
    import pandas as pd
    from py_mea_axion.stats.compare import pairwise_test

    df = _load_summary_csv(args)
    if df is None:
        return 1

    try:
        df = _apply_inline_plate_map(df, args)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    metrics = _resolve_metric_columns(
        df, args.metric, group_col=args.group_col, time_col=args.time_col,
    )
    if metrics is None:
        return 1

    if getattr(args, "group_order", None):
        groups = list(args.group_order)
    else:
        groups = sorted(df[args.group_col].dropna().unique().tolist())

    df = _maybe_aggregate_replicates(
        df, pool=getattr(args, "pool", False),
        group_col=args.group_col, time_col=args.time_col,
    )

    if args.div is not None:
        time_values = list(args.div)
    elif args.time_col in df.columns:
        time_values = sorted(df[args.time_col].dropna().unique().tolist())
    else:
        time_values = [None]

    compare_pairs = (
        [tuple(p) for p in args.compare] if getattr(args, "compare", None)
        else None
    )

    rows: List[Dict[str, object]] = []
    for metric in metrics:
        for tv in time_values:
            sub = df if tv is None else df[df[args.time_col] == tv]
            if len(sub) < 2:
                continue

            res = pairwise_test(
                sub, metric=metric,
                group_col=args.group_col,
                test=args.test,
                groups=groups,
                pairs=compare_pairs,
            )
            if res.empty:
                continue
            # Per-group n for the rows we're emitting.
            n_per_group: Dict[str, int] = {
                g: int(((sub[args.group_col] == g) & sub[metric].notna()).sum())
                for g in groups
            }
            test_label = {
                "tukey":       "tukey_hsd",
                "mannwhitney": "mannwhitney_pairwise",
                "kruskal":     "kruskal_dunn_pairwise",
            }.get(args.test, args.test)
            for _, row in res.iterrows():
                ga, gb = row["group_a"], row["group_b"]
                p_adj = float(row["p_adj"])
                rows.append({
                    "metric":        metric,
                    args.time_col:   tv,
                    "test":          test_label,
                    "group_a":       ga,
                    "group_b":       gb,
                    "mean_diff":     float(row["mean_diff"]),
                    "p_adj":         p_adj,
                    "significance":  _significance_label_text(p_adj),
                    "n_a":           n_per_group.get(ga, 0),
                    "n_b":           n_per_group.get(gb, 0),
                })

    if not rows:
        log.error("No statistical results were produced.  Check your --metric, "
                  "--div, --filter and --group-order arguments.")
        return 1

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, float_format="%.6f")
    log.info("Saved %d rows of statistical results to %s",
             len(out_df), out_path)
    return 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_version() -> str:
    try:
        from py_mea_axion import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"
