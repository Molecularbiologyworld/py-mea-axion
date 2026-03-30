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
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

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
        _save_figures(exp, fig_dir)

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


def _save_figures(exp, fig_dir: Path) -> None:
    """Export standard PNG figures for every well."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for well_id in exp.wells:
        # Electrode MFR heatmap.
        fig = exp.plot_heatmap(well_id, metric="mfr_hz")
        fig.savefig(fig_dir / f"{well_id}_heatmap_mfr.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Burst raster.
        fig = exp.plot_raster(well_id)
        fig.savefig(fig_dir / f"{well_id}_raster.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # STTC matrix.
        fig = exp.plot_sttc(well_id)
        fig.savefig(fig_dir / f"{well_id}_sttc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Network-burst timeline.
        fig = exp.plot_network_timeline(well_id)
        fig.savefig(fig_dir / f"{well_id}_network_timeline.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    log.info("Saved %d figures to %s", len(exp.wells) * 4, fig_dir)

    # Trajectory plot (only when metadata provides condition + DIV).
    if exp.metadata is not None:
        js = exp.joined_summary()
        if "condition" in js.columns and "DIV" in js.columns:
            for metric in ("mean_mfr_active_hz", "mean_sttc", "burst_rate_hz"):
                try:
                    fig = exp.plot_trajectory(metric)
                    fig.savefig(
                        fig_dir / f"trajectory_{metric}.png",
                        dpi=150, bbox_inches="tight",
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_version() -> str:
    try:
        from py_mea_axion import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"
