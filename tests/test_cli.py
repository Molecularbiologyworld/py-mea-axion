"""
tests/test_cli.py
==================
Tests for py_mea_axion.cli.

Tests are split into:
  - Parser tests: verify argparse accepts/rejects the right arguments.
  - Integration tests: drive the subcommand functions against synthetic
    data using tmp_path — no real .spk file required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from py_mea_axion.cli import (
    _cmd_run,
    _cmd_summary,
    _print_summary,
    _save_csvs,
    build_parser,
    main,
)
from py_mea_axion.pipeline import MEAExperiment


# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(99)
T = 30.0

_SPIKES = {
    "A1_11": np.sort(RNG.uniform(0, T, 60)),
    "A1_12": np.sort(RNG.uniform(0, T, 45)),
    "B1_11": np.sort(RNG.uniform(0, T, 50)),
    "B1_12": np.sort(RNG.uniform(0, T, 30)),
}

_META = pd.DataFrame({
    "well_id":      ["A1", "B1"],
    "condition":    ["WT", "KD"],
    "DIV":          [14, 14],
    "replicate_id": ["r1", "r2"],
})


@pytest.fixture(scope="module")
def ran_exp():
    exp = MEAExperiment.from_spikes(_SPIKES, total_time_s=T, metadata=_META)
    exp.run()
    return exp


@pytest.fixture()
def fake_spk(tmp_path):
    """A zero-byte file that exists (only used to test path checks)."""
    p = tmp_path / "fake.spk"
    p.write_bytes(b"")
    return p


# ── build_parser ──────────────────────────────────────────────────────────────

class TestBuildParser:
    def test_run_subcommand(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk"])
        assert args.command == "run"
        assert args.spk_file == "rec.spk"

    def test_summary_subcommand(self):
        p = build_parser()
        args = p.parse_args(["summary", "rec.spk"])
        assert args.command == "summary"

    def test_no_subcommand_exits(self):
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_wells_flag(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk", "--wells", "A1", "B2"])
        assert args.wells == ["A1", "B2"]

    def test_wells_short_flag(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk", "-w", "A1"])
        assert args.wells == ["A1"]

    def test_fs_override(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk", "--fs-override", "12500"])
        assert args.fs_override == pytest.approx(12500.0)

    def test_out_flag(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk", "--out", "/tmp/results"])
        assert args.out == "/tmp/results"

    def test_metadata_flag(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk", "--metadata", "plate.csv"])
        assert args.metadata == "plate.csv"

    def test_metadata_short(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk", "-m", "plate.csv"])
        assert args.metadata == "plate.csv"

    def test_no_figures_flag(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk", "--no-figures"])
        assert args.no_figures is True

    def test_no_figures_default_false(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk"])
        assert args.no_figures is False

    def test_max_isi(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk", "--max-isi", "0.05"])
        assert args.max_isi == pytest.approx(0.05)

    def test_min_spikes(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk", "--min-spikes", "3"])
        assert args.min_spikes == 3

    def test_sttc_dt(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk", "--sttc-dt", "0.1"])
        assert args.sttc_dt == pytest.approx(0.1)

    def test_active_threshold(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk", "--active-threshold", "0.5"])
        assert args.active_threshold == pytest.approx(0.5)

    def test_version_flag(self, capsys):
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--version"])


# ── _save_csvs ────────────────────────────────────────────────────────────────

class TestSaveCsvs:
    def test_creates_spike_metrics_csv(self, ran_exp, tmp_path):
        _save_csvs(ran_exp, tmp_path)
        assert (tmp_path / "spike_metrics.csv").exists()

    def test_creates_burst_table_csv(self, ran_exp, tmp_path):
        _save_csvs(ran_exp, tmp_path)
        assert (tmp_path / "burst_table.csv").exists()

    def test_creates_well_summary_csv(self, ran_exp, tmp_path):
        _save_csvs(ran_exp, tmp_path)
        assert (tmp_path / "well_summary.csv").exists()

    def test_spike_metrics_content(self, ran_exp, tmp_path):
        _save_csvs(ran_exp, tmp_path)
        df = pd.read_csv(tmp_path / "spike_metrics.csv")
        assert "well_id" in df.columns
        assert len(df) == len(_SPIKES)

    def test_well_summary_content(self, ran_exp, tmp_path):
        _save_csvs(ran_exp, tmp_path)
        df = pd.read_csv(tmp_path / "well_summary.csv")
        assert "mean_mfr_active_hz" in df.columns
        assert len(df) == 2   # A1, B1


# ── _print_summary ────────────────────────────────────────────────────────────

class TestPrintSummary:
    def test_prints_to_stdout(self, ran_exp, capsys):
        _print_summary(ran_exp)
        out = capsys.readouterr().out
        assert "well_id" in out

    def test_contains_well_ids(self, ran_exp, capsys):
        _print_summary(ran_exp)
        out = capsys.readouterr().out
        assert "A1" in out
        assert "B1" in out

    def test_contains_metric_name(self, ran_exp, capsys):
        _print_summary(ran_exp)
        out = capsys.readouterr().out
        assert "mean_mfr_active_hz" in out


# ── _cmd_run ──────────────────────────────────────────────────────────────────

class TestCmdRun:
    def _make_args(self, spk_path, out_dir, **overrides):
        """Build a Namespace that mimics argparse output for 'run'."""
        import argparse
        defaults = dict(
            spk_file=str(spk_path),
            wells=None,
            fs_override=None,
            active_threshold=0.1,
            metadata=None,
            out=str(out_dir),
            max_isi=0.1,
            min_spikes=5,
            sttc_dt=0.05,
            no_figures=True,   # skip figures by default in unit tests
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_missing_file_returns_one(self, tmp_path):
        import argparse
        args = argparse.Namespace(
            spk_file=str(tmp_path / "nonexistent.spk"),
            wells=None, fs_override=None, active_threshold=0.1,
            metadata=None, out=str(tmp_path / "out"),
            max_isi=0.1, min_spikes=5, sttc_dt=0.05, no_figures=True,
        )
        assert _cmd_run(args) == 1

    def test_run_creates_csvs(self, fake_spk, tmp_path):
        out = tmp_path / "out"

        # Patch MEAExperiment so we don't need a real .spk file.
        mock_exp = MagicMock()
        mock_exp.wells = ["A1"]
        mock_exp.spike_metrics = MEAExperiment.from_spikes(
            _SPIKES, total_time_s=T
        ).run().spike_metrics
        mock_exp.burst_table = MEAExperiment.from_spikes(
            _SPIKES, total_time_s=T
        ).run().burst_table
        mock_exp.well_summary = MEAExperiment.from_spikes(
            _SPIKES, total_time_s=T
        ).run().well_summary
        mock_exp.run.return_value = mock_exp

        with patch("py_mea_axion.cli.MEAExperiment", return_value=mock_exp):
            args = self._make_args(fake_spk, out)
            rc = _cmd_run(args)

        assert rc == 0
        assert (out / "spike_metrics.csv").exists()
        assert (out / "burst_table.csv").exists()
        assert (out / "well_summary.csv").exists()

    def test_run_default_out_dir(self, fake_spk, tmp_path):
        """When --out is omitted, output goes next to the .spk file."""
        mock_exp = MagicMock()
        mock_exp.wells = ["A1"]
        mock_exp.spike_metrics = pd.DataFrame(columns=["well_id"])
        mock_exp.burst_table = pd.DataFrame(columns=["well_id"])
        mock_exp.well_summary = pd.DataFrame(columns=["well_id"])
        mock_exp.run.return_value = mock_exp

        import argparse
        args = argparse.Namespace(
            spk_file=str(fake_spk),
            wells=None, fs_override=None, active_threshold=0.1,
            metadata=None,
            out=None,                  # <-- default: derive from spk_file
            max_isi=0.1, min_spikes=5, sttc_dt=0.05, no_figures=True,
        )
        with patch("py_mea_axion.cli.MEAExperiment", return_value=mock_exp):
            rc = _cmd_run(args)

        assert rc == 0
        expected = fake_spk.parent / fake_spk.stem
        assert expected.exists()

    def test_no_figures_skips_figure_dir(self, fake_spk, tmp_path):
        out = tmp_path / "out"
        mock_exp = MagicMock()
        mock_exp.wells = []
        mock_exp.spike_metrics = pd.DataFrame(columns=["well_id"])
        mock_exp.burst_table = pd.DataFrame(columns=["well_id"])
        mock_exp.well_summary = pd.DataFrame(columns=["well_id"])
        mock_exp.run.return_value = mock_exp

        with patch("py_mea_axion.cli.MEAExperiment", return_value=mock_exp):
            args = self._make_args(fake_spk, out, no_figures=True)
            _cmd_run(args)

        assert not (out / "figures").exists()


# ── _cmd_summary ──────────────────────────────────────────────────────────────

class TestCmdSummary:
    def test_missing_file_returns_one(self, tmp_path):
        import argparse
        args = argparse.Namespace(
            spk_file=str(tmp_path / "nope.spk"),
            wells=None, fs_override=None, active_threshold=0.1,
        )
        assert _cmd_summary(args) == 1

    def test_summary_prints_output(self, fake_spk, capsys):
        mock_exp = MagicMock()
        mock_exp.well_summary = pd.DataFrame({
            "well_id": ["A1"],
            "n_active": [4],
            "mean_mfr_active_hz": [1.5],
            "mean_sttc": [0.3],
            "burst_rate_hz": [0.1],
            "mean_burst_duration_s": [0.5],
            "n_network_bursts": [2],
            "mean_cv_isi": [1.2],
            "n_electrodes": [4],
        })
        mock_exp.run.return_value = mock_exp

        import argparse
        args = argparse.Namespace(
            spk_file=str(fake_spk),
            wells=None, fs_override=None, active_threshold=0.1,
        )

        with patch("py_mea_axion.cli.MEAExperiment", return_value=mock_exp):
            rc = _cmd_summary(args)

        assert rc == 0
        out = capsys.readouterr().out
        assert "A1" in out


# ── main() ────────────────────────────────────────────────────────────────────

class TestMain:
    def test_main_help_raises_system_exit(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_main_missing_file(self, tmp_path):
        rc = main(["run", str(tmp_path / "missing.spk"), "--no-figures",
                   "--out", str(tmp_path / "out")])
        assert rc == 1

    def test_main_summary_missing_file(self, tmp_path):
        rc = main(["summary", str(tmp_path / "missing.spk")])
        assert rc == 1

    def test_main_run_success(self, fake_spk, tmp_path):
        out = tmp_path / "out"
        mock_exp = MagicMock()
        mock_exp.wells = []
        mock_exp.spike_metrics = pd.DataFrame(columns=["well_id"])
        mock_exp.burst_table = pd.DataFrame(columns=["well_id"])
        mock_exp.well_summary = pd.DataFrame(columns=["well_id"])
        mock_exp.run.return_value = mock_exp

        with patch("py_mea_axion.cli.MEAExperiment", return_value=mock_exp):
            rc = main([
                "run", str(fake_spk),
                "--out", str(out),
                "--no-figures",
            ])
        assert rc == 0

    def test_main_summary_success(self, fake_spk, capsys):
        mock_exp = MagicMock()
        mock_exp.well_summary = pd.DataFrame({
            "well_id": ["A1"],
            "n_active": [4],
            "mean_mfr_active_hz": [1.5],
            "mean_sttc": [0.3],
            "burst_rate_hz": [0.1],
            "mean_burst_duration_s": [0.5],
            "n_network_bursts": [2],
            "mean_cv_isi": [1.2],
            "n_electrodes": [4],
        })
        mock_exp.run.return_value = mock_exp

        with patch("py_mea_axion.cli.MEAExperiment", return_value=mock_exp):
            rc = main(["summary", str(fake_spk)])
        assert rc == 0


# ── Real .spk smoke test ──────────────────────────────────────────────────────

class TestRealSpkSmoke:
    def test_run_on_real_file(self, tmp_path):
        spk = Path("LGI2 KD data/20251004_LGI2 KD_Plate 1_D28N(000).spk")
        if not spk.exists():
            pytest.skip("Real .spk file not available")

        rc = main([
            "run", str(spk),
            "--wells", "A1",
            "--fs-override", "12500",
            "--out", str(tmp_path / "out"),
            "--no-figures",
        ])
        assert rc == 0
        assert (tmp_path / "out" / "well_summary.csv").exists()
