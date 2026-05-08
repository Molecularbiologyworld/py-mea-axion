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
    _apply_inline_plate_map,
    _build_layout_from_wells,
    _cmd_run,
    _cmd_summary,
    _extract_plate_div,
    _fill_metric_nans,
    _load_layout_from_csv,
    _maybe_aggregate_replicates,
    _parse_inline_assignment,
    _parse_well_spec,
    _print_summary,
    _resolve_palette,
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

    def test_time_start(self):
        p = build_parser()
        args = p.parse_args(["run", "rec.spk", "--time-start", "300"])
        assert args.time_start == pytest.approx(300.0)

    def test_time_end(self):
        p = build_parser()
        args = p.parse_args(["summary", "rec.spk", "--time-end", "600"])
        assert args.time_end == pytest.approx(600.0)

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
            time_start=None,
            time_end=None,
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
            time_start=None, time_end=None,
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
            time_start=None, time_end=None,
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
            time_start=None, time_end=None,
        )
        assert _cmd_summary(args) == 1

    def test_summary_prints_output(self, fake_spk, capsys):
        mock_exp = MagicMock()
        mock_exp.well_summary = pd.DataFrame({
            "well_id": ["A1"],
            "n_active": [4],
            "mean_mfr_active_hz": [1.5],
            "mean_sttc": [0.3],
            "burst_freq_avg": [0.1],
            "burst_duration_avg": [0.5],
            "n_network_bursts": [2],
            "isi_cv_avg": [1.2],
            "n_electrodes": [4],
        })
        mock_exp.run.return_value = mock_exp

        import argparse
        args = argparse.Namespace(
            spk_file=str(fake_spk),
            wells=None, fs_override=None, active_threshold=0.1,
            time_start=None, time_end=None,
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
            "burst_freq_avg": [0.1],
            "burst_duration_avg": [0.5],
            "n_network_bursts": [2],
            "isi_cv_avg": [1.2],
            "n_electrodes": [4],
        })
        mock_exp.run.return_value = mock_exp

        with patch("py_mea_axion.cli.MEAExperiment", return_value=mock_exp):
            rc = main(["summary", str(fake_spk)])
        assert rc == 0


# ── Inline plate-map flags ────────────────────────────────────────────────────

class TestParseInlineAssignment:
    def test_single_value(self):
        assert _parse_inline_assignment("rep01=A1", flag="--x") == ("rep01", ["A1"])

    def test_multiple_values(self):
        assert _parse_inline_assignment(
            "SCRM=A1,A2,A3", flag="--condition",
        ) == ("SCRM", ["A1", "A2", "A3"])

    def test_strips_whitespace(self):
        assert _parse_inline_assignment(
            " SCRM = A1 , A2 ", flag="--condition",
        ) == ("SCRM", ["A1", "A2"])

    def test_drops_empty_items(self):
        assert _parse_inline_assignment(
            "SCRM=A1,,A2,", flag="--condition",
        ) == ("SCRM", ["A1", "A2"])

    def test_no_equals_raises(self):
        with pytest.raises(ValueError, match="expected KEY=VALUE"):
            _parse_inline_assignment("nope", flag="--condition")

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="missing key or values"):
            _parse_inline_assignment("=A1", flag="--condition")

    def test_empty_values_raises(self):
        with pytest.raises(ValueError, match="missing key or values"):
            _parse_inline_assignment("SCRM=", flag="--condition")

    def test_flag_name_in_error(self):
        with pytest.raises(ValueError, match=r"--condition"):
            _parse_inline_assignment("nope", flag="--condition")


def _make_plate_map_args(**overrides):
    """Build a minimal Namespace with inline-plate-map fields populated."""
    import argparse
    defaults = dict(
        condition=None,
        plate_replicate=None,
        replicate=None,
        plate=None,
        group_col="condition",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestApplyInlinePlateMap:
    def _df(self):
        return pd.DataFrame({
            "well_id": ["A1", "A2", "A3", "A1", "A2", "A3"],
            "plate":   [1, 1, 1, 2, 2, 2],
            "DIV":     [14, 14, 14, 21, 21, 21],
            "metric":  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })

    def test_returns_copy_when_no_flags(self):
        df = self._df()
        out = _apply_inline_plate_map(df, _make_plate_map_args())
        assert out is not df  # copy
        # All original columns preserved with same values.
        pd.testing.assert_frame_equal(out[df.columns], df)

    def test_replicate_auto_derived_from_plate_and_well(self):
        """Without any replicate flag, derive replicate_id from (plate, well_id)."""
        df = self._df()  # has plate + well_id, no replicate_id
        out = _apply_inline_plate_map(df, _make_plate_map_args())
        assert "replicate_id" in out.columns
        # Each unique (plate, well_id) -> distinct replicate.
        assert out["replicate_id"].nunique() == 6  # 2 plates x 3 wells
        # Same well on different plates -> different replicates.
        a1_p1 = out.loc[(out["plate"] == 1) & (out["well_id"] == "A1"), "replicate_id"].iloc[0]
        a1_p2 = out.loc[(out["plate"] == 2) & (out["well_id"] == "A1"), "replicate_id"].iloc[0]
        assert a1_p1 != a1_p2

    def test_replicate_auto_derive_falls_back_to_well_when_no_plate(self):
        df = self._df().drop(columns=["plate"])
        out = _apply_inline_plate_map(df, _make_plate_map_args())
        assert "replicate_id" in out.columns
        assert out["replicate_id"].nunique() == 3  # 3 unique wells
        a1 = out.loc[out["well_id"] == "A1", "replicate_id"].unique().tolist()
        assert a1 == ["A1"]

    def test_replicate_auto_derive_skipped_when_column_exists(self):
        df = self._df()
        df["replicate_id"] = "preset"
        out = _apply_inline_plate_map(df, _make_plate_map_args())
        # Don't overwrite the user's existing replicate_id column.
        assert (out["replicate_id"] == "preset").all()

    def test_replicate_flag_well_to_replicate(self):
        df = self._df()
        out = _apply_inline_plate_map(df, _make_plate_map_args(
            replicate=["rep01=A1,A2", "rep02=A3"],
        ))
        assert out.loc[out["well_id"] == "A1", "replicate_id"].tolist() == ["rep01"] * 2
        assert out.loc[out["well_id"] == "A2", "replicate_id"].tolist() == ["rep01"] * 2
        assert out.loc[out["well_id"] == "A3", "replicate_id"].tolist() == ["rep02"] * 2

    def test_replicate_flag_conflict_raises(self):
        df = self._df()
        with pytest.raises(ValueError, match="multiple replicates"):
            _apply_inline_plate_map(df, _make_plate_map_args(
                replicate=["rep01=A1", "rep02=A1"],
            ))

    def test_replicate_and_plate_replicate_mutually_exclusive(self):
        df = self._df()
        with pytest.raises(ValueError, match="mutually exclusive"):
            _apply_inline_plate_map(df, _make_plate_map_args(
                replicate=["rep01=A1"],
                plate_replicate=["1=rep01"],
            ))

    def test_condition_mapping_applied(self):
        df = self._df()
        out = _apply_inline_plate_map(df, _make_plate_map_args(
            condition=["SCRM=A1", "KD=A2,A3"],
        ))
        assert out.loc[out["well_id"] == "A1", "condition"].tolist() == ["SCRM"] * 2
        assert out.loc[out["well_id"] == "A2", "condition"].tolist() == ["KD"] * 2
        assert out.loc[out["well_id"] == "A3", "condition"].tolist() == ["KD"] * 2

    def test_condition_unmapped_well_becomes_nan(self):
        df = self._df()
        out = _apply_inline_plate_map(df, _make_plate_map_args(
            condition=["SCRM=A1"],
        ))
        # A2 and A3 are not assigned -> NaN.
        assert out.loc[out["well_id"] == "A2", "condition"].isna().all()
        assert out.loc[out["well_id"] == "A3", "condition"].isna().all()

    def test_condition_overwrites_existing_column(self):
        df = self._df()
        df["condition"] = "OLD"
        out = _apply_inline_plate_map(df, _make_plate_map_args(
            condition=["NEW=A1,A2,A3"],
        ))
        assert (out["condition"] == "NEW").all()

    def test_condition_respects_custom_group_col(self):
        df = self._df()
        out = _apply_inline_plate_map(df, _make_plate_map_args(
            condition=["SCRM=A1"], group_col="treatment",
        ))
        assert "treatment" in out.columns
        assert out.loc[out["well_id"] == "A1", "treatment"].tolist() == ["SCRM"] * 2

    def test_condition_conflict_raises(self):
        df = self._df()
        with pytest.raises(ValueError, match="multiple conditions"):
            _apply_inline_plate_map(df, _make_plate_map_args(
                condition=["SCRM=A1", "KD=A1"],
            ))

    def test_condition_without_well_id_column_raises(self):
        df = self._df().drop(columns=["well_id"])
        with pytest.raises(ValueError, match="well_id"):
            _apply_inline_plate_map(df, _make_plate_map_args(
                condition=["SCRM=A1"],
            ))

    def test_plate_replicate_applied(self):
        df = self._df()
        out = _apply_inline_plate_map(df, _make_plate_map_args(
            plate_replicate=["1=rep01", "2=rep02"],
        ))
        assert out.loc[out["plate"] == 1, "replicate_id"].tolist() == ["rep01"] * 3
        assert out.loc[out["plate"] == 2, "replicate_id"].tolist() == ["rep02"] * 3

    def test_plate_replicate_unmapped_becomes_nan(self):
        df = self._df()
        out = _apply_inline_plate_map(df, _make_plate_map_args(
            plate_replicate=["1=rep01"],
        ))
        assert out.loc[out["plate"] == 2, "replicate_id"].isna().all()

    def test_plate_replicate_overwrites_existing(self):
        df = self._df()
        df["replicate_id"] = "OLD"
        out = _apply_inline_plate_map(df, _make_plate_map_args(
            plate_replicate=["1=rep01", "2=rep02"],
        ))
        assert "OLD" not in out["replicate_id"].tolist()

    def test_plate_replicate_without_plate_column_raises(self):
        df = self._df().drop(columns=["plate"])
        with pytest.raises(ValueError, match="plate"):
            _apply_inline_plate_map(df, _make_plate_map_args(
                plate_replicate=["1=rep01"],
            ))

    def test_plate_replicate_duplicate_plate_raises(self):
        df = self._df()
        with pytest.raises(ValueError, match="more than once"):
            _apply_inline_plate_map(df, _make_plate_map_args(
                plate_replicate=["1=rep01", "1=rep02"],
            ))

    def test_plate_replicate_multi_value_raises(self):
        df = self._df()
        with pytest.raises(ValueError, match="exactly one value"):
            _apply_inline_plate_map(df, _make_plate_map_args(
                plate_replicate=["1=rep01,rep02"],
            ))

    def test_plate_flag_tags_all_rows(self):
        df = self._df().drop(columns=["plate"])
        out = _apply_inline_plate_map(df, _make_plate_map_args(plate="3"))
        assert (out["plate"] == "3").all()

    def test_plate_flag_enables_plate_replicate(self):
        df = self._df().drop(columns=["plate"])
        out = _apply_inline_plate_map(df, _make_plate_map_args(
            plate="1", plate_replicate=["1=rep01"],
        ))
        assert (out["replicate_id"] == "rep01").all()

    def test_invalid_assignment_raises(self):
        df = self._df()
        with pytest.raises(ValueError, match="--condition"):
            _apply_inline_plate_map(df, _make_plate_map_args(
                condition=["no_equals_sign"],
            ))


class TestResolvePalette:
    def _args(self, color):
        import argparse
        return argparse.Namespace(color=color)

    def test_returns_none_when_no_color_flag(self):
        assert _resolve_palette(self._args(None), ["SCRM", "KD"]) is None

    def test_assigns_specified_colors_in_group_order(self):
        args = self._args(["SCRM=#4477AA", "KD=#CC3311"])
        palette = _resolve_palette(args, ["SCRM", "KD"])
        assert palette == ["#4477AA", "#CC3311"]

    def test_unspecified_groups_fall_back_to_default(self):
        args = self._args(["SCRM=#4477AA"])
        palette = _resolve_palette(args, ["SCRM", "KD"])
        assert palette[0] == "#4477AA"
        assert palette[1] != "#4477AA"   # falls back to default cycle

    def test_named_color_accepted(self):
        args = self._args(["SCRM=red", "KD=blue"])
        palette = _resolve_palette(args, ["SCRM", "KD"])
        assert palette == ["red", "blue"]

    def test_duplicate_group_raises(self):
        args = self._args(["SCRM=#4477AA", "SCRM=#222222"])
        with pytest.raises(ValueError, match="more than once"):
            _resolve_palette(args, ["SCRM", "KD"])

    def test_multi_value_raises(self):
        args = self._args(["SCRM=#4477AA,#222222"])
        with pytest.raises(ValueError, match="exactly one value"):
            _resolve_palette(args, ["SCRM"])

    def test_invalid_assignment_raises(self):
        args = self._args(["no_equals_here"])
        with pytest.raises(ValueError, match="--color"):
            _resolve_palette(args, ["SCRM"])


class TestPlotColorParser:
    def test_trajectory_color_appends(self):
        p = build_parser()
        args = p.parse_args([
            "plot", "trajectory", "x.csv",
            "--color", "SCRM=#4477AA",
            "--color", "LGI2_KD4=#CC3311",
        ])
        assert args.color == ["SCRM=#4477AA", "LGI2_KD4=#CC3311"]

    def test_timepoint_color_appends(self):
        p = build_parser()
        args = p.parse_args([
            "plot", "timepoint", "x.csv",
            "--color", "SCRM=red",
        ])
        assert args.color == ["SCRM=red"]

    def test_pca_now_accepts_color(self):
        p = build_parser()
        args = p.parse_args([
            "plot", "pca", "x.csv",
            "--color", "SCRM=#4477AA",
        ])
        assert args.color == ["SCRM=#4477AA"]

    def test_pool_flag_default_false(self):
        p = build_parser()
        args = p.parse_args(["plot", "trajectory", "x.csv"])
        assert args.pool is False

    def test_pool_flag_set(self):
        p = build_parser()
        args = p.parse_args(["plot", "trajectory", "x.csv", "--pool"])
        assert args.pool is True

    def test_pool_on_timepoint_and_pca(self):
        p = build_parser()
        for cmd in ("timepoint", "pca"):
            args = p.parse_args(["plot", cmd, "x.csv", "--pool"])
            assert args.pool is True

    def test_group_order_default_none(self):
        p = build_parser()
        args = p.parse_args(["plot", "trajectory", "x.csv"])
        assert args.group_order is None

    def test_group_order_accepts_list(self):
        p = build_parser()
        args = p.parse_args([
            "plot", "trajectory", "x.csv",
            "--group-order", "SCRM", "LGI2_KD4", "LGI2_KD5",
        ])
        assert args.group_order == ["SCRM", "LGI2_KD4", "LGI2_KD5"]

    def test_group_order_on_timepoint_and_pca(self):
        p = build_parser()
        for cmd in ("timepoint", "pca"):
            args = p.parse_args([
                "plot", cmd, "x.csv",
                "--group-order", "A", "B",
            ])
            assert args.group_order == ["A", "B"]

    def test_compare_appends_pairs(self):
        p = build_parser()
        args = p.parse_args([
            "plot", "timepoint", "x.csv",
            "--compare", "LGI2_KD4", "SCRM",
            "--compare", "LGI2_KD5", "SCRM",
        ])
        assert args.compare == [["LGI2_KD4", "SCRM"], ["LGI2_KD5", "SCRM"]]

    def test_compare_default_none(self):
        p = build_parser()
        args = p.parse_args(["plot", "timepoint", "x.csv"])
        assert args.compare is None

    def test_pca_loadings_flag(self):
        p = build_parser()
        args = p.parse_args(["plot", "pca", "x.csv", "--loadings"])
        assert args.loadings is True

    def test_pca_loadings_default_false(self):
        p = build_parser()
        args = p.parse_args(["plot", "pca", "x.csv"])
        assert args.loadings is False

    def test_pca_top_default(self):
        p = build_parser()
        args = p.parse_args(["plot", "pca", "x.csv"])
        assert args.top == 10

    def test_pca_top_custom(self):
        p = build_parser()
        args = p.parse_args(["plot", "pca", "x.csv", "--top", "5"])
        assert args.top == 5

    def test_timepoint_test_default_tukey(self):
        p = build_parser()
        args = p.parse_args(["plot", "timepoint", "x.csv"])
        assert args.test == "tukey"

    def test_timepoint_test_choices(self):
        p = build_parser()
        for t in ("tukey", "mannwhitney", "kruskal"):
            args = p.parse_args(["plot", "timepoint", "x.csv", "--test", t])
            assert args.test == t

    def test_timepoint_test_invalid_choice_rejected(self):
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["plot", "timepoint", "x.csv", "--test", "welch"])

    def test_point_shape_default_none(self):
        p = build_parser()
        args = p.parse_args(["plot", "timepoint", "x.csv"])
        assert args.point_shape is None
        assert args.point_shape_map is None

    def test_point_shape_on_timepoint(self):
        p = build_parser()
        args = p.parse_args([
            "plot", "timepoint", "x.csv",
            "--point-shape", "batch",
            "--point-shape-map", "Batch1=o",
            "--point-shape-map", "Batch2=s",
            "--point-shape-map", "Batch3=^",
        ])
        assert args.point_shape == "batch"
        assert args.point_shape_map == ["Batch1=o", "Batch2=s", "Batch3=^"]

    def test_point_shape_on_pca(self):
        p = build_parser()
        args = p.parse_args([
            "plot", "pca", "x.csv",
            "--point-shape", "batch",
            "--point-shape-map", "Batch1=o",
        ])
        assert args.point_shape == "batch"
        assert args.point_shape_map == ["Batch1=o"]

    def test_filter_appends(self):
        p = build_parser()
        args = p.parse_args([
            "plot", "pca", "x.csv",
            "--filter", "condition=SCRM",
            "--filter", "DIV=14,28",
        ])
        assert args.filter == ["condition=SCRM", "DIV=14,28"]


class TestFilterEnd2End:
    def _csv(self, tmp_path: Path) -> Path:
        rows = []
        for cond in ("SCRM", "KD"):
            for div in (14, 21, 28):
                for rep in range(3):
                    rows.append({
                        "well_id":   f"W{rep}",
                        "plate":     1,
                        "DIV":       div,
                        "condition": cond,
                        "metric":    float(rep) + (10.0 if cond == "KD" else 0.0),
                    })
        p = tmp_path / "m.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    def test_filter_restricts_rows_to_one_condition(self, tmp_path):
        csv = self._csv(tmp_path)
        out = tmp_path / "tr"
        rc = main([
            "plot", "trajectory", str(csv),
            "--metric", "metric",
            "--filter", "condition=SCRM",
            "--out", str(out),
        ])
        assert rc == 0
        # File exists; content is implicitly verified by no errors.
        assert (out / "trajectory_metric.png").exists()

    def test_filter_with_multi_value_list(self, tmp_path):
        csv = self._csv(tmp_path)
        out = tmp_path / "tp"
        rc = main([
            "plot", "timepoint", str(csv),
            "--metric", "metric",
            "--filter", "DIV=14,28",
            "--out", str(out),
        ])
        assert rc == 0

    def test_filter_unknown_column_returns_one(self, tmp_path):
        csv = self._csv(tmp_path)
        out = tmp_path / "tr"
        rc = main([
            "plot", "trajectory", str(csv),
            "--metric", "metric",
            "--filter", "ghost_col=foo",
            "--out", str(out),
        ])
        assert rc == 1

    def test_filter_no_matches_returns_one(self, tmp_path):
        csv = self._csv(tmp_path)
        out = tmp_path / "tr"
        rc = main([
            "plot", "trajectory", str(csv),
            "--metric", "metric",
            "--filter", "condition=NOT_THERE",
            "--out", str(out),
        ])
        assert rc == 1


class TestPaletteHandlesNumericGroups:
    def test_numeric_group_matches_string_color_key(self):
        # _resolve_palette should match when groups are numeric and the
        # user typed string keys (e.g. --color 14=blue).
        import argparse
        args = argparse.Namespace(color=["14=blue", "28=red"])
        palette = _resolve_palette(args, [14.0, 28.0])
        assert palette == ["blue", "red"]

    def test_numeric_int_group_matches_string_color_key(self):
        import argparse
        args = argparse.Namespace(color=["14=blue", "28=red"])
        palette = _resolve_palette(args, [14, 28])
        assert palette == ["blue", "red"]


class TestPlotGroupOrderEnd2End:
    def _csv(self, tmp_path: Path) -> Path:
        rng = np.random.default_rng(0)
        rows = []
        for cond in ("SCRM", "LGI2_KD4", "LGI2_KD5"):
            for div in (14, 21):
                for rep in range(3):
                    rows.append({
                        "well_id":   f"W{rep}",
                        "plate":     1,
                        "DIV":       div,
                        "condition": cond,
                        "metric":    float(rng.normal(2.0, 0.3)),
                    })
        p = tmp_path / "m.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    def test_timepoint_respects_group_order(self, tmp_path):
        csv = self._csv(tmp_path)
        out = tmp_path / "tp"
        rc = main([
            "plot", "timepoint", str(csv),
            "--metric", "metric",
            "--div", "21",
            "--group-order", "SCRM", "LGI2_KD4", "LGI2_KD5",
            "--out", str(out),
        ])
        assert rc == 0
        assert (out / "metric" / "DIV_21.png").exists()

    def test_trajectory_respects_group_order(self, tmp_path):
        csv = self._csv(tmp_path)
        out = tmp_path / "tr"
        rc = main([
            "plot", "trajectory", str(csv),
            "--metric", "metric",
            "--group-order", "LGI2_KD5", "SCRM",  # subset + custom order
            "--out", str(out),
        ])
        assert rc == 0
        assert (out / "trajectory_metric.png").exists()


class TestPlotPlateMapParser:
    def test_trajectory_condition_appends(self):
        p = build_parser()
        args = p.parse_args([
            "plot", "trajectory", "x.csv",
            "--condition", "SCRM=A1,A2",
            "--condition", "KD=B1,B2",
        ])
        assert args.condition == ["SCRM=A1,A2", "KD=B1,B2"]

    def test_trajectory_plate_replicate_appends(self):
        p = build_parser()
        args = p.parse_args([
            "plot", "trajectory", "x.csv",
            "--plate-replicate", "1=rep01",
            "--plate-replicate", "2=rep02",
        ])
        assert args.plate_replicate == ["1=rep01", "2=rep02"]

    def test_trajectory_plate_default_none(self):
        p = build_parser()
        args = p.parse_args(["plot", "trajectory", "x.csv"])
        assert args.plate is None
        assert args.condition is None
        assert args.plate_replicate is None

    def test_timepoint_has_same_flags(self):
        p = build_parser()
        args = p.parse_args([
            "plot", "timepoint", "x.csv",
            "--condition", "SCRM=A1",
            "--plate-replicate", "1=rep01",
            "--plate", "1",
        ])
        assert args.condition == ["SCRM=A1"]
        assert args.plate_replicate == ["1=rep01"]
        assert args.plate == "1"

    def test_pca_does_not_have_plate_map_flags(self):
        """PCA was deliberately not given inline plate-map flags."""
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args([
                "plot", "pca", "x.csv",
                "--condition", "SCRM=A1",
            ])


class TestPlotInlinePlateMapEnd2End:
    def _write_csv(self, tmp_path: Path, *, with_plate: bool = True) -> Path:
        rng = np.random.default_rng(0)
        rows = []
        plates = [1, 2] if with_plate else [None]
        for plate in plates:
            for well in ["A1", "A2", "A3", "A4"]:
                for div in (14, 21, 28):
                    row = {
                        "well_id": well,
                        "DIV":     div,
                        "metric":  float(rng.normal(2.0, 0.3)),
                    }
                    if plate is not None:
                        row["plate"] = plate
                    rows.append(row)
        p = tmp_path / "raw.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    def test_trajectory_with_inline_map(self, tmp_path):
        csv = self._write_csv(tmp_path)
        out = tmp_path / "traj"
        rc = main([
            "plot", "trajectory", str(csv),
            "--metric", "metric",
            "--condition", "SCRM=A1,A2",
            "--condition", "KD=A3,A4",
            "--plate-replicate", "1=rep01",
            "--plate-replicate", "2=rep02",
            "--out", str(out),
        ])
        assert rc == 0
        assert (out / "trajectory_metric.png").exists()

    def test_timepoint_with_plate_flag_for_single_plate_csv(self, tmp_path):
        csv = self._write_csv(tmp_path, with_plate=False)
        out = tmp_path / "tp"
        rc = main([
            "plot", "timepoint", str(csv),
            "--metric", "metric",
            "--condition", "SCRM=A1,A2",
            "--condition", "KD=A3,A4",
            "--plate", "1",
            "--plate-replicate", "1=rep01",
            "--div", "21",
            "--out", str(out),
        ])
        assert rc == 0
        assert (out / "metric" / "DIV_21.png").exists()

    def test_inline_map_error_returns_one(self, tmp_path):
        csv = self._write_csv(tmp_path)
        out = tmp_path / "traj"
        rc = main([
            "plot", "trajectory", str(csv),
            "--metric", "metric",
            "--condition", "SCRM=A1",
            "--condition", "KD=A1",   # conflict
            "--out", str(out),
        ])
        assert rc == 1


# ── build subcommand: per-well parsing ────────────────────────────────────────

class TestParseWellSpec:
    def test_minimal_form(self):
        row = _parse_well_spec("1_A1=SCRM")
        assert row == {
            "plate": 1, "well_id": "A1", "condition": "SCRM",
            "bio_rep": None, "tech_rep": None, "batch": None,
        }

    def test_with_bio(self):
        row = _parse_well_spec("1_A1=SCRM,B1")
        assert row["bio_rep"] == "B1"
        assert row["tech_rep"] is None

    def test_full_form(self):
        row = _parse_well_spec("1_A1=SCRM,B1,T2,Batch1")
        assert row["bio_rep"] == "B1"
        assert row["tech_rep"] == "T2"
        assert row["batch"] == "Batch1"

    def test_plate_prefix_accepted(self):
        row = _parse_well_spec("Plate2_B3=KD,B1,T1")
        assert row["plate"] == 2 and row["well_id"] == "B3"

    def test_p_prefix_accepted(self):
        row = _parse_well_spec("P3_C2=KD,B1,T1")
        assert row["plate"] == 3 and row["well_id"] == "C2"

    def test_lowercase_accepted(self):
        row = _parse_well_spec("p1_a1=SCRM,b1,t1")
        assert row["plate"] == 1 and row["well_id"] == "A1"

    def test_no_equals_raises(self):
        with pytest.raises(ValueError, match="expected PLATE_WELL"):
            _parse_well_spec("1_A1_SCRM")

    def test_bad_key_raises(self):
        with pytest.raises(ValueError, match="expected like"):
            _parse_well_spec("garbage=SCRM")

    def test_missing_condition_raises(self):
        with pytest.raises(ValueError, match="missing condition"):
            _parse_well_spec("1_A1=")

    def test_bad_bio_rep_raises(self):
        with pytest.raises(ValueError, match="biological-replicate"):
            _parse_well_spec("1_A1=SCRM,X1,T1")

    def test_bad_tech_rep_raises(self):
        with pytest.raises(ValueError, match="technical-replicate"):
            _parse_well_spec("1_A1=SCRM,B1,X1")


class TestBuildLayoutFromWells:
    def test_returns_dataframe_with_expected_columns(self):
        df = _build_layout_from_wells([
            "1_A1=SCRM,B1,T1",
            "1_A2=KD,B2,T1",
        ])
        assert set(df.columns) >= {
            "plate", "well_id", "condition", "bio_rep", "tech_rep", "batch",
        }
        assert len(df) == 2

    def test_duplicate_entry_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            _build_layout_from_wells([
                "1_A1=SCRM,B1,T1",
                "Plate1_A1=KD,B2,T1",  # same (1, A1)
            ])

    def test_empty_specs_raises(self):
        with pytest.raises(ValueError, match="No --well"):
            _build_layout_from_wells([])


# ── build subcommand: filename parsing ────────────────────────────────────────

class TestExtractPlateDiv:
    def test_minimal(self):
        assert _extract_plate_div("Plate1_DIV14.spk") == (1, 14)

    def test_with_prefix(self):
        assert _extract_plate_div("20251002_LGI2_Plate2_DIV26.spk") == (2, 26)

    def test_case_insensitive(self):
        assert _extract_plate_div("plate3_div7.spk") == (3, 7)

    def test_no_match(self):
        assert _extract_plate_div("nothing_here.spk") is None


# ── build subcommand: layout-from CSV ─────────────────────────────────────────

class TestLoadLayoutFromCsv:
    def test_extracts_one_row_per_well(self, tmp_path):
        df = pd.DataFrame({
            "well_id": ["A1", "A1", "A2"],
            "plate":   [1, 1, 1],
            "DIV":     [14, 21, 14],
            "condition": ["SCRM", "SCRM", "KD"],
            "bio_rep":   ["B1", "B1", "B2"],
            "metric":  [1.0, 2.0, 3.0],
        })
        p = tmp_path / "m.csv"
        df.to_csv(p, index=False)
        layout = _load_layout_from_csv(p)
        assert len(layout) == 2  # collapsed across DIVs
        assert "metric" not in layout.columns

    def test_missing_required_column_raises(self, tmp_path):
        df = pd.DataFrame({"well_id": ["A1"]})
        p = tmp_path / "m.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required"):
            _load_layout_from_csv(p)


# ── --pool aggregation ────────────────────────────────────────────────────────

class TestFillMetricNans:
    def test_nan_in_metric_column_filled_with_zero(self):
        df = pd.DataFrame({
            "condition": ["A", "A", "B"],
            "DIV":       [14, 21, 14],
            "burst_freq_avg": [np.nan, 1.0, 2.0],
        })
        out = _fill_metric_nans(df, group_col="condition", time_col="DIV")
        assert out["burst_freq_avg"].tolist() == [0.0, 1.0, 2.0]

    def test_identifier_columns_untouched(self):
        # well_id is in the denylist; NaN there should NOT be filled.
        df = pd.DataFrame({
            "well_id":   ["A1", None, "B1"],
            "condition": ["A", "A", "B"],
            "DIV":       [14, 14, 14],
            "metric":    [np.nan, 1.0, 2.0],
        })
        out = _fill_metric_nans(df, group_col="condition", time_col="DIV")
        assert out["well_id"].tolist()[0] == "A1"
        # NaN in well_id stays NaN.
        assert pd.isna(out["well_id"].tolist()[1])
        # NaN in metric becomes 0.
        assert out["metric"].tolist() == [0.0, 1.0, 2.0]

    def test_time_and_group_columns_untouched(self):
        df = pd.DataFrame({
            "condition": ["A", None, "B"],
            "DIV":       [14, 21, np.nan],
            "metric":    [np.nan, 1.0, 2.0],
        })
        out = _fill_metric_nans(df, group_col="condition", time_col="DIV")
        # group_col / time_col NaNs stay NaN.
        assert pd.isna(out["condition"].tolist()[1])
        assert pd.isna(out["DIV"].tolist()[2])
        # metric NaN becomes 0.
        assert out["metric"].tolist() == [0.0, 1.0, 2.0]

    def test_nonnumeric_metric_columns_skipped(self):
        df = pd.DataFrame({
            "condition": ["A", "A"],
            "DIV":       [14, 21],
            "label":     ["x", None],   # string column, not a metric
            "metric":    [np.nan, 1.0],
        })
        out = _fill_metric_nans(df, group_col="condition", time_col="DIV")
        # String column with None unchanged.
        assert pd.isna(out["label"].tolist()[1])
        # Numeric metric filled.
        assert out["metric"].tolist() == [0.0, 1.0]


class TestMaybeAggregateReplicates:
    def _df(self):
        return pd.DataFrame({
            "well_id":   ["A1","A2","A3","B1","B2","B3"],
            "plate":     [1]*6,
            "DIV":       [14]*6,
            "condition": ["SCRM"]*6,
            "bio_rep":   ["B1"]*3 + ["B2"]*3,
            "tech_rep":  ["T1","T2","T3"]*2,
            "metric":    [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        })

    def test_pool_returns_unchanged(self):
        df = self._df()
        out = _maybe_aggregate_replicates(
            df, pool=True, group_col="condition", time_col="DIV",
        )
        # Length preserved, but groupby orders rows; just check row count.
        assert len(out) == len(df)

    def test_hierarchical_collapses_tech_reps(self):
        df = self._df()
        out = _maybe_aggregate_replicates(
            df, pool=False, group_col="condition", time_col="DIV",
        )
        # 2 bio reps -> 2 rows.
        assert len(out) == 2
        # Means of [1,2,3] and [10,11,12].
        assert sorted(out["metric"].tolist()) == [2.0, 11.0]

    def test_no_bio_rep_column_returns_unchanged(self):
        df = self._df().drop(columns=["bio_rep"])
        out = _maybe_aggregate_replicates(
            df, pool=False, group_col="condition", time_col="DIV",
        )
        assert len(out) == len(df)

    def test_groups_kept_separate(self):
        # Two conditions × two bio reps -> 4 rows.
        df = pd.DataFrame({
            "DIV":       [14]*8,
            "condition": ["A","A","A","A","B","B","B","B"],
            "bio_rep":   ["B1","B1","B2","B2"]*2,
            "metric":    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        })
        out = _maybe_aggregate_replicates(
            df, pool=False, group_col="condition", time_col="DIV",
        )
        assert len(out) == 4
        assert set(out["condition"]) == {"A", "B"}


# ── build / show-layout end-to-end ────────────────────────────────────────────

class TestBuildAndShowLayoutEnd2End:
    """Sanity-check the full build → show-layout round-trip without .spk files.

    We cannot run the build subcommand without real recordings, but we
    can exercise the parser and show-layout against a hand-built CSV.
    """

    def _master_csv(self, tmp_path: Path) -> Path:
        df = pd.DataFrame({
            "well_id":   ["A1","A2","A1"],
            "plate":     [1, 1, 2],
            "DIV":       [14, 14, 14],
            "condition": ["SCRM", "KD", "SCRM"],
            "bio_rep":   ["B1", "B1", "B2"],
            "tech_rep":  ["T1", "T1", "T1"],
            "batch":     ["Batch1", "Batch1", "Plate2"],
            "mean_mfr_active_hz": [0.1, 0.2, 0.15],
        })
        p = tmp_path / "master.csv"
        df.to_csv(p, index=False)
        return p

    def test_parser_accepts_build(self):
        p = build_parser()
        args = p.parse_args([
            "build", "data/",
            "--well", "1_A1=SCRM,B1,T1",
            "--out", "out.csv",
        ])
        assert args.command == "build"
        assert args.spk_path == "data/"
        assert args.well == ["1_A1=SCRM,B1,T1"]
        assert args.out == "out.csv"

    def test_parser_accepts_layout_from(self):
        p = build_parser()
        args = p.parse_args([
            "build", "data/",
            "--layout-from", "old.csv",
            "--out", "new.csv",
        ])
        assert args.layout_from == "old.csv"

    def test_parser_accepts_show_layout(self):
        p = build_parser()
        args = p.parse_args(["show-layout", "master.csv"])
        assert args.command == "show-layout"

    def test_show_layout_prints_well_flags(self, tmp_path, capsys):
        master = self._master_csv(tmp_path)
        rc = main(["show-layout", str(master)])
        assert rc == 0
        out = capsys.readouterr().out.splitlines()
        assert len(out) == 3
        assert all(line.startswith("--well ") for line in out)
        assert "1_A1=SCRM,B1,T1,Batch1" in out[0]
        assert "1_A2=KD,B1,T1,Batch1" in out[1]
        assert "2_A1=SCRM,B2,T1,Plate2" in out[2]

    def test_show_layout_missing_file_returns_one(self, tmp_path):
        rc = main(["show-layout", str(tmp_path / "nope.csv")])
        assert rc == 1

    def test_build_well_and_layout_from_mutually_exclusive(self, tmp_path):
        # Make a fake master CSV so --layout-from validation passes.
        master = self._master_csv(tmp_path)
        rc = main([
            "build", str(tmp_path),
            "--well", "1_A1=SCRM",
            "--layout-from", str(master),
            "--out", str(tmp_path / "out.csv"),
        ])
        assert rc == 1

    def test_build_no_layout_returns_one(self, tmp_path):
        rc = main([
            "build", str(tmp_path),
            "--out", str(tmp_path / "out.csv"),
        ])
        assert rc == 1


# ── stats subcommand ──────────────────────────────────────────────────────────

class TestStatsParser:
    def test_parser_recognises_stats(self):
        p = build_parser()
        args = p.parse_args([
            "stats", "x.csv", "--metric", "m", "--out", "out.csv",
        ])
        assert args.command == "stats"
        assert args.metric == ["m"]
        assert args.test == "tukey"

    def test_test_choice(self):
        p = build_parser()
        for t in ("tukey", "mannwhitney", "kruskal"):
            args = p.parse_args([
                "stats", "x.csv", "--test", t, "--out", "out.csv",
            ])
            assert args.test == t

    def test_compare_repeatable(self):
        p = build_parser()
        args = p.parse_args([
            "stats", "x.csv",
            "--compare", "A", "B",
            "--compare", "A", "C",
            "--out", "out.csv",
        ])
        assert args.compare == [["A", "B"], ["A", "C"]]

    def test_out_required(self):
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["stats", "x.csv"])  # no --out


class TestStatsEnd2End:
    def _csv(self, tmp_path: Path) -> Path:
        rng = np.random.default_rng(0)
        rows = []
        for div in (14, 28):
            for cond, mu in (("SCRM", 2.0), ("LGI2_KD4", 3.0), ("LGI2_KD5", 3.5)):
                for i in range(12):
                    rows.append({
                        "well_id":   f"W{i}",
                        "plate":     1 + (i // 4),
                        "DIV":       div,
                        "condition": cond,
                        "mfr":       float(rng.normal(mu, 0.3)),
                    })
        p = tmp_path / "master.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    def test_tukey_writes_one_row_per_pair(self, tmp_path):
        csv = self._csv(tmp_path)
        out = tmp_path / "stats.csv"
        rc = main([
            "stats", str(csv),
            "--metric", "mfr",
            "--div", "28",
            "--pool",
            "--out", str(out),
        ])
        assert rc == 0
        df = pd.read_csv(out)
        # 3 conditions -> 3 pairs by default.
        assert len(df) == 3
        assert set(df.columns) >= {
            "metric", "DIV", "test", "group_a", "group_b",
            "mean_diff", "p_adj", "significance", "n_a", "n_b",
        }
        assert (df["test"] == "tukey_hsd").all()

    def test_compare_filters_pairs(self, tmp_path):
        csv = self._csv(tmp_path)
        out = tmp_path / "stats.csv"
        rc = main([
            "stats", str(csv),
            "--metric", "mfr",
            "--div", "28",
            "--compare", "LGI2_KD4", "SCRM",
            "--compare", "LGI2_KD5", "SCRM",
            "--pool",
            "--out", str(out),
        ])
        assert rc == 0
        df = pd.read_csv(out)
        assert len(df) == 2
        # KD4-vs-KD5 must be absent.
        kd_pair = df[
            ((df["group_a"] == "LGI2_KD4") & (df["group_b"] == "LGI2_KD5")) |
            ((df["group_a"] == "LGI2_KD5") & (df["group_b"] == "LGI2_KD4"))
        ]
        assert kd_pair.empty

    def test_kruskal_test_works(self, tmp_path):
        csv = self._csv(tmp_path)
        out = tmp_path / "stats.csv"
        rc = main([
            "stats", str(csv),
            "--metric", "mfr",
            "--div", "28",
            "--test", "kruskal",
            "--pool",
            "--out", str(out),
        ])
        assert rc == 0
        df = pd.read_csv(out)
        # Either one omnibus row, or omnibus + posthoc rows.
        assert (df["test"].str.startswith("kruskal")).any()

    def test_multiple_metrics_and_divs(self, tmp_path):
        csv = self._csv(tmp_path)
        # Add a second metric column.
        df_in = pd.read_csv(csv)
        df_in["sttc"] = df_in["mfr"] * 0.1
        df_in.to_csv(csv, index=False)

        out = tmp_path / "stats.csv"
        rc = main([
            "stats", str(csv),
            "--metric", "mfr", "sttc",
            "--div", "14", "28",
            "--pool",
            "--out", str(out),
        ])
        assert rc == 0
        df = pd.read_csv(out)
        # 2 metrics × 2 DIVs × 3 pairs = 12 rows.
        assert len(df) == 12

    def test_filter_works_with_stats(self, tmp_path):
        csv = self._csv(tmp_path)
        out = tmp_path / "stats.csv"
        rc = main([
            "stats", str(csv),
            "--metric", "mfr",
            "--div", "28",
            "--filter", "condition=SCRM,LGI2_KD4",
            "--pool",
            "--out", str(out),
        ])
        assert rc == 0
        df = pd.read_csv(out)
        # Two groups remain → one pair.
        assert len(df) == 1


# ── Real .spk smoke test ──────────────────────────────────────────────────────

class TestRealSpkSmoke:
    def test_run_on_real_file(self, tmp_path):
        spk = Path("LGI2 KD data/Plate1_DIV28.spk")
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
