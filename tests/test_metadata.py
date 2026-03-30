"""
tests/test_metadata.py
======================
Tests for py_mea_axion.io.metadata.
"""

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from py_mea_axion.io.metadata import load_metadata

# ── Fixtures ──────────────────────────────────────────────────────────────────

MINIMAL_DICT = {
    "A1": {"condition": "WT", "DIV": 14, "replicate_id": "rep1"},
    "A2": {"condition": "KD", "DIV": 14, "replicate_id": "rep1"},
    "B1": {"condition": "WT", "DIV": 14, "replicate_id": "rep2"},
    "B2": {"condition": "KD", "DIV": 14, "replicate_id": "rep2"},
}

FULL_DICT = {
    "A1": {"condition": "WT", "DIV": 14, "replicate_id": "rep1", "plate_id": "P1"},
    "A2": {"condition": "KD", "DIV": 14, "replicate_id": "rep1", "plate_id": "P1"},
}

MINIMAL_CSV = textwrap.dedent("""\
    well_id,condition,DIV,replicate_id
    A1,WT,14,rep1
    A2,KD,14,rep1
    B1,WT,21,rep2
    B2,KD,21,rep2
""")

FULL_CSV = textwrap.dedent("""\
    well_id,condition,DIV,replicate_id,plate_id
    A1,WT,14,rep1,P1
    A2,KD,14,rep1,P1
    D6,WT,28,rep3,P2
""")


@pytest.fixture()
def minimal_csv(tmp_path: Path) -> Path:
    p = tmp_path / "meta.csv"
    p.write_text(MINIMAL_CSV)
    return p


@pytest.fixture()
def full_csv(tmp_path: Path) -> Path:
    p = tmp_path / "meta_full.csv"
    p.write_text(FULL_CSV)
    return p


# ── Dict source ───────────────────────────────────────────────────────────────

class TestFromDict:
    def test_row_count(self):
        df = load_metadata(MINIMAL_DICT)
        assert len(df) == 4

    def test_columns_present(self):
        df = load_metadata(MINIMAL_DICT)
        for col in ["well_id", "condition", "DIV", "replicate_id", "plate_id"]:
            assert col in df.columns

    def test_well_id_uppercased(self):
        df = load_metadata({"a1": {"condition": "WT", "DIV": 7, "replicate_id": "r1"}})
        assert df["well_id"].iloc[0] == "A1"

    def test_div_dtype_int64(self):
        df = load_metadata(MINIMAL_DICT)
        assert df["DIV"].dtype == "int64"

    def test_plate_id_none_when_absent(self):
        df = load_metadata(MINIMAL_DICT)
        assert df["plate_id"].iloc[0] is None

    def test_plate_id_preserved_when_present(self):
        df = load_metadata(FULL_DICT)
        assert df["plate_id"].iloc[0] == "P1"

    def test_condition_values(self):
        df = load_metadata(MINIMAL_DICT)
        assert set(df["condition"]) == {"WT", "KD"}

    def test_extra_columns_preserved(self):
        src = {
            "A1": {"condition": "WT", "DIV": 14, "replicate_id": "r1",
                   "experimenter": "Alice"}
        }
        df = load_metadata(src)
        assert "experimenter" in df.columns
        assert df["experimenter"].iloc[0] == "Alice"

    def test_reset_index(self):
        df = load_metadata(MINIMAL_DICT)
        assert list(df.index) == list(range(len(df)))


# ── CSV source ────────────────────────────────────────────────────────────────

class TestFromCsv:
    def test_row_count(self, minimal_csv):
        df = load_metadata(minimal_csv)
        assert len(df) == 4

    def test_columns_present(self, minimal_csv):
        df = load_metadata(minimal_csv)
        for col in ["well_id", "condition", "DIV", "replicate_id", "plate_id"]:
            assert col in df.columns

    def test_div_dtype_int64(self, minimal_csv):
        df = load_metadata(minimal_csv)
        assert df["DIV"].dtype == "int64"

    def test_plate_id_from_csv(self, full_csv):
        df = load_metadata(full_csv)
        assert "plate_id" in df.columns
        assert df.loc[df["well_id"] == "A1", "plate_id"].iloc[0] == "P1"

    def test_accepts_string_path(self, minimal_csv):
        df = load_metadata(str(minimal_csv))
        assert len(df) == 4

    def test_well_id_uppercased_from_csv(self, tmp_path):
        csv_text = "well_id,condition,DIV,replicate_id\na1,WT,7,r1\n"
        p = tmp_path / "lower.csv"
        p.write_text(csv_text)
        df = load_metadata(p)
        assert df["well_id"].iloc[0] == "A1"

    def test_strips_whitespace_from_columns(self, tmp_path):
        csv_text = " well_id , condition , DIV , replicate_id \nA1,WT,14,r1\n"
        p = tmp_path / "spaces.csv"
        p.write_text(csv_text)
        df = load_metadata(p)
        assert "well_id" in df.columns

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_metadata(tmp_path / "nonexistent.csv")


# ── Validation errors ─────────────────────────────────────────────────────────

class TestValidation:
    def test_missing_required_column(self):
        # Missing 'condition'
        with pytest.raises(ValueError, match="condition"):
            load_metadata({"A1": {"DIV": 14, "replicate_id": "r1"}})

    def test_invalid_well_id(self):
        with pytest.raises(ValueError, match="Invalid well_id"):
            load_metadata({"Z9": {"condition": "WT", "DIV": 14, "replicate_id": "r1"}})

    def test_non_numeric_div(self):
        with pytest.raises(ValueError, match="DIV"):
            load_metadata({"A1": {"condition": "WT", "DIV": "two_weeks", "replicate_id": "r1"}})

    def test_extra_columns_required(self):
        with pytest.raises(ValueError, match="experimenter"):
            load_metadata(MINIMAL_DICT, extra_columns=["experimenter"])

    def test_wrong_source_type(self):
        with pytest.raises(TypeError, match="dict, str, or Path"):
            load_metadata(42)


# ── Longitudinal multi-DIV dict ───────────────────────────────────────────────

class TestLongitudinal:
    """Simulate the LGI2 experiment: two conditions × multiple DIVs."""

    def _build_lgi2_meta(self) -> dict:
        meta = {}
        for div in [14, 21, 28]:
            for col, cond in [(1, "WT"), (2, "WT"), (3, "KD"), (4, "KD")]:
                well = f"A{col}"
                meta[f"{well}_D{div}"] = None  # placeholder key only
        # Use proper well IDs (A1–A4, one DIV per recording file)
        rows = {}
        for rep, (col, cond) in enumerate([(1, "WT"), (2, "WT"), (3, "KD"), (4, "KD")], 1):
            for div in [14, 21, 28]:
                well = f"A{col}"
                rows[well] = {"condition": cond, "DIV": div, "replicate_id": f"rep{rep}"}
        # Last DIV wins per well in a dict — acceptable for this shape test.
        return rows

    def test_conditions_extracted(self):
        rows = {}
        for rep, (col, cond) in enumerate([(1, "WT"), (2, "KD")], 1):
            rows[f"A{col}"] = {
                "condition": cond, "DIV": 14, "replicate_id": f"rep{rep}",
            }
        df = load_metadata(rows)
        assert set(df["condition"]) == {"WT", "KD"}

    def test_div_values_preserved(self):
        rows = {
            "A1": {"condition": "WT", "DIV": 14, "replicate_id": "r1"},
            "A2": {"condition": "WT", "DIV": 21, "replicate_id": "r1"},
            "A3": {"condition": "WT", "DIV": 28, "replicate_id": "r1"},
        }
        df = load_metadata(rows)
        assert sorted(df["DIV"].tolist()) == [14, 21, 28]
