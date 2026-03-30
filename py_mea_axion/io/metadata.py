"""
metadata.py
===========
Load and validate experiment metadata that maps well IDs to experimental
conditions (treatment group, DIV, replicate, plate).

The metadata drives all downstream grouping, statistical comparisons, and
longitudinal trajectory plots.

Public API
----------
load_metadata(source, extra_columns=None)
    Accept a dict or CSV path and return a validated pandas DataFrame.
"""

import csv
import io
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from py_mea_axion.io.spk_reader import build_well_list

# Columns that every metadata table must contain.
_REQUIRED_COLUMNS: List[str] = ["well_id", "condition", "DIV", "replicate_id"]

# Columns that are filled with a default if absent (plate_id is optional
# because single-plate experiments are common).
_OPTIONAL_DEFAULTS: Dict[str, object] = {"plate_id": None}

_VALID_WELL_IDS: frozenset = frozenset(build_well_list())


def load_metadata(
    source: Union[str, Path, Dict],
    extra_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load experiment metadata from a CSV file or a plain Python dict.

    The returned DataFrame uses consistent column naming throughout the
    package: ``well_id``, ``condition``, ``DIV``, ``replicate_id``,
    ``plate_id``.

    Parameters
    ----------
    source : str, Path, or dict
        Either:

        * A path to a CSV file whose first row is a header.  Required
          columns: ``well_id``, ``condition``, ``DIV``, ``replicate_id``.
          Optional column: ``plate_id``.  Any additional columns are
          preserved.
        * A :class:`dict` mapping well_id strings to sub-dicts that contain
          at minimum ``condition``, ``DIV``, and ``replicate_id`` keys::

              {
                  "A1": {"condition": "WT",  "DIV": 14, "replicate_id": 1},
                  "A2": {"condition": "KD",  "DIV": 14, "replicate_id": 1},
                  ...
              }

    extra_columns : list of str, optional
        Additional column names (beyond the standard set) that must be
        present.  A :class:`ValueError` is raised if any are missing.

    Returns
    -------
    pd.DataFrame
        One row per well.  Guaranteed columns and dtypes:

        ============  =======  =============================================
        Column        Dtype    Notes
        ============  =======  =============================================
        well_id       object   E.g. ``'A1'``, upper-cased
        condition     object   E.g. ``'WT'``, ``'KD'``
        DIV           int64    Days in vitro
        replicate_id  object   Biological replicate label
        plate_id      object   Plate identifier; ``None`` if not supplied
        ============  =======  =============================================

    Raises
    ------
    TypeError
        If *source* is not a str, Path, or dict.
    FileNotFoundError
        If *source* is a path that does not exist.
    ValueError
        If required columns are missing, well IDs are invalid, or DIV
        values cannot be coerced to integers.

    Examples
    --------
    From a dict:

    >>> meta = load_metadata({
    ...     "A1": {"condition": "WT", "DIV": 14, "replicate_id": "rep1"},
    ...     "A2": {"condition": "KD", "DIV": 14, "replicate_id": "rep1"},
    ... })
    >>> meta.shape
    (2, 5)

    From a CSV file:

    >>> meta = load_metadata("experiment_metadata.csv")
    """
    if isinstance(source, dict):
        df = _from_dict(source)
    elif isinstance(source, (str, Path)):
        df = _from_csv(Path(source))
    else:
        raise TypeError(
            f"source must be a dict, str, or Path; got {type(source).__name__}"
        )

    _validate(df, extra_columns or [])
    df = _coerce_dtypes(df)
    return df.reset_index(drop=True)


# ── Private helpers ───────────────────────────────────────────────────────────

def _from_dict(source: Dict) -> pd.DataFrame:
    """Convert a ``{well_id: {field: value, ...}}`` dict to a DataFrame."""
    rows = []
    for well_id, fields in source.items():
        row = {"well_id": str(well_id).upper()}
        row.update(fields)
        rows.append(row)
    return pd.DataFrame(rows)


def _from_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame, normalising column names."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    df = pd.read_csv(path, dtype=str)
    # Strip whitespace from column names and well_id values.
    df.columns = [c.strip() for c in df.columns]
    if "well_id" in df.columns:
        df["well_id"] = df["well_id"].str.strip().str.upper()
    return df


def _validate(df: pd.DataFrame, extra_columns: List[str]) -> None:
    """Raise ValueError for any structural problems in the DataFrame."""
    required = _REQUIRED_COLUMNS + extra_columns
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Metadata is missing required column(s): {missing_cols}. "
            f"Present columns: {list(df.columns)}"
        )

    # Validate well IDs.
    invalid = df.loc[~df["well_id"].isin(_VALID_WELL_IDS), "well_id"].tolist()
    if invalid:
        raise ValueError(
            f"Invalid well_id value(s): {invalid}. "
            f"Expected one of {sorted(_VALID_WELL_IDS)}."
        )

    # Validate DIV is numeric.
    try:
        pd.to_numeric(df["DIV"])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"DIV column contains non-numeric values: {exc}"
        ) from exc


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standard columns have the correct dtypes."""
    df = df.copy()
    df["well_id"] = df["well_id"].astype(str)
    df["condition"] = df["condition"].astype(str)
    df["DIV"] = pd.to_numeric(df["DIV"]).astype("int64")
    df["replicate_id"] = df["replicate_id"].astype(str)
    if "plate_id" not in df.columns:
        df["plate_id"] = None
    # Preserve object dtype for plate_id (may be None or string).
    return df
