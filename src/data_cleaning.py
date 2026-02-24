"""
data_cleaning.py
================
Functions for loading and cleaning the raw states_all.csv dataset.

Usage
-----
    from src.data_cleaning import load_raw, clean

    raw = load_raw()
    df  = clean(raw)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "states_all.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "states_clean.csv"

# ---------------------------------------------------------------------------
# Column groups (handy references for downstream modules)
# ---------------------------------------------------------------------------
REVENUE_COLS = ["TOTAL_REVENUE", "FEDERAL_REVENUE", "STATE_REVENUE", "LOCAL_REVENUE"]

EXPENDITURE_COLS = [
    "TOTAL_EXPENDITURE",
    "INSTRUCTION_EXPENDITURE",
    "SUPPORT_SERVICES_EXPENDITURE",
    "OTHER_EXPENDITURE",
    "CAPITAL_OUTLAY_EXPENDITURE",
]

GRADE_ENROLLMENT_COLS = [
    "GRADES_PK_G",
    "GRADES_KG_G",
    "GRADES_4_G",
    "GRADES_8_G",
    "GRADES_12_G",
    "GRADES_1_8_G",
    "GRADES_9_12_G",
    "GRADES_ALL_G",
]

SCORE_COLS = [
    "AVG_MATH_4_SCORE",
    "AVG_MATH_8_SCORE",
    "AVG_READING_4_SCORE",
    "AVG_READING_8_SCORE",
]

ALL_NUMERIC_COLS = REVENUE_COLS + EXPENDITURE_COLS + GRADE_ENROLLMENT_COLS + SCORE_COLS + ["ENROLL"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV and return a DataFrame with minimal type coercion.

    Parameters
    ----------
    path:
        Override the default path to ``states_all.csv``.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


def clean(df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
    """Apply the full cleaning pipeline to the raw DataFrame.

    Steps
    -----
    1. Standardise column names (upper-case, strip whitespace).
    2. Drop the redundant ``PRIMARY_KEY`` column.
    3. Coerce numeric columns; replace sentinel values with ``NaN``.
    4. Remove aggregate / non-state rows (e.g. "DODEA", "NATIONAL").
    5. Enforce correct dtypes (``STATE`` → string, ``YEAR`` → int).
    6. Drop fully-duplicate rows.
    7. Sort by STATE then YEAR and reset the index.
    8. Optionally persist the cleaned file to ``data/processed/``.

    Parameters
    ----------
    df:
        Raw DataFrame returned by :func:`load_raw`.
    save:
        If ``True``, write the result to ``PROCESSED_PATH``.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # 1. Standardise column names
    df.columns = df.columns.str.strip().str.upper()
    logger.info("Step 1 — columns standardised")

    # 2. Drop redundant primary key
    if "PRIMARY_KEY" in df.columns:
        df.drop(columns=["PRIMARY_KEY"], inplace=True)

    # 3. Coerce numerics
    numeric_present = [c for c in ALL_NUMERIC_COLS if c in df.columns]
    for col in numeric_present:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Replace zeroes that are clearly missing in enrollment / score columns
    zero_suspect = SCORE_COLS + ["ENROLL"]
    for col in [c for c in zero_suspect if c in df.columns]:
        df[col] = df[col].replace(0, np.nan)
    logger.info("Step 3 — numeric coercion done (%d cols)", len(numeric_present))

    # 4. Remove aggregate rows
    non_state = {"DODEA", "NATIONAL", "DISTRICT OF COLUMBIA"}   # keep DC if desired — remove from set
    non_state_present = df["STATE"].str.upper().isin(non_state)
    n_dropped = non_state_present.sum()
    df = df[~non_state_present].copy()
    logger.info("Step 4 — dropped %d aggregate rows", n_dropped)

    # 5. Dtypes
    df["STATE"] = df["STATE"].str.strip().str.upper().astype("string")
    df["YEAR"] = df["YEAR"].astype(int)

    # 6. Duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info("Step 6 — removed %d duplicate rows", before - len(df))

    # 7. Sort & reset index
    df.sort_values(["STATE", "YEAR"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Cleaning complete — %d rows × %d columns", *df.shape)

    # 8. Optionally save
    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        logger.info("Saved cleaned data to %s", PROCESSED_PATH)

    return df


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising missing values per column.

    Columns returned: ``missing_count``, ``missing_pct``, ``dtype``.
    """
    missing = df.isnull().sum()
    result = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_pct": (missing / len(df) * 100).round(2),
            "dtype": df.dtypes,
        }
    )
    return result[result["missing_count"] > 0].sort_values("missing_pct", ascending=False)
