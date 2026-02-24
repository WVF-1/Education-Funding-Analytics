"""
feature_engineering.py
======================
Derives analysis-ready features from the cleaned dataset.

Usage
-----
    from src.feature_engineering import engineer_features

    df = engineer_features(clean_df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline.

    New columns added
    -----------------
    Per-pupil financials
        ``REVENUE_PER_PUPIL``, ``EXPENDITURE_PER_PUPIL``,
        ``INSTRUCTION_PER_PUPIL``, ``SUPPORT_PER_PUPIL``

    Revenue composition (% of total revenue)
        ``PCT_FEDERAL_REVENUE``, ``PCT_STATE_REVENUE``, ``PCT_LOCAL_REVENUE``

    Expenditure composition (% of total expenditure)
        ``PCT_INSTRUCTION``, ``PCT_SUPPORT_SERVICES``,
        ``PCT_CAPITAL_OUTLAY``, ``PCT_OTHER_EXP``

    Composite test scores
        ``AVG_MATH_SCORE``    – mean of grade-4 and grade-8 math
        ``AVG_READING_SCORE`` – mean of grade-4 and grade-8 reading
        ``AVG_COMPOSITE_SCORE`` – overall mean across all four NAEP scores

    Spending efficiency proxies
        ``MATH_PER_1K_SPENT``    – composite math score per $1 k expenditure/pupil
        ``READING_PER_1K_SPENT`` – composite reading score per $1 k expenditure/pupil

    Year-over-year growth (within each state)
        ``EXPENDITURE_YOY_PCT``, ``REVENUE_YOY_PCT``, ``ENROLL_YOY_PCT``

    Decade label
        ``DECADE`` – e.g. "1990s", "2000s"

    Parameters
    ----------
    df:
        Cleaned DataFrame from :func:`src.data_cleaning.clean`.

    Returns
    -------
    pd.DataFrame with all original columns plus the new features.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Per-pupil financials
    # ------------------------------------------------------------------
    enrollment = df["ENROLL"].replace(0, np.nan)

    df["REVENUE_PER_PUPIL"] = df["TOTAL_REVENUE"] / enrollment
    df["EXPENDITURE_PER_PUPIL"] = df["TOTAL_EXPENDITURE"] / enrollment
    df["INSTRUCTION_PER_PUPIL"] = df["INSTRUCTION_EXPENDITURE"] / enrollment
    df["SUPPORT_PER_PUPIL"] = df["SUPPORT_SERVICES_EXPENDITURE"] / enrollment

    logger.info("Step 1 — per-pupil financials created")

    # ------------------------------------------------------------------
    # 2. Revenue composition (%)
    # ------------------------------------------------------------------
    total_rev = df["TOTAL_REVENUE"].replace(0, np.nan)

    df["PCT_FEDERAL_REVENUE"] = df["FEDERAL_REVENUE"] / total_rev * 100
    df["PCT_STATE_REVENUE"] = df["STATE_REVENUE"] / total_rev * 100
    df["PCT_LOCAL_REVENUE"] = df["LOCAL_REVENUE"] / total_rev * 100

    logger.info("Step 2 — revenue composition (%) created")

    # ------------------------------------------------------------------
    # 3. Expenditure composition (%)
    # ------------------------------------------------------------------
    total_exp = df["TOTAL_EXPENDITURE"].replace(0, np.nan)

    df["PCT_INSTRUCTION"] = df["INSTRUCTION_EXPENDITURE"] / total_exp * 100
    df["PCT_SUPPORT_SERVICES"] = df["SUPPORT_SERVICES_EXPENDITURE"] / total_exp * 100
    df["PCT_CAPITAL_OUTLAY"] = df["CAPITAL_OUTLAY_EXPENDITURE"] / total_exp * 100
    df["PCT_OTHER_EXP"] = df["OTHER_EXPENDITURE"] / total_exp * 100

    logger.info("Step 3 — expenditure composition (%) created")

    # ------------------------------------------------------------------
    # 4. Composite test scores
    # ------------------------------------------------------------------
    math_cols = ["AVG_MATH_4_SCORE", "AVG_MATH_8_SCORE"]
    reading_cols = ["AVG_READING_4_SCORE", "AVG_READING_8_SCORE"]
    all_score_cols = math_cols + reading_cols

    df["AVG_MATH_SCORE"] = df[math_cols].mean(axis=1, skipna=True)
    df["AVG_READING_SCORE"] = df[reading_cols].mean(axis=1, skipna=True)
    df["AVG_COMPOSITE_SCORE"] = df[all_score_cols].mean(axis=1, skipna=True)

    # Null out composites where ALL component scores were missing
    df.loc[df[all_score_cols].isnull().all(axis=1), "AVG_COMPOSITE_SCORE"] = np.nan
    df.loc[df[math_cols].isnull().all(axis=1), "AVG_MATH_SCORE"] = np.nan
    df.loc[df[reading_cols].isnull().all(axis=1), "AVG_READING_SCORE"] = np.nan

    logger.info("Step 4 — composite test scores created")

    # ------------------------------------------------------------------
    # 5. Spending efficiency proxies
    # ------------------------------------------------------------------
    exp_per_pupil_k = df["EXPENDITURE_PER_PUPIL"] / 1_000  # in $1k units

    df["MATH_PER_1K_SPENT"] = df["AVG_MATH_SCORE"] / exp_per_pupil_k
    df["READING_PER_1K_SPENT"] = df["AVG_READING_SCORE"] / exp_per_pupil_k

    logger.info("Step 5 — spending efficiency proxies created")

    # ------------------------------------------------------------------
    # 6. Year-over-year growth rates (within state)
    # ------------------------------------------------------------------
    df.sort_values(["STATE", "YEAR"], inplace=True)

    for col, new_col in [
        ("TOTAL_EXPENDITURE", "EXPENDITURE_YOY_PCT"),
        ("TOTAL_REVENUE", "REVENUE_YOY_PCT"),
        ("ENROLL", "ENROLL_YOY_PCT"),
    ]:
        df[new_col] = (
            df.groupby("STATE")[col]
            .pct_change() * 100
        )

    logger.info("Step 6 — YoY growth rates created")

    # ------------------------------------------------------------------
    # 7. Decade label
    # ------------------------------------------------------------------
    df["DECADE"] = (df["YEAR"] // 10 * 10).astype(str) + "s"

    logger.info("Step 7 — decade label created")

    df.reset_index(drop=True, inplace=True)
    logger.info("Feature engineering complete — %d rows × %d columns", *df.shape)

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_latest_year_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per state, keeping only the most recent year."""
    return df.loc[df.groupby("STATE")["YEAR"].idxmax()].reset_index(drop=True)


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot a single metric to a (YEAR × STATE) wide table.

    Parameters
    ----------
    df:
        Feature-engineered DataFrame.
    metric:
        Column name to pivot, e.g. ``"EXPENDITURE_PER_PUPIL"``.

    Returns
    -------
    pd.DataFrame indexed by YEAR with one column per STATE.
    """
    return df.pivot_table(index="YEAR", columns="STATE", values=metric)
