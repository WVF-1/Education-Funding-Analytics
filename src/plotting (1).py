"""
plotting.py
===========
Reusable, notebook-friendly plotting functions for the education-spending
analytics project.  Every function returns the ``matplotlib`` ``Figure``
object so callers can further customise or save it.

Usage
-----
    from src import plots

    fig = plots.spending_over_time(df)
    fig = plots.score_vs_spending(df, score_col="AVG_MATH_SCORE")
    fig = plots.revenue_composition(df, state="TEXAS")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"

PALETTE = "tab10"
STYLE = "whitegrid"
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 8)
FIGSIZE_TALL = (10, 12)


def _apply_style() -> None:
    sns.set_theme(style=STYLE, palette=PALETTE)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )


def _save(fig: plt.Figure, filename: Optional[str]) -> None:
    if filename:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")


# ---------------------------------------------------------------------------
# 1. Spending / revenue trends over time
# ---------------------------------------------------------------------------

def spending_over_time(
    df: pd.DataFrame,
    states: Optional[Iterable[str]] = None,
    metric: str = "EXPENDITURE_PER_PUPIL",
    title: str | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Line chart of a per-pupil spending metric over time.

    Parameters
    ----------
    df:
        Feature-engineered DataFrame.
    states:
        Subset of states to plot. If ``None``, plots the national median.
    metric:
        Column to plot on the y-axis.
    title:
        Optional override for the chart title.
    save_as:
        Filename (e.g. ``"spending_over_time.png"``) to save under
        ``outputs/figures/``.  ``None`` skips saving.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    if states is None:
        # National median
        national = df.groupby("YEAR")[metric].median().reset_index()
        ax.plot(national["YEAR"], national[metric], linewidth=2, label="National Median")
    else:
        states = [s.upper() for s in states]
        for state in states:
            subset = df[df["STATE"] == state].sort_values("YEAR")
            ax.plot(subset["YEAR"], subset[metric], linewidth=2, marker="o", markersize=3, label=state)
        ax.legend(loc="upper left", ncol=2)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Year")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric.replace('_', ' ').title()} Over Time")

    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 2. Score vs. spending scatter
# ---------------------------------------------------------------------------

def score_vs_spending(
    df: pd.DataFrame,
    score_col: str = "AVG_COMPOSITE_SCORE",
    spending_col: str = "EXPENDITURE_PER_PUPIL",
    hue_col: str = "DECADE",
    highlight_states: Optional[Iterable[str]] = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Scatter plot of test scores against per-pupil spending.

    Parameters
    ----------
    hue_col:
        Column used to colour the points (e.g. ``"DECADE"`` or ``"STATE"``).
    highlight_states:
        State names to annotate with a text label on the chart.
    """
    _apply_style()
    plot_df = df[[spending_col, score_col, hue_col, "STATE", "YEAR"]].dropna()

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.scatterplot(
        data=plot_df,
        x=spending_col,
        y=score_col,
        hue=hue_col,
        alpha=0.65,
        s=50,
        ax=ax,
    )

    # Regression line
    x = plot_df[spending_col].values
    y = plot_df[score_col].values
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5, linestyle="--", label="OLS trend")
    ax.legend()

    # Annotate selected states (latest year only)
    if highlight_states:
        latest = df.loc[df.groupby("STATE")["YEAR"].idxmax()]
        for state in [s.upper() for s in highlight_states]:
            row = latest[latest["STATE"] == state]
            if row.empty:
                continue
            ax.annotate(
                state,
                xy=(row[spending_col].values[0], row[score_col].values[0]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
            )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel(spending_col.replace("_", " ").title())
    ax.set_ylabel(score_col.replace("_", " ").title())
    ax.set_title(f"{score_col.replace('_', ' ').title()} vs. {spending_col.replace('_', ' ').title()}")

    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 3. Revenue composition stacked bar
# ---------------------------------------------------------------------------

def revenue_composition(
    df: pd.DataFrame,
    state: str | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Stacked bar chart of revenue source composition over time.

    If ``state`` is ``None``, uses the national median percentages.
    """
    _apply_style()
    pct_cols = ["PCT_FEDERAL_REVENUE", "PCT_STATE_REVENUE", "PCT_LOCAL_REVENUE"]
    labels = ["Federal", "State", "Local"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    if state:
        subset = df[df["STATE"] == state.upper()].sort_values("YEAR")
        plot_df = subset.set_index("YEAR")[pct_cols].dropna()
        title = f"Revenue Composition — {state.title()}"
    else:
        plot_df = df.groupby("YEAR")[pct_cols].median().dropna()
        title = "Revenue Composition — National Median"

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plot_df.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8, legend=False)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel("Year")
    ax.set_ylabel("% of Total Revenue")
    ax.set_title(title)
    ax.legend(labels, loc="upper right")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 4. State rankings bar chart (snapshot)
# ---------------------------------------------------------------------------

def state_rankings(
    df: pd.DataFrame,
    metric: str = "EXPENDITURE_PER_PUPIL",
    year: int | None = None,
    top_n: int = 20,
    ascending: bool = False,
    save_as: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart ranking states by a given metric in a single year.

    Parameters
    ----------
    year:
        Year to snapshot. Defaults to the most recent year in the data.
    top_n:
        Number of states to show.
    """
    _apply_style()
    if year is None:
        year = int(df["YEAR"].max())

    subset = df[df["YEAR"] == year][["STATE", metric]].dropna()
    subset = subset.sort_values(metric, ascending=ascending).head(top_n)

    fig, ax = plt.subplots(figsize=(10, top_n * 0.45 + 1))
    colors = sns.color_palette(PALETTE, len(subset))
    ax.barh(subset["STATE"], subset[metric], color=colors)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
        if "expenditure" in metric.lower() or "revenue" in metric.lower() or "per_pupil" in metric.lower()
        else mticker.ScalarFormatter()
    )
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} by State ({year})")
    ax.invert_yaxis()

    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 5. Correlation heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    save_as: str | None = None,
) -> plt.Figure:
    """Heatmap of Pearson correlations between key numeric columns."""
    _apply_style()
    default_cols = [
        "EXPENDITURE_PER_PUPIL",
        "REVENUE_PER_PUPIL",
        "PCT_FEDERAL_REVENUE",
        "PCT_STATE_REVENUE",
        "PCT_LOCAL_REVENUE",
        "PCT_INSTRUCTION",
        "AVG_MATH_SCORE",
        "AVG_READING_SCORE",
        "AVG_COMPOSITE_SCORE",
        "ENROLL",
    ]
    cols = cols or [c for c in default_cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # show lower triangle
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation Heatmap — Key Metrics")
    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 6. Score trends for selected states
# ---------------------------------------------------------------------------

def score_trends(
    df: pd.DataFrame,
    states: Iterable[str],
    score_col: str = "AVG_COMPOSITE_SCORE",
    save_as: str | None = None,
) -> plt.Figure:
    """Line chart comparing test-score trends across selected states."""
    _apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for state in [s.upper() for s in states]:
        subset = df[df["STATE"] == state].sort_values("YEAR").dropna(subset=[score_col])
        if subset.empty:
            continue
        ax.plot(subset["YEAR"], subset[score_col], linewidth=2, marker="o", markersize=3, label=state)

    ax.set_xlabel("Year")
    ax.set_ylabel(score_col.replace("_", " ").title())
    ax.set_title(f"{score_col.replace('_', ' ').title()} — State Comparison")
    ax.legend(loc="lower right", ncol=2)

    fig.tight_layout()
    _save(fig, save_as)
    return fig


# ---------------------------------------------------------------------------
# 7. Distribution of per-pupil spending by decade
# ---------------------------------------------------------------------------

def spending_distribution(
    df: pd.DataFrame,
    metric: str = "EXPENDITURE_PER_PUPIL",
    save_as: str | None = None,
) -> plt.Figure:
    """Box-plot of per-pupil spending distribution broken down by decade."""
    _apply_style()
    plot_df = df[["DECADE", metric]].dropna()
    decades = sorted(plot_df["DECADE"].unique())

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.boxplot(data=plot_df, x="DECADE", y=metric, order=decades, palette=PALETTE, ax=ax)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Decade")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Distribution of {metric.replace('_', ' ').title()} by Decade")

    fig.tight_layout()
    _save(fig, save_as)
    return fig
