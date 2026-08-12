"""
Reproduce the complete NexaVest Project B pipeline.

Run from the Project B root:

    python scripts/run_part_b.py

The script:

1. Reuses the student's Project A data foundation where available.
2. Builds the 12 base out-of-sample funds.
3. Builds the finance-aware sector sentiment index.
4. Applies the NexaVest coverage-adjusted sentiment extension.
5. Saves the final 14-fund outputs, audit tables and figures.
"""

from __future__ import annotations

import os
import pathlib
import sys

# ---------------------------------------------------------------------
# Keep numerical libraries lightweight for reproducible local/cloud runs
# ---------------------------------------------------------------------

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd  # noqa: E402
import src.fusion as fusion  # noqa: E402
import src.portfolios as portfolios  # noqa: E402
import src.sentiment as sentiment  # noqa: E402
from src import data_access, etl, features  # noqa: E402

# =====================================================================
# Project A foundation
# =====================================================================


def _read_wide_returns(path: pathlib.Path) -> pd.DataFrame:
    """
    Read a Project A wide return panel.

    The returned DataFrame uses a DatetimeIndex named 'date' and
    numeric asset-return columns.
    """
    frame = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    frame = frame.set_index("date").sort_index()

    frame = frame.apply(
        pd.to_numeric,
        errors="coerce",
    )

    frame.index.name = "date"

    return frame


def _project_a_result_candidates() -> list[pathlib.Path]:
    """
    Return possible locations of the student's Project A results.
    """
    candidates: list[pathlib.Path] = []

    # Optional explicit override.
    environment_path = os.getenv(
        "PROJECT_A_RESULTS_DIR"
    )

    if environment_path:
        candidates.append(
            pathlib.Path(
                environment_path
            )
            .expanduser()
            .resolve()
        )

    # Normal sibling-project layout:
    #
    # fins2026/
    #   z5537596_projectA/
    #   z5537596_projectB/
    #
    sibling_name = (
        PROJECT_ROOT.name[:-1] + "A"
        if PROJECT_ROOT.name.endswith("B")
        else "z5537596_projectA"
    )

    candidates.append(
        PROJECT_ROOT.parent
        / sibling_name
        / "results"
    )

    # Optional copy under Project B.
    candidates.append(
        PROJECT_ROOT
        / "part_a_results"
    )

    return candidates


def _load_project_a_artifacts(
    results_root: pathlib.Path,
) -> dict[str, object] | None:
    """
    Load the four Project A artifacts required by Project B.
    """
    paths = {
        "equity_returns":
            results_root
            / "data"
            / "equity_returns_wide.csv",

        "crypto_returns":
            results_root
            / "data"
            / "crypto_returns_wide.csv",

        "combined_returns":
            results_root
            / "data"
            / "combined_returns_panel.csv",

        "headline_panel":
            results_root
            / "data"
            / "headline_daily_panel.csv",
    }

    if not all(
        path.exists()
        for path in paths.values()
    ):
        return None

    equity_returns = _read_wide_returns(
        paths["equity_returns"]
    )

    crypto_returns = _read_wide_returns(
        paths["crypto_returns"]
    )

    combined_returns = _read_wide_returns(
        paths["combined_returns"]
    )

    headline_panel = pd.read_csv(
        paths["headline_panel"],
        parse_dates=["trading_date"],
    )

    return {
        "equity_returns": equity_returns,
        "crypto_returns": crypto_returns,
        "combined_returns": combined_returns,
        "headline_panel": headline_panel,
        "source": str(
            results_root.resolve()
        ),
    }


def _rebuild_project_a_foundation() -> dict[str, object]:
    """
    Rebuild the Project A Stations 1-2 foundation if saved artifacts
    cannot be found.
    """
    print(
        "Project A artifacts were not found. "
        "Rebuilding the data foundation..."
    )

    # Station 1: load and clean raw prices.
    equity_prices = etl.load_clean_equities()
    crypto_prices = etl.load_clean_crypto()

    # Station 2: calculate returns separately on native calendars.
    equity_returns = features.daily_returns(
        equity_prices
    )

    crypto_returns = features.daily_returns(
        crypto_prices
    )

    # Align already-computed crypto returns to the equity calendar.
    combined_returns = pd.concat(
        [
            equity_returns.add_prefix("EQ_"),
            crypto_returns
            .reindex(equity_returns.index)
            .add_prefix("CR_"),
        ],
        axis=1,
    )

    combined_returns.index.name = "date"

    # Build the trading-day headline panel.
    headline_panel = (
        features.assemble_headline_panel(
            data_access.load_news_headlines()
        )
    )

    return {
        "equity_returns": equity_returns,
        "crypto_returns": crypto_returns,
        "combined_returns": combined_returns,
        "headline_panel": headline_panel,
        "source": (
            "Rebuilt from the student's "
            "Project A data-access, ETL and "
            "feature modules."
        ),
    }


def _load_or_rebuild_project_a() -> dict[str, object]:
    """
    Prefer existing Project A artifacts; rebuild only when necessary.
    """
    for results_root in (
        _project_a_result_candidates()
    ):
        artifacts = (
            _load_project_a_artifacts(
                results_root
            )
        )

        if artifacts is not None:
            print(
                "Using Project A results:",
                results_root,
            )

            return artifacts

    return _rebuild_project_a_foundation()


# =====================================================================
# Helpers
# =====================================================================


def _announce(
    stage: int,
    name: str,
) -> None:
    """Print a clear pipeline-stage heading."""
    print()
    print("=" * 78)
    print(
        f"Project B Stage {stage}: {name}"
    )
    print("=" * 78)


def _add_sector_mapping(
    fund_weights: pd.DataFrame,
    ticker_day_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add equity-sector metadata required by the fusion layer.

    Crypto assets intentionally remain without a sector and therefore
    are not directly tilted by the sentiment signal.
    """
    output = fund_weights.copy()

    if "sector" in output.columns:
        return output

    mapping = (
        ticker_day_scores[
            [
                "ticker",
                "sector",
            ]
        ]
        .dropna()
        .drop_duplicates()
    )

    output = output.merge(
        mapping,
        left_on="underlying_ticker",
        right_on="ticker",
        how="left",
        suffixes=(
            "",
            "_sector_map",
        ),
    )

    if "ticker_sector_map" in output.columns:
        output = output.drop(
            columns=[
                "ticker_sector_map"
            ]
        )

    return output


# =====================================================================
# Main pipeline
# =====================================================================


def main() -> None:
    """
    Run the complete NexaVest Project B workflow.
    """

    # -----------------------------------------------------------------
    # Stage 0: Project A inputs
    # -----------------------------------------------------------------

    project_a = (
        _load_or_rebuild_project_a()
    )

    equity_returns = project_a[
        "equity_returns"
    ]

    crypto_returns = project_a[
        "crypto_returns"
    ]

    combined_returns = project_a[
        "combined_returns"
    ]

    headline_panel = project_a[
        "headline_panel"
    ]

    foundation_source = project_a[
        "source"
    ]

    print()
    print("=== PROJECT A INPUTS ===")
    print(
        "Equity returns:",
        equity_returns.shape,
    )
    print(
        "Crypto returns:",
        crypto_returns.shape,
    )
    print(
        "Combined returns:",
        combined_returns.shape,
    )
    print(
        "Headline panel:",
        headline_panel.shape,
    )


    # -----------------------------------------------------------------
    # Stage 1: Portfolios
    # -----------------------------------------------------------------

    _announce(
        1,
        "Portfolios: 12 base walk-forward OOS funds",
    )

    # Use the public starter entry point.
    # The internal Project-B action dispatch builds:
    #
    # Equity:
    #   Equal Weight
    #   Minimum Variance
    #   Maximum Sharpe
    #   Risk Parity
    #
    # Crypto:
    #   Equal Weight
    #   Minimum Variance
    #   Maximum Sharpe
    #   Risk Parity
    #
    # Combined:
    #   Equal Weight
    #   Minimum Variance
    #   Maximum Sharpe
    #   Risk Parity
    #
    base_outputs = portfolios._build_base_funds(
        equity_returns=equity_returns,
        crypto_returns=crypto_returns,
        combined_returns=combined_returns,
    )

    # Keep a compatibility alias for the fusion module.
    base_outputs["performance"] = base_outputs["performance_metrics"].copy()

    print(
        "Base funds:",
        base_outputs["performance_metrics"]["fund"].nunique(),
    )

    print(
        "Base funds:",
        base_outputs[
            "performance"
        ]["fund"].nunique(),
    )


    # -----------------------------------------------------------------
    # Stage 2: Sentiment
    # -----------------------------------------------------------------

    _announce(
        2,
        "Sentiment: NexaVest finance-aware sector index",
    )

    # Keep the project path as metadata so sentiment.py can save its
    # own audit tables and figures without changing its public API.
    sentiment_input = (
        headline_panel.copy()
    )

    sentiment_input.attrs[
        "project_root"
    ] = PROJECT_ROOT

    ticker_day_scores = (
        sentiment.score_headlines(
            sentiment_input
        )
    )

    sector_index = (
        sentiment.sector_sentiment_index(
            ticker_day_scores
        )
    )

    print(
        "Ticker-day sentiment rows:",
        len(ticker_day_scores),
    )

    print(
        "Sector-index rows:",
        len(sector_index),
    )

    print(
        "Sectors:",
        sector_index[
            "sector"
        ].nunique(),
    )


    # -----------------------------------------------------------------
    # Stage 3: Fusion
    # -----------------------------------------------------------------

    _announce(
        3,
        "Fusion: coverage-adjusted sentiment tilt",
    )

    # The portfolio output does not require sector in its base schema.
    # Add the mapping only for the fusion layer.
    fusion_base_outputs = {
        key: (
            value.copy()
            if isinstance(
                value,
                pd.DataFrame,
            )
            else value
        )
        for key, value
        in base_outputs.items()
    }

    fusion_base_outputs[
        "fund_weights"
    ] = _add_sector_mapping(
        base_outputs[
            "fund_weights"
        ],
        ticker_day_scores,
    )

    (
        final_returns,
        final_weights,
        final_performance,
    ) = fusion.build_fusion_funds(
        base_outputs=
            fusion_base_outputs,

        sector_sentiment=
            sector_index,

        equity_returns=
            equity_returns,

        combined_returns=
            combined_returns,
    )

    comparison = (
        fusion.save_fusion_outputs(
            fund_returns=
                final_returns,

            performance=
                final_performance,

            project_root=
                PROJECT_ROOT,
        )
    )

    print(
        "Final fund count:",
        final_performance[
            "fund"
        ].nunique(),
    )

    print()
    print(
        "Fusion comparison:"
    )

    print(
        comparison.to_string(
            index=False
        )
    )


    # -----------------------------------------------------------------
    # Stage 4: Save the final required portfolio outputs
    # -----------------------------------------------------------------

    _announce(
        4,
        "Save final outputs and portfolio audit",
    )

    # Use the portfolio module's public save workflow so schemas,
    # supporting tables and required figures remain consistent with
    # the portfolio implementation.
    final_results = {
        "fund_returns": final_returns,
        "fund_weights": final_weights,
        "performance_metrics": final_performance,
        "rebalance_audit": base_outputs["rebalance_audit"],
    }

    saved_paths = portfolios.save_portfolio_outputs(
        final_results,
        project_root=PROJECT_ROOT,
    )

    print("\nSaved portfolio outputs:")
    for name, path in saved_paths.items():
        print(f"{name}: {path}")


    # -----------------------------------------------------------------
    # Final validation summary
    # -----------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "NexaVest Project B build complete."
    )
    print("=" * 78)

    required_files = [
        PROJECT_ROOT
        / "results"
        / "data"
        / "fund_returns.csv",

        PROJECT_ROOT
        / "results"
        / "data"
        / "fund_weights.csv",

        PROJECT_ROOT
        / "results"
        / "data"
        / "sector_sentiment_index.csv",

        PROJECT_ROOT
        / "results"
        / "tables"
        / "performance_metrics.csv",
    ]

    for path in required_files:
        print(
            f"{path.relative_to(PROJECT_ROOT)}: "
            f"{'OK' if path.exists() else 'MISSING'}"
        )

    print()
    print(
        "Final funds:",
        final_performance[
            "fund"
        ].nunique(),
    )

    print(
        "Expected final funds: 14"
    )

    print()
    print(
        "Next:"
    )
    print(
        "  streamlit run streamlit_app.py"
    )
    print(
        "  python scripts/check_handin.py"
    )


if __name__ == "__main__":
    main()
