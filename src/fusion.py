"""Station 3 (extension) - fuse sentiment into the funds.

Tilt or factor: combine your sentiment signal with the portfolio weights,
look-ahead safe, then test whether it adds value. An honest negative result,
explained, is good work.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.portfolios as portfolios

# ============================================================
# NexaVest sentiment fusion design
# ============================================================

# Pre-specified before inspecting final OOS fusion performance.
SENTIMENT_TILT_ALPHA = 0.30

# Relative sentiment is clipped before being translated into a
# portfolio tilt so that one extreme sector observation cannot
# dominate the allocation.
SIGNAL_CLIP = 2.0
TRANSACTION_COST_BPS = 10.0


def _single_asset_cap(n_assets: int) -> float:
    """
    Match the diversification cap used in the base portfolio module.
    """
    if n_assets <= 12:
        return 0.30

    if n_assets <= 30:
        return 0.18

    return 0.10


def _scale_to_budget_with_cap(
    raw_weights: pd.Series,
    sleeve_budget: float,
    cap: float,
) -> pd.Series:
    """
    Scale non-negative raw weights back to a fixed sleeve budget
    while respecting a single-asset cap.
    """
    values = (
        pd.to_numeric(
            raw_weights,
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0)
        .copy()
    )

    if sleeve_budget <= 0 or values.empty:
        return pd.Series(
            0.0,
            index=values.index,
            dtype=float,
        )

    if values.sum() <= 0:
        values[:] = 1.0

    values = values / values.sum() * sleeve_budget

    for _ in range(100):
        over_cap = values > cap + 1e-12

        if not over_cap.any():
            break

        excess = float((values.loc[over_cap] - cap).sum())

        values.loc[over_cap] = cap

        free = ~over_cap

        capacity = (cap - values.loc[free]).clip(lower=0.0)

        if capacity.sum() <= 0:
            break

        values.loc[free] += excess * capacity / capacity.sum()

    difference = sleeve_budget - float(values.sum())

    if abs(difference) > 1e-12:
        free = values < cap - 1e-12

        if free.any():
            remaining_capacity = cap - values.loc[free]

            if remaining_capacity.sum() > 0:
                values.loc[free] += difference * remaining_capacity / remaining_capacity.sum()

    return values


def _latest_sector_signal(
    sentiment: pd.DataFrame,
    rebalance_dates: pd.Series,
) -> pd.DataFrame:
    """
    Match each rebalance date to the latest available sector signal.

    The signal is already lagged and standardised using prior history.
    Coverage is used to reduce the influence of low-information
    sector observations.
    """
    required = {
        "trading_date",
        "sector",
        "sentiment_z",
        "coverage_ratio",
    }

    missing = required.difference(sentiment.columns)

    if missing:
        raise ValueError(f"sentiment is missing columns: {sorted(missing)}")

    available = sentiment[
        [
            "trading_date",
            "sector",
            "sentiment_z",
            "coverage_ratio",
        ]
    ].copy()

    available["trading_date"] = pd.to_datetime(
        available["trading_date"],
        errors="coerce",
    )

    available["sentiment_z"] = pd.to_numeric(
        available["sentiment_z"],
        errors="coerce",
    )

    available["coverage_ratio"] = pd.to_numeric(
        available["coverage_ratio"],
        errors="coerce",
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    available = available.dropna(
        subset=[
            "trading_date",
            "sector",
        ]
    )

    # --------------------------------------------------------
    # NexaVest innovation:
    # reduce signal strength when sector headline coverage is low.
    # --------------------------------------------------------

    available["coverage_adjusted_signal"] = available["sentiment_z"] * np.sqrt(
        available["coverage_ratio"]
    )

    available["coverage_adjusted_signal"] = (
        available["coverage_adjusted_signal"].clip(
            lower=-SIGNAL_CLIP,
            upper=SIGNAL_CLIP,
        )
        / SIGNAL_CLIP
    )

    target_dates = pd.DataFrame(
        {"rebalance_date": pd.to_datetime(pd.Series(rebalance_dates).dropna().unique())}
    ).sort_values("rebalance_date")

    rows = []

    for sector_name, group in available.groupby(
        "sector",
        sort=True,
    ):
        group = group.sort_values("trading_date")

        matched = pd.merge_asof(
            target_dates,
            group,
            left_on="rebalance_date",
            right_on="trading_date",
            direction="backward",
            allow_exact_matches=True,
        )

        matched["sector"] = sector_name

        rows.append(
            matched[
                [
                    "rebalance_date",
                    "sector",
                    "sentiment_z",
                    "coverage_ratio",
                    "coverage_adjusted_signal",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "rebalance_date",
                "sector",
                "sentiment_z",
                "coverage_ratio",
                "coverage_adjusted_signal",
            ]
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def apply_sentiment(weights: pd.DataFrame, sentiment: pd.DataFrame):
    """TODO: your fusion rule (for example tilt weights toward high-sentiment names)."""
    required = {
        "rebalance_date",
        "ticker",
        "weight",
        "sector",
    }

    missing = required.difference(weights.columns)

    if missing:
        raise ValueError(f"weights is missing columns: {sorted(missing)}")

    adjusted = weights.copy()

    adjusted["rebalance_date"] = pd.to_datetime(
        adjusted["rebalance_date"],
        errors="coerce",
    )

    adjusted["weight"] = pd.to_numeric(
        adjusted["weight"],
        errors="coerce",
    )

    adjusted = adjusted.dropna(
        subset=[
            "rebalance_date",
            "ticker",
            "weight",
        ]
    )

    signals = _latest_sector_signal(
        sentiment,
        adjusted["rebalance_date"],
    )

    adjusted = adjusted.merge(
        signals,
        on=[
            "rebalance_date",
            "sector",
        ],
        how="left",
        validate="many_to_one",
    )

    # No usable sentiment -> no change.
    adjusted["coverage_adjusted_signal"] = adjusted["coverage_adjusted_signal"].fillna(0.0)

    adjusted["base_weight"] = adjusted["weight"]

    # Equity rows have sector labels.
    adjusted["is_equity"] = adjusted["sector"].notna()

    adjusted["tilt_multiplier"] = 1.0

    adjusted.loc[
        adjusted["is_equity"],
        "tilt_multiplier",
    ] = (
        1.0
        + SENTIMENT_TILT_ALPHA
        * adjusted.loc[
            adjusted["is_equity"],
            "coverage_adjusted_signal",
        ]
    )

    group_columns = [
        column
        for column in [
            "fund",
            "asset_family",
            "method",
            "rebalance_date",
        ]
        if column in adjusted.columns
    ]

    if "rebalance_date" not in group_columns:
        group_columns.append("rebalance_date")

    output_groups = []

    for _, group in adjusted.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        group = group.copy()

        equity_mask = group["is_equity"]

        crypto_mask = ~equity_mask

        equity_budget = float(
            group.loc[
                equity_mask,
                "base_weight",
            ].sum()
        )

        crypto_budget = float(
            group.loc[
                crypto_mask,
                "base_weight",
            ].sum()
        )

        tilted_equity = (
            group.loc[
                equity_mask,
                "base_weight",
            ]
            * group.loc[
                equity_mask,
                "tilt_multiplier",
            ]
        )

        if equity_mask.any() and tilted_equity.sum() > 0:
            cap = _single_asset_cap(len(group))

            group.loc[
                equity_mask,
                "weight",
            ] = _scale_to_budget_with_cap(
                tilted_equity,
                sleeve_budget=equity_budget,
                cap=cap,
            )

        # Preserve crypto sleeve exactly.
        if crypto_mask.any() and crypto_budget > 0:
            group.loc[
                crypto_mask,
                "weight",
            ] = group.loc[
                crypto_mask,
                "base_weight",
            ]

        total_weight = float(group["weight"].sum())

        if total_weight > 0:
            group["weight"] = group["weight"] / total_weight

        output_groups.append(group)

    result = pd.concat(
        output_groups,
        ignore_index=True,
    )

    result["weight_change_from_sentiment"] = result["weight"] - result["base_weight"]

    result.attrs["fusion_rule"] = (
        "coverage-adjusted sector sentiment tilt: "
        "signal = clip(z * sqrt(coverage), -2, 2) / 2; "
        "weight multiplier = 1 + alpha * signal; "
        "equity sleeve is renormalised while crypto weights "
        "and the Combined crypto budget are preserved."
    )

    result.attrs["alpha"] = SENTIMENT_TILT_ALPHA

    result.attrs["lookahead_policy"] = (
        "Only historical, lagged sentiment is matched backward to each rebalance date."
    )

    return result.drop(
        columns=[
            "is_equity",
        ]
    )
def _returns_from_weight_schedule(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """
    Apply a sequence of rebalance-date target weights to future returns.

    Each target portfolio is held from its rebalance date until the day
    before the next rebalance date. Transaction cost is charged only on
    the first day of each holding block.
    """
    panel = returns.copy()

    panel.index = pd.to_datetime(
        panel.index,
        errors="coerce",
    )

    panel = panel.loc[
        ~panel.index.isna()
    ].sort_index()

    schedule = weights[
        [
            "rebalance_date",
            "ticker",
            "weight",
        ]
    ].copy()

    schedule["rebalance_date"] = pd.to_datetime(
        schedule["rebalance_date"],
        errors="coerce",
    )

    schedule["weight"] = pd.to_numeric(
        schedule["weight"],
        errors="coerce",
    )

    schedule = schedule.dropna(
        subset=[
            "rebalance_date",
            "ticker",
            "weight",
        ]
    )

    rebalance_dates = sorted(
        schedule["rebalance_date"].unique()
    )

    if not rebalance_dates:
        raise ValueError(
            "No valid rebalance dates were found."
        )

    records = []

    previous_weights = None

    for i, rebalance_date in enumerate(
        rebalance_dates
    ):
        next_date = (
            rebalance_dates[i + 1]
            if i + 1 < len(rebalance_dates)
            else None
        )

        # Target weights for this rebalance.
        current = (
            schedule.loc[
                schedule["rebalance_date"].eq(
                    rebalance_date
                ),
                [
                    "ticker",
                    "weight",
                ],
            ]
            .drop_duplicates(
                subset="ticker",
                keep="last",
            )
            .set_index("ticker")["weight"]
        )

        # Future holding block.
        block = panel.loc[
            panel.index >= rebalance_date
        ]

        if next_date is not None:
            block = block.loc[
                block.index < next_date
            ]

        if block.empty:
            continue

        block_returns, _ = (
            portfolios
            ._portfolio_returns_from_weights(
                test_returns=block,
                weights=current,
                previous_weights=previous_weights,
                transaction_cost_bps=
                    transaction_cost_bps,
            )
        )

        block_returns[
            "rebalance_date"
        ] = pd.Timestamp(
            rebalance_date
        )

        records.append(
            block_returns
        )

        previous_weights = current

    if not records:
        raise ValueError(
            "No holding-period returns were generated."
        )

    return (
        pd.concat(
            records,
            ignore_index=True,
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

def _fused_fund_metric_row(
    daily_returns: pd.DataFrame,
    tilted_weights: pd.DataFrame,
    fund_name: str,
    asset_family: str,
) -> dict:
    """
    Build the performance row for one sentiment-enhanced fund.
    """
    metrics = portfolios.performance_metrics(
        daily_returns.set_index("date")["daily_return"],
        periods_per_year=252,
    )

    rebalance_turnover = daily_returns.loc[
        daily_returns["turnover"].gt(0),
        "turnover",
    ]

    current_holdings_count = int(
        tilted_weights.sort_values("rebalance_date")
        .groupby("ticker")["weight"]
        .last()
        .gt(1e-6)
        .sum()
    )

    metrics.update(
        {
            "periods_per_year": 252,
            "estimation_window": 252,
            "rebalance_step": 21,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "average_turnover": float(rebalance_turnover.mean())
            if not rebalance_turnover.empty
            else np.nan,
            "first_live_date": pd.Timestamp(daily_returns["date"].min()),
            "last_live_date": pd.Timestamp(daily_returns["date"].max()),
            "fund": fund_name,
            "asset_family": asset_family,
            "method": "min_variance_sentiment",
            "current_holdings_count": current_holdings_count,
        }
    )

    return metrics


def build_fusion_funds(
    base_outputs: dict[str, pd.DataFrame],
    sector_sentiment: pd.DataFrame,
    equity_returns: pd.DataFrame,
    combined_returns: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build Equity and Combined Minimum Variance sentiment-enhanced funds.
    """
    fund_returns = base_outputs["fund_returns"].copy()

    fund_weights = base_outputs["fund_weights"].copy()

    performance = base_outputs["performance"].copy()

    candidate_bases = [
        "NexaVest Equity Minimum Variance",
        "NexaVest Combined Minimum Variance",
    ]

    available_bases = [name for name in candidate_bases if name in set(fund_weights["fund"])]

    if not available_bases:
        return (
            fund_returns,
            fund_weights,
            performance,
        )

    base_weights = fund_weights.loc[fund_weights["fund"].isin(available_bases)].copy()

    tilted_all = apply_sentiment(
        base_weights,
        sector_sentiment,
    )

    fused_return_frames = []
    fused_weight_frames = []
    fused_metric_rows = []

    for base_name in available_bases:
        asset_family = "Combined" if base_name.startswith("NexaVest Combined") else "Equity"

        return_panel = combined_returns if asset_family == "Combined" else equity_returns

        enhanced_name = f"{base_name} + Sentiment"

        this_weights = tilted_all.loc[tilted_all["fund"].eq(base_name)].copy()

        this_weights["fund"] = enhanced_name

        this_weights["method"] = "min_variance_sentiment"

        daily = _returns_from_weight_schedule(
            returns=return_panel,
            weights=this_weights,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )


        daily["growth_of_one"] = (1.0 + daily["daily_return"]).cumprod()

        daily["drawdown"] = daily["growth_of_one"] / daily["growth_of_one"].cummax() - 1.0

        daily["fund"] = enhanced_name

        daily["asset_family"] = asset_family

        daily["method"] = "min_variance_sentiment"

        fused_return_frames.append(daily)

        fused_weight_frames.append(this_weights)

        fused_metric_rows.append(
            _fused_fund_metric_row(
                daily,
                this_weights,
                enhanced_name,
                asset_family,
            )
        )

    if fused_return_frames:
        fund_returns = pd.concat(
            [
                fund_returns,
                *fused_return_frames,
            ],
            ignore_index=True,
        )

        fund_weights = pd.concat(
            [
                fund_weights,
                *fused_weight_frames,
            ],
            ignore_index=True,
        )

        performance = pd.concat(
            [
                performance,
                pd.DataFrame(fused_metric_rows),
            ],
            ignore_index=True,
        )

    return (
        fund_returns,
        fund_weights,
        performance,
    )


def fusion_comparison_table(
    performance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare Minimum Variance base funds with sentiment-enhanced versions.
    """
    required = {
        "fund",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
    }

    missing = required.difference(performance.columns)

    if missing:
        raise ValueError(f"performance is missing columns: {sorted(missing)}")

    indexed = performance.set_index("fund")

    rows = []

    for asset_family in [
        "Equity",
        "Combined",
    ]:
        base_name = f"NexaVest {asset_family} Minimum Variance"

        enhanced_name = f"{base_name} + Sentiment"

        if base_name not in indexed.index or enhanced_name not in indexed.index:
            continue

        base = indexed.loc[base_name]

        enhanced = indexed.loc[enhanced_name]

        rows.append(
            {
                "asset_family": asset_family,
                "base_fund": base_name,
                "enhanced_fund": enhanced_name,
                "base_annualised_return": base["annualised_return"],
                "enhanced_annualised_return": enhanced["annualised_return"],
                "delta_annualised_return": enhanced["annualised_return"]
                - base["annualised_return"],
                "base_volatility": base["annualised_volatility"],
                "enhanced_volatility": enhanced["annualised_volatility"],
                "delta_volatility": enhanced["annualised_volatility"]
                - base["annualised_volatility"],
                "base_sharpe": base["sharpe_ratio"],
                "enhanced_sharpe": enhanced["sharpe_ratio"],
                "delta_sharpe": enhanced["sharpe_ratio"] - base["sharpe_ratio"],
                "base_max_drawdown": base["maximum_drawdown"],
                "enhanced_max_drawdown": enhanced["maximum_drawdown"],
                "delta_max_drawdown": enhanced["maximum_drawdown"] - base["maximum_drawdown"],
            }
        )

    return pd.DataFrame(rows)


def plot_fusion_before_after(
    fund_returns: pd.DataFrame,
    project_root: str | Path,
) -> Path:
    """
    Plot Combined Minimum Variance before and after sentiment fusion.
    """
    root = Path(project_root)

    figures_dir = root / "results" / "figures"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fund_names = [
        "NexaVest Combined Minimum Variance",
        "NexaVest Combined Minimum Variance + Sentiment",
    ]

    selected = fund_returns.loc[fund_returns["fund"].isin(fund_names)].copy()

    selected["date"] = pd.to_datetime(
        selected["date"],
        errors="coerce",
    )

    growth = selected.pivot(
        index="date",
        columns="fund",
        values="growth_of_one",
    )

    drawdown = selected.pivot(
        index="date",
        columns="fund",
        values="drawdown",
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        sharex=True,
    )

    for fund_name in growth.columns:
        axes[0].plot(
            growth.index,
            growth[fund_name],
            linewidth=1.5,
            label=fund_name,
        )

    axes[0].set_ylabel("Growth of $1")

    axes[0].set_title("NexaVest Fusion Before vs After: Combined Minimum Variance")

    axes[0].grid(alpha=0.25)

    axes[0].legend(
        frameon=False,
        fontsize=8,
    )

    for fund_name in drawdown.columns:
        axes[1].plot(
            drawdown.index,
            drawdown[fund_name] * 100.0,
            linewidth=1.4,
            label=fund_name,
        )

    axes[1].set_xlabel("Date")

    axes[1].set_ylabel("Drawdown (%)")

    axes[1].grid(alpha=0.25)

    figure.text(
        0.01,
        0.01,
        "Source: NexaVest walk-forward OOS results. "
        "Sentiment uses lagged historical sector information "
        "and only reallocates the equity sleeve.",
        fontsize=8,
    )

    figure.tight_layout(rect=(0, 0.04, 1, 1))

    path = figures_dir / "fusion_before_after.png"

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path


def save_fusion_outputs(
    fund_returns: pd.DataFrame,
    performance: pd.DataFrame,
    project_root: str | Path,
) -> pd.DataFrame:
    """
    Save the fusion comparison table and before/after figure.
    """
    root = Path(project_root)

    tables_dir = root / "results" / "tables"

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = fusion_comparison_table(performance)

    comparison.to_csv(
        tables_dir / "fusion_comparison.csv",
        index=False,
    )

    plot_fusion_before_after(
        fund_returns,
        project_root=root,
    )

    return comparison
