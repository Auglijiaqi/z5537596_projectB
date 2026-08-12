"""Station 3 - your funds: optimal portfolios + out-of-sample backtest.

Build at least a combined equity-plus-crypto fund with two optimisation methods.
Backtest rules: walk-forward, no look-ahead, weights from past data only, annualise
with 252 (equity) or 365 (crypto). See the brief, Part B.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class BacktestConfig:
    periods_per_year: int
    estimation_window: int
    rebalance_step: int
    transaction_cost_bps: float = 10.0
    crypto_cap: float | None = None


def _infer_periods_per_year(index: pd.Index) -> int:
    """Infer whether the return panel follows a 252- or 365-day calendar."""
    dates = pd.DatetimeIndex(index)

    if len(dates) < 2:
        return 252

    weekend_share = (dates.dayofweek >= 5).mean()

    return 365 if weekend_share > 0.05 else 252


def _config_for_returns(returns: pd.DataFrame) -> BacktestConfig:
    """Create the backtest configuration from the return panel calendar."""
    periods = _infer_periods_per_year(returns.index)

    if periods == 365:
        return BacktestConfig(
            periods_per_year=365,
            estimation_window=365,
            rebalance_step=30,
            crypto_cap=None,
        )

    columns = pd.Index(returns.columns.astype(str))

    has_equity = columns.str.startswith("EQ_").any()
    has_crypto = columns.str.startswith("CR_").any()

    crypto_cap = 0.25 if has_equity and has_crypto else None

    return BacktestConfig(
        periods_per_year=252,
        estimation_window=252,
        rebalance_step=21,
        crypto_cap=crypto_cap,
    )


def _clean_return_panel(returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardise a wide return panel."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("returns is empty.")

    clean = returns.copy()

    clean.index = pd.to_datetime(clean.index, errors="coerce")
    clean = clean.loc[clean.index.notna()].sort_index()

    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.dropna(how="all")
    columns = pd.Index(clean.columns.astype(str))

    equity_columns = columns[columns.str.startswith("EQ_")]

    crypto_columns = columns[columns.str.startswith("CR_")]

    # A Combined fund follows the equity trading calendar.
    # Rows with no equity return information should therefore
    # not count as valid Combined return observations.
    if len(equity_columns) > 0 and len(crypto_columns) > 0:
        clean = clean.loc[~clean[equity_columns].isna().all(axis=1)]

    if clean.shape[1] < 2:
        raise ValueError("At least two assets are required.")

    return clean


def _asset_cap(n_assets: int) -> float:
    """Return a diversification cap based on universe size."""
    if n_assets <= 12:
        return 0.30

    if n_assets <= 30:
        return 0.18

    return 0.10


def _normalise_with_cap(
    weights: np.ndarray,
    max_weight: float,
    target_sum: float = 1.0,
) -> np.ndarray:
    """
    Scale non-negative weights to a target budget while respecting
    a maximum weight per asset.
    """
    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("weights must be a non-empty one-dimensional array.")

    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")

    if target_sum < 0:
        raise ValueError("target_sum cannot be negative.")

    if len(weights) * max_weight < target_sum - 1e-12:
        raise ValueError("The requested target sum is infeasible under the asset cap.")

    if target_sum == 0:
        return np.zeros_like(weights)

    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.clip(weights, 0.0, None)

    if weights.sum() <= 0:
        weights = np.ones_like(weights)

    result = np.zeros_like(weights)
    active = np.ones(len(weights), dtype=bool)
    remaining_budget = float(target_sum)

    for _ in range(len(weights) + 1):
        if not active.any():
            break

        raw = weights[active]

        if raw.sum() <= 0:
            raw = np.ones_like(raw)

        proposed = raw / raw.sum() * remaining_budget
        capped = proposed > max_weight + 1e-12

        active_indices = np.where(active)[0]

        if not capped.any():
            result[active_indices] = proposed
            remaining_budget = 0.0
            break

        capped_indices = active_indices[capped]
        result[capped_indices] = max_weight

        remaining_budget -= max_weight * len(capped_indices)
        active[capped_indices] = False

    if abs(result.sum() - target_sum) > 1e-8:
        raise ValueError("Unable to construct feasible capped weights.")

    return result


def _optimisation_constraints(
    tickers: pd.Index,
    crypto_cap: float | None,
) -> list[dict]:
    """
    Build SLSQP constraints for a long-only, fully invested portfolio.
    """
    tickers = pd.Index(tickers.astype(str))

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
        }
    ]

    if crypto_cap is not None:
        crypto_mask = np.asarray(
            tickers.str.startswith("CR_"),
            dtype=float,
        )

        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w, mask=crypto_mask, cap=crypto_cap: float(cap - np.dot(mask, w)),
            }
        )

    return constraints


def _apply_crypto_sleeve_cap(
    weights: np.ndarray,
    tickers: pd.Index,
    crypto_cap: float | None,
    max_weight: float,
) -> np.ndarray:
    """
    Limit the total crypto allocation in a Combined portfolio while
    preserving a fully invested portfolio.
    """
    weights = np.asarray(weights, dtype=float)

    if crypto_cap is None:
        return _normalise_with_cap(weights, max_weight)

    tickers = pd.Index(tickers.astype(str))
    crypto_mask = np.asarray(
        tickers.str.startswith("CR_"),
        dtype=bool,
    )
    equity_mask = ~crypto_mask

    if not crypto_mask.any() or not equity_mask.any():
        return _normalise_with_cap(weights, max_weight)

    weights = _normalise_with_cap(weights, max_weight)

    crypto_total = float(weights[crypto_mask].sum())

    if crypto_total <= crypto_cap + 1e-12:
        return weights

    adjusted = np.zeros_like(weights)

    adjusted[crypto_mask] = _normalise_with_cap(
        weights[crypto_mask],
        max_weight=max_weight,
        target_sum=crypto_cap,
    )

    adjusted[equity_mask] = _normalise_with_cap(
        weights[equity_mask],
        max_weight=max_weight,
        target_sum=1.0 - crypto_cap,
    )

    return adjusted


def _equal_weight_weights(
    train: pd.DataFrame,
    crypto_cap: float | None = None,
) -> pd.Series:
    """
    Construct an equal-weight portfolio that respects the same
    diversification constraints as the optimised portfolios.
    """
    n_assets = train.shape[1]

    if n_assets == 0:
        raise ValueError("No assets are available for equal weighting.")

    tickers = pd.Index(train.columns.astype(str))
    max_weight = _asset_cap(n_assets)

    raw = np.repeat(1.0 / n_assets, n_assets)

    weights = _apply_crypto_sleeve_cap(
        raw,
        tickers=tickers,
        crypto_cap=crypto_cap,
        max_weight=max_weight,
    )

    output = pd.Series(
        weights,
        index=tickers,
        name="weight",
    )

    output.attrs["solver_success"] = True
    output.attrs["used_fallback"] = False
    output.attrs["solver_message"] = "Deterministic equal-weight rule."

    return output


def _minimum_variance_weights(
    train: pd.DataFrame,
    crypto_cap: float | None = None,
) -> pd.Series:
    """
    Estimate a long-only minimum-variance portfolio from past returns only.
    """
    clean_train = train.copy()

    # Remove assets with no usable observations in the current training window.
    valid_assets = clean_train.notna().sum() > 1
    clean_train = clean_train.loc[:, valid_assets]

    if clean_train.shape[1] < 2:
        raise ValueError("At least two assets with valid return observations are required.")

    # Remaining missing values are filled using each asset's training-window mean.
    clean_train = clean_train.apply(lambda column: column.fillna(column.mean()))

    tickers = pd.Index(clean_train.columns.astype(str))
    n_assets = len(tickers)
    max_weight = _asset_cap(n_assets)

    covariance = clean_train.cov().to_numpy(dtype=float)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ covariance @ weights)

    initial = _equal_weight_weights(
        clean_train,
        crypto_cap=crypto_cap,
    ).to_numpy(dtype=float)

    bounds = [(0.0, max_weight) for _ in range(n_assets)]

    constraints = _optimisation_constraints(
        tickers=tickers,
        crypto_cap=crypto_cap,
    )

    result = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 500,
            "ftol": 1e-10,
            "disp": False,
        },
    )

    if not result.success or not np.isfinite(result.x).all():
        fallback = _inverse_volatility_weights(
            clean_train,
            crypto_cap=crypto_cap,
        )

        fallback.attrs["solver_message"] = (
            f"Minimum-variance optimisation failed: "
            f"{result.message}. "
            f"Inverse-volatility fallback used."
        )

        return fallback

    weights = _apply_crypto_sleeve_cap(
        result.x,
        tickers=tickers,
        crypto_cap=crypto_cap,
        max_weight=max_weight,
    )

    output = pd.Series(
        weights,
        index=tickers,
        name="weight",
    )

    output.attrs["solver_success"] = bool(result.success)
    output.attrs["solver_message"] = str(result.message)

    return output


def _maximum_sharpe_weights(
    train: pd.DataFrame,
    crypto_cap: float | None = None,
) -> pd.Series:
    """
    Estimate a long-only maximum-Sharpe portfolio using only
    the returns available in the training window.

    The risk-free rate is assumed to be zero.
    """
    clean_train = train.copy()

    valid_assets = clean_train.notna().sum() > 1
    clean_train = clean_train.loc[:, valid_assets]

    if clean_train.shape[1] < 2:
        raise ValueError("At least two assets with valid return observations are required.")

    clean_train = clean_train.apply(lambda column: column.fillna(column.mean()))

    tickers = pd.Index(clean_train.columns.astype(str))
    n_assets = len(tickers)
    max_weight = _asset_cap(n_assets)

    mean_returns = clean_train.mean().to_numpy(dtype=float)
    covariance = clean_train.cov().to_numpy(dtype=float)

    def objective(weights: np.ndarray) -> float:
        portfolio_return = float(weights @ mean_returns)
        portfolio_variance = float(weights @ covariance @ weights)

        if portfolio_variance <= 0:
            return 1e6

        portfolio_volatility = np.sqrt(portfolio_variance)

        # Minimising the negative Sharpe ratio is equivalent
        # to maximising the Sharpe ratio.
        return -portfolio_return / portfolio_volatility

    initial = _equal_weight_weights(
        clean_train,
        crypto_cap=crypto_cap,
    ).to_numpy(dtype=float)

    bounds = [(0.0, max_weight) for _ in range(n_assets)]

    constraints = _optimisation_constraints(
        tickers=tickers,
        crypto_cap=crypto_cap,
    )

    result = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 500,
            "ftol": 1e-10,
            "disp": False,
        },
    )

    if not result.success or not np.isfinite(result.x).all():
        fallback = _inverse_volatility_weights(
            clean_train,
            crypto_cap=crypto_cap,
        )

        fallback.attrs["solver_message"] = (
            f"Maximum-Sharpe optimisation failed: "
            f"{result.message}. "
            f"Inverse-volatility fallback used."
        )

        return fallback

    weights = _apply_crypto_sleeve_cap(
        result.x,
        tickers=tickers,
        crypto_cap=crypto_cap,
        max_weight=max_weight,
    )

    output = pd.Series(
        weights,
        index=tickers,
        name="weight",
    )

    output.attrs["solver_success"] = bool(result.success)
    output.attrs["solver_message"] = str(result.message)

    return output


def _risk_parity_weights(
    train: pd.DataFrame,
    crypto_cap: float | None = None,
) -> pd.Series:
    """
    Construct a long-only risk-parity portfolio by making each
    asset's share of total portfolio variance as similar as possible.
    """
    clean_train = train.copy()

    valid_assets = clean_train.notna().sum() > 1
    clean_train = clean_train.loc[:, valid_assets]

    if clean_train.shape[1] < 2:
        raise ValueError("At least two assets with valid return observations are required.")

    clean_train = clean_train.apply(lambda column: column.fillna(column.mean()))

    tickers = pd.Index(clean_train.columns.astype(str))
    n_assets = len(tickers)
    max_weight = _asset_cap(n_assets)

    covariance = clean_train.cov().to_numpy(dtype=float)

    target_risk_share = np.repeat(
        1.0 / n_assets,
        n_assets,
    )

    def objective(weights: np.ndarray) -> float:
        marginal_risk = covariance @ weights
        component_variance = weights * marginal_risk
        total_variance = float(component_variance.sum())

        if total_variance <= 0:
            return 1e6

        risk_share = component_variance / total_variance

        return float(np.sum((risk_share - target_risk_share) ** 2))

    initial = _equal_weight_weights(
        clean_train,
        crypto_cap=crypto_cap,
    ).to_numpy(dtype=float)

    bounds = [(0.0, max_weight) for _ in range(n_assets)]

    constraints = _optimisation_constraints(
        tickers=tickers,
        crypto_cap=crypto_cap,
    )

    result = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-10,
            "disp": False,
        },
    )

    if not result.success or not np.isfinite(result.x).all():
        fallback = _inverse_volatility_weights(
            clean_train,
            crypto_cap=crypto_cap,
        )

        fallback.attrs["solver_message"] = (
            f"Risk-parity optimisation failed: {result.message}. Inverse-volatility fallback used."
        )

        return fallback

    weights = _apply_crypto_sleeve_cap(
        result.x,
        tickers=tickers,
        crypto_cap=crypto_cap,
        max_weight=max_weight,
    )

    output = pd.Series(
        weights,
        index=tickers,
        name="weight",
    )

    output.attrs["solver_success"] = bool(result.success)
    output.attrs["used_fallback"] = False
    output.attrs["solver_message"] = str(result.message)

    return output


def _inverse_volatility_weights(
    train: pd.DataFrame,
    crypto_cap: float | None = None,
) -> pd.Series:
    """
    Construct fallback weights inversely proportional to historical volatility.
    Used only when an optimisation method fails.
    """
    clean_train = train.copy()

    valid_assets = clean_train.notna().sum() > 1
    clean_train = clean_train.loc[:, valid_assets]

    if clean_train.shape[1] < 2:
        raise ValueError("At least two assets with valid return observations are required.")

    clean_train = clean_train.apply(lambda column: column.fillna(column.mean()))

    tickers = pd.Index(clean_train.columns.astype(str))
    n_assets = len(tickers)
    max_weight = _asset_cap(n_assets)

    volatility = clean_train.std(ddof=1).replace(0.0, np.nan)

    inverse_vol = 1.0 / volatility

    inverse_vol = inverse_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if inverse_vol.sum() <= 0:
        raw = np.ones(n_assets)
    else:
        raw = inverse_vol.to_numpy(dtype=float)

    weights = _apply_crypto_sleeve_cap(
        raw,
        tickers=tickers,
        crypto_cap=crypto_cap,
        max_weight=max_weight,
    )

    output = pd.Series(
        weights,
        index=tickers,
        name="weight",
    )

    output.attrs["solver_success"] = False
    output.attrs["used_fallback"] = True
    output.attrs["solver_message"] = "Inverse-volatility fallback used."

    return output


def _solve_weights(
    train: pd.DataFrame,
    method: str,
    crypto_cap: float | None = None,
) -> pd.Series:
    """
    Route a training window to the requested portfolio-construction method.
    """
    method_key = method.strip().lower()

    if method_key == "equal_weight":
        return _equal_weight_weights(
            train,
            crypto_cap=crypto_cap,
        )

    if method_key == "min_variance":
        return _minimum_variance_weights(
            train,
            crypto_cap=crypto_cap,
        )

    if method_key == "max_sharpe":
        return _maximum_sharpe_weights(
            train,
            crypto_cap=crypto_cap,
        )

    if method_key == "risk_parity":
        return _risk_parity_weights(
            train,
            crypto_cap=crypto_cap,
        )

    allowed = [
        "equal_weight",
        "min_variance",
        "max_sharpe",
        "risk_parity",
    ]

    raise ValueError(f"Unknown portfolio method '{method}'. Choose one of: {allowed}.")


def _portfolio_returns_from_weights(
    test_returns: pd.DataFrame,
    weights: pd.Series,
    previous_weights: pd.Series | None = None,
    transaction_cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, float]:
    """
    Apply target portfolio weights to the future holding-period returns.

    Transaction cost is charged once, on the first day of the holding block,
    based on portfolio turnover.
    """
    if test_returns.empty:
        raise ValueError("test_returns is empty.")

    aligned_weights = weights.reindex(test_returns.columns).fillna(0.0).astype(float)

    if aligned_weights.sum() <= 0:
        raise ValueError("Portfolio weights must sum to a positive value.")

    aligned_weights = aligned_weights / aligned_weights.sum()

    # First portfolio establishment is treated as full deployment of capital.
    if previous_weights is None:
        turnover = 1.0
    else:
        previous_aligned = previous_weights.reindex(test_returns.columns).fillna(0.0).astype(float)

        if previous_aligned.sum() > 0:
            previous_aligned = previous_aligned / previous_aligned.sum()

        turnover = 0.5 * float((aligned_weights - previous_aligned).abs().sum())

    transaction_cost = turnover * transaction_cost_bps / 10_000.0

    # We do not create artificial returns from missing asset observations.
    # Missing values contribute zero for that day only.
    gross_return = test_returns.fillna(0.0).mul(aligned_weights, axis=1).sum(axis=1)

    net_return = gross_return.copy()

    # Cost is deducted once, on the first holding-period observation.
    net_return.iloc[0] = net_return.iloc[0] - transaction_cost

    output = pd.DataFrame(
        {
            "date": test_returns.index,
            "gross_return": gross_return.to_numpy(),
            "daily_return": net_return.to_numpy(),
            "transaction_cost": 0.0,
            "turnover": 0.0,
        }
    )

    output.loc[0, "transaction_cost"] = transaction_cost
    output.loc[0, "turnover"] = turnover

    return output, turnover


def oos_backtest(
    returns: pd.DataFrame,
    method: str = "min_variance",
):
    """
    Run a walk-forward out-of-sample portfolio backtest.

    Portfolio weights are estimated only from observations available
    before each rebalance date. The resulting weights are then held
    over the following out-of-sample block.
    """
    panel = _clean_return_panel(returns)
    config = _config_for_returns(panel)

    if len(panel) <= config.estimation_window:
        raise ValueError("Not enough observations for the required estimation window.")

    daily_blocks = []
    weight_records = []
    audit_records = []

    previous_weights = None

    for start in range(
        config.estimation_window,
        len(panel),
        config.rebalance_step,
    ):
        end = min(
            start + config.rebalance_step,
            len(panel),
        )

        train = panel.iloc[start - config.estimation_window : start].copy()

        test = panel.iloc[start:end].copy()

        if test.empty:
            continue

        # Remove assets that have insufficient usable history
        # in this particular training window.
        valid_assets = train.notna().sum() > 1

        train_active = train.loc[:, valid_assets]
        test_active = test.reindex(columns=train_active.columns)

        if train_active.shape[1] < 2:
            continue

        target_weights = _solve_weights(
            train_active,
            method=method,
            crypto_cap=config.crypto_cap,
        )

        block, turnover = _portfolio_returns_from_weights(
            test_returns=test_active,
            weights=target_weights,
            previous_weights=previous_weights,
            transaction_cost_bps=config.transaction_cost_bps,
        )

        rebalance_date = pd.Timestamp(test.index[0])

        block["rebalance_date"] = rebalance_date

        daily_blocks.append(block)

        weight_frame = pd.DataFrame(
            {
                "rebalance_date": rebalance_date,
                "ticker": target_weights.index,
                "weight": target_weights.to_numpy().round(12),
            }
        )

        weight_records.append(weight_frame)

        transaction_cost = float(block.loc[0, "transaction_cost"])

        audit_records.append(
            {
                "rebalance_date": rebalance_date,
                "training_start": pd.Timestamp(train.index.min()),
                "training_end": pd.Timestamp(train.index.max()),
                "holding_end": pd.Timestamp(test.index.max()),
                "n_training_observations": int(len(train)),
                "n_active_assets": int(len(target_weights)),
                "turnover": float(turnover),
                "transaction_cost": transaction_cost,
                "solver_success": bool(
                    target_weights.attrs.get(
                        "solver_success",
                        True,
                    )
                ),
                "used_fallback": bool(
                    target_weights.attrs.get(
                        "used_fallback",
                        False,
                    )
                ),
                "solver_message": str(
                    target_weights.attrs.get(
                        "solver_message",
                        "Deterministic rule",
                    )
                ),
            }
        )

        previous_weights = target_weights.copy()

    if not daily_blocks:
        raise RuntimeError("The backtest produced no out-of-sample observations.")

    daily = (
        pd.concat(
            daily_blocks,
            ignore_index=True,
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    weights = pd.concat(
        weight_records,
        ignore_index=True,
    )

    audit = pd.DataFrame(audit_records)

    # Growth of $1 based on net returns.
    daily["growth_of_one"] = (1.0 + daily["daily_return"]).cumprod()

    running_peak = daily["growth_of_one"].cummax()

    daily["drawdown"] = daily["growth_of_one"] / running_peak - 1.0

    metrics = performance_metrics(
        daily["daily_return"],
        periods_per_year=config.periods_per_year,
    )

    metrics.update(
        {
            "periods_per_year": config.periods_per_year,
            "estimation_window": config.estimation_window,
            "rebalance_step": config.rebalance_step,
            "transaction_cost_bps": config.transaction_cost_bps,
            "average_turnover": float(audit["turnover"].mean()),
            "first_live_date": pd.Timestamp(daily["date"].min()),
            "last_live_date": pd.Timestamp(daily["date"].max()),
        }
    )

    return {
        "daily_returns": daily,
        "weights": weights,
        "metrics": metrics,
        "rebalance_audit": audit,
        "config": config,
    }


def performance_metrics(
    daily_returns: pd.Series,
    periods_per_year: int = 252,
) -> dict:
    """
    Calculate standard performance and downside-risk metrics
    from a series of out-of-sample daily fund returns.
    """
    clean = pd.to_numeric(
        daily_returns,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return {
            "n_observations": 0,
            "total_return": np.nan,
            "annualised_return": np.nan,
            "annualised_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "downside_deviation": np.nan,
            "historical_var_95": np.nan,
            "historical_cvar_95": np.nan,
            "positive_day_rate": np.nan,
            "worst_daily_return": np.nan,
            "maximum_drawdown": np.nan,
        }

    growth = (1.0 + clean).cumprod()

    total_return = float(growth.iloc[-1] - 1.0)

    annualised_return = float(growth.iloc[-1] ** (periods_per_year / len(clean)) - 1.0)

    daily_std = float(clean.std(ddof=1))

    annualised_volatility = float(daily_std * np.sqrt(periods_per_year))

    if daily_std > 0:
        sharpe_ratio = float(clean.mean() / daily_std * np.sqrt(periods_per_year))
    else:
        sharpe_ratio = np.nan

    downside_returns = np.minimum(
        clean.to_numpy(dtype=float),
        0.0,
    )

    downside_daily = float(np.sqrt(np.mean(downside_returns**2)))

    downside_deviation = float(downside_daily * np.sqrt(periods_per_year))

    if downside_daily > 0:
        sortino_ratio = float(clean.mean() / downside_daily * np.sqrt(periods_per_year))
    else:
        sortino_ratio = np.nan

    historical_var_95 = float(clean.quantile(0.05))

    tail = clean[clean <= historical_var_95]

    historical_cvar_95 = float(tail.mean()) if not tail.empty else np.nan

    running_peak = growth.cummax()

    drawdown = growth / running_peak - 1.0

    maximum_drawdown = float(drawdown.min())

    return {
        "n_observations": int(len(clean)),
        "total_return": total_return,
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "downside_deviation": downside_deviation,
        "historical_var_95": historical_var_95,
        "historical_cvar_95": historical_cvar_95,
        "positive_day_rate": float((clean > 0).mean()),
        "worst_daily_return": float(clean.min()),
        "maximum_drawdown": maximum_drawdown,
    }


METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "min_variance": "Minimum Variance",
    "max_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
}


def _attach_fund_fields(
    daily: pd.DataFrame,
    family: str,
    method: str,
) -> pd.DataFrame:
    """
    Add fund-identification fields to an out-of-sample return panel.
    """
    output = daily.copy()

    output["fund"] = f"NexaVest {family} {METHOD_LABELS[method]}"

    output["asset_family"] = family
    output["method"] = method

    return output


def _annotate_weights(
    weights: pd.DataFrame,
    family: str,
    method: str,
) -> pd.DataFrame:
    """
    Add fund and asset-class labels to target portfolio weights.
    """
    output = weights.copy()

    output["fund"] = f"NexaVest {family} {METHOD_LABELS[method]}"

    output["asset_family"] = family
    output["method"] = method

    if family == "Combined":
        output["asset_class"] = np.where(
            output["ticker"].astype(str).str.startswith("CR_"),
            "Crypto",
            "Equity",
        )

        output["underlying_ticker"] = (
            output["ticker"].astype(str).str.replace(r"^(EQ_|CR_)", "", regex=True)
        )

    else:
        output["asset_class"] = family
        output["underlying_ticker"] = output["ticker"]

    return output


def _metric_row(
    result: dict,
    family: str,
    method: str,
) -> dict:
    """
    Convert one fund's performance metrics into a flat table row.
    """
    metrics = dict(result["metrics"])

    metrics.update(
        {
            "fund": f"NexaVest {family} {METHOD_LABELS[method]}",
            "asset_family": family,
            "method": method,
        }
    )

    return metrics


def _build_base_funds(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
    combined_returns: pd.DataFrame,
) -> dict:
    """
    Build the 12 base NexaVest funds across three asset families
    and four portfolio-construction methods.
    """
    families = {
        "Equity": equity_returns,
        "Crypto": crypto_returns,
        "Combined": combined_returns,
    }

    methods = [
        "equal_weight",
        "min_variance",
        "max_sharpe",
        "risk_parity",
    ]

    return_frames = []
    weight_frames = []
    metric_rows = []
    audit_frames = []

    for family, returns in families.items():
        for method in methods:
            print(f"Running {family} - {METHOD_LABELS[method]}...")

            result = oos_backtest(
                returns,
                method=method,
            )

            daily = _attach_fund_fields(
                result["daily_returns"],
                family=family,
                method=method,
            )

            weights = _annotate_weights(
                result["weights"],
                family=family,
                method=method,
            )

            metrics = _metric_row(
                result,
                family=family,
                method=method,
            )

            audit = result["rebalance_audit"].copy()

            audit["fund"] = f"NexaVest {family} {METHOD_LABELS[method]}"
            audit["asset_family"] = family
            audit["method"] = method

            return_frames.append(daily)
            weight_frames.append(weights)
            metric_rows.append(metrics)
            audit_frames.append(audit)

    fund_returns = pd.concat(
        return_frames,
        ignore_index=True,
    )

    fund_weights = pd.concat(
        weight_frames,
        ignore_index=True,
    )

    performance = pd.DataFrame(metric_rows)

    rebalance_audit = pd.concat(
        audit_frames,
        ignore_index=True,
    )

    return {
        "fund_returns": fund_returns,
        "fund_weights": fund_weights,
        "performance_metrics": performance,
        "rebalance_audit": rebalance_audit,
    }


from pathlib import Path


def save_portfolio_outputs(
    results: dict,
    project_root: str | Path,
) -> dict[str, Path]:
    """
    Save required portfolio artifacts for Part B.
    """
    root = Path(project_root)

    data_dir = root / "results" / "data"
    tables_dir = root / "results" / "tables"

    data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    fund_returns = results["fund_returns"].copy()
    fund_weights = results["fund_weights"].copy()
    performance = results["performance_metrics"].copy()
    audit = results["rebalance_audit"].copy()

    # Clean floating-point representation before saving.
    if "weight" in fund_weights.columns:
        fund_weights["weight"] = fund_weights["weight"].round(12)

    paths = {
        "fund_returns": data_dir / "fund_returns.csv",
        "fund_weights": data_dir / "fund_weights.csv",
        "performance_metrics": tables_dir / "performance_metrics.csv",
        "rebalance_audit": tables_dir / "rebalance_audit.csv",
    }

    fund_returns.to_csv(
        paths["fund_returns"],
        index=False,
    )

    fund_weights.to_csv(
        paths["fund_weights"],
        index=False,
    )

    performance.to_csv(
        paths["performance_metrics"],
        index=False,
    )

    audit.to_csv(
        paths["rebalance_audit"],
        index=False,
    )

    return paths


def plot_growth_of_one(
    fund_returns: pd.DataFrame,
    project_root: str | Path,
) -> Path:
    """
    Plot out-of-sample growth of $1 for the four Combined base funds.
    """
    root = Path(project_root)

    figures_dir = root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    selected = fund_returns.loc[fund_returns["asset_family"].eq("Combined")].copy()

    if selected.empty:
        raise ValueError("No Combined fund returns were found.")

    selected["date"] = pd.to_datetime(
        selected["date"],
        errors="coerce",
    )

    pivot = selected.pivot(
        index="date",
        columns="fund",
        values="growth_of_one",
    )

    figure, axis = plt.subplots(figsize=(10.5, 5.8))

    method_order = [
        "NexaVest Combined Equal Weight",
        "NexaVest Combined Minimum Variance",
        "NexaVest Combined Maximum Sharpe",
        "NexaVest Combined Risk Parity",
    ]

    for fund_name in method_order:
        if fund_name in pivot.columns:
            axis.plot(
                pivot.index,
                pivot[fund_name],
                linewidth=1.5,
                label=fund_name.replace(
                    "NexaVest Combined ",
                    "",
                ),
            )


    axis.axhline(
        1.0,
        linewidth=0.8,
    )
    axis.set_title(
        "NexaVest Combined Funds: Out-of-Sample Growth of $1\n"
        "Walk-forward backtest, 4 January 2021–29 December 2023"
    )

    axis.set_title(
        "NexaVest Combined Funds: Out-of-Sample Growth of $1\n"
        "Walk-forward backtest, 4 January 2021–29 December 2023"
    )

    axis.set_xlabel("Date")
    axis.set_ylabel("Portfolio value ($)")

    axis.legend(
        frameon=False,
        fontsize=8,
    )

    axis.grid(
        alpha=0.20,
    )

    figure.text(
        0.01,
        0.01,
        "Source: walk-forward OOS fund returns; "
        "returns are net of the 10 bps turnover-cost assumption.",
        fontsize=8,
    )

    figure.tight_layout(rect=(0, 0.04, 1, 1))

    path = figures_dir / "growth_of_one_combined.png"

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path

def plot_drawdown_combined(
    fund_returns: pd.DataFrame,
    project_root: str | Path,
) -> Path:
    """
    Plot out-of-sample drawdowns for the four Combined base funds.
    """
    root = Path(project_root)

    figures_dir = root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    selected = fund_returns.loc[
        fund_returns["asset_family"].eq("Combined")
    ].copy()

    if selected.empty:
        raise ValueError(
            "No Combined fund returns were found."
        )

    selected["date"] = pd.to_datetime(
        selected["date"],
        errors="coerce",
    )

    pivot = selected.pivot(
        index="date",
        columns="fund",
        values="drawdown",
    )

    method_order = [
        "NexaVest Combined Equal Weight",
        "NexaVest Combined Minimum Variance",
        "NexaVest Combined Maximum Sharpe",
        "NexaVest Combined Risk Parity",
    ]

    figure, axis = plt.subplots(
        figsize=(10.5, 5.8)
    )

    for fund_name in method_order:
        if fund_name in pivot.columns:
            axis.plot(
                pivot.index,
                pivot[fund_name] * 100,
                linewidth=1.5,
                label=fund_name.replace(
                    "NexaVest Combined ",
                    "",
                ),
            )

    axis.axhline(
        0.0,
        linewidth=0.8,
    )

    axis.set_title(
        "NexaVest Combined Funds: Out-of-Sample Drawdown\n"
        "Walk-forward backtest, 4 January 2021–29 December 2023"
    )

    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown (%)")

    axis.legend(
        frameon=False,
        fontsize=8,
    )

    axis.grid(
        alpha=0.20,
    )

    figure.text(
        0.01,
        0.01,
        "Source: walk-forward OOS fund returns; "
        "drawdown is measured relative to each fund's previous wealth peak.",
        fontsize=8,
    )

    figure.tight_layout(
        rect=(0, 0.04, 1, 1)
    )

    path = (
        figures_dir
        / "drawdown_combined.png"
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path

def plot_portfolio_weights_over_time(
    fund_weights: pd.DataFrame,
    project_root: str | Path,
) -> Path:
    """
    Plot the evolution of portfolio weights for Combined funds.

    Panel A shows the total crypto sleeve across the four methods.
    Panel B shows the largest holdings over time for Combined Risk Parity.
    """
    root = Path(project_root)

    figures_dir = root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    weights = fund_weights.copy()

    weights["rebalance_date"] = pd.to_datetime(
        weights["rebalance_date"],
        errors="coerce",
    )

    combined = weights.loc[
        weights["asset_family"].eq("Combined")
    ].copy()

    if combined.empty:
        raise ValueError(
            "No Combined fund weights were found."
        )

    combined["is_crypto"] = (
        combined["asset_class"].eq("Crypto")
    )

    # ---------------------------------------------------------
    # Panel A: total crypto sleeve by method
    # ---------------------------------------------------------
    crypto_sleeve = (
        combined.loc[combined["is_crypto"]]
        .groupby(
            ["rebalance_date", "fund"],
            as_index=False,
        )["weight"]
        .sum()
    )

    sleeve_pivot = crypto_sleeve.pivot(
        index="rebalance_date",
        columns="fund",
        values="weight",
    )

    method_order = [
        "NexaVest Combined Equal Weight",
        "NexaVest Combined Minimum Variance",
        "NexaVest Combined Maximum Sharpe",
        "NexaVest Combined Risk Parity",
    ]

    # ---------------------------------------------------------
    # Panel B: major holdings of Combined Risk Parity
    # ---------------------------------------------------------
    risk_parity_name = (
        "NexaVest Combined Risk Parity"
    )

    rp = combined.loc[
        combined["fund"].eq(risk_parity_name)
    ].copy()

    if rp.empty:
        raise ValueError(
            "Combined Risk Parity weights were not found."
        )

    average_weights = (
        rp.groupby("underlying_ticker")["weight"]
        .mean()
        .sort_values(ascending=False)
    )

    top_tickers = average_weights.head(10).index.tolist()

    rp["display_ticker"] = np.where(
        rp["underlying_ticker"].isin(top_tickers),
        rp["underlying_ticker"],
        "Other",
    )

    holdings = (
        rp.groupby(
            ["rebalance_date", "display_ticker"],
            as_index=False,
        )["weight"]
        .sum()
    )

    holdings_pivot = holdings.pivot(
        index="rebalance_date",
        columns="display_ticker",
        values="weight",
    ).fillna(0.0)

    ordered_holdings = [
        ticker
        for ticker in top_tickers
        if ticker in holdings_pivot.columns
    ]

    if "Other" in holdings_pivot.columns:
        ordered_holdings.append("Other")

    holdings_pivot = holdings_pivot[
        ordered_holdings
    ]

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 8.0),
        sharex=True,
    )

    # Panel A
    for fund_name in method_order:
        if fund_name in sleeve_pivot.columns:
            axes[0].plot(
                sleeve_pivot.index,
                sleeve_pivot[fund_name] * 100,
                linewidth=1.4,
                label=fund_name.replace(
                    "NexaVest Combined ",
                    "",
                ),
            )

    axes[0].axhline(
        25.0,
        linestyle="--",
        linewidth=1.0,
        label="25% crypto cap",
    )

    axes[0].set_title(
        "A. Crypto Sleeve Across Combined-Fund Methods"
    )

    axes[0].set_ylabel(
        "Crypto weight (%)"
    )

    axes[0].legend(
        frameon=False,
        fontsize=8,
        ncol=2,
    )

    axes[0].grid(
        alpha=0.20,
    )

    # Panel B
    axes[1].stackplot(
        holdings_pivot.index,
        [
            holdings_pivot[column] * 100
            for column in holdings_pivot.columns
        ],
        labels=holdings_pivot.columns,
        alpha=0.85,
    )

    axes[1].set_title(
        "B. Combined Risk Parity: Major Holdings and Other"
    )

    axes[1].set_xlabel(
        "Rebalance date"
    )

    axes[1].set_ylabel(
        "Target weight (%)"
    )

    axes[1].legend(
        frameon=False,
        fontsize=8,
        ncol=5,
        loc="upper center",
    )

    axes[1].grid(
        alpha=0.20,
    )

    figure.suptitle(
        "NexaVest Combined Funds: Portfolio Weights Over Time\n"
        "Walk-forward target weights, 4 January 2021–29 December 2023",
        fontsize=14,
    )

    figure.text(
        0.01,
        0.01,
        "Source: walk-forward target portfolio weights estimated "
        "from trailing data only. Combined crypto exposure is capped at 25%.",
        fontsize=8,
    )

    figure.tight_layout(
        rect=(0, 0.04, 1, 0.95)
    )

    path = (
        figures_dir
        / "portfolio_weights_over_time.png"
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path

def plot_return_risk_comparison(
    performance: pd.DataFrame,
    project_root: str | Path,
) -> Path:
    """
    Compare annualised return and volatility across the 12 base funds.
    """
    root = Path(project_root)

    figures_dir = root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = performance.copy()

    required = {
        "fund",
        "asset_family",
        "method",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
    }

    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    figure, axis = plt.subplots(
        figsize=(10.5, 6.5)
    )

    family_markers = {
        "Equity": "o",
        "Crypto": "^",
        "Combined": "s",
    }

    label_offsets = {
        ("Equity", "equal_weight"): (5, 5),
        ("Equity", "min_variance"): (5, -14),
        ("Equity", "max_sharpe"): (5, -14),
        ("Equity", "risk_parity"): (5, -14),
        ("Combined", "equal_weight"): (5, 5),
        ("Combined", "min_variance"): (5, -14),
        ("Combined", "max_sharpe"): (5, 6),
        ("Combined", "risk_parity"): (5, 6),
        ("Crypto", "equal_weight"): (5, 5),
        ("Crypto", "min_variance"): (5, 5),
        ("Crypto", "max_sharpe"): (5, 5),
        ("Crypto", "risk_parity"): (5, 5),
    }

    for family, marker in family_markers.items():
        subset = data.loc[
            data["asset_family"].eq(family)
        ]

        axis.scatter(
            subset["annualised_volatility"] * 100,
            subset["annualised_return"] * 100,
            marker=marker,
            s=80,
            label=family,
        )

        for _, row in subset.iterrows():
            label = METHOD_LABELS.get(
                row["method"],
                row["method"],
            )

            axis.annotate(
                label,
                (
                    row["annualised_volatility"] * 100,
                    row["annualised_return"] * 100,
                ),
                xytext=label_offsets.get(
                    (family, row["method"]),
                    (5, 5),
                ),
                textcoords="offset points",
                fontsize=7,
            )

    axis.set_title(
        "NexaVest Base Funds: Out-of-Sample Return–Risk Comparison\n"
        "Walk-forward backtest, 2021–2023"
    )

    axis.set_xlabel(
        "Annualised volatility (%)"
    )

    axis.set_ylabel(
        "Annualised return (%)"
    )

    axis.legend(
        title="Asset family",
        frameon=False,
    )

    axis.grid(
        alpha=0.20,
    )

    figure.text(
        0.01,
        0.01,
        "Source: NexaVest walk-forward OOS results. "
        "Returns are net of the 10 bps turnover-cost assumption.",
        fontsize=8,
    )

    figure.tight_layout(
        rect=(0, 0.04, 1, 1)
    )

    path = (
        figures_dir
        / "return_risk_comparison.png"
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path
