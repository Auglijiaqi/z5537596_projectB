"""
NexaVest Investor Dashboard

Run locally from the Project B root:

    streamlit run streamlit_app.py

The deployed app is intentionally lightweight. It reads precomputed
Project B artifacts from results/ and does not rerun portfolio backtests
or the sentiment model.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_access  # noqa: E402


DATA_DIR = ROOT / "results" / "data"
TABLE_DIR = ROOT / "results" / "tables"


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="NexaVest",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 3.0rem;
        font-weight: 760;
        margin-bottom: 0.1rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666666;
        margin-bottom: 1.4rem;
    }

    .section-note {
        font-size: 0.90rem;
        color: #666666;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(120, 120, 120, 0.20);
        border-radius: 10px;
        padding: 12px;
    }

    .insight-box {
        border: 1px solid rgba(120, 120, 120, 0.22);
        border-radius: 10px;
        padding: 14px 16px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def _percent(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def _number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _currency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):,.2f}"


def _safe_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


# ---------------------------------------------------------------------
# Required Project B artifacts
# ---------------------------------------------------------------------

def _required_files() -> dict[str, pathlib.Path]:
    return {
        "fund_returns":
            DATA_DIR / "fund_returns.csv",

        "fund_weights":
            DATA_DIR / "fund_weights.csv",

        "sector_sentiment":
            DATA_DIR / "sector_sentiment_index.csv",

        "performance":
            TABLE_DIR / "performance_metrics.csv",
    }


@st.cache_data(show_spinner=False)
def _read_csv(
    path_string: str,
    date_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    path = pathlib.Path(path_string)

    frame = pd.read_csv(path)

    for column in date_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(
                frame[column],
                errors="coerce",
            )

    return frame


@st.cache_data(show_spinner="Loading NexaVest results...")
def _load_app_data():
    required = _required_files()

    missing = [
        name
        for name, path in required.items()
        if not path.exists()
    ]

    if missing:
        st.error(
            "The Project B app artifacts have not been built. "
            f"Missing: {', '.join(missing)}"
        )

        st.code(
            "python scripts/run_part_b.py"
        )

        st.stop()

    fund_returns = _read_csv(
        str(required["fund_returns"]),
        ("date", "rebalance_date"),
    )

    fund_weights = _read_csv(
        str(required["fund_weights"]),
        ("rebalance_date",),
    )

    sentiment = _read_csv(
        str(required["sector_sentiment"]),
        ("trading_date",),
    )

    performance = _read_csv(
        str(required["performance"]),
        ("first_live_date", "last_live_date"),
    )

    fusion_path = (
        TABLE_DIR / "fusion_comparison.csv"
    )

    fusion_comparison = (
        _read_csv(str(fusion_path))
        if fusion_path.exists()
        else pd.DataFrame()
    )

    audit_path = (
        TABLE_DIR / "sentiment_model_audit.csv"
    )

    sentiment_audit = (
        _read_csv(str(audit_path))
        if audit_path.exists()
        else pd.DataFrame()
    )

    lexicon_path = (
        TABLE_DIR / "finance_lexicon.csv"
    )

    finance_lexicon = (
        _read_csv(str(lexicon_path))
        if lexicon_path.exists()
        else pd.DataFrame()
    )

    return (
        fund_returns,
        fund_weights,
        sentiment,
        performance,
        fusion_comparison,
        sentiment_audit,
        finance_lexicon,
    )


@st.cache_data(
    ttl=86_400,
    show_spinner="Loading hosted equity data...",
)
def _load_hosted_equities():
    return data_access.load_equity_prices()


(
    fund_returns,
    fund_weights,
    sentiment,
    performance,
    fusion_comparison,
    sentiment_audit,
    finance_lexicon,
) = _load_app_data()


# ---------------------------------------------------------------------
# Basic validation / convenience
# ---------------------------------------------------------------------

if "fund" not in performance.columns:
    st.error(
        "performance_metrics.csv does not contain a 'fund' column."
    )
    st.stop()


fund_names = sorted(
    performance["fund"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.markdown(
    '<div class="main-title">NexaVest</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Systematic investing with transparent evidence — compare equity,
    crypto and multi-asset strategies using walk-forward
    out-of-sample results.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Investor Lens")

    st.write(
        "Designed for financially literate, self-directed investors "
        "who want transparent systematic-fund comparisons."
    )

    st.divider()

    st.markdown("**Backtest assumptions**")

    st.write("• Walk-forward, past-data only")
    st.write("• Long-only and fully invested")
    st.write("• Monthly-style rebalancing")
    st.write("• 10 bps turnover-cost assumption")
    st.write("• Risk-free rate = 0")
    st.write("• Combined crypto sleeve ≤ 25%")

    st.divider()

    st.caption(
        "Historical evidence only. "
        "NexaVest does not provide personal financial advice."
    )


# ---------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------

(
    tab_compare,
    tab_fact,
    tab_allocation,
    tab_sentiment,
    tab_method,
    tab_data,
) = st.tabs(
    [
        "Compare Funds",
        "Fund Fact Sheet",
        "Build Allocation",
        "Sentiment Lens",
        "Methodology",
        "Data Check",
    ]
)


# =====================================================================
# 1. Compare Funds
# =====================================================================

with tab_compare:

    st.subheader("Compare the NexaVest fund range")

    st.caption(
        "All performance figures are based on walk-forward "
        "out-of-sample fund returns and include the stated "
        "turnover-cost assumption."
    )

    family_values = (
        performance["asset_family"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if "asset_family" in performance.columns
        else []
    )

    family_options = [
        "All"
    ] + sorted(family_values)

    selected_family = st.selectbox(
        "Asset family",
        family_options,
        key="compare_family",
    )

    compare_frame = performance.copy()

    if (
        selected_family != "All"
        and "asset_family" in compare_frame.columns
    ):
        compare_frame = compare_frame.loc[
            compare_frame["asset_family"]
            .astype(str)
            .eq(selected_family)
        ].copy()

    if "sharpe_ratio" in compare_frame.columns:
        compare_frame = compare_frame.sort_values(
            "sharpe_ratio",
            ascending=False,
        )

    display_columns = [
        column
        for column in [
            "fund",
            "asset_family",
            "method",
            "annualised_return",
            "annualised_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "average_turnover",
        ]
        if column in compare_frame.columns
    ]

    table = compare_frame[
        display_columns
    ].copy()

    percentage_columns = [
        "annualised_return",
        "annualised_volatility",
        "maximum_drawdown",
        "average_turnover",
    ]

    for column in percentage_columns:
        if column in table.columns:
            table[column] = (
                table[column] * 100
            )

    rename_map = {
        "fund": "Fund",
        "asset_family": "Asset Family",
        "method": "Method",
        "annualised_return":
            "Annual Return (%)",
        "annualised_volatility":
            "Volatility (%)",
        "sharpe_ratio":
            "Sharpe",
        "maximum_drawdown":
            "Max Drawdown (%)",
        "average_turnover":
            "Average Turnover (%)",
    }

    table = table.rename(
        columns=rename_map
    )

    st.dataframe(
        table.round(3),
        hide_index=True,
        width="stretch",
    )

    st.markdown("### Risk–Return Map")

    if {
        "annualised_volatility",
        "annualised_return",
    }.issubset(compare_frame.columns):

        risk_return = compare_frame[
            [
                "fund",
                "annualised_volatility",
                "annualised_return",
            ]
        ].copy()

        risk_return[
            "Annualised volatility (%)"
        ] = (
            risk_return[
                "annualised_volatility"
            ]
            * 100
        )

        risk_return[
            "Annualised return (%)"
        ] = (
            risk_return[
                "annualised_return"
            ]
            * 100
        )

        chart_data = (
            risk_return
            .set_index("fund")[
                [
                    "Annualised volatility (%)",
                    "Annualised return (%)",
                ]
            ]
        )

        st.scatter_chart(
            chart_data,
            x="Annualised volatility (%)",
            y="Annualised return (%)",
            height=430,
        )

        st.caption(
            "Funds further to the right have higher realised "
            "annualised volatility; funds higher on the chart "
            "have higher realised annualised return."
        )

    if (
        not fusion_comparison.empty
        and "asset_family"
        in fusion_comparison.columns
    ):
        st.markdown(
            "### Sentiment-enhanced strategies"
        )

        st.write(
            "NexaVest reports the sentiment extension even when "
            "it does not improve performance, avoiding selection "
            "of only favourable results."
        )

        fusion_small = (
            fusion_comparison.copy()
        )

        change_columns = [
            "delta_annualised_return",
            "delta_volatility",
            "delta_max_drawdown",
        ]

        for column in change_columns:
            if column in fusion_small.columns:
                fusion_small[column] = (
                    fusion_small[column]
                    * 100
                )

        fusion_rename = {
            "asset_family":
                "Asset Family",
            "base_fund":
                "Base Fund",
            "enhanced_fund":
                "Sentiment-Enhanced Fund",
            "delta_annualised_return":
                "Return Change (pp)",
            "delta_volatility":
                "Volatility Change (pp)",
            "delta_sharpe":
                "Sharpe Change",
            "delta_max_drawdown":
                "Max Drawdown Change (pp)",
        }

        useful = [
            column
            for column in fusion_rename
            if column in fusion_small.columns
        ]

        st.dataframe(
            fusion_small[
                useful
            ]
            .rename(
                columns=fusion_rename
            )
            .round(3),
            hide_index=True,
            width="stretch",
        )


# =====================================================================
# 2. Fund Fact Sheet
# =====================================================================

with tab_fact:

    st.subheader("Open a fund fact sheet")

    selected_fund = st.selectbox(
        "Fund",
        fund_names,
        key="fact_fund",
    )

    fund_metric = (
        performance.loc[
            performance["fund"]
            .astype(str)
            .eq(selected_fund)
        ]
        .iloc[0]
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Annual Return",
        _percent(
            fund_metric.get(
                "annualised_return",
                np.nan,
            )
        ),
    )

    c2.metric(
        "Annualised Volatility",
        _percent(
            fund_metric.get(
                "annualised_volatility",
                np.nan,
            )
        ),
    )

    c3.metric(
        "Sharpe Ratio",
        _number(
            fund_metric.get(
                "sharpe_ratio",
                np.nan,
            ),
            3,
        ),
    )

    c4.metric(
        "Maximum Drawdown",
        _percent(
            fund_metric.get(
                "maximum_drawdown",
                np.nan,
            )
        ),
    )

    if "asset_family" in fund_metric.index:
        st.caption(
            f"Asset family: "
            f"{fund_metric.get('asset_family', 'N/A')} | "
            f"Method: "
            f"{fund_metric.get('method', 'N/A')}"
        )

    fund_history = (
        fund_returns.loc[
            fund_returns["fund"]
            .astype(str)
            .eq(selected_fund)
        ]
        .sort_values("date")
        .copy()
    )

    left, right = st.columns(2)

    with left:

        st.markdown("### Growth of $1")

        if (
            not fund_history.empty
            and "growth_of_one"
            in fund_history.columns
        ):
            growth = (
                fund_history[
                    [
                        "date",
                        "growth_of_one",
                    ]
                ]
                .dropna()
                .set_index("date")
            )

            growth.columns = [
                "Growth of $1"
            ]

            st.line_chart(
                growth,
                height=340,
            )
        else:
            st.info(
                "Growth-of-$1 history is unavailable."
            )

    with right:

        st.markdown("### Drawdown")

        if (
            not fund_history.empty
            and "drawdown"
            in fund_history.columns
        ):
            drawdown = (
                fund_history[
                    [
                        "date",
                        "drawdown",
                    ]
                ]
                .dropna()
                .copy()
            )

            drawdown[
                "Drawdown (%)"
            ] = (
                drawdown[
                    "drawdown"
                ]
                * 100
            )

            drawdown = (
                drawdown[
                    [
                        "date",
                        "Drawdown (%)",
                    ]
                ]
                .set_index("date")
            )

            st.line_chart(
                drawdown,
                height=340,
            )
        else:
            st.info(
                "Drawdown history is unavailable."
            )

    st.markdown("### Current Target Holdings")

    selected_weights = (
        fund_weights.loc[
            fund_weights["fund"]
            .astype(str)
            .eq(selected_fund)
        ]
        .copy()
    )

    if selected_weights.empty:
        st.info(
            "No portfolio holdings are available "
            "for this fund."
        )

    else:

        if "rebalance_date" in selected_weights.columns:
            latest_date = (
                selected_weights[
                    "rebalance_date"
                ].max()
            )

            selected_weights = (
                selected_weights.loc[
                    selected_weights[
                        "rebalance_date"
                    ].eq(latest_date)
                ]
                .copy()
            )

            st.caption(
                "Latest target weights: "
                f"{pd.Timestamp(latest_date).date()}"
            )

        holding_columns = [
            column
            for column in [
                "underlying_ticker",
                "ticker",
                "asset_class",
                "sector",
                "weight",
            ]
            if column in selected_weights.columns
        ]

        holdings = (
            selected_weights[
                holding_columns
            ]
            .copy()
        )

        ticker_col = (
            "underlying_ticker"
            if "underlying_ticker"
            in holdings.columns
            else "ticker"
        )

        if "weight" in holdings.columns:
            holdings[
                "weight"
            ] = (
                holdings[
                    "weight"
                ]
                * 100
            )

            holdings = (
                holdings.sort_values(
                    "weight",
                    ascending=False,
                )
            )

        holdings = holdings.rename(
            columns={
                ticker_col:
                    "Ticker",
                "ticker":
                    "Portfolio Ticker",
                "asset_class":
                    "Asset Class",
                "sector":
                    "Sector",
                "weight":
                    "Target Weight (%)",
            }
        )

        st.dataframe(
            holdings.round(3),
            hide_index=True,
            width="stretch",
        )

    with st.expander(
        "Advanced risk and implementation metrics"
    ):

        advanced_fields = {
            "sortino_ratio":
                "Sortino Ratio",

            "downside_deviation":
                "Downside Deviation",

            "historical_var_95":
                "Historical VaR 95%",

            "historical_cvar_95":
                "Historical CVaR 95%",

            "positive_day_rate":
                "Positive Day Rate",

            "worst_daily_return":
                "Worst Daily Return",

            "average_turnover":
                "Average Turnover",

            "n_observations":
                "OOS Observations",
        }

        rows = []

        for column, label in (
            advanced_fields.items()
        ):
            if column in fund_metric.index:

                value = fund_metric.get(
                    column
                )

                if column in {
                    "downside_deviation",
                    "historical_var_95",
                    "historical_cvar_95",
                    "positive_day_rate",
                    "worst_daily_return",
                    "average_turnover",
                }:
                    display_value = (
                        _percent(value)
                    )
                else:
                    display_value = (
                        _number(value, 3)
                    )

                rows.append(
                    {
                        "Metric": label,
                        "Value": display_value,
                    }
                )

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
            )


# =====================================================================
# 3. Build Allocation
# =====================================================================

with tab_allocation:

    st.subheader(
        "Build an allocation across NexaVest funds"
    )

    st.caption(
        "This tool illustrates how a user-defined allocation "
        "could be distributed across funds. It does not optimise "
        "or recommend an allocation."
    )

    default_candidates = [
        "NexaVest Equity Minimum Variance",
        "NexaVest Combined Risk Parity",
        "NexaVest Combined Maximum Sharpe",
    ]

    defaults = [
        fund
        for fund in default_candidates
        if fund in fund_names
    ]

    if not defaults:
        defaults = fund_names[:3]

    chosen_funds = st.multiselect(
        "Choose funds",
        options=fund_names,
        default=defaults,
    )

    investment_amount = st.number_input(
        "Investment amount ($)",
        min_value=100.0,
        value=10_000.0,
        step=500.0,
    )

    if not chosen_funds:

        st.info(
            "Choose at least one fund."
        )

    else:

        raw_allocations = {}

        st.markdown(
            "### Your preferred percentages"
        )

        columns = st.columns(
            min(
                len(chosen_funds),
                3,
            )
        )

        equal_default = (
            100.0
            / len(chosen_funds)
        )

        for i, fund in enumerate(
            chosen_funds
        ):
            with columns[
                i % len(columns)
            ]:
                raw_allocations[
                    fund
                ] = st.number_input(
                    fund,
                    min_value=0.0,
                    max_value=100.0,
                    value=float(
                        round(
                            equal_default,
                            2,
                        )
                    ),
                    step=1.0,
                    key=f"allocation_{fund}",
                )

        total_input = sum(
            raw_allocations.values()
        )

        if total_input <= 0:

            st.warning(
                "Enter at least one positive allocation."
            )

        else:

            allocation_rows = []

            for fund, raw_value in (
                raw_allocations.items()
            ):

                normalised = (
                    raw_value
                    / total_input
                )

                allocation_rows.append(
                    {
                        "Fund": fund,
                        "Normalised Allocation (%)":
                            normalised * 100,

                        "Dollar Allocation ($)":
                            normalised
                            * investment_amount,
                    }
                )

            allocation = pd.DataFrame(
                allocation_rows
            )

            st.markdown(
                "### Allocation Snapshot"
            )

            st.dataframe(
                allocation.round(2),
                hide_index=True,
                width="stretch",
            )

            chart = (
                allocation
                .set_index("Fund")[
                    [
                        "Dollar Allocation ($)"
                    ]
                ]
            )

            st.bar_chart(
                chart,
                height=350,
            )

            selected_metrics = (
                performance.loc[
                    performance[
                        "fund"
                    ].isin(
                        chosen_funds
                    )
                ]
                .copy()
            )

            if (
                not selected_metrics.empty
                and "annualised_volatility"
                in selected_metrics.columns
            ):

                highest_risk = (
                    selected_metrics
                    .sort_values(
                        "annualised_volatility",
                        ascending=False,
                    )
                    .iloc[0]
                )

                lowest_risk = (
                    selected_metrics
                    .sort_values(
                        "annualised_volatility",
                        ascending=True,
                    )
                    .iloc[0]
                )

                c1, c2, c3 = (
                    st.columns(3)
                )

                c1.metric(
                    "Funds selected",
                    len(chosen_funds),
                )

                c2.metric(
                    "Highest-volatility fund",
                    highest_risk[
                        "fund"
                    ],
                )

                c3.metric(
                    "Lowest-volatility fund",
                    lowest_risk[
                        "fund"
                    ],
                )

            st.caption(
                "The allocation preview is descriptive only. "
                "NexaVest does not infer suitability or investor risk tolerance."
            )


# =====================================================================
# 4. Sentiment Lens
# =====================================================================

with tab_sentiment:

    st.subheader(
        "Finance-aware equity-sector sentiment"
    )

    st.write(
        "NexaVest extends VADER with a finance-specific lexicon, "
        "aggregates ticker-day scores into sector signals, "
        "standardises each sector relative to its own history, "
        "lags the signal, and adjusts its strength for news coverage."
    )

    if sentiment.empty:

        st.info(
            "Sector sentiment data are unavailable."
        )

    else:

        sector_values = sorted(
            sentiment[
                "sector"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_sector = (
            st.selectbox(
                "Sector",
                sector_values,
                key="sentiment_sector",
            )
        )

        sector_frame = (
            sentiment.loc[
                sentiment[
                    "sector"
                ]
                .astype(str)
                .eq(selected_sector)
            ]
            .sort_values(
                "trading_date"
            )
            .copy()
        )

        latest = (
            sector_frame
            .dropna(
                subset=[
                    "trading_date"
                ]
            )
            .iloc[-1]
        )

        raw_column = _safe_column(
            sector_frame,
            [
                "sentiment_score",
                "score_100",
            ],
        )

        relative_column = (
            _safe_column(
                sector_frame,
                [
                    "sentiment_z_21d_mean",
                    "sentiment_z_expanding",
                    "lagged_sentiment_score",
                ],
            )
        )

        coverage_column = (
            _safe_column(
                sector_frame,
                [
                    "coverage_ratio",
                ],
            )
        )

        signal_column = (
            _safe_column(
                sector_frame,
                [
                    "coverage_adjusted_signal",
                    "lagged_sentiment_signal",
                    "lagged_sentiment_score",
                ],
            )
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        if raw_column:
            raw_value = latest[
                raw_column
            ]

            if raw_column == "score_100":
                raw_display = (
                    f"{raw_value:.1f}/100"
                )
            else:
                raw_display = (
                    _number(
                        raw_value,
                        3,
                    )
                )

            c1.metric(
                "Latest Raw Sentiment",
                raw_display,
            )

        else:
            c1.metric(
                "Latest Raw Sentiment",
                "N/A",
            )

        c2.metric(
            "Relative Signal",
            (
                _number(
                    latest[
                        relative_column
                    ],
                    3,
                )
                if relative_column
                else "N/A"
            ),
        )

        c3.metric(
            "News Coverage",
            (
                _percent(
                    latest[
                        coverage_column
                    ],
                    1,
                )
                if coverage_column
                else "N/A"
            ),
        )

        c4.metric(
            "Coverage-Adjusted Signal",
            (
                _number(
                    latest[
                        signal_column
                    ],
                    3,
                )
                if signal_column
                else "N/A"
            ),
        )

        st.markdown(
            "### Sector Sentiment History"
        )

        chart_candidates = [
            column
            for column in [
                "sentiment_z_expanding",
                "sentiment_z_21d_mean",
                "lagged_sentiment_score",
                "coverage_adjusted_signal",
            ]
            if column
            in sector_frame.columns
        ]

        if chart_candidates:

            sentiment_chart = (
                sector_frame[
                    [
                        "trading_date",
                        *chart_candidates,
                    ]
                ]
                .dropna(
                    subset=[
                        "trading_date"
                    ]
                )
                .set_index(
                    "trading_date"
                )
            )

            sentiment_chart = (
                sentiment_chart.rename(
                    columns={
                        "sentiment_z_expanding":
                            "Historical relative sentiment",

                        "sentiment_z_21d_mean":
                            "21-day relative mean",

                        "lagged_sentiment_score":
                            "Lagged sentiment",

                        "coverage_adjusted_signal":
                            "Coverage-adjusted signal",
                    }
                )
            )

            st.line_chart(
                sentiment_chart,
                height=390,
            )

        st.markdown(
            "### Why coverage matters"
        )

        st.markdown(
            """
            <div class="insight-box">
            A strong sentiment reading based on broad sector news
            coverage carries more information than the same reading
            based on only a small fraction of sector tickers.
            NexaVest therefore moderates the usable signal when
            coverage is sparse.
            </div>
            """,
            unsafe_allow_html=True,
        )


    # -----------------------------------------------------------------
    # Finance-aware VADER audit
    # -----------------------------------------------------------------

    if not sentiment_audit.empty:

        st.markdown(
            "### Finance-Aware VADER Validation"
        )

        audit = (
            sentiment_audit.copy()
        )

        if {
            "metric",
            "value",
        }.issubset(
            audit.columns
        ):

            audit_lookup = (
                audit.set_index(
                    "metric"
                )[
                    "value"
                ]
                .to_dict()
            )

            plain_rate = (
                audit_lookup.get(
                    "plain_vader_neutral_rate",
                    np.nan,
                )
            )

            finance_rate = (
                audit_lookup.get(
                    "finance_vader_neutral_rate",
                    np.nan,
                )
            )

            changed = (
                audit_lookup.get(
                    "headline_scores_changed",
                    np.nan,
                )
            )

            lexicon_count = (
                audit_lookup.get(
                    "finance_lexicon_terms",
                    np.nan,
                )
            )

            c1, c2, c3, c4 = (
                st.columns(4)
            )

            c1.metric(
                "Plain VADER Neutral",
                _percent(
                    plain_rate,
                    1,
                ),
            )

            c2.metric(
                "Finance-Aware Neutral",
                _percent(
                    finance_rate,
                    1,
                ),
            )

            c3.metric(
                "Headline Scores Changed",
                (
                    f"{int(changed):,}"
                    if not pd.isna(
                        changed
                    )
                    else "N/A"
                ),
            )

            c4.metric(
                "Finance Lexicon Terms",
                (
                    f"{int(lexicon_count)}"
                    if not pd.isna(
                        lexicon_count
                    )
                    else "N/A"
                ),
            )

            if (
                not pd.isna(
                    plain_rate
                )
                and not pd.isna(
                    finance_rate
                )
            ):

                validation = (
                    pd.DataFrame(
                        {
                            "Neutral headline rate (%)":
                                [
                                    plain_rate
                                    * 100,

                                    finance_rate
                                    * 100,
                                ]
                        },
                        index=[
                            "Plain VADER",
                            "Finance-aware VADER",
                        ],
                    )
                )

                st.bar_chart(
                    validation,
                    height=300,
                )

            st.caption(
                "A lower neutral rate indicates broader recognition "
                "of finance-specific language. It is a vocabulary-coverage "
                "diagnostic, not evidence of return predictability."
            )

        if not finance_lexicon.empty:

            with st.expander(
                "View finance lexicon audit"
            ):
                st.dataframe(
                    finance_lexicon,
                    hide_index=True,
                    width="stretch",
                )


    # -----------------------------------------------------------------
    # Fusion results
    # -----------------------------------------------------------------

    if not fusion_comparison.empty:

        st.markdown(
            "### Did Sentiment Add Value?"
        )

        st.write(
            "The fusion test compares the fixed Minimum Variance "
            "base strategy with the sentiment-tilted version. "
            "Negative results are retained rather than removed."
        )

        fusion_display = (
            fusion_comparison.copy()
        )

        percent_delta_columns = [
            "delta_annualised_return",
            "delta_volatility",
            "delta_max_drawdown",
        ]

        for column in percent_delta_columns:
            if column in fusion_display.columns:
                fusion_display[
                    column
                ] = (
                    fusion_display[
                        column
                    ]
                    * 100
                )

        selected_columns = [
            column
            for column in [
                "asset_family",
                "base_fund",
                "enhanced_fund",
                "delta_annualised_return",
                "delta_volatility",
                "delta_sharpe",
                "delta_max_drawdown",
            ]
            if column in fusion_display.columns
        ]

        fusion_display = (
            fusion_display[
                selected_columns
            ]
            .rename(
                columns={
                    "asset_family":
                        "Asset Family",

                    "base_fund":
                        "Base Fund",

                    "enhanced_fund":
                        "Enhanced Fund",

                    "delta_annualised_return":
                        "Return Change (pp)",

                    "delta_volatility":
                        "Volatility Change (pp)",

                    "delta_sharpe":
                        "Sharpe Change",

                    "delta_max_drawdown":
                        "Max Drawdown Change (pp)",
                }
            )
        )

        st.dataframe(
            fusion_display.round(3),
            hide_index=True,
            width="stretch",
        )

        if (
            "delta_sharpe"
            in fusion_comparison.columns
        ):

            mean_delta = (
                fusion_comparison[
                    "delta_sharpe"
                ]
                .mean()
            )

            if mean_delta > 0:

                message = (
                    "Across the reported fusion tests, "
                    "the sentiment extension improved "
                    "average Sharpe."
                )

                st.success(message)

            else:

                message = (
                    "In the reported out-of-sample test, "
                    "the sentiment tilt changed portfolio "
                    "allocations but did not improve average "
                    "risk-adjusted performance."
                )

                st.info(message)


# =====================================================================
# 5. Methodology
# =====================================================================

with tab_method:

    st.subheader(
        "How NexaVest was built"
    )

    st.markdown(
        """
        ### 1. Data Foundation

        NexaVest reuses the cleaned equity, crypto and
        trading-day headline panels produced in Project A.
        Equity and crypto returns are calculated on their
        native calendars before crypto returns are aligned to
        the equity calendar for Combined funds.

        ### 2. Portfolio Engine

        The product tests four systematic portfolio methods:

        - Equal Weight
        - Minimum Variance
        - Maximum Sharpe
        - Risk Parity

        They are applied across Equity, Crypto and Combined
        investment universes.

        Portfolio performance is evaluated using a walk-forward
        out-of-sample design. Weights are estimated from historical
        observations only and then held over the following
        out-of-sample block.

        Transaction costs are modelled using a 10 bps
        turnover-based assumption.

        ### 3. Sentiment Engine

        Equity news headlines are scored using finance-aware VADER.
        A manually reviewed finance lexicon extends the base model
        with terms that are common in financial news.

        Ticker-day sentiment is aggregated into equal-weighted
        equity-sector sentiment. Each sector is then evaluated
        relative to its own historical distribution.

        The signal is lagged before it can affect portfolio
        decisions, preventing future information from entering
        the allocation.

        ### 4. Coverage-Aware Fusion

        NexaVest applies sentiment only to the equity sleeve.

        The sentiment signal adjusts relative equity weights,
        while the Combined fund's crypto budget is preserved.
        News coverage is incorporated so that sparse sector
        coverage produces a weaker usable signal.

        The sentiment tilt was specified before evaluating the
        final out-of-sample fusion result.

        ### 5. Deployment

        The Streamlit app reads precomputed artifacts stored under
        `results/`.

        It does not rerun the backtest or sentiment model in the
        deployed environment.
        """
    )


# =====================================================================
# 6. Data Check
# =====================================================================

with tab_data:

    st.subheader(
        "Hosted Equity Data Check"
    )

    st.write(
        "This page preserves the starter's hosted-data check. "
        "The investor dashboard itself reads precomputed Project B "
        "artifacts and does not rerun portfolio or sentiment models."
    )

    if st.button(
        "Load hosted equity sample",
        key="load_equity_sample",
    ):

        equities = (
            _load_hosted_equities()
        )

        if "date" in equities.columns:
            equities[
                "date"
            ] = pd.to_datetime(
                equities[
                    "date"
                ],
                errors="coerce",
            )

        ticker_count = (
            equities[
                "ticker"
            ].nunique()
            if "ticker"
            in equities.columns
            else np.nan
        )

        first_date = (
            equities[
                "date"
            ].min()
            if "date"
            in equities.columns
            else None
        )

        last_date = (
            equities[
                "date"
            ].max()
            if "date"
            in equities.columns
            else None
        )

        st.write(
            f"Equity prices: "
            f"{len(equities):,} rows, "
            f"{ticker_count} tickers"
        )

        if (
            first_date is not None
            and last_date is not None
        ):
            st.caption(
                f"Coverage: "
                f"{pd.Timestamp(first_date).date()} "
                f"to "
                f"{pd.Timestamp(last_date).date()}"
            )

        st.dataframe(
            equities.head(20),
            width="stretch",
        )

    else:

        st.info(
            "Click the button to verify the hosted equity source. "
            "This check is optional during normal app use."
        )

    st.caption(
        "Historical data coverage ends in 2023. "
        "Historical performance does not guarantee future performance."
    )
