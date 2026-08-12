"""Station 3 - your sentiment model and index from news headlines.

This is the model step: score each headline, aggregate to a daily per-ticker score,
then to an equal-weight sector index. Headlines are a noisy proxy, so lag to avoid
look-ahead.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_NLTK_DATA = PROJECT_ROOT / ".nltk_data"

if LOCAL_NLTK_DATA.exists():
    nltk.data.path.insert(
        0,
        str(LOCAL_NLTK_DATA),
    )

# ============================================================
# NexaVest finance-aware sentiment extension
# ============================================================

NEUTRAL_THRESHOLD = 0.05


# These terms are informed by the descriptive text exploration
# completed in Project A and then manually reviewed for whether
# their directional meaning is reasonably clear in financial news.
FINANCE_LEXICON = {
    # Positive
    "upgrade": 2.4,
    "upgraded": 2.4,
    "beat": 2.0,
    "beats": 2.0,
    "surge": 2.2,
    "surges": 2.2,
    "recovery": 1.8,
    "profit": 1.6,
    "gains": 1.5,
    "gain": 1.5,
    "strong": 1.5,

    # Negative
    "downgrade": -2.4,
    "downgraded": -2.4,
    "miss": -2.0,
    "misses": -2.0,
    "crash": -3.0,
    "loss": -1.8,
    "losses": -1.8,
    "weak": -1.6,
    "decline": -1.5,
    "declines": -1.5,
}


# Multi-word expressions are converted to single tokens before
# finance-aware VADER scoring.
PHRASE_MAP = {
    "earnings miss": "earnings_miss",
    "profit warning": "profit_warning",
    "record revenue": "record_revenue",
    "record profit": "record_profit",
}


PHRASE_LEXICON = {
    "earnings_miss": -2.6,
    "profit_warning": -2.8,
    "record_revenue": 2.3,
    "record_profit": 2.4,
}

def _build_analyzers() -> tuple[
    SentimentIntensityAnalyzer,
    SentimentIntensityAnalyzer,
]:
    """
    Build a plain VADER baseline and a NexaVest finance-aware VADER.
    """
    plain = SentimentIntensityAnalyzer()

    finance = SentimentIntensityAnalyzer()

    finance.lexicon.update(FINANCE_LEXICON)
    finance.lexicon.update(PHRASE_LEXICON)

    return plain, finance

#3.
def _prepare_finance_text(text: str) -> tuple[str, int]:
    """
    Replace selected multi-word financial expressions with
    single tokens so VADER can apply phrase-level valence.
    """
    output = str(text)

    phrase_hits = 0

    for phrase, token in PHRASE_MAP.items():
        pattern = re.compile(
            rf"\b{re.escape(phrase)}\b",
            flags=re.IGNORECASE,
        )

        output, hits = pattern.subn(
            token,
            output,
        )

        phrase_hits += hits

    return output, phrase_hits

#4.
def _explode_headlines(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the Project A ticker-day headline panel back into
    individual headline observations.
    """
    required = {
        "trading_date",
        "ticker",
        "sector",
        "headlines_joined",
    }

    missing = required.difference(
        panel.columns
    )

    if missing:
        raise ValueError(
            f"Headline panel is missing columns: {sorted(missing)}"
        )

    frame = panel[
        [
            "trading_date",
            "ticker",
            "sector",
            "headlines_joined",
        ]
    ].copy()

    frame["trading_date"] = pd.to_datetime(
        frame["trading_date"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "trading_date",
            "ticker",
            "sector",
            "headlines_joined",
        ]
    )

    # Project A joined individual headlines with " || ".
    frame["headline"] = (
        frame["headlines_joined"]
        .astype(str)
        .str.split(r"\s*\|\|\s*")
    )

    frame = frame.explode(
        "headline"
    )

    frame["headline"] = (
        frame["headline"]
        .astype(str)
        .str.strip()
    )

    frame = frame.loc[
        frame["headline"].ne("")
    ].copy()

    return frame[
        [
            "trading_date",
            "ticker",
            "sector",
            "headline",
        ]
    ].reset_index(drop=True)

def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply a sentiment model (VADER or another) to the assembled headlines.

    TODO: return a per-headline or per-ticker-day sentiment score. VADER uses
    casing, punctuation, and negation, so do not strip them. VADER also needs a
    one-time nltk.download('vader_lexicon') before it scores (a build step, not the
    deployed app).
    """
    if not isinstance(
        panel,
        pd.DataFrame,
    ) or panel.empty:
        raise ValueError(
            "panel must be a non-empty DataFrame."
        )

    headlines = _explode_headlines(
        panel
    )

    plain_analyzer, finance_analyzer = (
        _build_analyzers()
    )

    # --------------------------------------------------------
    # Score each unique headline only once.
    # --------------------------------------------------------

    unique_headlines = (
        headlines[["headline"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    records = []

    for headline in unique_headlines["headline"]:

        finance_text, phrase_hits = (
            _prepare_finance_text(
                headline
            )
        )

        plain_score = (
            plain_analyzer
            .polarity_scores(headline)
        )

        finance_score = (
            finance_analyzer
            .polarity_scores(finance_text)
        )

        records.append(
            {
                "headline": headline,

                "plain_compound":
                    float(
                        plain_score["compound"]
                    ),

                "finance_compound":
                    float(
                        finance_score["compound"]
                    ),

                "finance_positive":
                    float(
                        finance_score["pos"]
                    ),

                "finance_negative":
                    float(
                        finance_score["neg"]
                    ),

                "finance_neutral":
                    float(
                        finance_score["neu"]
                    ),

                "finance_phrase_hits":
                    int(phrase_hits),
            }
        )

    headline_scores = pd.DataFrame(
        records
    )

    scored = headlines.merge(
        headline_scores,
        on="headline",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Headline-level classification for validation only.
    # Compound remains the continuous sentiment measure.
    # --------------------------------------------------------

    scored["sentiment_class"] = np.select(
        [
            scored["finance_compound"]
            >= NEUTRAL_THRESHOLD,

            scored["finance_compound"]
            <= -NEUTRAL_THRESHOLD,
        ],
        [
            "positive",
            "negative",
        ],
        default="neutral",
    )

    # --------------------------------------------------------
    # Aggregate individual headlines into ticker-day scores.
    # --------------------------------------------------------

    ticker_day = (
        scored
        .groupby(
            [
                "trading_date",
                "ticker",
                "sector",
            ],
            as_index=False,
        )
        .agg(
            n_headlines=(
                "headline",
                "size",
            ),

            plain_sentiment=(
                "plain_compound",
                "mean",
            ),

            sentiment_score=(
                "finance_compound",
                "mean",
            ),

            positive_share=(
                "sentiment_class",
                lambda values:
                    float(
                        (
                            values
                            == "positive"
                        ).mean()
                    ),
            ),

            negative_share=(
                "sentiment_class",
                lambda values:
                    float(
                        (
                            values
                            == "negative"
                        ).mean()
                    ),
            ),

            neutral_share=(
                "sentiment_class",
                lambda values:
                    float(
                        (
                            values
                            == "neutral"
                        ).mean()
                    ),
            ),

            finance_phrase_hits=(
                "finance_phrase_hits",
                "sum",
            ),
        )
        .sort_values(
            [
                "trading_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    ticker_day["score_100"] = (
        (
            ticker_day["sentiment_score"]
            + 1.0
        )
        * 50.0
    )

    # --------------------------------------------------------
    # Store model diagnostics as attrs for later saving.
    # --------------------------------------------------------

    plain_neutral = (
        unique_headlines
        .merge(
            headline_scores,
            on="headline",
            how="left",
        )["plain_compound"]
        .abs()
        < NEUTRAL_THRESHOLD
    ).mean()

    finance_neutral = (
        headline_scores[
            "finance_compound"
        ].abs()
        < NEUTRAL_THRESHOLD
    ).mean()

    model_audit = pd.DataFrame(
        [
            {
                "metric":
                    "headline_rows_scored",
                "value":
                    len(scored),
            },
            {
                "metric":
                    "unique_headlines_scored",
                "value":
                    len(unique_headlines),
            },
            {
                "metric":
                    "finance_lexicon_terms",
                "value":
                    len(FINANCE_LEXICON)
                    + len(PHRASE_LEXICON),
            },
            {
                "metric":
                    "plain_vader_neutral_rate",
                "value":
                    float(plain_neutral),
            },
            {
                "metric":
                    "finance_vader_neutral_rate",
                "value":
                    float(finance_neutral),
            },
            {
                "metric":
                    "headline_scores_changed",
                "value":
                    int(
                        (
                            np.abs(
                                headline_scores[
                                    "finance_compound"
                                ]
                                -
                                headline_scores[
                                    "plain_compound"
                                ]
                            )
                            > 1e-12
                        ).sum()
                    ),
            },
        ]
    )

    ticker_day.attrs[
        "headline_scores"
    ] = scored

    ticker_day.attrs[
        "model_audit"
    ] = model_audit

    ticker_day.attrs[
        "finance_lexicon"
    ] = pd.DataFrame(
        [
            {
                "term": term,
                "valence": value,
                "source":
                    "Project A informed, manually reviewed",
            }
            for term, value
            in {
                **FINANCE_LEXICON,
                **PHRASE_LEXICON,
            }.items()
        ]
    )

    return ticker_day

def sector_sentiment_index(scores: pd.DataFrame) -> pd.DataFrame:
    """TODO: build a daily sentiment index per sector (equal-weight across names)."""
    required = {
        "trading_date",
        "ticker",
        "sector",
        "sentiment_score",
        "n_headlines",
    }

    missing = required.difference(
        scores.columns
    )

    if missing:
        raise ValueError(
            f"scores is missing columns: {sorted(missing)}"
        )

    clean = scores.copy()

    clean["trading_date"] = pd.to_datetime(
        clean["trading_date"],
        errors="coerce",
    )

    clean = clean.dropna(
        subset=[
            "trading_date",
            "ticker",
            "sector",
        ]
    )

    # --------------------------------------------------------
    # Fixed ticker-sector membership.
    # --------------------------------------------------------

    ticker_sector = (
        clean[
            [
                "ticker",
                "sector",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "sector",
                "ticker",
            ]
        )
    )

    trading_dates = (
        pd.DatetimeIndex(
            clean[
                "trading_date"
            ].dropna().unique()
        )
        .sort_values()
    )

    # --------------------------------------------------------
    # Full trading-date × ticker grid.
    # --------------------------------------------------------

    full_grid = (
        pd.MultiIndex
        .from_product(
            [
                trading_dates,
                ticker_sector[
                    "ticker"
                ].sort_values(),
            ],
            names=[
                "trading_date",
                "ticker",
            ],
        )
        .to_frame(
            index=False
        )
    )

    full_grid = full_grid.merge(
        ticker_sector,
        on="ticker",
        how="left",
        validate="many_to_one",
    )

    grid = full_grid.merge(
        clean[
            [
                "trading_date",
                "ticker",
                "sentiment_score",
                "n_headlines",
            ]
        ],
        on=[
            "trading_date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # No-news policy:
    # missing ticker-day sentiment is neutral, not carried forward.
    # --------------------------------------------------------

    grid[
        "has_news"
    ] = (
        grid["sentiment_score"]
        .notna()
    )

    grid[
        "sentiment_neutral_filled"
    ] = (
        grid[
            "sentiment_score"
        ]
        .fillna(0.0)
    )

    grid[
        "n_headlines"
    ] = (
        grid["n_headlines"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Equal-weight ticker aggregation within each sector.
    # --------------------------------------------------------

    sector = (
        grid
        .groupby(
            [
                "trading_date",
                "sector",
            ],
            as_index=False,
        )
        .agg(
            sentiment_score=(
                "sentiment_neutral_filled",
                "mean",
            ),

            covered_tickers=(
                "has_news",
                "sum",
            ),

            total_tickers=(
                "ticker",
                "nunique",
            ),

            headline_count=(
                "n_headlines",
                "sum",
            ),
        )
        .sort_values(
            [
                "sector",
                "trading_date",
            ]
        )
        .reset_index(drop=True)
    )

    sector[
        "coverage_ratio"
    ] = (
        sector["covered_tickers"]
        /
        sector["total_tickers"]
    )

    sector[
        "score_100"
    ] = (
        (
            sector[
                "sentiment_score"
            ]
            + 1.0
        )
        * 50.0
    )

    # --------------------------------------------------------
    # Strict 1-trading-day lag.
    # --------------------------------------------------------

    sector[
        "lagged_sentiment_score"
    ] = (
        sector
        .groupby("sector")[
            "sentiment_score"
        ]
        .shift(1)
    )

    sector[
        "lagged_score_100"
    ] = (
        (
            sector[
                "lagged_sentiment_score"
            ]
            + 1.0
        )
        * 50.0
    )

    # --------------------------------------------------------
    # Expanding historical z-score.
    #
    # The benchmark uses only values available BEFORE date t.
    # No current-day observation enters its own benchmark.
    # --------------------------------------------------------

    sector_frames = []

    for _, group in sector.groupby(
        "sector",
        sort=False,
    ):

        group = (
            group
            .sort_values(
                "trading_date"
            )
            .copy()
        )

        historical = (
            group[
                "lagged_sentiment_score"
            ]
        )

        prior_mean = (
            historical
            .expanding(
                min_periods=60
            )
            .mean()
            .shift(1)
        )

        prior_std = (
            historical
            .expanding(
                min_periods=60
            )
            .std(ddof=1)
            .shift(1)
        )

        group[
            "sentiment_z"
        ] = (
            (
                historical
                - prior_mean
            )
            /
            prior_std.replace(
                0.0,
                np.nan,
            )
        )

        # 21-trading-day smoothed relative sentiment is
        # primarily for visualisation and interpretation.
        group[
            "sentiment_z_21d_mean"
        ] = (
            group[
                "sentiment_z"
            ]
            .rolling(
                window=21,
                min_periods=5,
            )
            .mean()
        )

        sector_frames.append(
            group
        )

    sector = (
        pd.concat(
            sector_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "trading_date",
                "sector",
            ]
        )
        .reset_index(drop=True)
    )

    sector.attrs[
        "missing_news_policy"
    ] = (
        "Ticker-days with no headline are treated as neutral (0) "
        "before equal-weight sector aggregation; coverage is "
        "reported separately."
    )

    sector.attrs[
        "lookahead_policy"
    ] = (
        "Sector sentiment is lagged by one trading day. "
        "Historical standardisation uses prior observations only."
    )

    return sector

def save_sentiment_outputs(
    ticker_day_scores: pd.DataFrame,
    sector_index: pd.DataFrame,
    project_root: str | Path,
) -> dict[str, Path]:
    """
    Save the required sentiment artifact and supporting audit tables.
    """
    root = Path(project_root)

    data_dir = (
        root
        / "results"
        / "data"
    )

    tables_dir = (
        root
        / "results"
        / "tables"
    )

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "sector_sentiment_index":
            data_dir
            / "sector_sentiment_index.csv",

        "ticker_day_sentiment":
            tables_dir
            / "ticker_day_sentiment.csv",

        "sentiment_model_audit":
            tables_dir
            / "sentiment_model_audit.csv",

        "finance_lexicon":
            tables_dir
            / "finance_lexicon.csv",
    }

    ticker_day = (
        ticker_day_scores.copy()
    )

    sector = (
        sector_index.copy()
    )

    ticker_day[
        "trading_date"
    ] = pd.to_datetime(
        ticker_day[
            "trading_date"
        ],
        errors="coerce",
    )

    sector[
        "trading_date"
    ] = pd.to_datetime(
        sector[
            "trading_date"
        ],
        errors="coerce",
    )

    ticker_day.to_csv(
        paths[
            "ticker_day_sentiment"
        ],
        index=False,
    )

    sector.to_csv(
        paths[
            "sector_sentiment_index"
        ],
        index=False,
    )

    model_audit = (
        ticker_day_scores
        .attrs
        .get(
            "model_audit",
            pd.DataFrame(),
        )
    )

    model_audit.to_csv(
        paths[
            "sentiment_model_audit"
        ],
        index=False,
    )

    finance_lexicon = (
        ticker_day_scores
        .attrs
        .get(
            "finance_lexicon",
            pd.DataFrame(),
        )
    )

    finance_lexicon.to_csv(
        paths[
            "finance_lexicon"
        ],
        index=False,
    )

    return paths

def plot_sector_sentiment(
    sector_index: pd.DataFrame,
    project_root: str | Path,
) -> Path:
    """
    Plot 21-day smoothed relative sentiment for the 10 equity sectors.
    """
    root = Path(project_root)
    figures_dir = root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = sector_index.copy()
    data["trading_date"] = pd.to_datetime(
        data["trading_date"],
        errors="coerce",
    )

    sectors = sorted(
        data["sector"].dropna().unique()
    )

    figure, axes = plt.subplots(
        5,
        2,
        figsize=(11, 12),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for axis, sector_name in zip(
        axes,
        sectors,
    ):
        subset = data.loc[
            data["sector"].eq(sector_name)
        ].sort_values("trading_date")

        axis.plot(
            subset["trading_date"],
            subset["sentiment_z_21d_mean"],
            linewidth=1.2,
        )

        axis.axhline(
            0.0,
            linewidth=0.8,
        )

        axis.set_title(
            sector_name,
            fontsize=10,
        )

        axis.grid(
            alpha=0.20,
        )

    for axis in axes[len(sectors):]:
        axis.axis("off")

    figure.suptitle(
        "NexaVest Finance-Aware Relative Sector Sentiment\n"
        "21-trading-day mean of lagged expanding z-scores, 2020–2023",
        fontsize=14,
    )

    figure.supxlabel("Trading date")
    figure.supylabel("Relative sentiment (z-score)")

    figure.text(
        0.01,
        0.01,
        "Source: equity news headlines. Sector scores are equal-weighted "
        "across fixed ticker sets; no-news ticker-days are neutral-filled. "
        "Relative sentiment uses lagged, prior-history standardisation.",
        fontsize=8,
    )

    figure.tight_layout(
        rect=(0.03, 0.04, 1, 0.96)
    )

    path = (
        figures_dir
        / "sector_sentiment_index.png"
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path

def plot_sentiment_validation(
    model_audit: pd.DataFrame,
    project_root: str | Path,
) -> Path:
    """
    Compare neutral rates of plain and finance-aware VADER.
    """
    root = Path(project_root)
    figures_dir = root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    lookup = dict(
        zip(
            model_audit["metric"],
            model_audit["value"],
        )
    )

    plain_rate = float(
        lookup[
            "plain_vader_neutral_rate"
        ]
    )

    finance_rate = float(
        lookup[
            "finance_vader_neutral_rate"
        ]
    )

    labels = [
        "Plain VADER",
        "Finance-aware VADER",
    ]

    values = [
        plain_rate * 100,
        finance_rate * 100,
    ]

    figure, axis = plt.subplots(
        figsize=(7.5, 4.8)
    )

    bars = axis.bar(
        labels,
        values,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axis.set_ylabel(
        "Neutral headline rate (%)"
    )

    axis.set_title(
        "NexaVest Sentiment Model Validation\n"
        "Plain vs finance-aware VADER"
    )

    axis.grid(
        axis="y",
        alpha=0.20,
    )

    figure.text(
        0.01,
        0.01,
        "Source: 105,329 unique equity-news headlines. "
        "A lower neutral rate indicates broader finance vocabulary coverage, "
        "not evidence of return predictability.",
        fontsize=8,
    )

    figure.tight_layout(
        rect=(0, 0.05, 1, 1)
    )

    path = (
        figures_dir
        / "sentiment_model_validation.png"
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path
