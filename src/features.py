"""Station 2 - your features: return features and headline text assembly.

Build your return (and optional risk) features here, and assemble the headlines
into a daily text panel. Part A is assembly only - scoring the text into sentiment
(and lagging the signal) is the Station 3 model, built in Part B, not here.
"""

import pandas as pd

from src import data_access


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Simple daily returns per ticker. Use adjClose.

    TODO: pivot to wide (date x ticker) or keep long, your choice; pct_change
    within each ticker.
    """

    required_columns = {"ticker", "date", price_col}
    missing_columns = sorted(required_columns.difference(prices.columns))

    if missing_columns:
        raise ValueError(f"Missing required columns for return calculation: {missing_columns}")


    # Verification
    if not pd.api.types.is_datetime64_any_dtype(prices["date"]):
        raise TypeError("date must be datetime type; please standardize dates in src/etl.py.")

    if not pd.api.types.is_numeric_dtype(prices[price_col]):
        raise TypeError(f"{price_col} must be numeric; please standardize it in src/etl.py.")

    duplicate_count = int(prices.duplicated(subset=["ticker", "date"]).sum())

    if duplicate_count > 0:
        raise ValueError(
            "daily_returns input still contains "
            f"{duplicate_count} duplicate ticker-date rows; "
            "please handle them in src/etl.py."
        )

    # Check missing key values
    missing_key_rows = int(prices[["ticker", "date", price_col]].isna().any(axis=1).sum())

    if missing_key_rows > 0:
        raise ValueError(
            "daily_returns input still contains missing ticker, date, or price values; "
            "please handle them in src/etl.py."
        )
    df = prices.copy()

    # Sort by ticker and date
    df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    # 1. Daily simple return
    df["return"] = df.groupby("ticker", sort=False)[price_col].pct_change(fill_method=None)

    # 2. 21-day rolling volatility
    df["rolling_volatility_21d"] = df.groupby("ticker", sort=False)["return"].transform(
        lambda x: x.rolling(
            window=21,
            min_periods=21,
        ).std()
    )

    # 3. 21-day momentum
    df["momentum_21d"] = df.groupby("ticker", sort=False)["return"].transform(
        lambda x: (
            (1 + x)
            .rolling(
                window=21,
                min_periods=21,
            )
            .apply(
                lambda values: values.prod() - 1,
                raw=True,
            )
        )
    )

    # 4. Drawdown from the running historical peak
    running_max = df.groupby("ticker", sort=False)[price_col].cummax()

    df["drawdown"] = df[price_col] / running_max - 1

    # 5. Downside return: keep only negative returns
    df["downside_return"] = df["return"].where(
        df["return"].isna() | (df["return"] < 0),
        0.0,
    )

    # 6. 21-day downside volatility
    df["downside_volatility_21d"] = df.groupby("ticker", sort=False)["downside_return"].transform(
        lambda x: x.rolling(
            window=21,
            min_periods=21,
        ).std()
    )

    df = df.drop(columns=["downside_return"])
    df.attrs["feature_definitions"] = {
        "return": "Simple daily return calculated from adjusted close prices.",
        "rolling_volatility_21d": ("21-observation rolling standard deviation of daily returns."),
        "momentum_21d": ("Cumulative return over the previous 21 observations."),
        "drawdown": ("Percentage decline from each asset's running historical peak."),
        "downside_volatility_21d": ("21-observation rolling volatility based on downside returns."),
    }

    df.attrs["feature_parameters"] = {
        "price_field": price_col,
        "rolling_window": 21,
        "panel_format": "long",
    }

    return df


def assemble_headline_panel(headlines: pd.DataFrame) -> pd.DataFrame:
    """Assemble the headlines into a daily panel per ticker and sector.

    Station 2 (Part A) is assembly only: structure the text and date-align it to
    the trading calendar (a weekend headline maps to the next trading day).
    Scoring the text - and lagging the signal - is the Station 3 sentiment model,
    built in Part B, not here.
    """

    # 1. Check required columns
    required_columns = {
        "date",
        "ticker",
        "sector",
        "title",
        "url",
        "publisher",
    }

    missing_columns = sorted(
        required_columns.difference(headlines.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required news columns: {missing_columns}"
        )

    # 2. Copy raw data
    clean = headlines.copy()
    raw_rows = len(clean)

    # 3. Standardize news timestamps to UTC calendar dates
    clean["date"] = (
        pd.to_datetime(
            clean["date"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )

    # 4. Standardize text fields
    clean["ticker"] = (
        clean["ticker"]
        .astype("string")
        .str.strip()
    )

    clean["sector"] = (
        clean["sector"]
        .astype("string")
        .str.strip()
    )

    clean["title"] = (
        clean["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    clean["url"] = (
        clean["url"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    clean["publisher"] = (
        clean["publisher"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Keep original headline text for later Part B use
    clean["text_raw"] = clean["title"]

    # 5. Record data-quality issues before cleaning
    duplicate_headlines = int(
        clean.duplicated(
            subset=["ticker", "date", "title"]
        ).sum()
    )

    missing_dates = int(
        clean["date"].isna().sum()
    )

    missing_tickers = int(
        clean["ticker"].isna().sum()
    )

    missing_sectors = int(
        clean["sector"].isna().sum()
    )

    blank_titles = int(
        clean["title"].eq("").sum()
    )

    blank_publishers = int(
        clean["publisher"].eq("").sum()
    )

    # 6. Remove unusable key rows
    clean = clean.dropna(
        subset=["date", "ticker", "sector"]
    ).copy()

    clean = clean.loc[
        clean["title"].ne("")
    ].copy()

    # 7. Remove exact logical duplicates
    clean = clean.drop_duplicates(
        subset=["ticker", "date", "title"],
        keep="first",
    ).copy()

    # 8. Restrict to the Part A sample period
    clean = clean.loc[
        clean["date"].between(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2023-12-31"),
        )
    ].copy()

    # 9. Add headline-level descriptive information
    clean["headline_words"] = (
        clean["text_raw"]
        .map(lambda text: len(str(text).split()))
    )

    # 10. Load the real equity trading calendar
    equity_prices = (
        data_access.load_equity_prices()[["date"]]
        .copy()
    )

    equity_prices["date"] = (
        pd.to_datetime(
            equity_prices["date"],
            errors="coerce",
        )
        .dt.normalize()
    )

    trading_calendar = (
        equity_prices["date"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    trading_calendar_df = pd.DataFrame({"trading_date": trading_calendar})

    # Make sure both merge keys use the same datetime precision
    clean["date"] = clean["date"].astype("datetime64[ns]")

    trading_calendar_df["trading_date"] = trading_calendar_df["trading_date"].astype(
        "datetime64[ns]"
    )

    # Map each headline to the same or next equity trading day
    mapped = pd.merge_asof(
        clean.sort_values("date"),
        trading_calendar_df.sort_values("trading_date"),
        left_on="date",
        right_on="trading_date",
        direction="forward",
        allow_exact_matches=True,
    )

    unmapped_after_last_trading_day = int(
        mapped["trading_date"].isna().sum()
    )

    mapped = mapped.dropna(
        subset=["trading_date"]
    ).copy()

    mapped["trading_date"] = pd.to_datetime(
        mapped["trading_date"]
    ).dt.normalize()

    # 12. Aggregate to ticker x sector x trading-day panel
    panel = (
        mapped.groupby(
            ["trading_date", "ticker", "sector"],
            as_index=False,
        )
        .agg(
            headline_count=(
                "text_raw",
                "size",
            ),
            headlines_joined=(
                "text_raw",
                lambda values: " || ".join(
                    values.astype(str)
                ),
            ),
            total_headline_words=(
                "headline_words",
                "sum",
            ),
            mean_headline_words=(
                "headline_words",
                "mean",
            ),
            unique_publishers=(
                "publisher",
                lambda values: (
                    values[values.ne("")]
                    .nunique()
                ),
            ),
        )
        .sort_values(
            ["trading_date", "sector", "ticker"]
        )
        .reset_index(drop=True)
    )

    # 13. Add simple information-intensity indicator
    panel["headline_density"] = (
        panel["total_headline_words"]
        / panel["headline_count"]
    )

    # 14. Record headline audit metadata
    panel.attrs["headline_audit"] = {
        "dataset": "news_headlines",
        "raw_rows": raw_rows,
        "clean_rows_after_deduplication": len(clean),
        "mapped_rows": len(mapped),
        "daily_panel_rows": len(panel),
        "duplicate_ticker_date_title_rows": duplicate_headlines,
        "missing_dates": missing_dates,
        "missing_tickers": missing_tickers,
        "missing_sectors": missing_sectors,
        "blank_titles": blank_titles,
        "blank_publishers": blank_publishers,
        "unmapped_after_last_trading_day": (
            unmapped_after_last_trading_day
        ),
        "n_tickers": int(
            clean["ticker"].nunique()
        ),
        "n_sectors": int(
            clean["sector"].nunique()
        ),
        "resolution": (
            "Standardized UTC dates, removed unusable key rows and "
            "ticker-date-title duplicates, preserved missing publishers, "
            "and mapped non-trading-day headlines to the next equity "
            "trading day."
        ),
    }

    # 15. Record feature definitions
    panel.attrs["feature_definitions"] = {
        "headline_count": (
            "Number of headlines mapped to a ticker on a trading day."
        ),
        "headlines_joined": (
            "Original headline text concatenated for later Part B analysis."
        ),
        "total_headline_words": (
            "Total number of words across all headlines for a ticker-day."
        ),
        "mean_headline_words": (
            "Average headline length for a ticker-day."
        ),
        "unique_publishers": (
            "Number of distinct non-empty publishers contributing headlines."
        ),
        "headline_density": (
            "Average text volume per headline, calculated as total headline "
            "words divided by headline count."
        ),
    }

    # 16. Record provenance
    panel.attrs["provenance"] = {
        "source": (
            "src.data_access.load_news_headlines"
        ),
        "frequency": (
            "headline-level raw data aggregated to ticker-trading-day"
        ),
        "coverage": (
            "2020-01-01 to 2023-12-31"
        ),
        "processing": (
            "UTC date normalization, ticker-date-title deduplication, "
            "same-or-next equity trading-day mapping, and daily text assembly."
        ),
        "sentiment_scoring": (
            "Not performed in Part A."
        ),
    }

    return panel

def align_crypto_to_equity_calendar(
    equity_features: pd.DataFrame,
    crypto_features: pd.DataFrame,
) -> pd.DataFrame:
    """Align crypto returns to the equity trading calendar."""

    required_equity_columns = {
        "ticker",
        "date",
        "return",
    }

    required_crypto_columns = {
        "ticker",
        "date",
        "return",
    }

    missing_equity = sorted(
        required_equity_columns.difference(
            equity_features.columns
        )
    )

    missing_crypto = sorted(
        required_crypto_columns.difference(
            crypto_features.columns
        )
    )

    if missing_equity:
        raise ValueError(
            f"Missing required equity columns: {missing_equity}"
        )

    if missing_crypto:
        raise ValueError(
            f"Missing required crypto columns: {missing_crypto}"
        )

    equity_calendar = (
        equity_features[["date"]]
        .drop_duplicates()
        .sort_values("date")
        .reset_index(drop=True)
    )

    crypto_aligned = (
        equity_calendar.merge(
            crypto_features,
            on="date",
            how="left",
        )
        .sort_values(
            ["ticker", "date"]
        )
        .reset_index(drop=True)
    )

    return crypto_aligned
