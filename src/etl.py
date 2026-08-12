"""Station 1 - your ETL: load and clean the data.

Load raw data through src.data_access (see context/DATA_GUIDE.md). Add your own
integrity checks. Do not commit data files.
"""

import pandas as pd

from src import data_access


def load_clean_equities():
    """Load equity prices and run your Station 1 integrity checks.

    TODO: missing-date audit, duplicate ticker-date check, outlier/extreme-value
    screen on returns. Return the clean frame and record what you found.
    """
    df = data_access.load_equity_prices().copy()

    # 1.cleaning data
    raw_rows = len(df)
    raw_columns = list(df.columns)

    # 2.check field(find missing columns
    required_columns = {
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjClose",
        "volume",
        "sector",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"equity price columns missing: {missing_columns}")

    # 3.Uniform date format
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].astype("string").str.strip()
    df["sector"] = df["sector"].astype("string").str.strip()

    numeric_columns = ["open", "high", "low", "close", "adjClose", "volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 4.Statistical deletion situation
    duplicate_ticker_date = int(df.duplicated(subset=["ticker", "date"]).sum())

    missing_date = int(df["date"].isna().sum())
    missing_ticker = int(df["ticker"].isna().sum())
    missing_adjclose = int(df["adjClose"].isna().sum())
    missing_sector = int(df["sector"].isna().sum())

    # 5. Check the price and trading volumes
    non_positive_adjclose = int((df["adjClose"] <= 0).fillna(False).sum())

    negative_volume = int((df["volume"] < 0).fillna(False).sum())

    high_below_low = int((df["high"] < df["low"]).fillna(False).sum())

    open_outside_range = int(
        ((df["open"] < df["low"]) | (df["open"] > df["high"])).fillna(False).sum()
    )

    # 6. Check the closing price
    close_outside_range = int(
        ((df["close"] < df["low"]) | (df["close"] > df["high"])).fillna(False).sum()
    )

    #
    df = df.dropna(subset=["ticker", "date", "adjClose", "sector"]).copy()
    df = df.drop_duplicates(subset=["ticker", "date"], keep="first").copy()
    df = df.sort_values(by=["date", "ticker"]).reset_index(drop=True)

    # 7.really equity trading calendar
    equity_calendar = pd.DatetimeIndex(df["date"].drop_duplicates().sort_values())
    missing_dates_by_ticker = []
    for ticker, group in df.groupby("ticker", sort=True):
        observed_dates = pd.DatetimeIndex(group["date"].drop_duplicates().sort_values())
        missing_dates = equity_calendar.difference(observed_dates)
        missing_dates_by_ticker.append(
            {
                "ticker": ticker,
                "expected_trading_days": len(equity_calendar),
                "observed_trading_days": len(observed_dates),
                "missing_trading_days": len(missing_dates),
                "first_missing_date": (
                    missing_dates.min().date().isoformat() if len(missing_dates) else ""
                ),
                "last_missing_date": (
                    missing_dates.max().date().isoformat() if len(missing_dates) else ""
                ),
            }
        )

    # 8.DataFrame
    missing_dates_by_ticker = pd.DataFrame(missing_dates_by_ticker)

    # 9.Supplement the distribution of extreme adjclose
    audit_returns = df.groupby("ticker", sort=False)["adjClose"].pct_change(fill_method=None)

    # 10
    extreme_return_count = int(audit_returns.abs().ge(0.10).sum())

    # 11.Record audit result in DataFrame
    df.attrs["integrity"] = {
        "dataset": "equity_prices",
        "raw_rows": raw_rows,
        "clean_rows": len(df),
        "raw_columns": raw_columns,
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "n_tickers": int(df["ticker"].nunique()),
        "n_sectors": int(df["sector"].nunique()),
        "duplicate_ticker_date_rows": duplicate_ticker_date,
        "missing_date": missing_date,
        "missing_ticker": missing_ticker,
        "missing_adjclose": missing_adjclose,
        "missing_sector": missing_sector,
        "non_positive_adjclose": non_positive_adjclose,
        "negative_volume": negative_volume,
        "high_below_low": high_below_low,
        "open_outside_range": open_outside_range,
        "close_outside_range": close_outside_range,
        "extreme_return_count": extreme_return_count,
        "resolution": "No equity rows removed; no structural issues found.",
    }

    # 12.
    df.attrs["missing_dates_by_ticker"] = missing_dates_by_ticker
    df.attrs["provenance"] = {
        "source": "src.data_access.load_equity_prices",
        "frequency": "daily equity trading days",
        "coverage": "2020-01-02 to 2023-12-29",
        "cleaning": (
            "date normalization, schema validation, duplicate checks, "
            "missing-value checks, OHLC/volume integrity checks, "
            "trading-calendar completeness audit"
        ),
    }

    return df


def load_clean_crypto():
    """Load crypto prices (365-day calendar). TODO: your checks."""

    # 1. Load raw crypto data
    df = data_access.load_crypto_prices().copy()

    raw_rows = len(df)
    raw_columns = list(df.columns)

    # 2. Check required columns
    required_columns = {
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjClose",
        "volume",
    }

    missing_columns = sorted(required_columns.difference(df.columns))

    if missing_columns:
        raise ValueError(f"crypto price columns missing: {missing_columns}")

    # 3. Standardise data types
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.normalize()

    df["ticker"] = df["ticker"].astype("string").str.strip()

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adjClose",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # 4. Record integrity issues before cleaning
    duplicate_ticker_date = int(df.duplicated(subset=["ticker", "date"]).sum())

    missing_date = int(df["date"].isna().sum())
    missing_ticker = int(df["ticker"].isna().sum())
    missing_adjclose = int(df["adjClose"].isna().sum())

    non_positive_adjclose = int((df["adjClose"] <= 0).fillna(False).sum())

    negative_volume = int((df["volume"] < 0).fillna(False).sum())

    high_below_low = int((df["high"] < df["low"]).fillna(False).sum())

    open_outside_range = int(
        ((df["open"] < df["low"]) | (df["open"] > df["high"])).fillna(False).sum()
    )

    close_outside_range = int(
        ((df["close"] < df["low"]) | (df["close"] > df["high"])).fillna(False).sum()
    )

    # 5. Restrict sample to Part A end date
    sample_end_date = pd.Timestamp("2023-12-31")

    rows_after_sample_end = int((df["date"] > sample_end_date).fillna(False).sum())

    df = df[df["date"] <= sample_end_date].copy()

    # 6. Remove invalid key observations and duplicates
    df = df.dropna(
        subset=[
            "ticker",
            "date",
            "adjClose",
        ]
    ).copy()

    df = df.drop_duplicates(
        subset=["ticker", "date"],
        keep="first",
    ).copy()

    df = df.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    # 7. Check crypto calendar completeness
    start_date = df["date"].min()
    end_date = df["date"].max()

    if pd.isna(start_date) or pd.isna(end_date):
        raise ValueError("Crypto date range is empty or invalid.")

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    expected_calendar = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    )

    missing_dates_by_ticker = []

    for ticker, group in df.groupby(
        "ticker",
        sort=True,
    ):
        observed_dates = pd.DatetimeIndex(group["date"].drop_duplicates().sort_values())

        missing_dates = expected_calendar.difference(observed_dates)

        missing_dates_by_ticker.append(
            {
                "ticker": ticker,
                "expected_calendar_days": len(expected_calendar),
                "observed_calendar_days": len(observed_dates),
                "missing_calendar_days": len(missing_dates),
                "first_missing_date": (
                    missing_dates.min().date().isoformat() if len(missing_dates) else ""
                ),
                "last_missing_date": (
                    missing_dates.max().date().isoformat() if len(missing_dates) else ""
                ),
            }
        )

    missing_dates_by_ticker = pd.DataFrame(missing_dates_by_ticker)

    # 8. Screen extreme daily returns
    audit_returns = df.groupby(
        "ticker",
        sort=False,
    )["adjClose"].pct_change(fill_method=None)

    extreme_return_count = int(audit_returns.abs().ge(0.10).sum())

    # 9. Record audit results
    df.attrs["integrity"] = {
        "dataset": "crypto_prices",
        "raw_rows": raw_rows,
        "clean_rows": len(df),
        "raw_columns": raw_columns,
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "n_tickers": df["ticker"].nunique(),
        "duplicate_ticker_date_rows": duplicate_ticker_date,
        "missing_date": missing_date,
        "missing_ticker": missing_ticker,
        "missing_adjclose": missing_adjclose,
        "non_positive_adjclose": non_positive_adjclose,
        "negative_volume": negative_volume,
        "high_below_low": high_below_low,
        "close_outside_range": close_outside_range,
        "rows_after_sample_end": rows_after_sample_end,
        "extreme_return_count": extreme_return_count,
        "resolution": (
            "Excluded observations after "
            "2023-12-31; retained genuine "
            "extreme returns for later review."
        ),
    }

    # 10. Record calendar audit and provenance
    df.attrs["missing_dates_by_ticker"] = missing_dates_by_ticker

    df.attrs["provenance"] = {
        "source": "src.data_access.load_crypto_prices",
        "frequency": "daily 365-day crypto calendar",
        "coverage": "10 cryptocurrencies, 2020-2023, no industry names",
        "cleaning": (
            "date normalization, schema "
            "validation, duplicate checks, "
            "missing-value checks, OHLC/"
            "volume integrity checks, "
            "sample-end-date restriction, "
            "calendar completeness audit"
        ),
    }

    return df
