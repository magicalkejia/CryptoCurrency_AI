"""Annualization helpers shared by stock and crypto backtests.

The legacy backtest API uses the name ``annual_days``.  In practice the value is
"periods per year": 252 for daily equities, 2190 for 4h crypto, 8760 for 1h
crypto, and so on.  Keep the old name as an alias, but centralize inference here
so callers do not hard-code 252 or 2190 throughout the codebase.
"""
from __future__ import annotations

import re
from typing import Literal

import pandas as pd


Market = Literal["stock", "crypto"]

STOCK_TRADING_DAYS_PER_YEAR = 252
STOCK_TRADING_HOURS_PER_DAY = 6.5
CRYPTO_CALENDAR_DAYS_PER_YEAR = 365


def normalize_market(market: str | None) -> Market:
    value = (market or "stock").strip().lower()
    if value in {"crypto", "cryptocurrency", "digital_asset", "digital_assets", "24/7", "247"}:
        return "crypto"
    if value in {"stock", "stocks", "equity", "equities", "share", "shares"}:
        return "stock"
    raise ValueError(f"Unsupported market={market!r}; expected 'stock' or 'crypto'.")


def timeframe_to_timedelta(timeframe: str | None) -> pd.Timedelta:
    """Parse common bar strings such as 1m, 15m, 1h, 4h, 1d and 1w."""
    if timeframe is None:
        timeframe = "1d"
    text = str(timeframe).strip().lower()
    aliases = {
        "d": "1d",
        "day": "1d",
        "daily": "1d",
        "h": "1h",
        "hour": "1h",
        "hourly": "1h",
        "m": "1m",
        "min": "1m",
        "minute": "1m",
        "w": "1w",
        "week": "1w",
        "weekly": "1w",
    }
    text = aliases.get(text, text)

    match = re.fullmatch(r"(\d+)?\s*([a-z]+)", text)
    if not match:
        raise ValueError(f"Unsupported timeframe={timeframe!r}.")
    n = int(match.group(1) or 1)
    unit = match.group(2)

    if unit in {"m", "min", "mins", "minute", "minutes"}:
        return pd.Timedelta(minutes=n)
    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        return pd.Timedelta(hours=n)
    if unit in {"d", "day", "days"}:
        return pd.Timedelta(days=n)
    if unit in {"w", "wk", "wks", "week", "weeks"}:
        return pd.Timedelta(weeks=n)
    if unit in {"mo", "mon", "month", "months"}:
        # Calendar months are variable length; use a standard accounting month.
        return pd.Timedelta(days=30 * n)
    if unit in {"y", "yr", "year", "years"}:
        return pd.Timedelta(days=365 * n)
    raise ValueError(f"Unsupported timeframe unit in {timeframe!r}.")


def infer_annual_periods(
    timeframe: str | None = "1d",
    market: str | None = "stock",
    *,
    stock_trading_days_per_year: int = STOCK_TRADING_DAYS_PER_YEAR,
    stock_trading_hours_per_day: float = STOCK_TRADING_HOURS_PER_DAY,
    crypto_calendar_days_per_year: int = CRYPTO_CALENDAR_DAYS_PER_YEAR,
) -> int:
    """Infer periods per year for the given market and bar frequency.

    For crypto, the market is assumed to trade continuously.  For stocks, daily
    and weekly bars use exchange trading days; intraday bars use the regular
    trading session length.  The function returns at least 1.
    """
    mkt = normalize_market(market)
    bar = timeframe_to_timedelta(timeframe)
    bar_seconds = bar.total_seconds()
    if bar_seconds <= 0:
        raise ValueError(f"timeframe must be positive, got {timeframe!r}.")

    if mkt == "crypto":
        year_seconds = crypto_calendar_days_per_year * 24 * 60 * 60
        return max(1, int(round(year_seconds / bar_seconds)))

    # Stock/equity path.  A daily bar means one exchange trading day, not one
    # calendar day; intraday bars are scaled by regular session hours.
    one_day = pd.Timedelta(days=1).total_seconds()
    one_week = pd.Timedelta(weeks=1).total_seconds()
    if bar_seconds >= one_week:
        return max(1, int(round(52 / (bar_seconds / one_week))))
    if bar_seconds >= one_day:
        return max(1, int(round(stock_trading_days_per_year / (bar_seconds / one_day))))

    trading_seconds_per_day = stock_trading_hours_per_day * 60 * 60
    return max(1, int(round(stock_trading_days_per_year * trading_seconds_per_day / bar_seconds)))


def resolve_annual_periods(
    annual_periods: int | None = None,
    annual_days: int | None = None,
    *,
    market: str | None = "stock",
    timeframe: str | None = "1d",
) -> int:
    """Resolve explicit or inferred periods per year.

    ``annual_days`` is accepted as a backward-compatible alias.  Prefer
    ``annual_periods`` or ``market`` + ``timeframe`` in new code.
    """
    value = annual_periods if annual_periods is not None else annual_days
    if value is not None:
        value = int(value)
        if value <= 0:
            raise ValueError(f"annual periods must be positive, got {value}.")
        return value
    return infer_annual_periods(timeframe=timeframe, market=market)
