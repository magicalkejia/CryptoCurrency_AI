"""
crypto.adapters
==================
Bridges the EXISTING etl.data_loader.DataLoader to the v6 bars schema, WITHOUT
modifying any existing code.

  * adds ts_close and availability_ts (= ts_close + kline_lag) per audit #16,
    so PIT is unambiguous (a bar opened at 12:00 is only usable after 13:00+lag).
  * builds the 4h decision-time grid = bar_close + decision_offset_minutes
    (v6 §1.2).

The existing loader returns crypto frames indexed by 'timestamp' (= bar open)
with columns open/high/low/close/volume/taker_buy_vol/net_taker_vol.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def to_bars_schema(df: pd.DataFrame, timeframe: str, kline_lag_seconds: int = 30) -> pd.DataFrame:
    """
    df: index = bar-open timestamp, columns include open/high/low/close/volume.
    Returns a frame indexed by ts_open with ts_close + availability_ts added.
    """
    bar_td = pd.Timedelta(timeframe.replace("h", "H").replace("m", "min"))
    out = df.copy()
    out.index.name = "ts_open"
    out["ts_close"] = out.index + bar_td
    out["availability_ts"] = out["ts_close"] + pd.Timedelta(seconds=kline_lag_seconds)
    return out


def decision_time_grid(bars_4h: pd.DataFrame, offset_minutes: int = 1) -> pd.DatetimeIndex:
    """v6 §1.2: decision_time = 4h bar close + offset_minutes."""
    if "ts_close" not in bars_4h.columns:
        raise ValueError("bars_4h must have ts_close (use to_bars_schema first)")
    return pd.DatetimeIndex(bars_4h["ts_close"] + pd.Timedelta(minutes=offset_minutes))


def load_crypto_bars(loader, symbol: str, timeframe: str = "1h",
                     start_date: Optional[str] = None, end_date: Optional[str] = None,
                     kline_lag_seconds: int = 30) -> Optional[pd.DataFrame]:
    """Use the existing DataLoader, then attach the v6 schema columns."""
    df = loader.get_crypto_kline_data(symbol=symbol, timeframe=timeframe,
                                      start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return None
    return to_bars_schema(df, timeframe, kline_lag_seconds)
