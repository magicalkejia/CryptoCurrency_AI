"""
crypto.features.derivatives
==============================
Turn the newly-fetched funding-rate / open-interest data into point-in-time
features (v6 §5) and provide a funding Series for the triple-barrier cost model.

PIT discipline:
  * funding is known at its settlement time -> availability_ts = funding_time.
  * OI from Binance has only ~30d history (data-source limit, see notes); we
    expose what exists and let it be NaN before that (degrades gracefully).
  * all z-scores are rolling (right-aligned), never full-sample.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def load_funding(loader_processed_dir, symbol: str) -> Optional[pd.Series]:
    """Load RAW/PROCESSED funding parquet -> Series indexed by settlement time."""
    from pathlib import Path
    p = Path(loader_processed_dir) / f"{symbol.replace('/', '')}_funding.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    s = df.set_index("timestamp")["funding_rate"].sort_index()
    return s


def funding_features(funding: pd.Series, decision_times: pd.DatetimeIndex,
                     z_window: int = 90) -> pd.DataFrame:
    """
    For each decision_time, the most recent funding known strictly before it
    (PIT), plus a rolling z-score / change. z_window in number of funding points
    (90 * 8h ~ 30 days).
    """
    if funding is None or len(funding) == 0:
        return pd.DataFrame(index=decision_times,
                            columns=["funding_rate", "funding_rate_z", "funding_rate_chg"], dtype=float)
    f = funding.sort_index()
    z = (f - f.rolling(z_window, min_periods=10).mean()) / (f.rolling(z_window, min_periods=10).std() + 1e-12)
    chg = f.diff()
    feat = pd.DataFrame({"funding_rate": f, "funding_rate_z": z, "funding_rate_chg": chg})
    # asof: latest funding with index <= decision_time (PIT)
    out = feat.reindex(feat.index.union(decision_times)).sort_index().ffill().reindex(decision_times)
    out.index = decision_times
    return out


def oi_features(loader_processed_dir, symbol: str, decision_times: pd.DatetimeIndex,
                z_window: int = 180, availability_lag_min: int = 5) -> pd.DataFrame:
    """OI change / z-score, PIT-aligned with a small availability lag."""
    from pathlib import Path
    p = Path(loader_processed_dir) / f"{symbol.replace('/', '')}_oi.parquet"
    cols = ["oi", "oi_chg", "oi_z"]
    if not p.exists():
        return pd.DataFrame(index=decision_times, columns=cols, dtype=float)
    df = pd.read_parquet(p).set_index("timestamp").sort_index()
    oi = df["open_interest"]
    oi.index = oi.index + pd.Timedelta(minutes=availability_lag_min)  # availability lag
    z = (oi - oi.rolling(z_window, min_periods=10).mean()) / (oi.rolling(z_window, min_periods=10).std() + 1e-12)
    feat = pd.DataFrame({"oi": oi, "oi_chg": oi.pct_change(), "oi_z": z})
    out = feat.reindex(feat.index.union(decision_times)).sort_index().ffill().reindex(decision_times)
    out.index = decision_times
    return out
