"""
crypto.live.market_feed
==========================
Real-time market data with point-in-time snapshot consistency (v6 §14.7).

  * MarketFeed interface: latest(symbol) -> (event_time, price).
  * ReplayFeed: offline, deterministic — replays historical bars as the "latest"
    snapshot at a simulated wall-clock; makes the staleness logic testable.
  * CCXTProFeed: live skeleton (needs ccxt.pro) that maintains the latest
    trade/orderbook per symbol from a websocket stream.
  * snapshot_consistency_check(): for a decision, returns per-symbol staleness
    (normal / stale_warning / skip) using age = read_time - event_time, and the
    list of symbols safe to trade — exactly the §14.7 gate before the OMS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from crypto.live.risk_guard import check_staleness, StalenessResult


class MarketFeed:
    def latest(self, symbol: str) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
        raise NotImplementedError


class ReplayFeed(MarketFeed):
    """Replays bars: latest() returns the most recent bar <= the simulated now."""

    def __init__(self, bars_by_symbol: Dict[str, pd.DataFrame],
                 read_delay_s: float = 0.5):
        self.bars = {s: b.sort_index() for s, b in bars_by_symbol.items()}
        self.now: Optional[pd.Timestamp] = None
        self.read_delay_s = read_delay_s

    def set_now(self, ts):
        self.now = pd.Timestamp(ts)

    def latest(self, symbol):
        b = self.bars.get(symbol)
        if b is None or self.now is None:
            return None, None
        sub = b.index[b.index <= self.now]
        if len(sub) == 0:
            return None, None
        t = sub[-1]
        return t, float(b.loc[t, "close"])

    def read_time(self) -> pd.Timestamp:
        # simulate that we read the snapshot read_delay_s after `now`
        return self.now + pd.Timedelta(seconds=self.read_delay_s)


def snapshot_consistency_check(feed: MarketFeed, symbols: List[str],
                               read_time: pd.Timestamp,
                               warn_s: float = 2.0, skip_s: float = 10.0) -> dict:
    """
    For each symbol, compute staleness of its latest snapshot vs read_time and
    decide tradeability (v6 §14.7).  Returns {symbol: StalenessResult} + the
    list of tradeable symbols.
    """
    out: Dict[str, StalenessResult] = {}
    tradeable: List[str] = []
    for s in symbols:
        ev, _ = feed.latest(s)
        if ev is None:
            out[s] = StalenessResult("skip", float("inf"), False, False, 0.0)
            continue
        r = check_staleness(ev, read_time, warn_s, skip_s)
        out[s] = r
        if r.allow_trade:
            tradeable.append(s)
    return {"per_symbol": out, "tradeable": tradeable}


# --------------------------------------------------------------------------- #
# Live skeleton (needs ccxt.pro). Maintains latest snapshot from a ws stream.
# --------------------------------------------------------------------------- #
class CCXTProFeed(MarketFeed):
    def __init__(self, exchange_id: str = "binanceusdm"):
        try:
            import ccxt.pro as ccxtpro  # noqa
        except Exception as e:
            raise RuntimeError("ccxt.pro required for live websocket feed") from e
        self._ccxtpro = ccxtpro
        self.ex = getattr(ccxtpro, exchange_id)({"enableRateLimit": True})
        self._latest: Dict[str, Tuple[pd.Timestamp, float]] = {}

    async def stream_trades(self, symbol: str):
        """Background task: keep updating the latest trade snapshot."""
        while True:
            trades = await self.ex.watch_trades(symbol)
            if trades:
                t = trades[-1]
                self._latest[symbol] = (pd.Timestamp(t["timestamp"], unit="ms"),
                                        float(t["price"]))

    def latest(self, symbol):
        return self._latest.get(symbol, (None, None))
