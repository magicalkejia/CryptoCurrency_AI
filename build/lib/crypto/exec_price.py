"""
crypto.exec_price  &  crypto.costs (combined module)
==========================================================
Split entry / exit execution price (audit fix #2, #3) and the cost model
(slippage / funding / decomposition: audit fix #11, #12, #13).

Convention (chosen explicitly to avoid the double-count the auditor warned about):
  * entry_ref_price / exit_ref_price are RAW reference prices (no slippage).
  * label_entry_price == entry_ref_price (so labels & backtest share one function).
  * ALL execution friction (fee, spread, impact slippage, rounding) is deducted
    once in the cost term of net_exit_return.  Prices are never slippage-adjusted
    inside the labelling path -> impact slippage is counted exactly once.
  * get_executable_price() is provided separately for the live/backtest *fill*
    path, where you do want a slippage-adjusted price.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Execution price (reference + executable)
# --------------------------------------------------------------------------- #
def get_entry_price(
    bars: pd.DataFrame,
    decision_time: pd.Timestamp,
    rule: Literal["next_1h_open", "next_1m_open"] = "next_1h_open",
    bars_1m: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    """
    Return (entry_time, entry_ref_price) = the first tradable bar's OPEN after
    decision_time.  RAW reference price, no slippage (audit #2/#3).

    `bars` is a 1h OHLC frame indexed by ts_open. For next_1m_open, bars_1m is
    required (audit #4).
    """
    if rule == "next_1m_open":
        if bars_1m is None:
            raise ValueError("bars_1m required for next_1m_open")
        src = bars_1m
    else:
        src = bars
    nxt = src.index[src.index > decision_time]
    if len(nxt) == 0:
        return None, None
    t = nxt[0]
    return t, float(src.loc[t, "open"])


def apply_slippage(ref_price: float, side: Literal["long", "short"],
                   action: Literal["entry", "exit"], slippage_bps: float) -> float:
    """Side/action-aware executable price (for live/backtest fills)."""
    adj = slippage_bps / 1e4
    # entry long buys -> pays up; entry short sells -> receives less
    # exit  long sells -> receives less; exit short buys -> pays up
    if (side == "long" and action == "entry") or (side == "short" and action == "exit"):
        return ref_price * (1 + adj)
    return ref_price * (1 - adj)


def get_executable_price(ref_price: float, side: Literal["long", "short"],
                         action: Literal["entry", "exit"], slippage_bps: float) -> float:
    return apply_slippage(ref_price, side, action, slippage_bps)


# --------------------------------------------------------------------------- #
# Slippage model (audit #11): unit-consistent bps
# --------------------------------------------------------------------------- #
def slippage_bps(
    realized_vol_short_bps: float,
    vol_benchmark_bps: float,
    order_notional: float,
    avg_depth_proxy: float,
    cost_cfg,
) -> float:
    """
    slippage_bps = base + k_vol*max(0,(rv-bench)/bench) + k_liq*depth_ratio
    All terms in bps. depth_ratio is dimensionless. (audit #11)
    """
    if avg_depth_proxy is None or not np.isfinite(avg_depth_proxy) or avg_depth_proxy <= 0:
        # depth unknown -> treat as fully illiquid penalty cap
        depth_ratio = 1.0
    else:
        depth_ratio = order_notional / avg_depth_proxy
    vol_term = 0.0
    if vol_benchmark_bps and vol_benchmark_bps > 0:
        vol_term = cost_cfg.k_vol_bps * max(0.0, (realized_vol_short_bps - vol_benchmark_bps) / vol_benchmark_bps)
    return cost_cfg.base_slippage_bps + vol_term + cost_cfg.k_liq_bps * depth_ratio


# --------------------------------------------------------------------------- #
# Funding PnL (audit #12): mark notional at each funding timestamp, signed
# --------------------------------------------------------------------------- #
def funding_return(
    side: Literal["long", "short"],
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_ref_price: float,
    funding: Optional[pd.Series],
    mark_price: Optional[pd.Series] = None,
) -> float:
    """
    Sign convention (linear USDT perp):
      funding_rate > 0 -> long PAYS, short RECEIVES.
    funding_pnl_k = - side_sign * notional_k * rate_k ; notional_k uses mark price.
    Returned as a *return on entry notional* so it slots into net return additively
    (can be + or -). audit #12.
    """
    if funding is None or len(funding) == 0:
        return 0.0
    mask = (funding.index > entry_time) & (funding.index <= exit_time)
    rates = funding[mask]
    if len(rates) == 0:
        return 0.0
    side_sign = 1.0 if side == "long" else -1.0
    entry_notional = entry_ref_price  # per 1 unit qty
    pnl = 0.0
    for ts, rate in rates.items():
        mk = float(mark_price.asof(ts)) if mark_price is not None else entry_ref_price
        notional_k = abs(mk)  # 1 unit qty
        pnl += -side_sign * notional_k * float(rate)
    return pnl / entry_notional


# --------------------------------------------------------------------------- #
# Net return (audit #3, #13): PnL-additive, no funding sign reversal,
# spread & impact slippage decomposed and not double-counted.
# --------------------------------------------------------------------------- #
def net_return(
    raw_return_side: float,
    side: Literal["long", "short"],
    cost_cfg,
    funding_ret: float = 0.0,
    impact_slippage_bps: Optional[float] = None,
) -> float:
    """
    net = raw + funding_return - (fee + spread + impact_slippage + rounding)
    fee/spread/impact charged on BOTH legs (entry+exit). (audit #3, #13)
    funding_ret is already signed (can be + or -); we ADD it (never subtract a
    signed quantity -> avoids the sign reversal the auditor flagged).
    """
    imp = cost_cfg.base_slippage_bps if impact_slippage_bps is None else impact_slippage_bps
    fee_ret = 2.0 * cost_cfg.fee_bps / 1e4
    spread_ret = 2.0 * cost_cfg.spread_proxy_bps / 1e4
    impact_ret = 2.0 * imp / 1e4
    rounding_ret = cost_cfg.rounding_bps / 1e4
    return raw_return_side + funding_ret - (fee_ret + spread_ret + impact_ret + rounding_ret)


def depth_proxy(volume_same_bucket: pd.Series, cost_cfg) -> float:
    """
    avg_depth_proxy = k_depth * median(past-30d same 4h-bucket volume).
    Caller is responsible for passing a PIT-safe (strictly past) series
    (audit #12 / v6 8.5.1). Returns NaN if too few samples (audit #10).
    """
    s = pd.Series(volume_same_bucket).dropna()
    if len(s) < cost_cfg.min_depth_samples:
        return float("nan")
    return cost_cfg.k_depth * float(s.median())
