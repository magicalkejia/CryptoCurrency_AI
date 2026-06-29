"""
crypto.risk.portfolio
========================
Portfolio-level risk overlay (v6 §8.2).  The per-symbol Risk skill
(`risk_size_and_gate`) sizes each symbol IN ISOLATION (edge, per-symbol vol
target, confidence haircut, circuit-breaker level).  This module takes the
collection of per-symbol *intents* and applies the CROSS-symbol constraints the
per-symbol path cannot see:

  1. correlation haircut   — names that move together get shrunk, so the book is
                             not secretly one big BTC bet (avg |corr| to the rest
                             above `corr_floor` -> multiplicative penalty).
  2. cluster cap           — sum|w| within a correlated cluster <= max_cluster_weight.
  3. gross cap             — sum|w| across the whole book <= gross_cap.
  4. portfolio vol target  — scale the whole book down (never up, by default) so the
                             covariance-implied annualized vol <= target_portfolio_vol.
  5. drawdown scaler       — smooth de-risking as the portfolio equity drawdown grows
                             (a continuous companion to the discrete circuit breaker).

DELIBERATELY NOT here: the discrete circuit-breaker multiplier (cb_level).  That is
already applied once, per symbol, inside `risk_size_and_gate`; re-applying it here
would double-count.  This overlay only adds the portfolio-level pieces.

Everything is right-aligned / PIT: the rolling covariance uses only `close_panel`
rows up to (and including) the decision bar that the caller passes in.  Pure
numpy/pandas, fully unit-testable offline.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Default correlation clusters for the DIVERSIFIED_10 universe (base tickers).
# Tunable / override via `clusters=` arg; kept out of FrozenConfig so it does not
# bloat the config hash (it is structural, like the feature list).
DEFAULT_CLUSTERS: Dict[str, str] = {
    "BTC": "majors", "ETH": "majors",
    "SOL": "alt_l1", "BNB": "alt_l1", "ADA": "alt_l1", "TRX": "alt_l1",
    "XRP": "payments", "LTC": "payments",
    "DOGE": "high_beta", "LINK": "high_beta",
}


def base_ticker(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC'; 'BTCUSDT' -> 'BTC'; 'BTC' -> 'BTC'."""
    s = str(symbol).upper().replace("/", "")
    return s[:-4] if s.endswith("USDT") else s


def cluster_of(symbol: str, clusters: Optional[Dict[str, str]] = None) -> str:
    return (clusters or DEFAULT_CLUSTERS).get(base_ticker(symbol), "other")


def equity_risk_metrics(returns, bars_per_day: int = 6,
                        lookback_days: int = 30,
                        rolling_dd_days: int = None) -> Tuple[float, float, pd.Series]:
    """From a per-bar return series, derive the inputs the CircuitBreaker wants:
        drawdown               : peak-to-current equity drawdown (fraction, >=0)
        daily_loss             : loss over the most recent day (fraction, >=0)
        rolling_abs_daily_ret  : |daily return| over the last `lookback_days` days
    Used by B1 to feed CircuitBreaker.evaluate(...) instead of a hardcoded cb_level=0.

    rolling_dd_days: if set, drawdown is measured vs the peak over the trailing
    `rolling_dd_days` days (not the inception-to-date peak), so a single early >20%
    drop does not pin the snapshot at L3 forever. None = legacy inception peak.
    """
    r = pd.Series(list(returns)).dropna().astype(float)
    if len(r) == 0:
        return 0.0, 0.0, pd.Series(dtype=float)
    eq = (1.0 + r).cumprod()
    if rolling_dd_days:
        win = max(2, int(rolling_dd_days * bars_per_day))
        peak = float(eq.rolling(win, min_periods=1).max().iloc[-1])
    else:
        peak = float(eq.cummax().iloc[-1])
    dd = float(1.0 - eq.iloc[-1] / max(peak, 1e-12))
    grp = np.arange(len(r)) // int(max(bars_per_day, 1))
    daily = (1.0 + r).groupby(grp).prod() - 1.0          # per-day compounded return
    daily_loss = float(max(0.0, -daily.iloc[-1])) if len(daily) else 0.0
    rolling_abs = daily.abs().tail(int(lookback_days)).reset_index(drop=True)
    return dd, daily_loss, rolling_abs


def _dd_scaler(dd: float, rcfg) -> float:
    a, b, floor = rcfg.dd_scale_start, rcfg.dd_scale_stop, rcfg.dd_scale_floor
    if dd <= a:
        return 1.0
    if dd >= b:
        return float(floor)
    frac = (dd - a) / max(1e-9, (b - a))
    return float(1.0 - frac * (1.0 - floor))


def _rolling_corr_cov(close_panel: pd.DataFrame, symbols, window: int):
    """Right-aligned rolling corr & covariance (per-bar) over the last `window` bars.
    Returns (corr_df, cov_df, used_symbols) or (None, None, []) if insufficient data."""
    cols = [s for s in symbols if s in close_panel.columns]
    if len(cols) < 2:
        return None, None, []
    px = close_panel[cols].astype(float)
    ret = px.pct_change().replace([np.inf, -np.inf], np.nan)
    ret = ret.tail(int(window)).dropna(axis=1, how="any")
    if ret.shape[0] < 5 or ret.shape[1] < 2:
        return None, None, []
    return ret.corr(), ret.cov(), list(ret.columns)


def apply_portfolio_overlay(intents: Dict[str, dict],
                            close_panel: Optional[pd.DataFrame],
                            rcfg,
                            *,
                            equity_drawdown: float = 0.0,
                            bars_per_year: int = 2190,
                            clusters: Optional[Dict[str, str]] = None
                            ) -> Tuple[Dict[str, float], dict]:
    """
    intents : {symbol: {"target_position": signed_float, "direction": str, ...}}
              (the per-symbol, post-`risk_size_and_gate` positions)
    close_panel : wide close-price frame indexed by time, columns = symbols, sliced
                  by the caller so its LAST row is the decision bar (PIT).
    rcfg : a RiskConfig (fcfg.risk).

    Returns (adjusted_positions, report).  `adjusted_positions` maps every input
    symbol to its final signed target weight after the overlay.
    """
    w: Dict[str, float] = {s: float(v.get("target_position", 0.0) or 0.0) for s, v in intents.items()}
    report: dict = {"steps": []}
    gross_before = float(sum(abs(x) for x in w.values()))
    report["gross_before"] = gross_before

    active = [s for s, x in w.items() if abs(x) > 1e-12]
    if not active:
        report["note"] = "all_flat"
        report["gross_after"] = 0.0
        return {s: 0.0 for s in w}, report

    # ---- 1. correlation haircut ---------------------------------------- #
    corr, cov, used = (None, None, [])
    if close_panel is not None:
        corr, cov, used = _rolling_corr_cov(close_panel, active, int(rcfg.corr_window))
    if corr is not None:
        for s in used:
            others = [o for o in used if o != s]
            avg_abs = float(np.mean([abs(corr.loc[s, o]) for o in others])) if others else 0.0
            excess = max(0.0, avg_abs - rcfg.corr_floor)
            penalty = 1.0 / (1.0 + rcfg.corr_penalty * excess / max(1e-9, (1.0 - rcfg.corr_floor)))
            w[s] *= penalty
        off = corr.values[~np.eye(len(used), dtype=bool)]
        report["avg_pair_corr"] = float(np.nanmean(off)) if off.size else float("nan")
        report["steps"].append("corr_haircut")
    else:
        report["avg_pair_corr"] = float("nan")
        report["steps"].append("corr_haircut_skipped_insufficient_history")

    # ---- 2. cluster cap ------------------------------------------------- #
    members_by_cluster = defaultdict(list)
    for s in w:
        members_by_cluster[cluster_of(s, clusters)].append(s)
    cluster_scales = {}
    for c, members in members_by_cluster.items():
        g = sum(abs(w[m]) for m in members)
        if g > rcfg.max_cluster_weight and g > 1e-12:
            f = rcfg.max_cluster_weight / g
            for m in members:
                w[m] *= f
            cluster_scales[c] = round(f, 4)
    if cluster_scales:
        report["cluster_scaled"] = cluster_scales
        report["steps"].append("cluster_cap")
    report["cluster_gross"] = {c: round(sum(abs(w[m]) for m in ms), 4)
                               for c, ms in members_by_cluster.items()}

    # ---- 3. gross cap --------------------------------------------------- #
    gross = sum(abs(x) for x in w.values())
    if gross > rcfg.gross_cap and gross > 1e-12:
        f = rcfg.gross_cap / gross
        for s in w:
            w[s] *= f
        report["gross_cap_scale"] = round(f, 4)
        report["steps"].append("gross_cap")

    # ---- 4. portfolio vol target (delever only) ------------------------ #
    port_vol_ann = float("nan")
    if cov is not None and used:
        vec = np.array([w.get(s, 0.0) for s in used], dtype=float)
        var_bar = float(vec @ cov.loc[used, used].values @ vec)
        port_vol_ann = math.sqrt(max(var_bar, 0.0)) * math.sqrt(bars_per_year)
        if port_vol_ann > rcfg.target_portfolio_vol and port_vol_ann > 1e-12:
            f = min(rcfg.portfolio_vol_max_scale, rcfg.target_portfolio_vol / port_vol_ann)
            for s in w:
                w[s] *= f
            report["portfolio_vol_scale"] = round(f, 4)
            report["steps"].append("portfolio_vol_target")
    report["portfolio_vol_ann_pre"] = round(port_vol_ann, 4) if np.isfinite(port_vol_ann) else None

    # ---- 5. drawdown smooth scaler ------------------------------------- #
    dd_scaler = _dd_scaler(float(equity_drawdown), rcfg)
    if dd_scaler < 1.0:
        for s in w:
            w[s] *= dd_scaler
        report["steps"].append("drawdown_scaler")
    report["equity_drawdown"] = round(float(equity_drawdown), 4)
    report["dd_scaler"] = round(dd_scaler, 4)

    # ---- safety: per-symbol clip --------------------------------------- #
    for s in w:
        w[s] = float(np.clip(w[s], -rcfg.max_pos_per_symbol, rcfg.max_pos_per_symbol))

    report["gross_after"] = round(float(sum(abs(x) for x in w.values())), 4)
    report["n_active_after"] = int(sum(1 for x in w.values() if abs(x) > 1e-9))
    return w, report
