"""
crypto.benchmark.tsmom
=========================
Volatility-parity time-series-momentum portfolio benchmark (v6 §9.1.1) with
audit fixes:
  #9  risk_weight = inv_vol_i / sum(inv_vol); signed_w = dir * risk_weight;
      port_vol = sqrt(w' Sigma w) using rolling covariance (captures the high
      correlation of the four coins) -> no understated portfolio vol.
  #15 eps guard: all-zero direction or port_vol<=eps -> weights 0 (no div-by-0).
  #8  main = sign(); optional no-trade band for the robustness benchmark.
  scale clipped to max_vol_scale + gross cap (no implicit leverage).

All computations are right-aligned / PIT (uses data up to and including t).
Returns a target-weight wide frame to feed the EXISTING backtest engine.

The inner risk engine is factored out as `_vol_parity_from_direction` and is
ALSO reused to deploy an ML signal on equal footing with the benchmark via
`vol_parity_weights_from_signal` (same inverse-vol + covariance vol-target +
gross cap), instead of naive per-symbol position sizing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _vol_parity_from_direction(
    close: pd.DataFrame,
    direction: pd.DataFrame,         # values in [-1, 1] (sign for TSMOM, conviction for ML)
    vol_window: int = 30,
    cov_window: int = 30,
    target_portfolio_vol: float = 0.30,
    bars_per_year: int = 2190,
    max_vol_scale: float = 3.0,
    gross_cap: float = 1.0,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Inverse-vol risk weighting + covariance portfolio-vol targeting + gross cap.
    `direction` is a per-symbol signed exposure panel in [-1, 1] aligned (or
    reindexed) to `close`. This is the shared risk engine for both the TSMOM
    baseline and the ML deployment."""
    close = close.sort_index()
    rets = close.pct_change()
    direction = direction.reindex(index=close.index, columns=close.columns)

    ann = np.sqrt(bars_per_year)
    vol = rets.rolling(vol_window, min_periods=vol_window).std() * ann
    inv_vol = 1.0 / vol.replace(0.0, np.nan)

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    cols = close.columns
    times = close.index

    for i, t in enumerate(times):
        d = direction.loc[t]
        iv = inv_vol.loc[t]
        if d.isna().all() or iv.isna().all():
            continue
        d = d.fillna(0.0)
        iv = iv.fillna(0.0)
        if iv.sum() <= eps or (d == 0).all():
            continue
        risk_w = iv / iv.sum()
        signed_w = (d * risk_w).to_numpy(dtype=float)

        # portfolio vol with covariance (audit #9)
        if i < cov_window:
            port_vol = float(np.sqrt(np.sum((signed_w * vol.loc[t].fillna(0).to_numpy()) ** 2)))
        else:
            window_rets = rets.iloc[i - cov_window + 1: i + 1][cols].dropna(how="any")
            if len(window_rets) < 2:
                continue
            cov = window_rets.cov().to_numpy() * bars_per_year
            port_vol = float(np.sqrt(max(signed_w @ cov @ signed_w, 0.0)))

        if port_vol <= eps:   # audit #15
            continue
        scale = min(target_portfolio_vol / port_vol, max_vol_scale)
        w = signed_w * scale

        gross = np.abs(w).sum()
        if gross > gross_cap and gross > eps:
            w = w * (gross_cap / gross)
        weights.loc[t, cols] = w

    return weights


def vol_parity_tsmom_weights(
    close: pd.DataFrame,             # index=time, columns=symbol
    lookback_mom: int = 90,          # in *rows* (bars); caller sizes per frequency
    vol_window: int = 30,
    cov_window: int = 30,
    target_portfolio_vol: float = 0.30,
    bars_per_year: int = 2190,       # 4h crypto: 365*6
    max_vol_scale: float = 3.0,
    gross_cap: float = 1.0,
    no_trade_band: float = 0.0,      # >0 -> robustness benchmark
    eps: float = 1e-9,
) -> pd.DataFrame:
    close = close.sort_index()
    mom = close.pct_change(lookback_mom)
    if no_trade_band > 0:
        direction = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        direction[mom > no_trade_band] = 1.0
        direction[mom < -no_trade_band] = -1.0
    else:
        direction = np.sign(mom)
    return _vol_parity_from_direction(
        close, direction, vol_window=vol_window, cov_window=cov_window,
        target_portfolio_vol=target_portfolio_vol, bars_per_year=bars_per_year,
        max_vol_scale=max_vol_scale, gross_cap=gross_cap, eps=eps)


def vol_parity_weights_from_signal(
    close: pd.DataFrame,             # index=time, columns=symbol
    signal: pd.DataFrame,            # per-symbol signed CONVICTION panel, values in [-1, 1]
    vol_window: int = 30,
    cov_window: int = 30,
    target_portfolio_vol: float = 0.30,
    bars_per_year: int = 2190,
    max_vol_scale: float = 3.0,
    gross_cap: float = 1.0,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Deploy an arbitrary per-symbol directional CONVICTION panel (values in
    [-1, 1]) through the SAME inverse-vol + covariance vol-targeting + gross-cap
    risk engine as the TSMOM baseline. This puts an ML signal on an equal risk
    footing with the benchmark instead of naive per-symbol position sizing."""
    signal = signal.clip(-1.0, 1.0)
    return _vol_parity_from_direction(
        close, signal, vol_window=vol_window, cov_window=cov_window,
        target_portfolio_vol=target_portfolio_vol, bars_per_year=bars_per_year,
        max_vol_scale=max_vol_scale, gross_cap=gross_cap, eps=eps)
