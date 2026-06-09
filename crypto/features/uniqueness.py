"""
crypto.features.uniqueness
=============================
Average-uniqueness sample weights (AFML ch.7-8) per v6 §6.3.1.

  * 1h granularity, half-open interval [t0, t1)  (audit #5: off-by-one).
  * event-stream + prefix sum -> O(N+T).
  * within_symbol (per-asset models) or pooled (cross-asset) with asset-balance
    weighting to stop a label-dense asset (BTC) dominating (audit #11 / simp #6).
  * computed only within a single label_config_hash (caller guarantees).
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


def _avg_uniqueness_one_pool(t0: pd.Series, t1: pd.Series, freq: str = "1h") -> np.ndarray:
    """Core: average uniqueness over a single pool, half-open [t0, t1)."""
    t0 = pd.to_datetime(pd.Series(t0).reset_index(drop=True))
    t1 = pd.to_datetime(pd.Series(t1).reset_index(drop=True))
    n = len(t0)
    if n == 0:
        return np.array([])

    # build a unified grid covering all intervals
    grid = pd.date_range(t0.min(), t1.max(), freq=freq)
    if len(grid) == 0:
        return np.ones(n)
    # event stream on grid: +1 at t0, -1 at t1  (half-open -> t1 not counted)
    events = np.zeros(len(grid) + 1, dtype=np.int64)
    gi0 = grid.searchsorted(t0.values, side="left")
    gi1 = grid.searchsorted(t1.values, side="left")  # half-open: drop t1 bar
    for a, b in zip(gi0, gi1):
        events[a] += 1
        events[b] -= 1
    concurrency = np.cumsum(events)[:len(grid)]
    concurrency = np.where(concurrency <= 0, 1, concurrency)  # guard

    inv = 1.0 / concurrency
    out = np.empty(n)
    for i in range(n):
        a, b = gi0[i], gi1[i]
        if b <= a:               # zero/negative span -> fully unique
            out[i] = 1.0
        else:
            out[i] = inv[a:b].mean()
    return out


def average_uniqueness(
    t0: pd.Series,
    t1: pd.Series,
    grid: str = "1h",
    scope: Literal["within_symbol", "pooled"] = "within_symbol",
    symbol: Optional[pd.Series] = None,
    normalize: bool = True,
) -> pd.Series:
    """
    Returns a weight Series aligned to the input index.

    within_symbol : uniqueness computed per symbol (independent models).
    pooled        : uniqueness over the whole pool, then asset-balance weight
                    w_i_raw = u_i * N_total / (N_symbols * N_symbol_i)  (simp #6).
    """
    idx = pd.Series(t0).index
    t0 = pd.Series(t0).reset_index(drop=True)
    t1 = pd.Series(t1).reset_index(drop=True)
    n = len(t0)
    if n == 0:
        return pd.Series([], dtype=float)

    w = np.ones(n)
    if scope == "within_symbol":
        if symbol is None:
            w = _avg_uniqueness_one_pool(t0, t1, grid)
        else:
            sym = pd.Series(symbol).reset_index(drop=True)
            for s in sym.unique():
                m = (sym == s).to_numpy()
                w[m] = _avg_uniqueness_one_pool(t0[m], t1[m], grid)
    else:  # pooled
        u = _avg_uniqueness_one_pool(t0, t1, grid)
        if symbol is None:
            w = u
        else:
            sym = pd.Series(symbol).reset_index(drop=True)
            counts = sym.value_counts()
            n_symbols = len(counts)
            n_total = n
            balance = sym.map(lambda s: n_total / (n_symbols * counts[s])).to_numpy()
            w = u * balance

    if normalize and w.sum() > 0:
        w = w * (n / w.sum())  # sum to N
    return pd.Series(w, index=idx)
