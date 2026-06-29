"""
etl.cross_sectional_features
============================
PIT-safe CROSS-SECTIONAL (relative) features for a market-neutral (xs_neutral)
book. The existing market features are almost all *own-asset absolute* quantities
(this coin's own return / vol). A relative book ranks coins against each other, so
it needs *relative* inputs: how a coin is doing versus the cross-section and versus
BTC at the SAME timestamp.

These are computed purely from columns already in the supervised dataset (zero new
data collection). They are PIT-safe: every feature at decision_time t uses only the
contemporaneous cross-section at t (all symbols' already-backward-looking features
at t) — no future information and no other timestamp.

Design keeps the set SMALL and non-redundant (avoids repeating the existing
10-collinear-returns problem): demeaned + ranked relative strength at two horizons,
return-vs-BTC, and relative volatility.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

# (new_col, base_col, kind) — kind in {"demean", "rank", "vs_btc"}
# base_col must exist in the dataset; specs whose base is missing are skipped.
XS_SPECS = [
    ("xs_demean_ret_24h", "ret_24h", "demean"),
    ("xs_rank_ret_24h",   "ret_24h", "rank"),
    ("xs_demean_ret_7d",  "ret_7d",  "demean"),
    ("ret_vs_btc_24h",    "ret_24h", "vs_btc"),
    ("xs_demean_vol_30d", "vol_30d", "demean"),
]

BTC_SYMBOLS = ("BTC/USDT", "BTCUSDT", "BTC")


def _btc_key(symbols) -> str:
    s = set(symbols)
    for b in BTC_SYMBOLS:
        if b in s:
            return b
    return ""


def add_cross_sectional_features(ds: pd.DataFrame,
                                 time_col: str = "decision_time",
                                 symbol_col: str = "symbol"
                                 ) -> Tuple[pd.DataFrame, List[str]]:
    """Append cross-sectional relative features to the supervised dataset `ds`
    (rows = (symbol, decision_time)). Returns (ds_with_new_cols, new_col_names).

    - demean : x_i(t) - mean_j x_j(t)        (relative strength vs the cross-section)
    - rank   : centered percentile rank in [-0.5, 0.5] at t (outlier-robust)
    - vs_btc : x_i(t) - x_BTC(t)             (relative to the market leader)
    """
    ds = ds.copy()
    out_cols: List[str] = []
    btc = _btc_key(ds[symbol_col].unique())

    for new_col, base, kind in XS_SPECS:
        if base not in ds.columns:
            continue
        grp = ds.groupby(time_col)[base]
        if kind == "demean":
            ds[new_col] = ds[base] - grp.transform("mean")
        elif kind == "rank":
            # pct rank in (0,1] -> center to [-0.5, 0.5]; NaN-safe per timestamp
            ds[new_col] = grp.transform(lambda s: s.rank(pct=True)) - 0.5
        elif kind == "vs_btc":
            if not btc:
                continue
            btc_at_t = (ds.loc[ds[symbol_col] == btc, [time_col, base]]
                        .rename(columns={base: "_btc_base"}))
            ds = ds.merge(btc_at_t, on=time_col, how="left")
            ds[new_col] = ds[base] - ds["_btc_base"]
            ds.drop(columns=["_btc_base"], inplace=True)
        else:
            continue
        # contemporaneous cross-section can still leave NaN (e.g. only 1 symbol at t)
        ds[new_col] = ds[new_col].astype(float).fillna(0.0)
        out_cols.append(new_col)

    return ds, out_cols
