"""
etl.feature_builder
===================
Multi-timeframe MARKET feature / dataset builder (fills the previously-empty
placeholder module the project author reserved here).

It assembles ONE point-in-time supervised dataset from the already-processed
parquet, honouring the intended multi-timeframe contract:

    main sequence   : 4h OHLCV          (decision frequency, v6 §1.2)
    auxiliary state : derived from 1h   (short-period features, PIT as-of)
    environment     : derived from 1d   (mid/long-term regime, PIT as-of)

This module does NOT re-implement any methodology. It REUSES the project's own
functions (`to_bars_schema`, `decision_time_grid`, `compute_triple_barrier`,
`run_patchtst`, `load_funding`, `funding_features`, `average_uniqueness`,
`make_supervised_dataset`, `audit_lookahead`) and only wires them together with
the multi-timeframe market features. On-chain / narrative modalities are left as
empty hooks (a colleague is producing that data; the pipeline degrades
gracefully and the experiment ladder auto-skips those steps).

PIT discipline (v6 §4):
  * 4h core features are right-aligned on the 4h close (available by decision_time).
  * 1h / 1d features are merged AS-OF using availability_ts = bar_close + lag, so
    a decision at `dt` only sees bars whose data was truly available at/before `dt`.
  * all z-scores / percentiles are rolling (right-aligned), never full-sample (§4.3).
  * funding_rate is OPTIONAL — if `{SYMBOL}_funding.parquet` is absent the funding
    columns are NaN and get dropped downstream (this is the colleague's hook).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import config
from crypto.adapters import to_bars_schema, decision_time_grid
from crypto.labels.triple_barrier import compute_triple_barrier
from crypto.features.uniqueness import average_uniqueness
from crypto.features.derivatives import load_funding, funding_features
from crypto.models.patchtst import run_patchtst
from crypto.pit import make_supervised_dataset, audit_lookahead

# 4h "main" core features — kept identical to the original demos so prior results
# remain comparable after the migration to real data.
CORE_4H_COLS = ["ret_1", "ret_6", "ret_24", "vol_24", "mom_z"]
AUX_1H_COLS = ["h1_ret_24", "h1_vol_24", "h1_rsi_14"]
ENV_1D_COLS = ["d1_trend_20", "d1_ret_5", "d1_vol_20"]
# order-flow factors derived from taker_buy_vol / net_taker_vol (data you already
# have but were only fed to PatchTST channels — now PIT tabular market features).
FLOW_COLS = ["of_buy_frac", "of_net_z", "of_buy_frac_chg"]
FUNDING_COLS = ["funding_rate", "funding_rate_z", "funding_rate_chg"]

_BAR_TD = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


# --------------------------------------------------------------------------- #
# small, right-aligned indicators (rolling only — no full-sample stats)
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    roll_up = up.rolling(window, min_periods=window).mean()
    roll_dn = dn.rolling(window, min_periods=window).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _asof_align(frame: pd.DataFrame, timeframe: str,
                decision_times: pd.DatetimeIndex, kline_lag_seconds: int = 30) -> pd.DataFrame:
    """As-of align a frame indexed by ts_open onto decision_times using
    availability_ts = ts_open + bar_length + kline_lag (PIT). For each
    decision_time take the most recent row whose availability_ts <= decision_time.
    (Same mechanism funding_features / onchain_factors already use.)"""
    avail = frame.copy()
    avail.index = avail.index + _BAR_TD[timeframe] + pd.Timedelta(seconds=kline_lag_seconds)
    union = avail.index.union(decision_times)
    out = avail.reindex(union).sort_index().ffill().reindex(decision_times)
    out.index = decision_times
    return out


# --------------------------------------------------------------------------- #
# per-symbol feature frame (4h core + 1h aux + 1d env + funding hook)
# --------------------------------------------------------------------------- #
def build_symbol_features(bars_4h: pd.DataFrame, bars_1h: pd.DataFrame,
                          bars_1d: Optional[pd.DataFrame], symbol: str,
                          decision_times: pd.DatetimeIndex,
                          funding: Optional[pd.Series] = None) -> pd.DataFrame:
    # ---- 4h core (right-aligned on 4h close) ----
    c4 = bars_4h["close"]
    f = pd.DataFrame(index=bars_4h.index)
    f["ret_1"] = c4.pct_change()
    f["ret_6"] = c4.pct_change(6)
    f["ret_24"] = c4.pct_change(24)
    f["vol_24"] = c4.pct_change().rolling(24).std()
    m = c4.pct_change(12)
    f["mom_z"] = (m - m.rolling(48).mean()) / (m.rolling(48).std() + 1e-9)
    f = f.reset_index().rename(columns={"index": "ts_open"})
    # decision_times is already (4h close + offset), one per 4h bar, same order.
    f["decision_time"] = np.asarray(decision_times)[: len(f)]
    f["symbol"] = symbol

    # ---- 1h auxiliary (short-period state, PIT as-of) ----
    c1 = bars_1h["close"]
    aux = pd.DataFrame(index=bars_1h.index)
    aux["h1_ret_24"] = c1.pct_change(24)
    aux["h1_vol_24"] = c1.pct_change().rolling(24).std()
    aux["h1_rsi_14"] = _rsi(c1, 14)
    aux_aligned = _asof_align(aux, "1h", pd.DatetimeIndex(f["decision_time"]))
    for col in AUX_1H_COLS:
        f[col] = aux_aligned[col].values

    # ---- 1d environment (mid/long-term regime, PIT as-of) ----
    if bars_1d is not None and len(bars_1d) > 0:
        cd = bars_1d["close"]
        env = pd.DataFrame(index=bars_1d.index)
        env["d1_trend_20"] = cd / cd.rolling(20, min_periods=20).mean() - 1.0
        env["d1_ret_5"] = cd.pct_change(5)
        env["d1_vol_20"] = cd.pct_change().rolling(20, min_periods=20).std() * np.sqrt(365)
        env_aligned = _asof_align(env, "1d", pd.DatetimeIndex(f["decision_time"]))
        for col in ENV_1D_COLS:
            f[col] = env_aligned[col].values
    else:
        for col in ENV_1D_COLS:
            f[col] = np.nan

    # ---- order-flow factors from 4h taker volumes (right-aligned, rolling only) ----
    if {"taker_buy_vol", "net_taker_vol", "volume"}.issubset(bars_4h.columns):
        vol = bars_4h["volume"]
        buy_frac = (bars_4h["taker_buy_vol"] / (vol + 1e-9)).clip(0.0, 1.0)   # aggressive-buy share
        net = bars_4h["net_taker_vol"]
        net_z = (net - net.rolling(48).mean()) / (net.rolling(48).std() + 1e-9)  # normalized net flow
        buy_frac_chg = buy_frac - buy_frac.rolling(24).mean()                 # deviation from recent norm
        flow = {"of_buy_frac": buy_frac.values, "of_net_z": net_z.values,
                "of_buy_frac_chg": buy_frac_chg.values}
        for col in FLOW_COLS:
            f[col] = flow[col][: len(f)]
    else:
        for col in FLOW_COLS:
            f[col] = np.nan

    # ---- funding hook (optional; NaN -> dropped downstream) ----
    ff = funding_features(funding, pd.DatetimeIndex(f["decision_time"]))
    for col in FUNDING_COLS:
        f[col] = ff[col].values if col in ff.columns else np.nan

    f["max_feature_availability_ts"] = f["decision_time"]  # conservative (all <= decision)
    return f


# --------------------------------------------------------------------------- #
# loaders (real parquet) — prefer DataLoader, fall back to direct read
# --------------------------------------------------------------------------- #
def _read_parquet_bars(processed_dir, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    sym = symbol.replace("/", "")
    p = Path(processed_dir) / f"{sym}_{timeframe}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"])).sort_index()
        df = df.loc[:, [c for c in df.columns if c != "timestamp"]]
    return df


def load_symbol_bars(symbol: str, timeframe: str, processed_dir, loader=None) -> Optional[pd.DataFrame]:
    """v6-schema bar frame (ts_close + availability_ts attached). Tries the
    project's DataLoader (duckdb) first, then a direct pandas read."""
    raw = None
    if loader is not None:
        try:
            raw = loader.get_crypto_kline_data(symbol=symbol, timeframe=timeframe)
        except Exception:
            raw = None
    if raw is None or getattr(raw, "empty", True):
        raw = _read_parquet_bars(processed_dir, symbol, timeframe)
    if raw is None or raw.empty:
        return None
    return to_bars_schema(raw, timeframe)


# --------------------------------------------------------------------------- #
# result container + top-level builder
# --------------------------------------------------------------------------- #
@dataclass
class MarketDataset:
    dataset: pd.DataFrame
    feature_cols: List[str]
    modality_cols: Dict[str, List[str]]
    tabular_cols: List[str]        # market modality only (the "A" config in A/B/C/D)
    close_panel: pd.DataFrame      # 4h close, index = decision_time
    audit: dict
    per_symbol: Dict[str, dict] = field(default_factory=dict)


def build_market_dataset(symbols: List[str], fcfg, processed_dir=None, loader=None,
                         patchtst_lookback: int = 96, patchtst_emb_dim: int = 8,
                         bars_provider: Optional[Callable[[str, str], Optional[pd.DataFrame]]] = None
                         ) -> MarketDataset:
    """
    Build the full multi-symbol supervised dataset from real parquet.

    bars_provider(symbol, timeframe) -> v6-schema bars OR None.
      * default reads the processed parquet (real data path).
      * a synthetic provider can be injected for a quick offline wiring self-test
        (no pyarrow / heavy libs needed) — see run_on_real_data.py --synthetic.
    """
    processed_dir = processed_dir or config.PathConfig.PROCESSED
    if bars_provider is None:
        def bars_provider(sym, tf):
            return load_symbol_bars(sym, tf, processed_dir, loader=loader)

    all_labels, all_feats, close_map, per_symbol = [], [], {}, {}
    for sym in symbols:
        bars_1h = bars_provider(sym, "1h")
        bars_4h = bars_provider(sym, "4h")
        bars_1d = bars_provider(sym, "1d")
        if bars_1h is None or bars_4h is None:
            per_symbol[sym] = {"status": "missing 1h or 4h parquet — skipped"}
            continue

        dts = decision_time_grid(bars_4h, fcfg.decision_offset_minutes)
        funding = load_funding(processed_dir, sym) if processed_dir else None

        labels = compute_triple_barrier(
            bars_1h, dts, sym, fcfg.label, fcfg.cost, funding=funding,
            label_config_hash=fcfg.label_config_hash(), cost_model_hash=fcfg.cost_model_hash())
        feats = build_symbol_features(bars_4h, bars_1h, bars_1d, sym, dts, funding=funding)

        pt = run_patchtst(bars_1h, dts, sym, fcfg, lookback=patchtst_lookback, emb_dim=patchtst_emb_dim)
        if not pt.empty:
            feats = feats.merge(pt, on=["symbol", "decision_time"], how="left")

        all_labels.append(labels)
        all_feats.append(feats)
        close_map[sym] = pd.Series(bars_4h["close"].values, index=dts)  # index=decision_time
        per_symbol[sym] = {"status": "ok", "bars_1h": len(bars_1h), "bars_4h": len(bars_4h),
                           "bars_1d": (0 if bars_1d is None else len(bars_1d)),
                           "decisions": len(dts), "labels": len(labels),
                           "funding": "present" if funding is not None else "absent (hook)"}

    if not all_feats:
        raise RuntimeError("no symbol produced data — check parquet paths / filenames")

    labels = pd.concat(all_labels, ignore_index=True).dropna(subset=["entry_time", "exit_time"])
    feats = pd.concat(all_feats, ignore_index=True)
    labels["uniqueness_weight"] = average_uniqueness(
        labels["entry_time"], labels["exit_time"], scope="pooled", symbol=labels["symbol"]).values

    audit = audit_lookahead(feats)
    ds = make_supervised_dataset(feats, labels, require_pit=True)

    patch_cols = [c for c in ds.columns if c.startswith("patchtst_")]
    flow_present = [c for c in FLOW_COLS if c in ds.columns and ds[c].notna().any()]
    funding_present = [c for c in FUNDING_COLS if c in ds.columns and ds[c].notna().any()]
    market_cols = [c for c in (CORE_4H_COLS + AUX_1H_COLS + ENV_1D_COLS + flow_present + funding_present)
                   if c in ds.columns]
    feature_cols = [c for c in (market_cols + patch_cols) if c in ds.columns]

    ds = ds.dropna(subset=feature_cols).reset_index(drop=True)

    modality_cols = {"market": market_cols, "onchain": [], "narrative": [], "patchtst": patch_cols}
    close_panel = pd.DataFrame({s: close_map[s] for s in close_map})

    return MarketDataset(dataset=ds, feature_cols=feature_cols, modality_cols=modality_cols,
                         tabular_cols=market_cols, close_panel=close_panel,
                         audit=audit, per_symbol=per_symbol)
