"""
Run the v6 Phase-1b pipeline on REAL data and validate PIT + label distribution.

Run in YOUR environment (needs pyarrow + the project's data):
    python run_on_real_data.py --symbols BTC/USDT ETH/USDT SOL/USDT BNB/USDT

It reads PROCESSED/{SYMBOL}_1h.parquet (via the existing DataLoader if available,
else directly with pandas), builds 4h decision grid, triple-barrier labels,
PatchTST features (uses torch if installed, else sklearn fallback), funding
features (if {SYMBOL}_funding.parquet exists), uniqueness, the supervised
dataset, and Phase-1b.  Prints:
  * PIT audit report
  * tb_label / primary_direction distributions
  * ECE (calibration) and fold count
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from crypto.schemas import FrozenConfig
from crypto.adapters import to_bars_schema, decision_time_grid
from crypto.labels.triple_barrier import compute_triple_barrier
from crypto.features.uniqueness import average_uniqueness
from crypto.features.derivatives import load_funding, funding_features
from crypto.features.onchain import onchain_factors
from crypto.models.patchtst import run_patchtst
from crypto.pit import make_supervised_dataset, audit_lookahead
from crypto.pipeline_1b import run_phase1b


def load_1h(symbol: str) -> pd.DataFrame | None:
    sym = symbol.replace("/", "")
    # prefer the project's DataLoader; fall back to direct parquet read
    try:
        from etl.data_loader import DataLoader
        df = DataLoader().get_crypto_kline_data(symbol=symbol, timeframe="1h")
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"(DataLoader unavailable: {e}; trying direct read)")
    p = Path(config.PathConfig.PROCESSED) / f"{sym}_1h.parquet"
    if not p.exists():
        print(f"⚠️ not found: {p}")
        return None
    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"])).drop(columns=["timestamp"])
    return df


def build_features(bars_4h, symbol, dts):
    c = bars_4h["close"]
    f = pd.DataFrame(index=bars_4h.index)
    f["ret_1"] = c.pct_change(); f["ret_6"] = c.pct_change(6); f["ret_24"] = c.pct_change(24)
    f["vol_24"] = c.pct_change().rolling(24).std()
    m = c.pct_change(12)
    f["mom_z"] = (m - m.rolling(48).mean()) / (m.rolling(48).std() + 1e-9)
    f = f.reset_index().rename(columns={"index": "ts_open"})
    f["decision_time"] = bars_4h["ts_close"].values + pd.Timedelta(minutes=1)
    f["symbol"] = symbol
    f["max_feature_availability_ts"] = f["decision_time"]
    return f[f["decision_time"].isin(dts)].dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
    args = ap.parse_args()
    fcfg = FrozenConfig()

    all_labels, all_feats, close_4h = [], [], {}
    for sym in args.symbols:
        raw = load_1h(sym)
        if raw is None:
            continue
        bars_1h = to_bars_schema(raw, "1h")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        bars_4h = to_bars_schema(
            bars_1h.resample("4h", label="left", closed="left").agg(agg).dropna(), "4h")
        dts = decision_time_grid(bars_4h, fcfg.decision_offset_minutes)

        funding = load_funding(config.PathConfig.PROCESSED, sym) or \
            load_funding(config.PathConfig.RAW, sym)
        lbl = compute_triple_barrier(bars_1h, dts, sym, fcfg.label, fcfg.cost,
                                     funding=funding,
                                     label_config_hash=fcfg.label_config_hash(),
                                     cost_model_hash=fcfg.cost_model_hash())
        feat = build_features(bars_4h, sym, dts)
        pt = run_patchtst(bars_1h, dts, sym, fcfg, lookback=96, emb_dim=8)
        if not pt.empty:
            feat = feat.merge(pt, on=["symbol", "decision_time"], how="left")
        fund_feat = funding_features(funding, pd.DatetimeIndex(feat["decision_time"]))
        for ccol in fund_feat.columns:
            feat[ccol] = fund_feat[ccol].values
        # onchain degrades gracefully if no data
        oc = onchain_factors(None, pd.DatetimeIndex(feat["decision_time"]))

        all_labels.append(lbl)
        all_feats.append(feat)
        close_4h[sym] = bars_4h["close"]
        print(f"  {sym}: bars_1h={len(bars_1h)}  4h decisions={len(dts)}  labels={len(lbl)}")

    labels = pd.concat(all_labels, ignore_index=True).dropna(subset=["entry_time", "exit_time"])
    feats = pd.concat(all_feats, ignore_index=True)
    labels["uniqueness_weight"] = average_uniqueness(
        labels["entry_time"], labels["exit_time"], scope="pooled", symbol=labels["symbol"]).values

    print("\n--- PIT audit ---")
    print(audit_lookahead(feats))

    ds = make_supervised_dataset(feats, labels, require_pit=True)
    base = ["ret_1", "ret_6", "ret_24", "vol_24", "mom_z",
            "funding_rate", "funding_rate_z", "funding_rate_chg"]
    patch = [c for c in ds.columns if c.startswith("patchtst_")]
    feat_cols = [c for c in base + patch if c in ds.columns]
    ds = ds.dropna(subset=[c for c in feat_cols if c in ds.columns])
    print(f"\ndataset rows={len(ds)}  n_features={len(feat_cols)}")

    signals, diag = run_phase1b(ds, feat_cols, fcfg)
    print("\n--- Phase-1b diagnostics (REAL data) ---")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    print("\nNOTE: inspect tb_label_distribution — heavy skew means tp/sl asymmetry "
          "or trending regime; calibrate tp_mult/sl_mult & thresholds before any conclusion.")


if __name__ == "__main__":
    main()
