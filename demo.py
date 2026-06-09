"""
crypto end-to-end demo on synthetic data.
Run: python demo.py

Exercises: triple-barrier labels -> uniqueness weights -> make_supervised_dataset
-> run_phase1b (OOF stage1 -> direction -> meta-label -> stage2 -> calibration)
-> build target weights -> EXISTING backtest engine vs vol-parity TSMOM benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto.schemas import FrozenConfig
from crypto.adapters import to_bars_schema, decision_time_grid
from crypto.labels.triple_barrier import compute_triple_barrier
from crypto.features.uniqueness import average_uniqueness
from crypto.pit import make_supervised_dataset, feature_matrix_columns
from crypto.pipeline_1b import run_phase1b
from crypto.benchmark.tsmom import vol_parity_tsmom_weights
from crypto.models.patchtst import run_patchtst

from backtest.engine import run_vector_backtest, BacktestConfig  # EXISTING engine


def synth_bars(symbol_seed: int, n_hours: int = 24 * 400) -> pd.DataFrame:
    rng = np.random.default_rng(symbol_seed)
    idx = pd.date_range("2022-01-01", periods=n_hours, freq="1h")
    # trend-switching geometric random walk
    drift = np.zeros(n_hours)
    regime_len = 24 * 20
    for s in range(0, n_hours, regime_len):
        drift[s:s + regime_len] = rng.normal(0, 0.0004)
    ret = drift + rng.normal(0, 0.01, n_hours)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n_hours)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n_hours)))
    op = np.concatenate([[close[0]], close[:-1]])
    vol = rng.lognormal(10, 0.5, n_hours)
    df = pd.DataFrame({"open": op, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
    return df


def build_features(bars_4h: pd.DataFrame, symbol: str, dts: pd.DatetimeIndex) -> pd.DataFrame:
    c = bars_4h["close"]
    feat = pd.DataFrame(index=bars_4h.index)
    feat["ret_1"] = c.pct_change()
    feat["ret_6"] = c.pct_change(6)
    feat["ret_24"] = c.pct_change(24)
    feat["vol_24"] = c.pct_change().rolling(24).std()
    feat["mom_z"] = (c.pct_change(12) - c.pct_change(12).rolling(48).mean()) / (
        c.pct_change(12).rolling(48).std() + 1e-9)
    feat = feat.reset_index().rename(columns={"index": "ts_open"})
    feat["decision_time"] = bars_4h["ts_close"].values + pd.Timedelta(minutes=1)
    feat["symbol"] = symbol
    feat["max_feature_availability_ts"] = feat["decision_time"]   # features built from bars <= decision
    feat = feat[feat["decision_time"].isin(dts)]
    return feat.dropna()


def main():
    fcfg = FrozenConfig()
    print(f"config_hash={fcfg.config_hash()}  env={__import__('crypto.schemas', fromlist=['environment_hash']).environment_hash()}")

    symbols = ["BTC/USDT", "ETH/USDT"]
    all_labels, all_feats, close_4h = [], [], {}
    for i, sym in enumerate(symbols):
        bars_1h = to_bars_schema(synth_bars(i + 1), "1h")
        # resample to 4h (left/closed-left, matches existing data_processor)
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        bars_4h = bars_1h.resample("4h", label="left", closed="left").agg(agg).dropna()
        bars_4h = to_bars_schema(bars_4h, "4h")
        dts = decision_time_grid(bars_4h, fcfg.decision_offset_minutes)

        lbl = compute_triple_barrier(
            bars_1h, dts, sym, fcfg.label, fcfg.cost,
            label_config_hash=fcfg.label_config_hash(), cost_model_hash=fcfg.cost_model_hash())
        all_labels.append(lbl)

        feat = build_features(bars_4h, sym, dts)
        # PatchTST OOF features (multi-horizon forecast + embedding), merged in
        pt = run_patchtst(bars_1h, dts, sym, fcfg, lookback=96, emb_dim=4)
        if not pt.empty:
            feat = feat.merge(pt, on=["symbol", "decision_time"], how="left")
        all_feats.append(feat)
        close_4h[sym] = bars_4h["close"].copy()
        close_4h[sym].index = dts[:len(bars_4h)] if len(dts) >= len(bars_4h) else close_4h[sym].index

    labels = pd.concat(all_labels, ignore_index=True)
    feats = pd.concat(all_feats, ignore_index=True)

    # uniqueness (pooled, asset-balanced)
    labels = labels.dropna(subset=["entry_time", "exit_time"]).reset_index(drop=True)
    labels["uniqueness_weight"] = average_uniqueness(
        labels["entry_time"], labels["exit_time"], scope="pooled", symbol=labels["symbol"]).values

    ds = make_supervised_dataset(feats, labels, require_pit=True)
    base_feats = ["ret_1", "ret_6", "ret_24", "vol_24", "mom_z"]
    patch_feats = [c for c in ds.columns if c.startswith("patchtst_")]
    feat_cols = [c for c in base_feats + patch_feats if c in ds.columns]
    # drop rows where patchtst OOF is NaN (warmup / fold edges)
    ds = ds.dropna(subset=feat_cols)
    print(f"dataset rows={len(ds)}  features={feat_cols}")

    signals, diag = run_phase1b(ds, feat_cols, fcfg)
    print("\n--- Phase-1b diagnostics ---")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # build target weights from signals: direction * sized by meta prob, then vol-target lite
    sig = signals.dropna(subset=["meta_trade_prob_calibrated"]).copy()
    sig["dir"] = sig["primary_direction"].map({"long": 1, "short": -1, "flat": 0}).fillna(0)
    sig["edge"] = (sig["meta_trade_prob_calibrated"] - fcfg.risk.p_threshold).clip(lower=0)
    sig["w"] = sig["dir"] * (sig["edge"] / fcfg.risk.edge_cap).clip(upper=1.0) * fcfg.risk.max_pos_per_symbol
    tw = sig.pivot_table(index="decision_time", columns="symbol", values="w", aggfunc="last").fillna(0)

    close_df = pd.DataFrame({s: close_4h[s] for s in symbols}).reindex(tw.index).ffill().dropna()
    tw = tw.reindex(close_df.index).fillna(0)

    # vol-parity TSMOM benchmark on same close panel (4h -> bars_per_year 2190)
    bench_w = vol_parity_tsmom_weights(close_df, lookback_mom=90, vol_window=30, cov_window=30,
                                       bars_per_year=2190)
    cfg = BacktestConfig(fee_rate=0.0004, slippage_rate=0.0003, execution_lag=1, annual_days=2190)
    strat = run_vector_backtest(close_df, tw, config=cfg, strategy_name="v6_meta")
    bench = run_vector_backtest(close_df, bench_w, config=cfg, strategy_name="tsmom_bench")

    def summ(res):
        r = res["returns"]
        ann = (1 + r).prod() ** (2190 / max(len(r), 1)) - 1
        sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(2190)
        return ann, sharpe, res["equity_curve"].iloc[-1] / 1e6 - 1

    sa, ss, st = summ(strat)
    ba, bs, bt = summ(bench)
    print("\n--- backtest (synthetic, NOT a real result) ---")
    print(f"  v6_meta   : annual={sa:+.2%}  sharpe={ss:+.2f}  total={st:+.2%}")
    print(f"  tsmom_bench: annual={ba:+.2%}  sharpe={bs:+.2f}  total={bt:+.2%}")
    print("\nOK: full Phase-1b pipeline + existing backtest engine ran end-to-end.")


if __name__ == "__main__":
    main()
