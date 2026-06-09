"""
run_agent.py — command-line ENTRY POINT for the Agent + Skills system.

This is how the agents are invoked programmatically:
    python run_agent.py --symbol BTC/USDT            # run latest decision
    python run_agent.py --symbol BTC/USDT --n 5      # last 5 decisions
    python run_agent.py --cb 3                        # simulate circuit-breaker L3 (Risk veto)

It builds (or, with --real, loads) features+labels, fits a ModelBundle once,
constructs the 7-agent TradingGraph, and runs decisions, printing the structured
decision JSON (v6 §1.4) and the audited skill-call log.

The SAME entry (build_graph + graph.run_decision) backs the web UI (app/server.py).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from crypto.schemas import FrozenConfig
from crypto.adapters import to_bars_schema, decision_time_grid
from crypto.labels.triple_barrier import compute_triple_barrier
from crypto.features.uniqueness import average_uniqueness
from crypto.models.patchtst import run_patchtst
from crypto.pit import make_supervised_dataset
from crypto.models.bundle import ModelBundle
from crypto.live.oms import PaperBroker
from crypto.orchestration.graph import TradingGraph, decision_to_json


def _synth_bars(seed, n=24 * 200):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h")
    ret = rng.normal(0, 0.01, n)
    for s in range(0, n, 24 * 15):
        ret[s:s + 24 * 15] += rng.normal(0, 0.0006)
    close = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": close * 1.003,
                         "low": close * 0.997, "close": close,
                         "volume": rng.lognormal(10, .4, n),
                         "taker_buy_vol": rng.lognormal(9, .4, n),
                         "net_taker_vol": rng.normal(0, 1, n)}, index=idx)


def build_graph(symbols=("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"),
                fcfg=None, quality_threshold=0.6):
    """Construct a ready-to-use TradingGraph + close map + decision index.
    Returns (graph, close_map, feats, fcols, fcfg). Used by CLI and web UI."""
    fcfg = fcfg or FrozenConfig()
    labels, feats, close_map = [], [], {}
    for i, s in enumerate(symbols):
        b1 = to_bars_schema(_synth_bars(i + 1), "1h")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        b4 = to_bars_schema(b1.resample("4h", label="left", closed="left").agg(agg).dropna(), "4h")
        dts = decision_time_grid(b4, fcfg.decision_offset_minutes)
        labels.append(compute_triple_barrier(b1, dts, s, fcfg.label, fcfg.cost,
                      label_config_hash=fcfg.label_config_hash(),
                      cost_model_hash=fcfg.cost_model_hash()))
        f = b4["close"].pct_change()
        feat = pd.DataFrame({"ret_1": f, "ret_6": b4["close"].pct_change(6),
                             "vol_24": f.rolling(24).std()})
        m = b4["close"].pct_change(12)
        feat["mom_z"] = (m - m.rolling(48).mean()) / (m.rolling(48).std() + 1e-9)
        feat = feat.reset_index().rename(columns={"index": "ts_open"})
        feat["decision_time"] = b4["ts_close"].values + pd.Timedelta(minutes=1)
        feat["symbol"] = s
        feat["max_feature_availability_ts"] = feat["decision_time"]
        feat = feat[feat["decision_time"].isin(dts)].dropna()
        pt = run_patchtst(b1, dts, s, fcfg, lookback=96, emb_dim=4)
        if not pt.empty:
            feat = feat.merge(pt, on=["symbol", "decision_time"], how="left")
        feats.append(feat)
        cc = b4["close"].copy(); cc.index = dts[:len(b4)]
        close_map[s] = cc

    labels = pd.concat(labels, ignore_index=True).dropna(subset=["entry_time", "exit_time"])
    feats = pd.concat(feats, ignore_index=True)
    labels["uniqueness_weight"] = average_uniqueness(
        labels["entry_time"], labels["exit_time"], scope="pooled", symbol=labels["symbol"]).values
    ds = make_supervised_dataset(feats, labels, require_pit=True)
    fcols = [c for c in ds.columns if c in ["ret_1", "ret_6", "vol_24", "mom_z"]
             or c.startswith("patchtst_")]
    ds = ds.dropna(subset=fcols)
    bundle = ModelBundle(fcfg).fit(ds, fcols)
    graph = TradingGraph(fcfg, bundle, feats, fcols, quality_threshold=quality_threshold)
    return graph, close_map, feats, fcols, fcfg


def run_one(graph, close_map, feats, fcols, fcfg, symbol, decision_time=None,
            cb_level=0, broker=None):
    sub = feats[feats["symbol"] == symbol].dropna(subset=fcols)
    if sub.empty:
        return None
    dt = decision_time or sub["decision_time"].iloc[-1]
    ref = float(close_map[symbol].asof(dt))
    # broker=None -> fresh throwaway broker (read-only run). Pass a persistent
    # broker to accumulate the running position across decisions.
    b = broker if broker is not None else PaperBroker(max_slippage_bps=3)
    st = graph.run_decision(symbol, dt, b, ref_price=ref, cb_level=cb_level)
    return {"decision": decision_to_json(st, fcfg),
            "current_position": float(b.get_position(symbol)),
            "audit_log": [{"skill": r.skill, "category": r.category, "ok": r.ok,
                           "ms": round(r.duration_ms, 2)} for r in st["audit_log"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--cb", type=int, default=0, help="circuit breaker level (3 -> Risk veto)")
    args = ap.parse_args()

    print("building agent graph (synthetic data)...")
    graph, close_map, feats, fcols, fcfg = build_graph()
    print(f"orchestration backend = {graph.backend}\n")

    sub = feats[feats["symbol"] == args.symbol].dropna(subset=fcols).tail(args.n)
    for _, row in sub.iterrows():
        out = run_one(graph, close_map, feats, fcols, fcfg, args.symbol,
                      decision_time=row["decision_time"], cb_level=args.cb)
        print(json.dumps(out["decision"], ensure_ascii=False))
        print("  audit:", [a["skill"] for a in out["audit_log"]], "\n")


if __name__ == "__main__":
    main()
