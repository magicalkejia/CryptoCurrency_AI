"""
Agent + Skills orchestration demo (v6 §8/§9/§10) on synthetic data.
Run: python agent_demo.py

Builds features+labels, fits a ModelBundle, then runs the 7-agent state machine
(Data->quality_gate->Signal->Narrative->Fusion->Risk->Execution->Review) for a
few decisions, printing the structured decision JSON (v6 §1.4) and the audited
skill-call log. Also demonstrates the Risk agent's veto under a circuit breaker.
"""
from __future__ import annotations

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


def synth_bars(seed, n=24 * 300):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h")
    ret = rng.normal(0, 0.01, n)
    for s in range(0, n, 24 * 15):
        ret[s:s + 24 * 15] += rng.normal(0, 0.0005)
    close = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]],
                         "high": close * 1.003, "low": close * 0.997,
                         "close": close, "volume": rng.lognormal(10, .4, n),
                         "taker_buy_vol": rng.lognormal(9, .4, n),
                         "net_taker_vol": rng.normal(0, 1, n)}, index=idx)


def build_feats(bars_4h, sym, dts):
    c = bars_4h["close"]
    f = pd.DataFrame(index=bars_4h.index)
    f["ret_1"] = c.pct_change(); f["ret_6"] = c.pct_change(6); f["vol_24"] = c.pct_change().rolling(24).std()
    m = c.pct_change(12); f["mom_z"] = (m - m.rolling(48).mean()) / (m.rolling(48).std() + 1e-9)
    f = f.reset_index().rename(columns={"index": "ts_open"})
    f["decision_time"] = bars_4h["ts_close"].values + pd.Timedelta(minutes=1)
    f["symbol"] = sym; f["max_feature_availability_ts"] = f["decision_time"]
    return f[f["decision_time"].isin(dts)].dropna()


def main():
    fcfg = FrozenConfig()
    symbols = ["BTC/USDT", "ETH/USDT"]
    labels, feats, close_map = [], [], {}
    for i, s in enumerate(symbols):
        b1 = to_bars_schema(synth_bars(i + 1), "1h")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        b4 = to_bars_schema(b1.resample("4h", label="left", closed="left").agg(agg).dropna(), "4h")
        dts = decision_time_grid(b4, fcfg.decision_offset_minutes)
        labels.append(compute_triple_barrier(b1, dts, s, fcfg.label, fcfg.cost,
                      label_config_hash=fcfg.label_config_hash(), cost_model_hash=fcfg.cost_model_hash()))
        f = build_feats(b4, s, dts)
        pt = run_patchtst(b1, dts, s, fcfg, lookback=96, emb_dim=4)
        if not pt.empty:
            f = f.merge(pt, on=["symbol", "decision_time"], how="left")
        feats.append(f)
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
    graph = TradingGraph(fcfg, bundle, feats, fcols, quality_threshold=0.6)
    print(f"orchestration backend = {graph.backend}")
    print(f"agents = Data->Signal->Narrative->Fusion->Risk->Execution->Review")
    print(f"registered skills by category:")
    from crypto.skills.registry import REGISTRY
    for cat in ["data", "narrative", "fusion", "risk", "execution", "review"]:
        print(f"  {cat}: {REGISTRY.list(cat)}")

    broker = PaperBroker(max_slippage_bps=3)
    # run the state machine for the last few decisions of BTC
    btc = feats[feats["symbol"] == "BTC/USDT"].dropna(subset=fcols)
    sample = btc.tail(3)
    print("\n================ AGENT DECISIONS ================")
    for _, row in sample.iterrows():
        dt = row["decision_time"]
        ref = float(close_map["BTC/USDT"].asof(dt))
        st = graph.run_decision("BTC/USDT", dt, broker, ref_price=ref, cb_level=0)
        print(json.dumps(decision_to_json(st, fcfg), ensure_ascii=False))
        print("  audit_log:", [r.skill for r in st["audit_log"]])

    # demonstrate Risk veto under circuit breaker L3
    print("\n================ RISK VETO (circuit breaker L3) ================")
    dt = sample.iloc[-1]["decision_time"]; ref = float(close_map["BTC/USDT"].asof(dt))
    st = graph.run_decision("BTC/USDT", dt, broker, ref_price=ref, cb_level=3)
    print(f"  action={st['action']} approved={st['risk_approved']} pos={st['target_position']:.3f} "
          f"reason='{st['reason']}'  (Risk agent vetoed despite signal)")

    print("\nOK: 7-agent + skills orchestration ran end-to-end with audited skill calls.")


if __name__ == "__main__":
    main()
