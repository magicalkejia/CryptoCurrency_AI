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


# Demo drawdown injection: map specific symbols to a target trailing drawdown
# (measured over the last ~90 days / 540 4h-bars) so the data-driven circuit
# breaker fires a real L1/L2/L3 in synthetic mode. Thresholds (live CircuitBreaker
# defaults): L1 > 10% drawdown, L2 > 15%, L3 > 20%.
DEMO_DRAWDOWN = {
    "SOL/USDT": 0.12,   # -> L1 (warn)   : >10%
    "DOGE/USDT": 0.18,  # -> L2 (delever): >15%
    "XRP/USDT": 0.26,   # -> L3 (halt)   : >20%
}

def _synth_bars(seed, n=24 * 200, symbol=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h")
    # Calm, gently-rising baseline with LOW volatility so non-demo coins have a
    # shallow trailing drawdown (stay at circuit-breaker NORMAL). Hourly vol ~0.2%
    # with a small positive drift keeps the random-walk drawdown modest.
    ret = rng.normal(0.00005, 0.002, n)
    close = 100 * np.exp(np.cumsum(ret))
    # For designated demo symbols, overwrite the final ~90 days (540 4h-bars =
    # 2160 1h-bars) with a clean monotone decline from the prior peak to exactly
    # (1 - target_dd), so the trailing-90d drawdown measured by auto_cb == target.
    target_dd = DEMO_DRAWDOWN.get(symbol)
    if target_dd:
        w = min(2160, n // 2)
        peak = float(close[-w])
        decline = peak * (1.0 - np.linspace(0.0, target_dd, w))
        # add tiny noise so it doesn't look perfectly synthetic, but keep monotone-ish
        decline = decline * (1.0 + rng.normal(0, 0.0008, w))
        close[-w:] = decline
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": close * 1.002,
                         "low": close * 0.998, "close": close,
                         "volume": rng.lognormal(10, .4, n),
                         "taker_buy_vol": rng.lognormal(9, .4, n),
                         "net_taker_vol": rng.normal(0, 1, n)}, index=idx)


def build_graph(symbols=("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"),
                fcfg=None, quality_threshold=0.6, real=False):
    """Construct a ready-to-use TradingGraph + close map + decision index.
    Returns (graph, close_map, feats, fcols, fcfg). Used by CLI and web UI.

    real=False (default): build an offline synthetic feature frame (no data files
        needed) — fine for a wiring smoke-test / offline demo.
    real=True: load the SAME real processed parquet the experiment uses, via
        etl.dataset_builder.build_market_dataset, so the live agent demo runs on
        real market data. Falls back to synthetic if the data isn't available."""
    fcfg = fcfg or FrozenConfig()
    if real:
        try:
            from etl.dataset_builder import build_market_dataset
            md = build_market_dataset(list(symbols), fcfg)
            ds = md.dataset.dropna(subset=["combined_alpha"]) if "combined_alpha" in md.dataset.columns else md.dataset
            feats = md.dataset.copy()
            fcols = list(md.feature_cols)
            ds_fit = feats.dropna(subset=fcols)
            bundle = ModelBundle(fcfg).fit(ds_fit, fcols)
            graph = TradingGraph(fcfg, bundle, feats, fcols, quality_threshold=quality_threshold)
            close_map = md.close_map if hasattr(md, "close_map") else {}
            return graph, close_map, feats, fcols, fcfg
        except Exception as e:
            print(f"[build_graph] real data load failed ({e}); falling back to synthetic")

    labels, feats, close_map = [], [], {}
    for i, s in enumerate(symbols):
        b1 = to_bars_schema(_synth_bars(i + 1, symbol=s), "1h")
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
            cb_level=0, broker=None, auto_cb=False):
    sub = feats[feats["symbol"] == symbol].dropna(subset=fcols)
    if sub.empty:
        return None
    dt = decision_time or sub["decision_time"].iloc[-1]
    # Reference price: prefer close_map, but fall back gracefully if this symbol
    # was skipped during dataset build (missing parquet) or keyed differently.
    ref = None
    series = close_map.get(symbol) if hasattr(close_map, "get") else None
    if series is None and hasattr(close_map, "get"):
        # tolerate alternate key formats e.g. 'BTCUSDT' vs 'BTC/USDT'
        alt = symbol.replace("/", "")
        series = close_map.get(alt) or next(
            (v for k, v in close_map.items() if k.replace("/", "") == alt), None)
    if series is not None:
        try:
            val = series.asof(dt)
            ref = float(val) if val == val else None  # NaN guard
        except Exception:
            ref = None
    if ref is None:
        # last-resort fallback: use a close-like column from feats, else 1.0
        for col in ("close", "ref_price", "px_close"):
            if col in sub.columns:
                v = sub[col].iloc[-1]
                if v == v:
                    ref = float(v); break
        if ref is None:
            ref = 1.0  # purely a scaling reference; decision logic is scale-free

    # Data-driven circuit breaker (auto_cb): compute the symbol's trailing rolling
    # drawdown at the decision time from its own price series and let the circuit
    # breaker evaluate a REAL level, instead of using the injected what-if level.
    real_drawdown = 0.0
    if auto_cb and series is not None:
        try:
            import numpy as _np
            px = series[series.index <= dt].astype(float)
            if len(px) > 5:
                window = px.tail(540)               # ~90 days of 4h bars
                roll_peak = window.cummax()
                dd = 1.0 - (window / roll_peak)
                real_drawdown = float(dd.iloc[-1]) if dd.iloc[-1] == dd.iloc[-1] else 0.0
        except Exception:
            real_drawdown = 0.0

    b = broker if broker is not None else PaperBroker(max_slippage_bps=3)
    if auto_cb:
        # cb_level=None -> graph computes the level from the real drawdown
        st = graph.run_decision(symbol, dt, b, ref_price=ref, cb_level=None,
                                drawdown=real_drawdown)
    else:
        st = graph.run_decision(symbol, dt, b, ref_price=ref, cb_level=cb_level)
    out = {"decision": decision_to_json(st, fcfg),
           "current_position": float(b.get_position(symbol)),
           "real_drawdown": round(real_drawdown, 4),
           "audit_log": [{"skill": r.skill, "category": r.category, "ok": r.ok,
                          "ms": round(r.duration_ms, 2)} for r in st["audit_log"]]}
    return out


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
