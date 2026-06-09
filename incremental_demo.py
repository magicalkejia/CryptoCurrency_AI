"""
Incremental ablation study demo (§7.1 ladder). Run: python incremental_demo.py

Builds a synthetic multimodal dataset (market + on-chain + narrative + PatchTST),
then runs Step0->Step6 and prints, per step, the research question it answers,
IC, annualized Sharpe, Deflated Sharpe, the incremental Newey-West t vs the
previous step, and whether the increment is statistically included.

All numbers are SYNTHETIC and NOT a real result — this demonstrates the
experimental machinery; real conclusions require real data + a locked holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto.schemas import FrozenConfig
from crypto.adapters import to_bars_schema, decision_time_grid
from crypto.labels.triple_barrier import compute_triple_barrier
from crypto.features.uniqueness import average_uniqueness
from crypto.models.patchtst import run_patchtst
from crypto.pit import make_supervised_dataset
from crypto.experiments.incremental_study import run_incremental_study


def synth_bars(seed, n=24 * 260):
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


def main():
    fcfg = FrozenConfig()
    symbols = ["BTC/USDT", "ETH/USDT"]
    labels, feats, close_map = [], [], {}
    rng = np.random.default_rng(123)
    for i, s in enumerate(symbols):
        b1 = to_bars_schema(synth_bars(i + 1), "1h")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        b4 = to_bars_schema(b1.resample("4h", label="left", closed="left").agg(agg).dropna(), "4h")
        dts = decision_time_grid(b4, fcfg.decision_offset_minutes)
        lbl = compute_triple_barrier(b1, dts, s, fcfg.label, fcfg.cost,
                                     label_config_hash=fcfg.label_config_hash(),
                                     cost_model_hash=fcfg.cost_model_hash())
        labels.append(lbl)
        c = b4["close"]
        f = pd.DataFrame({"ret_1": c.pct_change(), "ret_6": c.pct_change(6),
                          "vol_24": c.pct_change().rolling(24).std()})
        m = c.pct_change(12); f["mom_z"] = (m - m.rolling(48).mean()) / (m.rolling(48).std() + 1e-9)
        f = f.reset_index().rename(columns={"index": "ts_open"})
        f["decision_time"] = b4["ts_close"].values + pd.Timedelta(minutes=1)
        f["symbol"] = s
        # synthetic on-chain & narrative columns (weak signal, for demonstration only)
        fut = c.shift(-6).reindex(b4.index)
        sig = (fut / c - 1).fillna(0).to_numpy()
        nrow = len(f)
        f["onchain_active_z"] = (0.4 * sig[:nrow] + rng.normal(0, 1, nrow))
        f["onchain_flow_z"] = (0.3 * sig[:nrow] + rng.normal(0, 1, nrow))
        f["narrative_sentiment"] = (0.25 * sig[:nrow] + rng.normal(0, 1, nrow))
        f["narrative_event_risk"] = rng.normal(0, 1, nrow)
        f["max_feature_availability_ts"] = f["decision_time"]
        f = f[f["decision_time"].isin(dts)].dropna()
        pt = run_patchtst(b1, dts, s, fcfg, lookback=96, emb_dim=4)
        if not pt.empty:
            f = f.merge(pt, on=["symbol", "decision_time"], how="left")
        feats.append(f)
        cc = c.copy(); cc.index = dts[:len(b4)]
        close_map[s] = cc

    labels = pd.concat(labels, ignore_index=True).dropna(subset=["entry_time", "exit_time"])
    feats = pd.concat(feats, ignore_index=True)
    labels["uniqueness_weight"] = average_uniqueness(
        labels["entry_time"], labels["exit_time"], scope="pooled", symbol=labels["symbol"]).values
    ds = make_supervised_dataset(feats, labels, require_pit=True)

    modality_cols = {
        "market": ["ret_1", "ret_6", "vol_24", "mom_z"],
        "onchain": ["onchain_active_z", "onchain_flow_z"],
        "narrative": ["narrative_sentiment", "narrative_event_risk"],
        "patchtst": [c for c in ds.columns if c.startswith("patchtst_")],
    }
    ds = ds.dropna(subset=[c for cols in modality_cols.values() for c in cols if c in ds.columns])
    close_panel = pd.DataFrame({s: close_map[s] for s in symbols})

    print("running incremental study Step0->Step6 (synthetic)...\n")
    table = run_incremental_study(ds, close_panel, modality_cols, fcfg, max_pos=0.2)
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(table.round(4).to_string())
    print("\n判读:incr_NW_t 是'本步相对上一步的逐期收益差'的 Newey-West t 值;")
    print("included=True 表示该模态/组件带来了统计显著的增量(默认阈值 t>2)。")
    print("(合成数据,NOT a real result;真实结论需在真实数据 + 锁定 holdout 上跑。)")


if __name__ == "__main__":
    main()
