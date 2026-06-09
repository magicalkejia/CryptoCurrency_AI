"""
Experiment + governance demo (v6 §7.4 / §9.2).  Run: python experiment_demo.py

Demonstrates, on synthetic data:
  1. PatchTST A/B/C/D ablation on a shared purged CV (IC + Newey-West t + DSR).
  2. PBO (CSCV) over the four configs' OOF return streams.
  3. Holdout-A freeze (frozen_config.json + hash) and a Confirmatory
     pre-registration gate.
All numbers are synthetic and NOT a real result.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from crypto.schemas import FrozenConfig
from crypto.experiments.patchtst_ablation import run_ablation, _oof_alpha
from crypto.cv.purged_kfold import purged_embargoed_splits, default_embargo_delta
from crypto.governance.pbo import cscv_pbo
from crypto.governance.holdout import dev_holdout_split, freeze_config, load_frozen
from crypto.governance.registry import pre_register, assert_preregistered


def synth(n=600, seed=7):
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2022-01-01", periods=n, freq="4h")
    sig = rng.normal(0, 1, n)
    fwd = 0.012 * np.tanh(sig) + rng.normal(0, 0.012, n)
    tb = np.sign(fwd); tb[np.abs(fwd) < 0.006] = 0
    df = pd.DataFrame({"symbol": "BTC/USDT", "decision_time": dt, "entry_time": dt,
                       "exit_time": dt + pd.Timedelta(hours=8), "tb_label": tb.astype(int),
                       "uniqueness_weight": 1.0, "raw_exit_return_long": fwd,
                       "tab_feat": sig + rng.normal(0, .5, n),
                       "patchtst_forecast_4h": .5 * sig + rng.normal(0, .6, n),
                       "patchtst_forecast_24h": .5 * sig + rng.normal(0, .6, n)})
    for j in range(4):
        df[f"patchtst_emb_{j}"] = sig * rng.normal() + rng.normal(0, 1, n)
    return df


def main():
    fcfg = FrozenConfig()
    df = synth()

    # ---- dev / holdout split (freeze before touching holdout) ----
    dev, hold = dev_holdout_split(df["decision_time"], df["decision_time"].iloc[int(len(df) * 0.8)])
    dev_df = df.iloc[dev].reset_index(drop=True)
    print(f"dev rows={len(dev)}  holdout rows={len(hold)} (holdout untouched during ablation)")

    # ---- 1. A/B/C/D ablation on DEV ----
    res = run_ablation(dev_df, ["tab_feat"], fcfg, max_label_horizon_bars=2)
    print("\n=== PatchTST A/B/C/D ablation (DEV, synthetic) ===")
    print(res.round(4).to_string())

    # ---- 2. PBO over the four configs' OOF pnl streams ----
    splits = purged_embargoed_splits(dev_df["decision_time"], dev_df["entry_time"],
                                     dev_df["exit_time"], fcfg.cv.n_splits,
                                     default_embargo_delta(fcfg.cv), symbol=dev_df["symbol"])
    fwd = dev_df["raw_exit_return_long"].to_numpy()
    cols = {"A": ["tab_feat"],
            "B": [c for c in dev_df if c.startswith("patchtst_forecast_")],
            "C": ["tab_feat"] + [c for c in dev_df if c.startswith("patchtst_emb_")],
            "D": ["tab_feat"] + [c for c in dev_df if c.startswith("patchtst_")]}
    pnl_streams = []
    for cfg_cols in cols.values():
        a = _oof_alpha(dev_df, cfg_cols, fcfg, splits)
        pnl = np.where(np.isfinite(a), np.sign(np.nan_to_num(a)) * fwd, 0.0)
        pnl_streams.append(pnl)
    R = np.column_stack(pnl_streams)
    pbo = cscv_pbo(R, n_blocks=8)
    print(f"\n=== PBO (CSCV over A/B/C/D) ===\n  PBO={pbo['pbo']:.3f}  over {pbo['n_combinations']} combinations")

    # ---- 3. freeze + pre-register before Holdout-A ----
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "frozen_config.json"
        h = freeze_config(fcfg, fp)
        load_frozen(fp)  # verifies hash
        reg = Path(d) / "registry.json"
        pre_register(fcfg.to_dict(), reg, label="holdout_A_confirmatory")
        assert_preregistered(fcfg.to_dict(), reg)
        print(f"\n=== Holdout-A governance ===\n  frozen config_hash={h}  pre-registration OK")
        print("  (Holdout-A would now run ONCE with this frozen config; no retuning.)")

    print("\nOK: ablation + PBO + freeze + pre-registration ran end-to-end.")


if __name__ == "__main__":
    main()
