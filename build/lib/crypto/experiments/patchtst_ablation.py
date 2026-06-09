"""
crypto.experiments.patchtst_ablation
=======================================
Formal A/B/C/D comparison for PatchTST (v6 §7.4), all on the SAME purged CV.

  A. LightGBM on tabular features only.
  B. PatchTST forecast columns only.
  C. PatchTST embedding + tabular (embedding + LightGBM).
  D. A + C + forecast (full fusion).

For each config: walk-forward OOF combined_alpha (3-class P(up)-P(down)), then
overlap-aware significance: Spearman IC + Newey-West t-stat (lag = max label
horizon in bars) and a Deflated Sharpe of a sign(alpha)-sized strategy with
n_trials = number of configs (penalizes the selection).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from crypto.cv.purged_kfold import purged_embargoed_splits, default_embargo_delta
from crypto.models.base_lgb import MultiClassLearner
from crypto.eval.significance import (information_coefficient, newey_west_tstat,
                                         deflated_sharpe_ratio, block_bootstrap_sharpe)


def _oof_alpha(df, feat_cols, fcfg, splits):
    X = df[feat_cols].to_numpy(float)
    y = df["tb_label"].to_numpy(int)
    w = df["uniqueness_weight"].fillna(1.0).to_numpy(float)
    alpha = np.full(len(df), np.nan)
    for f in splits:
        if len(f.train_idx) < 30 or len(f.test_idx) == 0:
            continue
        m = MultiClassLearner(fcfg.model).fit(X[f.train_idx], y[f.train_idx], w[f.train_idx])
        pdn, pne, pup = m.predict_proba_df(X[f.test_idx])
        alpha[f.test_idx] = pup - pdn
    return alpha


def run_ablation(dataset: pd.DataFrame, tabular_cols: List[str], fcfg,
                 fwd_return_col: str = "raw_exit_return_long",
                 bars_per_year: int = 2190, max_label_horizon_bars: int = 30) -> pd.DataFrame:
    df = dataset.reset_index(drop=True).copy()
    forecast_cols = [c for c in df.columns if c.startswith("patchtst_forecast_")]
    emb_cols = [c for c in df.columns if c.startswith("patchtst_emb_")]

    configs = {
        "A_tabular": tabular_cols,
        "B_patchtst_forecast": forecast_cols,
        "C_emb_plus_tabular": tabular_cols + emb_cols,
        "D_full_fusion": tabular_cols + emb_cols + forecast_cols,
    }
    configs = {k: v for k, v in configs.items() if len(v) > 0}

    splits = purged_embargoed_splits(df["decision_time"], df["entry_time"], df["exit_time"],
                                     fcfg.cv.n_splits, default_embargo_delta(fcfg.cv),
                                     symbol=df["symbol"])
    fwd = df[fwd_return_col].to_numpy(float)
    n_trials = len(configs)

    rows = []
    for name, cols in configs.items():
        alpha = _oof_alpha(df, cols, fcfg, splits)
        mask = np.isfinite(alpha) & np.isfinite(fwd)
        ic = information_coefficient(alpha[mask], fwd[mask], method="spearman")
        # strategy: position = sign(alpha); pnl = pos * fwd
        pnl = np.sign(alpha[mask]) * fwd[mask]
        t_nw = newey_west_tstat(pnl, lag=max_label_horizon_bars)
        sd = pnl.std()
        sharpe = (pnl.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else np.nan
        bb = block_bootstrap_sharpe(pnl, block=max_label_horizon_bars, n_boot=500)
        dsr = deflated_sharpe_ratio(sharpe / np.sqrt(bars_per_year) if np.isfinite(sharpe) else np.nan,
                                    n_obs=mask.sum(), n_trials=n_trials)
        rows.append({"config": name, "n_features": len(cols), "n_oof": int(mask.sum()),
                     "IC": ic, "IC_NW_t": t_nw, "sharpe_ann": sharpe,
                     "sharpe_ci": (round(bb["ci_low"], 3), round(bb["ci_high"], 3)),
                     "deflated_sharpe": dsr})
    res = pd.DataFrame(rows).set_index("config")
    return res
