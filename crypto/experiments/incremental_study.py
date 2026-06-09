"""
crypto.experiments.incremental_study
====================================
The §7.1 incremental-proof ladder, as a dedicated, scientific experiment.

It answers, one controlled step at a time, whether each added component carries
**statistically significant incremental** value over the previous step — the
thing that makes the project "like research".

Ladder (each step adds exactly one thing; everything else held fixed):
  Step0  baseline                  波动率平价 TSMOM(及格线,无 ML)
  Step1  + market/derivatives      结构化数据模型能否产生 alpha?
  Step2  + on-chain                链上数据是否提升预测能力?
  Step3  + narrative               文本情绪/事件因子是否有增量信息?
  Step4  + PatchTST                时序表征是否有增量?
  Step5  fusion(多模态融合)        Agent 融合系统是否优于单模型?
  Step6  + meta-labeling 门控       元标签/概率门控是否提升风险收益比?

Comparability: EVERY step is turned into a target-weight panel and run through
the SAME existing backtest engine on the SAME close panel, so all steps yield
comparable per-period portfolio returns (no apples-to-oranges).

Incremental significance (overlap-aware): for step_k vs step_{k-1} we form the
per-period return-difference series d_t = r_k − r_{k-1} and test mean(d)=0 with a
Newey-West t-stat (lag = max label horizon in bars). A step is marked
`included=True` only if NW-t(d) > t_threshold (default 2.0). Absolute strategy
quality is reported via annualized Sharpe + Deflated Sharpe (n_trials = #steps,
penalizing the multiple comparisons). Exploratory vs Confirmatory is a flag;
Confirmatory runs should pass through governance.registry.assert_preregistered.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from crypto.cv.purged_kfold import purged_embargoed_splits, default_embargo_delta
from crypto.models.base_lgb import MultiClassLearner, BinaryLearner
from crypto.labels.meta_label import build_meta_label
from crypto.benchmark.tsmom import vol_parity_tsmom_weights
from crypto.eval.significance import (newey_west_tstat, deflated_sharpe_ratio,
                                      information_coefficient, block_bootstrap_sharpe)
from backtest.engine import run_vector_backtest, BacktestConfig


RESEARCH_QUESTION = {
    "Step0_baseline_tsmom": "及格线基准(波动率平价 TSMOM,无 ML)",
    "Step1_market": "结构化数据模型能否产生 alpha?",
    "Step2_+onchain": "链上数据是否提升预测能力?",
    "Step3_+narrative": "文本情绪/事件因子是否有增量信息?",
    "Step4_+patchtst": "PatchTST 时序表征是否有增量?",
    "Step5_fusion": "多模态融合是否优于单模型?",
    "Step6_meta_gate": "元标签/概率门控是否提升风险收益比?",
}


def _oof_alpha(df: pd.DataFrame, cols: List[str], fcfg, splits) -> np.ndarray:
    """OOF combined_alpha = P(up)-P(down) from a single 3-class model on `cols`."""
    if not cols:
        return np.full(len(df), np.nan)
    X = df[cols].to_numpy(float)
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


def _alpha_to_returns(df, alpha, close_panel, bars_per_year, max_pos,
                      meta_prob=None, p_threshold=0.55, vol_target=False, fcfg=None):
    """Turn an OOF alpha (+ optional meta gate) into per-period portfolio returns
    via the EXISTING backtest engine (so all steps are comparable)."""
    pos = np.sign(np.nan_to_num(alpha)) * max_pos
    if meta_prob is not None:
        pos = pos * (np.nan_to_num(meta_prob) > p_threshold).astype(float)
    tmp = df[["symbol", "decision_time"]].copy()
    tmp["w"] = pos
    w = tmp.pivot_table(index="decision_time", columns="symbol", values="w", aggfunc="last").fillna(0)
    cp = close_panel.reindex(w.index).ffill().dropna()
    w = w.reindex(cp.index).fillna(0.0)
    if len(cp) < 5:
        return pd.Series(dtype=float)
    cfg = BacktestConfig(fee_rate=0.0004, slippage_rate=0.0003, execution_lag=1,
                         annual_days=bars_per_year)
    res = run_vector_backtest(cp, w, config=cfg, strategy_name="step")
    return res["returns"]


def _tsmom_returns(close_panel, bars_per_year):
    w = vol_parity_tsmom_weights(close_panel, lookback_mom=90, vol_window=30,
                                 cov_window=30, bars_per_year=bars_per_year)
    cfg = BacktestConfig(fee_rate=0.0004, slippage_rate=0.0003, execution_lag=1,
                         annual_days=bars_per_year)
    return run_vector_backtest(close_panel, w, config=cfg, strategy_name="tsmom")["returns"]


def _fusion_alpha(df, modality_cols, fcfg, splits):
    """Leakage-free fusion: mean of each modality's OOF combined_alpha (z-scored).
    (A *fitted* meta-combiner would require nested CV; this unfitted combiner is a
    clean 'fusion vs single-model' comparison.)  Returns (fused_alpha, base_oof)."""
    base = {}
    for mod, cols in modality_cols.items():
        if cols:
            base[mod] = _oof_alpha(df, cols, fcfg, splits)
    if not base:
        return np.full(len(df), np.nan), {}
    mat = []
    for a in base.values():
        s = np.nan_to_num(a)
        sd = s.std()
        mat.append(s / sd if sd > 1e-12 else s)
    fused = np.mean(np.column_stack(mat), axis=1)
    return fused, base


def _meta_gate(df, fused_alpha, base_oof, fcfg, splits):
    """Stage-2 meta gate on the fusion alpha (OOF direction -> meta-label -> OOF prob)."""
    direction = np.where(fused_alpha > fcfg.theta_long, "long",
                         np.where(fused_alpha < -fcfg.theta_short, "short", "flat"))
    meta = build_meta_label(pd.Series(direction), df["net_exit_return_long"],
                            df["net_exit_return_short"], source="oof")
    y = pd.Series(np.nan, index=df.index)
    y.loc[meta.index] = meta["meta_label"].to_numpy()
    feats = np.column_stack([np.nan_to_num(fused_alpha)] +
                            [np.nan_to_num(v) for v in base_oof.values()])
    prob = np.full(len(df), np.nan)
    yv = y.to_numpy()
    nonflat = ~np.isnan(yv)
    for f in splits:
        tr = [i for i in f.train_idx if nonflat[i]]
        te = [i for i in f.test_idx if nonflat[i]]
        if len(tr) < 30 or len(te) == 0 or len(np.unique(yv[tr])) < 2:
            continue
        b = BinaryLearner(fcfg.model).fit(feats[tr], yv[tr].astype(int))
        prob[te] = b.predict_proba(feats[te])
    return direction, prob


def run_incremental_study(
    dataset: pd.DataFrame,
    close_panel: pd.DataFrame,
    modality_cols: Dict[str, List[str]],
    fcfg,
    fwd_col: str = "raw_exit_return_long",
    bars_per_year: int = 2190,
    t_threshold: float = 2.0,
    max_pos: float = 0.2,
    kind: str = "exploratory",
) -> pd.DataFrame:
    """
    modality_cols keys expected: market / onchain / narrative / patchtst (any may
    be empty -> that step is 'skipped (no features)').
    Returns a per-step table with research question, IC, Sharpe, DSR, and the
    incremental NW-t vs the previous step + an `included` decision.
    """
    df = dataset.reset_index(drop=True).copy()
    splits = purged_embargoed_splits(df["decision_time"], df["entry_time"], df["exit_time"],
                                     fcfg.cv.n_splits, default_embargo_delta(fcfg.cv),
                                     symbol=df["symbol"])
    fwd = df[fwd_col].to_numpy(float)
    horizon_bars = max(2, int(fcfg.label.vertical_days * 6))  # 4h bars in vertical window

    mk = modality_cols.get("market", [])
    oc = modality_cols.get("onchain", [])
    nv = modality_cols.get("narrative", [])
    pt = modality_cols.get("patchtst", [])

    # build each step's (alpha, returns)
    steps = []  # (name, alpha_or_None, returns_series)

    # Step0 baseline TSMOM
    steps.append(("Step0_baseline_tsmom", None, _tsmom_returns(close_panel, bars_per_year)))

    cumulative = []
    for name, group in [("Step1_market", mk), ("Step2_+onchain", oc),
                        ("Step3_+narrative", nv), ("Step4_+patchtst", pt)]:
        new_cols = [c for c in group if c in df.columns]
        if not new_cols:                      # modality has no data -> carry previous
            steps.append((name, None, None))
            continue
        cumulative = cumulative + new_cols
        a = _oof_alpha(df, cumulative, fcfg, splits)
        r = _alpha_to_returns(df, a, close_panel, bars_per_year, max_pos)
        steps.append((name, a, r))

    # Step5 fusion (mean of per-modality OOF alphas)
    fused, base_oof = _fusion_alpha(df, {"market": mk, "onchain": oc,
                                         "narrative": nv, "patchtst": pt}, fcfg, splits)
    steps.append(("Step5_fusion", fused, _alpha_to_returns(df, fused, close_panel, bars_per_year, max_pos)))

    # Step6 meta gate
    direction, meta_prob = _meta_gate(df, fused, base_oof, fcfg, splits)
    steps.append(("Step6_meta_gate", fused,
                  _alpha_to_returns(df, fused, close_panel, bars_per_year, max_pos,
                                    meta_prob=meta_prob, p_threshold=fcfg.risk.p_threshold)))

    # assemble metrics + incremental significance vs previous *non-skipped* step
    n_steps = sum(1 for _, _, r in steps if r is not None)
    rows, prev_ret = [], None
    for name, alpha, ret in steps:
        if ret is None or len(ret) < 5:
            rows.append({"step": name, "research_question": RESEARCH_QUESTION.get(name, ""),
                         "status": "skipped (no features)", "IC": np.nan, "sharpe_ann": np.nan,
                         "deflated_sharpe": np.nan, "incr_NW_t": np.nan, "included": False})
            continue
        sd = ret.std()
        sharpe = ret.mean() / sd * np.sqrt(bars_per_year) if sd > 0 else np.nan
        dsr = deflated_sharpe_ratio(sharpe / np.sqrt(bars_per_year) if np.isfinite(sharpe) else np.nan,
                                    n_obs=len(ret), n_trials=max(n_steps, 1))
        ic = np.nan
        if alpha is not None:
            m = np.isfinite(alpha) & np.isfinite(fwd)
            ic = information_coefficient(alpha[m], fwd[m])
        incr_t, included = np.nan, None
        if prev_ret is not None:
            common = ret.index.intersection(prev_ret.index)
            if len(common) > horizon_bars + 2:
                d = (ret.reindex(common) - prev_ret.reindex(common)).to_numpy()
                incr_t = newey_west_tstat(d, lag=horizon_bars)
                included = bool(np.isfinite(incr_t) and incr_t > t_threshold)
        else:
            included = True  # baseline always "in"
        rows.append({"step": name, "research_question": RESEARCH_QUESTION.get(name, ""),
                     "status": kind, "IC": ic, "sharpe_ann": sharpe,
                     "deflated_sharpe": dsr, "incr_NW_t": incr_t, "included": included})
        prev_ret = ret

    return pd.DataFrame(rows).set_index("step")
