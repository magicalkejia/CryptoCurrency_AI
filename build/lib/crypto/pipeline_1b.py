"""
crypto.pipeline_1b
=====================
Phase-1b orchestrator (v6 §7.5/§7.6): the two-stage meta-labelling pipeline with
walk-forward OOF, wired with the audit fixes.

Flow (per fold, walk-forward, purged):
  1. Stage-1 multiclass model -> OOF P(down/neutral/up); combined_alpha = P(up)-P(down)
  2. primary_direction_oof from combined_alpha & theta thresholds (long/short/flat)
  3. build_meta_label(primary_direction_oof, net_long, net_short)  (source='oof')
  4. Stage-2 binary model -> OOF meta_trade_prob_raw
  5. calibrate (Platt) on OOF -> meta_trade_prob_calibrated; report ECE

Returns a tidy `signals` frame + diagnostics (class distribution, ECE, etc).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto.cv.purged_kfold import purged_embargoed_splits, default_embargo_delta
from crypto.labels.meta_label import build_meta_label
from crypto.models.base_lgb import MultiClassLearner, BinaryLearner
from crypto.models.calibrate import fit_calibrator, compute_ece


def _primary_direction(combined_alpha: np.ndarray, theta_long: float, theta_short: float) -> np.ndarray:
    out = np.full(len(combined_alpha), "flat", dtype=object)
    out[combined_alpha > theta_long] = "long"
    out[combined_alpha < -theta_short] = "short"
    return out


def run_phase1b(
    dataset: pd.DataFrame,
    feature_cols: list,
    fcfg,
    meta_feature_cols: list | None = None,
):
    """
    dataset must contain: symbol, decision_time, entry_time(t0), exit_time(t1),
    tb_label, net_exit_return_long, net_exit_return_short, uniqueness_weight,
    plus feature_cols.
    """
    df = dataset.reset_index(drop=True).copy()
    n = len(df)
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["tb_label"].to_numpy(dtype=int)
    w = df["uniqueness_weight"].fillna(1.0).to_numpy(dtype=float)

    splits = purged_embargoed_splits(
        decision_time=df["decision_time"], t0=df["entry_time"], t1=df["exit_time"],
        n_splits=fcfg.cv.n_splits, embargo_delta=default_embargo_delta(fcfg.cv),
        symbol=df["symbol"],
    )

    # ---- Stage-1 OOF ----
    combined_alpha = np.full(n, np.nan)
    for fold in splits:
        if len(fold.train_idx) < 30 or len(fold.test_idx) == 0:
            continue
        m = MultiClassLearner(fcfg.model)
        m.fit(X[fold.train_idx], y[fold.train_idx], sample_weight=w[fold.train_idx])
        p_down, p_neu, p_up = m.predict_proba_df(X[fold.test_idx])
        combined_alpha[fold.test_idx] = p_up - p_down

    df["combined_alpha"] = combined_alpha
    df["primary_direction"] = _primary_direction(
        np.nan_to_num(combined_alpha), fcfg.theta_long, fcfg.theta_short)

    tb_dist = pd.Series(y).value_counts(normalize=True).to_dict()
    dir_dist = pd.Series(df["primary_direction"]).value_counts(normalize=True).to_dict()

    # ---- Stage-2 meta-label from OOF direction (audit #5) ----
    meta = build_meta_label(
        df["primary_direction"], df["net_exit_return_long"], df["net_exit_return_short"],
        source="oof")
    df["meta_label"] = np.nan
    df.loc[meta.index, "meta_label"] = meta["meta_label"].to_numpy()

    meta_cols = meta_feature_cols or (feature_cols + ["combined_alpha"])
    nonflat = df["meta_label"].notna()
    df["meta_trade_prob_raw"] = np.nan

    # Stage-2 OOF (reuse same purged splits, restricted to non-flat)
    Xm = df[meta_cols].to_numpy(dtype=float)
    ym = df["meta_label"].to_numpy()
    for fold in splits:
        tr = [i for i in fold.train_idx if nonflat.iloc[i]]
        te = [i for i in fold.test_idx if nonflat.iloc[i]]
        if len(tr) < 30 or len(te) == 0 or len(np.unique(ym[tr])) < 2:
            continue
        b = BinaryLearner(fcfg.model)
        b.fit(Xm[tr], ym[tr].astype(int))
        df.loc[df.index[te], "meta_trade_prob_raw"] = b.predict_proba(Xm[te])

    # ---- calibration on OOF (audit #16/#21) ----
    cal_mask = df["meta_trade_prob_raw"].notna() & df["meta_label"].notna()
    ece_raw = ece_cal = float("nan")
    df["meta_trade_prob_calibrated"] = df["meta_trade_prob_raw"]
    if cal_mask.sum() > 50:
        cal = fit_calibrator(df.loc[cal_mask, "meta_trade_prob_raw"],
                             df.loc[cal_mask, "meta_label"], method="platt")
        df.loc[cal_mask, "meta_trade_prob_calibrated"] = cal.transform(
            df.loc[cal_mask, "meta_trade_prob_raw"])
        ece_raw = compute_ece(df.loc[cal_mask, "meta_label"], df.loc[cal_mask, "meta_trade_prob_raw"])
        ece_cal = compute_ece(df.loc[cal_mask, "meta_label"], df.loc[cal_mask, "meta_trade_prob_calibrated"])

    signals = df[["symbol", "decision_time", "combined_alpha", "primary_direction",
                  "meta_trade_prob_raw", "meta_trade_prob_calibrated", "meta_label"]]
    diagnostics = {
        "n_samples": n,
        "tb_label_distribution": tb_dist,
        "primary_direction_distribution": dir_dist,
        "n_folds": len(splits),
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "backend": MultiClassLearner(fcfg.model).backend,
    }
    return signals, diagnostics
