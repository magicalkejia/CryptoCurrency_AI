"""
crypto.models.bundle
=======================
A fitted bundle (stage1 + stage2 + calibrator) for INFERENCE inside the Agent
graph.  Training still builds the Stage-2 meta-label from OOF direction (audit
#5), so the deployed model is trained correctly; the bundle then exposes a
single-row `infer()` used by the FusionAgent at decision time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto.cv.purged_kfold import purged_embargoed_splits, default_embargo_delta
from crypto.labels.meta_label import build_meta_label
from crypto.models.base_lgb import MultiClassLearner, BinaryLearner
from crypto.models.calibrate import fit_calibrator


def _primary_direction(alpha, tl, ts):
    out = np.full(len(alpha), "flat", dtype=object)
    out[alpha > tl] = "long"
    out[alpha < -ts] = "short"
    return out


class ModelBundle:
    def __init__(self, fcfg):
        self.fcfg = fcfg
        self.stage1 = None
        self.stage2 = None
        self.calibrator = None
        self.feature_cols = None
        self.meta_cols = None
        self.fitted = False

    def fit(self, dataset: pd.DataFrame, feature_cols, meta_cols=None):
        df = dataset.reset_index(drop=True).copy()
        self.feature_cols = feature_cols
        self.meta_cols = meta_cols or (feature_cols + ["combined_alpha"])
        X = df[feature_cols].to_numpy(float)
        y = df["tb_label"].to_numpy(int)
        w = df["uniqueness_weight"].fillna(1.0).to_numpy(float)
        splits = purged_embargoed_splits(df["decision_time"], df["entry_time"], df["exit_time"],
                                         self.fcfg.cv.n_splits, default_embargo_delta(self.fcfg.cv),
                                         symbol=df["symbol"])
        # OOF stage1 -> direction -> meta_label (audit #5)
        alpha = np.full(len(df), np.nan)
        for f in splits:
            if len(f.train_idx) < 30 or len(f.test_idx) == 0:
                continue
            m = MultiClassLearner(self.fcfg.model).fit(X[f.train_idx], y[f.train_idx], w[f.train_idx])
            pd_, pn_, pu_ = m.predict_proba_df(X[f.test_idx])
            alpha[f.test_idx] = pu_ - pd_
        df["combined_alpha"] = alpha
        df["primary_direction"] = _primary_direction(np.nan_to_num(alpha),
                                                      self.fcfg.theta_long, self.fcfg.theta_short)
        meta = build_meta_label(df["primary_direction"], df["net_exit_return_long"],
                                df["net_exit_return_short"], source="oof")
        df["meta_label"] = np.nan
        df.loc[meta.index, "meta_label"] = meta["meta_label"].to_numpy()

        # OOF stage2 (for calibration) + final fits
        Xm = df[self.meta_cols].to_numpy(float)
        ym = df["meta_label"].to_numpy()
        nonflat = df["meta_label"].notna()
        oof_p = np.full(len(df), np.nan)
        for f in splits:
            tr = [i for i in f.train_idx if nonflat.iloc[i]]
            te = [i for i in f.test_idx if nonflat.iloc[i]]
            if len(tr) < 30 or len(te) == 0 or len(np.unique(ym[tr])) < 2:
                continue
            b = BinaryLearner(self.fcfg.model).fit(Xm[tr], ym[tr].astype(int))
            oof_p[te] = b.predict_proba(Xm[te])

        # final stage1 on all data (deployed model)
        self.stage1 = MultiClassLearner(self.fcfg.model).fit(X, y, w)
        nf = nonflat.to_numpy()
        if nf.sum() > 30 and len(np.unique(ym[nf])) == 2:
            self.stage2 = BinaryLearner(self.fcfg.model).fit(Xm[nf], ym[nf].astype(int))
            cal_mask = (~np.isnan(oof_p)) & nf
            if cal_mask.sum() > 50:
                self.calibrator = fit_calibrator(oof_p[cal_mask], ym[cal_mask], method="platt")
        self.fitted = True
        return self

    def infer(self, feat_row: dict) -> dict:
        """One decision's feature row -> alpha/direction/meta_trade_prob_calibrated."""
        X = np.array([[feat_row[c] for c in self.feature_cols]], float)
        p_d, p_n, p_u = self.stage1.predict_proba_df(X)
        alpha = float(p_u[0] - p_d[0])
        direction = ("long" if alpha > self.fcfg.theta_long else
                     "short" if alpha < -self.fcfg.theta_short else "flat")
        meta_raw = np.nan
        if self.stage2 is not None and direction != "flat":
            row = dict(feat_row); row["combined_alpha"] = alpha
            Xm = np.array([[row.get(c, 0.0) for c in self.meta_cols]], float)
            meta_raw = float(self.stage2.predict_proba(Xm)[0])
        meta_cal = meta_raw
        if self.calibrator is not None and not np.isnan(meta_raw):
            meta_cal = float(self.calibrator.transform([meta_raw])[0])
        return {"combined_alpha": alpha, "primary_direction": direction,
                "meta_trade_prob_raw": meta_raw, "meta_trade_prob_calibrated": meta_cal}
