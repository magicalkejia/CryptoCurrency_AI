"""
crypto.models.calibrate  &  crypto.eval.metrics (ece)
==========================================================
Probability calibration (v6 §7.6) + ECE (audit #16/#21/#24).

  * small samples -> Platt (logistic) preferred; isotonic only when ample.
  * calibrator fit on (purged) OOF only; never on holdout.
  * compute_ece: quantile bins by default (fewer empty bins on small samples).
  * ECE<=0.05 is a *dev-set pre-registration gate*, NOT a holdout knob (audit #21):
    enforcement lives in the pipeline, holdout ECE is report-only.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    def __init__(self, method: Literal["platt", "isotonic"] = "platt"):
        self.method = method
        self._m = None

    def fit(self, prob_oof, y):
        prob_oof = np.asarray(prob_oof).reshape(-1, 1)
        y = np.asarray(y)
        if self.method == "isotonic":
            self._m = IsotonicRegression(out_of_bounds="clip")
            self._m.fit(prob_oof.ravel(), y)
        else:
            self._m = LogisticRegression(max_iter=1000)
            self._m.fit(prob_oof, y)
        return self

    def transform(self, prob):
        prob = np.asarray(prob)
        if self.method == "isotonic":
            return self._m.predict(prob.ravel())
        return self._m.predict_proba(prob.reshape(-1, 1))[:, 1]


def fit_calibrator(prob_oof, y, method="platt", min_samples_isotonic: int = 1000):
    """audit #16: prefer Platt on small samples."""
    n = len(np.asarray(y))
    if method == "isotonic" and n < min_samples_isotonic:
        method = "platt"
    return Calibrator(method).fit(prob_oof, y)


def compute_ece(y_true, prob, n_bins: int = 10,
                strategy: Literal["uniform", "quantile"] = "quantile") -> float:
    """Expected Calibration Error. quantile bins by default (audit #24)."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    n = len(prob)
    if n == 0:
        return float("nan")
    if strategy == "quantile":
        edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    if len(edges) < 2:
        return abs(prob.mean() - y_true.mean())
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob > lo) & (prob <= hi) if lo > edges[0] else (prob >= lo) & (prob <= hi)
        if m.sum() == 0:
            continue
        conf = prob[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)
