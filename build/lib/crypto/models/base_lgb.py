"""
crypto.models.base_lgb
=========================
Thin learner wrapper.  Prefers LightGBM (the v6 production choice); falls back
to sklearn GradientBoosting / LogisticRegression so the pipeline runs anywhere.
Interface is identical regardless of backend.
"""
from __future__ import annotations

import numpy as np

try:
    import lightgbm as lgb  # noqa
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class _ConstantProbaModel:
    """Fallback for a training fold that contains only ONE class.
    Predicts that class with probability 1 (any backend would otherwise crash)."""
    def __init__(self, cls):
        self.classes_ = np.array([cls])

    def predict_proba(self, X):
        return np.ones((len(X), 1))


class MultiClassLearner:
    """Down/neutral/up classifier with a uniform predict_proba API.

    Robust to the label set actually present: 2 classes -> binary objective,
    3 classes -> multiclass, 1 class -> constant model. We do NOT hardcode
    objective='multiclass' (that requires num_class>1 and breaks on binary data
    -- the cause of 'Number of classes should be ... greater than 1').
    """

    def __init__(self, model_cfg, prefer: str = "auto"):
        self.cfg = model_cfg
        self.backend = "lightgbm" if (_HAS_LGB and prefer in ("auto", "lightgbm")) else "sklearn"
        self.model = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        classes = np.unique(y)
        # degenerate fold: only one class present -> constant model (no crash)
        if len(classes) < 2:
            self.model = _ConstantProbaModel(int(classes[0]))
            self.classes_ = self.model.classes_
            return self
        if self.backend == "lightgbm":
            # let LightGBM infer binary vs multiclass from y (do NOT force objective)
            self.model = lgb.LGBMClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                learning_rate=self.cfg.learning_rate,
                random_state=self.cfg.random_seed,
                verbose=-1,
                min_child_samples=5,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                learning_rate=self.cfg.learning_rate,
                random_state=self.cfg.random_seed,
            )
        self.model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model.classes_
        return self

    def predict_proba_df(self, X):
        """Return P(down), P(neutral), P(up) aligned to classes {-1,0,1}.
        Missing classes (e.g. binary data without a neutral class) -> zeros."""
        proba = self.model.predict_proba(X)
        cols = {}
        for j, c in enumerate(self.model.classes_):
            cols[int(c)] = proba[:, j]
        n = len(X)
        p_down = cols.get(-1, np.zeros(n))
        p_neutral = cols.get(0, np.zeros(n))
        p_up = cols.get(1, np.zeros(n))
        return p_down, p_neutral, p_up


class BinaryLearner:
    """Binary classifier for the Stage-2 meta-model."""

    def __init__(self, model_cfg, prefer: str = "auto"):
        self.cfg = model_cfg
        self.backend = "lightgbm" if (_HAS_LGB and prefer in ("auto", "lightgbm")) else "sklearn"
        self.model = None

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self._single = None
        if len(np.unique(y)) < 2:
            # degenerate fold: predict the single label's probability as constant
            self._single = float(int(np.unique(y)[0]))
            return self
        if self.backend == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=self.cfg.n_estimators, max_depth=self.cfg.max_depth,
                learning_rate=self.cfg.learning_rate, random_state=self.cfg.random_seed,
                verbose=-1, min_child_samples=5)
        else:
            # small samples -> logistic is more robust than GB
            self.model = LogisticRegression(max_iter=1000, random_state=self.cfg.random_seed)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        if getattr(self, "_single", None) is not None:
            return np.full(len(X), self._single)
        return self.model.predict_proba(X)[:, 1]
