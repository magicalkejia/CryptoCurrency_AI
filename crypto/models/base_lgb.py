"""
crypto.models.base_lgb
=========================
Thin learner wrapper.  Prefers LightGBM (the v6 production choice); falls back
to sklearn GradientBoosting / LogisticRegression so the pipeline runs anywhere.
Interface is identical regardless of backend.

Note: X is internally wrapped into a DataFrame with stable column names (f0..fn)
for BOTH fit and predict.  LightGBM >= 4 otherwise emits a cosmetic sklearn
UserWarning ("X does not have valid feature names ...") when fitted on a numpy
array and predicted on a numpy array; consistent names remove it. Numbers are
unchanged either way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb  # noqa
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def _as_named_df(X) -> pd.DataFrame:
    """Wrap array-like into a DataFrame with stable string column names so fit
    and predict see identical feature names -> no sklearn/LightGBM warning."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])


class _ConstantProbaModel:
    """Fallback for a training fold that contains only ONE class."""
    def __init__(self, cls):
        self.classes_ = np.array([cls])

    def predict_proba(self, X):
        return np.ones((len(X), 1))


class MultiClassLearner:
    """Down/neutral/up classifier with a uniform predict_proba API.

    2 classes -> binary objective, 3 classes -> multiclass, 1 class -> constant.
    We do NOT hardcode objective='multiclass' (that needs num_class>1 and breaks
    on binary data -- the 'classes should be greater than 1' error)."""

    def __init__(self, model_cfg, prefer: str = "auto"):
        self.cfg = model_cfg
        self.backend = "lightgbm" if (_HAS_LGB and prefer in ("auto", "lightgbm")) else "sklearn"
        self.model = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) < 2:
            self.model = _ConstantProbaModel(int(classes[0]))
            self.classes_ = self.model.classes_
            return self
        Xdf = _as_named_df(X)
        if self.backend == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=self.cfg.n_estimators, max_depth=self.cfg.max_depth,
                learning_rate=self.cfg.learning_rate, random_state=self.cfg.random_seed,
                verbose=-1, min_child_samples=getattr(self.cfg, "min_child_samples", 5),
                subsample=getattr(self.cfg, "subsample", 1.0),
                colsample_bytree=getattr(self.cfg, "colsample_bytree", 1.0),
                reg_alpha=getattr(self.cfg, "reg_alpha", 0.0),
                reg_lambda=getattr(self.cfg, "reg_lambda", 0.0))
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=self.cfg.n_estimators, max_depth=self.cfg.max_depth,
                learning_rate=self.cfg.learning_rate, random_state=self.cfg.random_seed,
                subsample=getattr(self.cfg, "subsample", 1.0),
                min_samples_leaf=getattr(self.cfg, "min_child_samples", 5))
        self.model.fit(Xdf, y, sample_weight=sample_weight)
        self.classes_ = self.model.classes_
        return self

    def predict_proba_df(self, X):
        n = np.asarray(X).shape[0] if not hasattr(X, "__len__") else len(X)
        if isinstance(self.model, _ConstantProbaModel):
            proba = self.model.predict_proba(X)
        else:
            proba = self.model.predict_proba(_as_named_df(X))
        cols = {}
        for j, c in enumerate(self.model.classes_):
            cols[int(c)] = proba[:, j]
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
            self._single = float(int(np.unique(y)[0]))
            return self
        Xdf = _as_named_df(X)
        if self.backend == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=self.cfg.n_estimators, max_depth=self.cfg.max_depth,
                learning_rate=self.cfg.learning_rate, random_state=self.cfg.random_seed,
                verbose=-1, min_child_samples=getattr(self.cfg, "min_child_samples", 5),
                subsample=getattr(self.cfg, "subsample", 1.0),
                colsample_bytree=getattr(self.cfg, "colsample_bytree", 1.0),
                reg_alpha=getattr(self.cfg, "reg_alpha", 0.0),
                reg_lambda=getattr(self.cfg, "reg_lambda", 0.0))
        else:
            self.model = LogisticRegression(max_iter=1000, random_state=self.cfg.random_seed)
        self.model.fit(Xdf, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        n = np.asarray(X).shape[0] if not hasattr(X, "__len__") else len(X)
        if getattr(self, "_single", None) is not None:
            return np.full(n, self._single)
        return self.model.predict_proba(_as_named_df(X))[:, 1]
