"""
crypto.pit
=============
Point-in-time guards (v6 §4) and supervised-dataset assembly.

  * audit_lookahead: detects features whose availability_ts > decision_time.
  * make_supervised_dataset: merge features+labels on (symbol, decision_time);
    require_pit checks max_feature_availability_ts <= decision_time (audit #19);
    refuses to leak label columns into the feature matrix (audit #17 / T1b_20).
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from crypto.labels.triple_barrier import LABEL_COLUMNS

_LABEL_LEAK_COLS = set(LABEL_COLUMNS) - {"symbol", "decision_time"}


def audit_lookahead(features: pd.DataFrame,
                    availability_col: str = "max_feature_availability_ts",
                    decision_col: str = "decision_time") -> dict:
    """Return an audit report; future_function_checks_passed=False if any leak."""
    viol = 0
    if availability_col in features.columns:
        viol = int((pd.to_datetime(features[availability_col]) >
                    pd.to_datetime(features[decision_col])).sum())
    return {
        "future_function_checks_passed": viol == 0,
        "availability_lag_violations": viol,
        "n_rows": len(features),
    }


def make_supervised_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    on: Optional[List[str]] = None,
    require_pit: bool = True,
    availability_col: str = "max_feature_availability_ts",
) -> pd.DataFrame:
    """
    Merge features & labels. require_pit checks FEATURE availability only
    (labels legitimately come from the future and must never be live features).
    Raises on PIT violation or if any label column sneaks into the feature set.
    """
    on = on or ["symbol", "decision_time"]

    if require_pit and availability_col in features.columns:
        rep = audit_lookahead(features, availability_col)
        if not rep["future_function_checks_passed"]:
            raise ValueError(
                f"PIT violation: {rep['availability_lag_violations']} feature rows "
                f"have availability_ts > decision_time")

    feat_cols = [c for c in features.columns if c not in (on + [availability_col, "feature_set_hash"])]
    leaked = _LABEL_LEAK_COLS.intersection(feat_cols)
    if leaked:
        raise ValueError(f"label columns leaked into features: {sorted(leaked)}")

    merged = features.merge(labels, on=on, how="inner", suffixes=("", "_lbl"))
    return merged


def feature_matrix_columns(merged: pd.DataFrame) -> List[str]:
    """Columns safe to feed a model (excludes ids, label cols, availability)."""
    exclude = set(["symbol", "decision_time", "max_feature_availability_ts",
                   "feature_set_hash"]) | _LABEL_LEAK_COLS
    return [c for c in merged.columns if c not in exclude]
