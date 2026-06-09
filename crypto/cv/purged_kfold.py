"""
crypto.cv.purged_kfold
=========================
Purged + embargoed, multi-asset time-block cross-validation (v6 §6.3.2) with
audit fixes:
  #6  double-sided overlap purge: t0_train < t1_test AND t1_train > t0_test.
  #7  embargo applied AFTER the test block (default), not symmetric.
  #9  splits by time-block, not row index; returns rich FoldSplit metadata.
  #18 multi-asset sync: same test time-block -> all symbols are test;
      no same-block sample of any symbol in train (cross-sectional leakage).
  half-open semantics: endpoints equal => NOT overlapping (test T1b_18).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class FoldSplit:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    purged_idx: np.ndarray
    embargoed_idx: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices_hash: str
    test_indices_hash: str


def _hash_idx(idx: np.ndarray) -> str:
    return hashlib.sha256(np.sort(idx).tobytes()).hexdigest()[:12]


def purged_embargoed_splits(
    decision_time: pd.Series,
    t0: pd.Series,
    t1: pd.Series,
    n_splits: int,
    embargo_delta: pd.Timedelta,
    symbol: Optional[pd.Series] = None,
    split_unit: str = "time_block",
) -> List[FoldSplit]:
    """
    Returns list[FoldSplit]. test blocks are contiguous spans of unique
    decision_time; all symbols inside a block go to test together.
    """
    decision_time = pd.to_datetime(pd.Series(decision_time).reset_index(drop=True))
    t0 = pd.to_datetime(pd.Series(t0).reset_index(drop=True))
    t1 = pd.to_datetime(pd.Series(t1).reset_index(drop=True))
    pos = np.arange(len(decision_time))

    uniq_times = np.array(sorted(decision_time.unique()))
    blocks = np.array_split(uniq_times, n_splits)

    folds: List[FoldSplit] = []
    for k, block in enumerate(blocks):
        if len(block) == 0:
            continue
        b_start, b_end = pd.Timestamp(block[0]), pd.Timestamp(block[-1])
        test_mask = (decision_time >= b_start) & (decision_time <= b_end)
        test_pos = pos[test_mask.to_numpy()]
        if len(test_pos) == 0:
            continue

        # label-interval span of the test block
        test_t0_min = t0[test_mask].min()
        test_t1_max = t1[test_mask].max()

        candidate = pos[~test_mask.to_numpy()]
        # purge: double-sided overlap (half-open). overlap iff t0<t1_test AND t1>t0_test
        ov = (t0.to_numpy() < np.datetime64(test_t1_max)) & (t1.to_numpy() > np.datetime64(test_t0_min))
        purged = pos[ov & (~test_mask.to_numpy())]
        # embargo: training samples whose decision_time falls in (test_t1_max, test_t1_max+embargo]
        emb_mask = (decision_time.to_numpy() > np.datetime64(test_t1_max)) & \
                   (decision_time.to_numpy() <= np.datetime64(test_t1_max + embargo_delta))
        embargoed = pos[emb_mask & (~test_mask.to_numpy())]

        drop = set(purged.tolist()) | set(embargoed.tolist())
        train_pos = np.array([p for p in candidate if p not in drop], dtype=int)

        folds.append(FoldSplit(
            fold_id=k,
            train_idx=train_pos,
            test_idx=test_pos,
            purged_idx=purged,
            embargoed_idx=embargoed,
            test_start=b_start,
            test_end=b_end,
            train_indices_hash=_hash_idx(train_pos),
            test_indices_hash=_hash_idx(test_pos),
        ))
    return folds


def default_embargo_delta(cv_cfg) -> pd.Timedelta:
    """v6 §6.3.2: default = max_label_horizon + buffer; conservative = max(180d,...)."""
    base = pd.Timedelta(days=cv_cfg.embargo_days + cv_cfg.data_lag_buffer_days)
    if cv_cfg.conservative_embargo:
        return max(base, pd.Timedelta(days=cv_cfg.max_feature_lookback_days))
    return base
