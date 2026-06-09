"""
crypto.governance.pbo
========================
Probability of Backtest Overfitting via Combinatorially-Symmetric Cross-
Validation (Bailey, Borwein, Lopez de Prado, Zhu), v6 §9.2.3.

Input: a [T, N] matrix of per-period returns for N candidate configs/strategies.
Split T into S contiguous blocks; for every way to choose S/2 blocks as IS and
the rest as OOS, pick the IS-best config, find its OOS performance rank, map to
a logit; PBO = fraction of cases where the IS-best config ranks below the OOS
median (logit <= 0).

Run in the DEVELOPMENT period only; never on Final Holdout-A.
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np


def _sharpe(x: np.ndarray) -> float:
    sd = x.std()
    return x.mean() / sd if sd > 0 else 0.0


def cscv_pbo(returns_matrix: np.ndarray, n_blocks: int = 8,
             perf_fn: Callable[[np.ndarray], float] = _sharpe) -> dict:
    """
    returns_matrix: [T, N]. Returns {pbo, n_combinations, logits}.
    """
    R = np.asarray(returns_matrix, float)
    T, N = R.shape
    if N < 2:
        return {"pbo": float("nan"), "n_combinations": 0, "logits": []}
    n_blocks = n_blocks if n_blocks % 2 == 0 else n_blocks - 1
    blocks = np.array_split(np.arange(T), n_blocks)
    half = n_blocks // 2

    logits = []
    for is_sel in itertools.combinations(range(n_blocks), half):
        is_rows = np.concatenate([blocks[b] for b in is_sel])
        oos_rows = np.concatenate([blocks[b] for b in range(n_blocks) if b not in is_sel])
        is_perf = np.array([perf_fn(R[is_rows, j]) for j in range(N)])
        oos_perf = np.array([perf_fn(R[oos_rows, j]) for j in range(N)])
        n_star = int(np.argmax(is_perf))
        # relative OOS rank of the IS-best config (0..1)
        order = oos_perf.argsort().argsort()       # ranks 0..N-1
        rank = order[n_star]
        w = (rank + 1) / (N + 1)                   # in (0,1)
        logits.append(float(np.log(w / (1 - w))))
    logits = np.array(logits)
    pbo = float(np.mean(logits <= 0.0))            # P(IS-best underperforms OOS median)
    return {"pbo": pbo, "n_combinations": len(logits), "logits": logits.tolist()}
