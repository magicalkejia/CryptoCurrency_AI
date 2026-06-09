"""
crypto.eval.significance
===========================
Overlap-aware significance (v6 §9.2.2): IC Newey-West t-stat, block bootstrap
Sharpe CI, and a Deflated Sharpe Ratio approximation.  Lightweight (numpy/scipy).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def information_coefficient(pred: np.ndarray, fwd_ret: np.ndarray, method: str = "spearman") -> float:
    pred = np.asarray(pred, float); fwd_ret = np.asarray(fwd_ret, float)
    m = np.isfinite(pred) & np.isfinite(fwd_ret)
    if m.sum() < 3:
        return float("nan")
    if method == "spearman":
        return float(stats.spearmanr(pred[m], fwd_ret[m]).statistic)
    return float(np.corrcoef(pred[m], fwd_ret[m])[0, 1])


def newey_west_tstat(series: np.ndarray, lag: int) -> float:
    """t-stat of mean(series) with Newey-West HAC correction for overlap."""
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1)
        cov = (e[l:] @ e[:-l]) / n
        var += 2.0 * w * cov
    se = np.sqrt(max(var, 1e-18) / n)
    return float(mu / se) if se > 0 else float("nan")


def block_bootstrap_sharpe(returns: np.ndarray, block: int, n_boot: int = 1000,
                           seed: int = 0) -> dict:
    """Block-bootstrap CI for the (per-bar) Sharpe ratio under overlap."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < block + 1:
        return {"sharpe": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    sharpes = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        sd = sample.std()
        sharpes.append(sample.mean() / sd if sd > 0 else 0.0)
    sharpes = np.array(sharpes)
    base = r.mean() / r.std() if r.std() > 0 else float("nan")
    return {"sharpe": float(base),
            "ci_low": float(np.percentile(sharpes, 2.5)),
            "ci_high": float(np.percentile(sharpes, 97.5))}


def deflated_sharpe_ratio(sharpe: float, n_obs: int, n_trials: int,
                          skew: float = 0.0, kurt: float = 3.0) -> float:
    """Bailey & Lopez de Prado DSR (probability the true SR>0 after trials)."""
    if not np.isfinite(sharpe) or n_obs < 2 or n_trials < 1:
        return float("nan")
    emax = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials) + \
        np.euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    sr_std = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4.0 * sharpe ** 2) / (n_obs - 1))
    if sr_std <= 0:
        return float("nan")
    return float(stats.norm.cdf((sharpe - emax * sr_std) / sr_std))
