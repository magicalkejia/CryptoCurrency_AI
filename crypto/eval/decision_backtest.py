"""
crypto.eval.decision_backtest
=============================
Bridge between the AGENT DECISION STACK and the existing vectorized backtest
engine (backtest/). NOTHING in backtest/ is modified — this module only PRODUCES
the `(close, target_weight)` pair that `backtest.engine.run_vector_backtest`
already expects, by running the real decision skills across the whole history.

Why this is needed
------------------
The incremental-study ladder already feeds the backtest engine, but it deploys the
raw model ALPHA (conviction sizing), deliberately bypassing risk to measure signal
value cleanly. The agent stack we built (per-symbol `risk_size_and_gate` + A2
`portfolio_risk_overlay` + B1 circuit breaker) previously only ran on the LATEST
snapshot (`_latest_signals`), so the realistic P&L of what the agents would ACTUALLY
trade was never backtested. This bridge closes that gap.

How
---
1. For every (symbol, decision_time) in `signals`, run the SAME skills the live path
   uses -> a per-symbol target position (edge gate, vol target, confidence haircut).
2. Per timestamp, apply `portfolio_risk_overlay` (correlation / cluster / gross /
   portfolio-vol). These pieces are path-INDEPENDENT (trailing PRICE stats only), so
   they belong in the one-shot panel. -> pass-1 weight panel W1.
3. Circuit breaker (B1) and the drawdown scaler ARE path-dependent (they react to the
   strategy's own equity). The engine is vectorized/one-shot, so we use the standard
   TWO-PASS decoupling: backtest W1 -> equity -> per-bar drawdown/daily-loss ->
   CircuitBreaker level + dd-scaler per bar -> multiply W1 -> W2. (Approximation: the
   breaker reacts to pass-1 equity, not post-breaker equity. A bit-faithful version
   needs a sequential event loop and is intentionally out of scope to keep the engine
   untouched.)
4. Hand the final panel to `run_vector_backtest(close, weight, config)` AS-IS.

Symbols, the 4h grid, execution_lag=1 and periods_per_year=bars_per_year all match the
convention `_panel_to_returns` already uses in incremental_study, so results are
directly comparable to the ladder steps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import run_vector_backtest, BacktestConfig
from backtest.metrics import calc_full_metrics

from crypto.skills.catalog import (check_data_quality, compute_confidence,
                                   risk_size_and_gate, portfolio_risk_overlay)
from crypto.risk.portfolio import equity_risk_metrics, _dd_scaler
from crypto.live.risk_guard import CircuitBreaker
from crypto.experiments.incremental_study import _cost_rates

# circuit-breaker level -> position multiplier (mirrors risk_size_and_gate's reaction)
CB_POS_MULT = {0: 1.0, 1: 1.0, 2: 0.5, 3: 0.0, 4: 0.0}


def _bt_config(fcfg, bars_per_year: int) -> BacktestConfig:
    fee, slip = _cost_rates(fcfg)
    return BacktestConfig(fee_rate=fee, slippage_rate=slip, execution_lag=1,
                          annual_periods=int(bars_per_year), market="crypto", timeframe="4h")


def _panel_returns(close_panel, w, fcfg, bars_per_year):
    cp = close_panel.reindex(w.index).ffill().dropna()
    w = w.reindex(cp.index).fillna(0.0)
    if len(cp) < 5:
        return pd.Series(dtype=float)
    res = run_vector_backtest(cp, w, config=_bt_config(fcfg, bars_per_year),
                              strategy_name="decision_stack")
    return res["returns"]


def build_decision_weight_panel(dataset, signals, close_panel, fcfg, *,
                                bars_per_year: int = 2190, apply_overlay: bool = True,
                                use_meta_gate: bool = False, sizing_mode: str = "xs_neutral",
                                smooth_bars: int = 24, deadband: float = 0.05):
    """Pass-1 target-weight panel, then portfolio overlay (cb_level=0, drawdown=0).
    Returns a signed wide panel (index=decision_time, columns=symbol).

    sizing_mode:
      "xs_neutral" (default) -- size by the CROSS-SECTIONAL rank of a SMOOTHED, BOUNDED
        conviction (NOT raw alpha). At each timestamp demean conviction across symbols
        and assign dollar-neutral weights (clipped to max_pos, scaled to gross_cap).
        This faithfully mirrors the pure-signal Step5 book. CRITICAL: the conviction is
        bounded (sign*clip(|a|/edge_cap,0,1)) and EWMA-smoothed over `smooth_bars`, then
        the final weights are smoothed AGAIN -- exactly like Step5. Using RAW alpha with
        NO smoothing (the earlier version) makes the book react to every noise wiggle and
        churn its full gross each bar; the resulting turnover/cost + noise-trading is what
        drove the stack to ~-126% even though it was dollar-neutral.
      "per_symbol" -- legacy: sizes each symbol independently via risk_size_and_gate.

    use_meta_gate (per_symbol mode only): gate on meta prob vs direction-only.
    smooth_bars/deadband: conviction transform params; match the deployment defaults so
        the decision stack and the pure-signal ladder see the same smoothed signal.
    """
    merged = dataset.merge(signals, on=["symbol", "decision_time"], how="left",
                           suffixes=("", "_sig")).dropna(subset=["combined_alpha"])

    # Pre-compute a SMOOTHED, BOUNDED conviction panel (mirrors Step5's _alpha_to_weight_panel)
    # so the per-bar xs_neutral sizing below uses the same low-turnover signal, not raw alpha.
    conv_panel = None
    if sizing_mode == "xs_neutral":
        edge_cap = float(getattr(fcfg.risk, "edge_cap", 0.2)) or 0.2
        _a = merged[["symbol", "decision_time", "combined_alpha"]].copy()
        _a["c"] = np.sign(_a["combined_alpha"].values) * np.clip(
            np.abs(_a["combined_alpha"].values) / max(edge_cap, 1e-9), 0.0, 1.0)
        _a.loc[np.abs(_a["combined_alpha"].values) < deadband, "c"] = 0.0
        conv_panel = _a.pivot_table(index="decision_time", columns="symbol",
                                    values="c", aggfunc="last").sort_index()
        if smooth_bars and smooth_bars > 1:
            conv_panel = conv_panel.ewm(span=int(smooth_bars), adjust=False).mean()
    dq_cols = [c for c in dataset.columns]

    funnel = {"rows": 0, "flat_or_no_meta": 0, "meta_gated": 0, "zero_size": 0,
              "approved_intents": 0, "timestamps_total": 0, "timestamps_with_intent": 0,
              "overlay_zeroed_ts": 0, "sizing_mode": sizing_mode}
    max_pos = float(fcfg.risk.max_pos_per_symbol)
    gross = float(fcfg.risk.gross_cap)
    weights_by_t = {}
    for dt, g in merged.groupby("decision_time"):
        funnel["timestamps_total"] += 1
        intents = {}
        if sizing_mode == "xs_neutral":
            # cross-sectional dollar-neutral sizing from SMOOTHED conviction (mirrors Step5)
            syms, alphas = [], []
            for row in g.itertuples(index=False):
                funnel["rows"] += 1
                r = row._asdict() if hasattr(row, "_asdict") else dict(zip(g.columns, row))
                sym = r["symbol"]
                # use the pre-smoothed bounded conviction for this (symbol, dt), not raw alpha
                cval = 0.0
                if conv_panel is not None and dt in conv_panel.index and sym in conv_panel.columns:
                    cv = conv_panel.at[dt, sym]
                    cval = float(cv) if cv == cv else 0.0   # NaN -> 0
                syms.append(sym); alphas.append(cval)
            a = np.array(alphas, dtype=float)
            a = np.where(np.isfinite(a), a, 0.0)
            if len(a) >= 2:
                a_dm = a - a.mean()                       # cross-sectional demean -> neutral
                denom = np.abs(a_dm).sum()
                if denom > 1e-12:
                    w = np.clip(a_dm / denom * gross, -max_pos, max_pos)
                    w = w - w.mean()                      # re-demean after clip -> restore neutrality
                    for sym, wi in zip(syms, w):
                        if abs(wi) > 1e-12:
                            funnel["approved_intents"] += 1
                            intents[sym] = {"target_position": float(wi),
                                            "direction": "long" if wi > 0 else "short"}
                        else:
                            funnel["zero_size"] += 1
        else:
            for row in g.itertuples(index=False):
                funnel["rows"] += 1
                r = row._asdict() if hasattr(row, "_asdict") else dict(zip(g.columns, row))
                fusion_out = {
                    "primary_direction": r.get("primary_direction", "flat"),
                    "combined_alpha": float(r.get("combined_alpha", 0.0) or 0.0),
                    "meta_trade_prob_calibrated": float(r.get("meta_trade_prob_calibrated", np.nan)),
                }
                dq = check_data_quality(r, [c for c in dq_cols if c in r])
                conf = compute_confidence(r, fusion_out, dq["data_quality_score"])["confidence"]
                risk = risk_size_and_gate(fusion_out, conf, r, fcfg, cb_level=0,
                                          use_meta_gate=use_meta_gate,
                                          bars_per_year=bars_per_year)
                reason = str(risk.get("reason", ""))
                if risk["risk_approved"] and abs(risk["target_position"]) > 1e-12:
                    funnel["approved_intents"] += 1
                    intents[r["symbol"]] = {"target_position": risk["target_position"],
                                            "direction": fusion_out["primary_direction"]}
                elif reason.startswith("flat") or reason == "flat_or_no_meta":
                    funnel["flat_or_no_meta"] += 1
                elif reason.startswith("meta_prob"):
                    funnel["meta_gated"] += 1
                else:
                    funnel["zero_size"] += 1
        if not intents:
            continue
        funnel["timestamps_with_intent"] += 1
        if apply_overlay:
            cp_t = close_panel.loc[close_panel.index <= dt]
            adj = portfolio_risk_overlay(intents, cp_t, fcfg, equity_drawdown=0.0,
                                         bars_per_year=bars_per_year)["adjusted_positions"]
            # The portfolio overlay's correlation haircut multiplies each symbol by a
            # DIFFERENT penalty, which breaks the dollar-neutrality of an xs_neutral book
            # (longs and shorts get scaled unequally -> a residual NET directional tilt).
            # No later overlay step re-centers it, so that unintended net exposure
            # compounds into large directional P&L (it flips the +Sharpe neutral book into
            # a big loss). Re-demean here to restore net=0 while keeping the overlay's
            # relative risk scaling intact.
            if sizing_mode == "xs_neutral" and adj:
                _vals = [v for v in adj.values() if v is not None]
                if _vals:
                    _m = float(np.mean(list(adj.values())))
                    adj = {s: (float(v) - _m) for s, v in adj.items()}
        else:
            adj = {s: v["target_position"] for s, v in intents.items()}
        if not any(abs(v) > 1e-12 for v in adj.values()):
            funnel["overlay_zeroed_ts"] += 1
        weights_by_t[dt] = adj

    if not weights_by_t:
        empty = pd.DataFrame(columns=close_panel.columns)
        empty.attrs["funnel"] = funnel
        return empty
    W1 = pd.DataFrame(weights_by_t).T.sort_index()
    W1 = W1.reindex(columns=close_panel.columns).fillna(0.0)
    # Second smoothing pass on the assembled weights + re-demean (mirrors Step5's final
    # EWMA): damps cross-sectional rank-flip churn that survives the per-bar construction,
    # then restores exact dollar-neutrality. This is the other half of Step5's turnover
    # control and is essential to stop the book from bleeding to costs.
    if sizing_mode == "xs_neutral" and smooth_bars and smooth_bars > 1 and len(W1) > 1:
        W1 = W1.ewm(span=int(smooth_bars), adjust=False).mean()
        W1 = W1.sub(W1.mean(axis=1), axis=0)
    W1.attrs["funnel"] = funnel
    return W1


def apply_circuit_breaker_to_panel(W1, close_panel, fcfg, *, bars_per_year: int = 2190,
                                   bars_per_day: int = 6, rolling_dd_days: int = 90,
                                   cb_dd_l1: float = None, cb_dd_l2: float = None,
                                   cb_dd_l3: float = None):
    """Two-pass B1: backtest W1 -> equity -> per-bar drawdown/daily-loss -> stateful
    CircuitBreaker level + drawdown scaler -> per-bar position multiplier. Returns
    (W2, multiplier_series, drawdown_series).

    cb_dd_l1/l2/l3: drawdown trip points for the breaker's L1/L2/L3. Defaults (None)
    fall back to the breaker's own defaults (0.10/0.15/0.20). Loosening these (e.g.
    0.25/0.35/0.45) is the lever for the risk-sensitivity analysis: crypto's 60%+ annual
    vol makes a 20% drawdown normal, so tight defaults over-trip and crush returns; the
    drawdown SCALER (dd_scale_start/stop) is moved in lockstep so the smooth delever and
    the discrete breaker stay consistent.

    Drawdown is measured against a ROLLING peak over the trailing `rolling_dd_days`
    days, NOT the inception-to-date peak. With an all-time peak, a single >20% drop
    early in the backtest would pin the breaker at L3 (mult=0) for the entire rest of
    the run -> the whole book stays flat forever. A rolling peak lets the breaker
    release once recent equity recovers or the old peak slides out of the window."""
    r1 = _panel_returns(close_panel, W1, fcfg, bars_per_year)
    if r1.empty:
        return W1, pd.Series(1.0, index=W1.index), pd.Series(0.0, index=W1.index)
    eq = (1.0 + r1).cumprod()
    win = max(2, int(rolling_dd_days * bars_per_day))
    roll_peak = eq.rolling(win, min_periods=1).max()
    dd_vec = (1.0 - eq / roll_peak)
    daily = (1.0 + r1).rolling(int(bars_per_day)).apply(np.prod, raw=True) - 1.0
    daily_loss_vec = (-daily).clip(lower=0.0).fillna(0.0)

    # build breaker with (optionally) loosened trip points
    cb_kwargs = {}
    if cb_dd_l1 is not None: cb_kwargs["dd_l1"] = cb_dd_l1
    if cb_dd_l2 is not None: cb_kwargs["dd_l2"] = cb_dd_l2
    if cb_dd_l3 is not None: cb_kwargs["dd_l3"] = cb_dd_l3
    cb = CircuitBreaker(**cb_kwargs)            # stateful: recovery hysteresis preserved
    # keep the smooth drawdown scaler consistent with the discrete breaker: move its
    # start/stop to the L1/L3 trip points when those are overridden.
    rcfg = fcfg.risk
    if cb_dd_l1 is not None or cb_dd_l3 is not None:
        import copy as _copy
        rcfg = _copy.copy(fcfg.risk)
        try:
            if cb_dd_l1 is not None: object.__setattr__(rcfg, "dd_scale_start", float(cb_dd_l1))
            if cb_dd_l3 is not None: object.__setattr__(rcfg, "dd_scale_stop", float(cb_dd_l3))
        except Exception:
            pass
    mult = pd.Series(1.0, index=r1.index)
    for t in r1.index:
        ddt = float(dd_vec.loc[t])
        lvl = int(cb.evaluate(drawdown=ddt, daily_loss=float(daily_loss_vec.loc[t])))
        mult.loc[t] = CB_POS_MULT.get(lvl, 1.0) * _dd_scaler(ddt, rcfg)
    mult = mult.reindex(W1.index).fillna(1.0)
    W2 = W1.mul(mult, axis=0)
    return W2, mult, dd_vec.reindex(W1.index)


def run_decision_backtest(dataset, signals, close_panel, fcfg, *, bars_per_year: int = 2190,
                          apply_overlay: bool = True, apply_circuit_breaker: bool = True,
                          rolling_dd_days: int = 90, use_meta_gate: bool = False,
                          sizing_mode: str = "xs_neutral",
                          smooth_bars: int = 24, deadband: float = 0.05,
                          cb_dd_l1: float = None, cb_dd_l2: float = None, cb_dd_l3: float = None,
                          strategy_name: str = "decision_stack", full_report: bool = False,
                          output_root: str = "data_storage/backtest_results"):
    """Backtest the FULL agent decision stack through the existing engine.

    rolling_dd_days: window (days) for the circuit-breaker drawdown peak; 90 by default
    so an early drawdown does not permanently lock the breaker (see
    apply_circuit_breaker_to_panel).

    use_meta_gate: False by default. The dev ladder (Step6) shows the meta-label gate
    hurts this book, so the decision stack sizes by signal conviction instead. Set True
    to reproduce the meta-gated behavior.

    full_report=True routes through backtest.quick.quick_backtest (quantstats HTML +
    plots; needs those deps). Default uses run_vector_backtest + calc_full_metrics
    (pure pandas/numpy) and returns a dict.
    """
    W1 = build_decision_weight_panel(dataset, signals, close_panel, fcfg,
                                     bars_per_year=bars_per_year, apply_overlay=apply_overlay,
                                     use_meta_gate=use_meta_gate, sizing_mode=sizing_mode,
                                     smooth_bars=smooth_bars, deadband=deadband)
    info = {"n_rebalances": int(len(W1)),
            "avg_gross": float(W1.abs().sum(axis=1).mean()) if len(W1) else 0.0,
            "funnel": W1.attrs.get("funnel", {})}
    W = W1
    if apply_circuit_breaker and len(W1):
        W, mult, dd = apply_circuit_breaker_to_panel(W1, close_panel, fcfg,
                                                     bars_per_year=bars_per_year,
                                                     rolling_dd_days=rolling_dd_days,
                                                     cb_dd_l1=cb_dd_l1, cb_dd_l2=cb_dd_l2,
                                                     cb_dd_l3=cb_dd_l3)
        info.update(cb_active_frac=float((mult < 1.0).mean()),
                    pass1_max_drawdown=float(dd.max()))

    cp = close_panel.reindex(W.index).ffill().dropna()
    W = W.reindex(cp.index).fillna(0.0)
    fee, slip = _cost_rates(fcfg)

    if full_report:
        from backtest.quick import quick_backtest
        res = quick_backtest(cp, W, strategy_name=strategy_name, fee_rate=fee, slippage_rate=slip,
                             execution_lag=1, annual_periods=bars_per_year,
                             market="crypto", timeframe="4h", output_root=output_root,
                             save=True, display=False)
        return {"result": res, "metrics": res.metrics, "weights": W, "info": info}

    raw = run_vector_backtest(cp, W, config=_bt_config(fcfg, bars_per_year),
                              strategy_name=strategy_name)
    metrics = calc_full_metrics(raw["returns"], turnover=raw["turnover"], cost=raw["cost"],
                                weights=raw["weights"], annual_periods=bars_per_year)
    return {"raw": raw, "metrics": metrics, "weights": W, "info": info}


def run_directional_decision_backtest(W_signal, close_panel, fcfg, *, bars_per_year: int = 2190,
                                      net_exposure_cap: float = 1.5, target_portfolio_vol: float = None,
                                      apply_circuit_breaker: bool = True, rolling_dd_days: int = 90,
                                      cb_dd_l1: float = None, cb_dd_l2: float = None, cb_dd_l3: float = None,
                                      strategy_name: str = "directional_stack"):
    """DIRECTIONAL decision stack for a NET-EXPOSURE book (e.g. Step7 ML+TSMOM).

    A directional book (net exposure != 0) needs the DIRECTIONAL risk paradigm, NOT the
    cross-sectional neutral overlay. Applying the neutral overlay's correlation-haircut /
    cluster-cap to a directional book would neutralize exactly the net-direction exposure
    that is its alpha. So here the controls are the directional-book equivalents:

      1. per-symbol cap        |w_i| <= max_pos_per_symbol            (concentration)
      2. net-exposure cap      |sum_i w_i| <= net_exposure_cap        (how directional)
      3. gross cap             sum_i |w_i| <= gross-budget            (leverage)
      4. total portfolio vol target: scale the whole book so realized vol = target
      5. circuit breaker:      rolling-drawdown delever               (tail protection)

    Takes a SIGNED weight panel (the strategy's intended positions) rather than a
    per-symbol alpha, because Step7 is already a constructed book (neutral sleeve +
    TSMOM sleeve). Returns the same dict shape as run_decision_backtest.
    """
    W = W_signal.sort_index().copy()
    max_pos = float(fcfg.risk.max_pos_per_symbol)
    gross_cap = float(fcfg.risk.gross_cap)
    tgt_vol = float(target_portfolio_vol if target_portfolio_vol is not None
                    else fcfg.risk.target_portfolio_vol)

    # 1) per-symbol concentration cap
    W = W.clip(lower=-max_pos, upper=max_pos)
    # 2) net-exposure cap: if |net| exceeds the cap at a bar, shrink the COMMON (net)
    #    component only, preserving the cross-sectional (relative) structure.
    net = W.sum(axis=1)
    n_sym = max(W.shape[1], 1)
    over = net.abs() > net_exposure_cap
    if over.any():
        excess = (net.abs() - net_exposure_cap).clip(lower=0.0) * np.sign(net)
        W = W.sub((excess / n_sym).where(over, 0.0), axis=0)
    # 3) gross cap
    gross = W.abs().sum(axis=1).replace(0.0, np.nan)
    scale = (gross_cap / gross).clip(upper=1.0).fillna(1.0)
    W = W.mul(scale, axis=0)
    # 4) total portfolio vol target (single scalar from realized vol; risk normalization,
    #    no directional lookahead)
    r0 = _panel_returns(close_panel, W, fcfg, bars_per_year)
    rv = float(r0.std() * (bars_per_year ** 0.5)) if len(r0) else 0.0
    if rv > 1e-9:
        W = W * float(min(tgt_vol / rv, float(fcfg.risk.max_vol_scalar)))

    info = {"avg_gross": float(W.abs().sum(axis=1).mean()) if len(W) else 0.0,
            "avg_net_exposure": float(W.sum(axis=1).mean()) if len(W) else 0.0,
            "avg_abs_net_exposure": float(W.sum(axis=1).abs().mean()) if len(W) else 0.0,
            "net_exposure_cap": net_exposure_cap, "target_portfolio_vol": tgt_vol,
            "paradigm": "directional"}
    if apply_circuit_breaker and len(W):
        W, mult, dd = apply_circuit_breaker_to_panel(W, close_panel, fcfg,
                                                     bars_per_year=bars_per_year,
                                                     rolling_dd_days=rolling_dd_days,
                                                     cb_dd_l1=cb_dd_l1, cb_dd_l2=cb_dd_l2,
                                                     cb_dd_l3=cb_dd_l3)
        info.update(cb_active_frac=float((mult < 1.0).mean()), pass1_max_drawdown=float(dd.max()))

    cp = close_panel.reindex(W.index).ffill().dropna()
    W = W.reindex(cp.index).fillna(0.0)
    raw = run_vector_backtest(cp, W, config=_bt_config(fcfg, bars_per_year), strategy_name=strategy_name)
    metrics = calc_full_metrics(raw["returns"], turnover=raw["turnover"], cost=raw["cost"],
                                weights=raw["weights"], annual_periods=bars_per_year)
    return {"raw": raw, "metrics": metrics, "weights": W, "info": info}


# --------------------------------------------------------------------------- #
#  Decision-stack  vs  pure-signal ladder steps  (for the defense slide)
# --------------------------------------------------------------------------- #
def _ann_metrics(returns, bars_per_year: int) -> dict:
    """Annualized headline metrics from a bare per-bar return series."""
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 5:
        return {"n": int(len(r)), "sharpe": float("nan"), "ann_return": float("nan"),
                "ann_vol": float("nan"), "max_drawdown": float("nan"), "calmar": float("nan")}
    mu = float(r.mean() * bars_per_year)
    vol = float(r.std(ddof=1) * (bars_per_year ** 0.5))
    eq = (1.0 + r).cumprod()
    mdd = float((1.0 - eq / eq.cummax()).max())
    return {"n": int(len(r)),
            "sharpe": float(mu / vol) if vol > 1e-12 else float("nan"),
            "ann_return": mu, "ann_vol": vol, "max_drawdown": mdd,
            "calmar": float(mu / mdd) if mdd > 1e-12 else float("nan")}


def compare_decision_vs_signal(decision_returns, signal_returns: dict,
                               bars_per_year: int = 2190, align: bool = True) -> dict:
    """Side-by-side: the FULL decision stack (per-symbol risk sizing + portfolio
    overlay + circuit breaker) vs the PURE-SIGNAL ladder deployments (Step5_fusion,
    Step6_meta_gate, Step0 TSMOM benchmark). This isolates exactly what the risk
    machinery did to the same underlying signal — smoothed the equity, or killed it.

    align=True restricts every series to the bars it shares with the decision stack,
    so the comparison is apples-to-apples over identical timestamps.
    """
    dec = pd.Series(decision_returns).dropna().astype(float)
    series = {"decision_stack(risk+port+CB)": dec}
    for name, s in (signal_returns or {}).items():
        if s is None:
            continue
        series[name] = pd.Series(s).dropna().astype(float)

    common = dec.index
    if align:
        for s in series.values():
            common = common.intersection(s.index)
        if len(common) >= 5:
            series = {k: v.reindex(common).dropna() for k, v in series.items()}

    table = {k: _ann_metrics(v, bars_per_year) for k, v in series.items()}
    # deltas of the decision stack vs each pure-signal series
    base = table["decision_stack(risk+port+CB)"]
    deltas = {}
    for k, m in table.items():
        if k == "decision_stack(risk+port+CB)":
            continue
        deltas[k] = {"d_sharpe": base["sharpe"] - m["sharpe"],
                     "d_ann_return": base["ann_return"] - m["ann_return"],
                     "d_max_drawdown": base["max_drawdown"] - m["max_drawdown"]}
    return {"aligned": bool(align), "aligned_bars": int(len(common)) if align else None,
            "bars_per_year": bars_per_year, "metrics": table, "delta_vs_signal": deltas}


def format_comparison_table(cmp: dict) -> str:
    """Pretty fixed-width table for the console / defense screenshot."""
    m = cmp["metrics"]
    hdr = f"{'strategy':<34}{'Sharpe':>8}{'AnnRet':>9}{'Vol':>8}{'MaxDD':>8}{'Calmar':>8}"
    lines = ["", f"    DECISION STACK vs PURE SIGNAL  (aligned bars={cmp.get('aligned_bars')})",
             "    " + hdr, "    " + "-" * len(hdr)]
    for name, v in m.items():
        lines.append("    " + f"{name:<34}{v['sharpe']:>8.2f}{100*v['ann_return']:>8.1f}%"
                     f"{100*v['ann_vol']:>7.1f}%{100*v['max_drawdown']:>7.1f}%{v['calmar']:>8.2f}")
    d = cmp.get("delta_vs_signal", {})
    if d:
        lines.append("    " + "-" * len(hdr))
        for name, dd in d.items():
            lines.append("    " + f"{'Δ stack − ' + name:<34}{dd['d_sharpe']:>8.2f}"
                         f"{100*dd['d_ann_return']:>8.1f}%{'':>8}{100*dd['d_max_drawdown']:>7.1f}%{'':>8}")
        lines.append("    (Δ MaxDD > 0 means the stack REDUCED drawdown; Δ Sharpe < 0 means it "
                     "cost risk-adjusted return.)")
    return "\n".join(lines)
