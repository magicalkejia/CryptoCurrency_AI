"""
run_on_real_data.py — REAL-DATA market experiment entry point
=============================================================
This is THE entry point for the market-only experiment (LightGBM + PatchTST) on
your already-processed parquet. It supersedes the synthetic *_demo.py scripts:
the experiment machinery (incremental ladder / A·B·C·D ablation / PBO) that those
demos exercised on SYNTHETIC data now runs here on your REAL multi-timeframe data.

Data contract (matches your processed parquet):
    main sequence   : {SYMBOL}_4h.parquet   (decision frequency, §1.2)
    auxiliary state : {SYMBOL}_1h.parquet    (short-period features)
    environment     : {SYMBOL}_1d.parquet    (mid/long-term regime)
    funding (hook)  : {SYMBOL}_funding.parquet (optional; auto-detected when ready)

Pipeline (all reuse existing project functions; nothing re-implemented):
  1. multi-timeframe PIT dataset  (etl.dataset_builder.build_market_dataset)
  2. PIT leakage audit            (crypto.pit.audit_lookahead)
  3. incremental proof ladder     (crypto.experiments.incremental_study) — onchain/
     narrative auto-skipped until the colleague's data arrives
  4. PatchTST A/B/C/D ablation    (crypto.experiments.patchtst_ablation)
  5. PBO (CSCV)                    (crypto.governance.pbo)
  6. two-stage meta-label signals (crypto.pipeline_1b.run_phase1b) + ECE
  7. §1.4 structured signals       (crypto.skills.catalog)
  8. governance freeze + register (crypto.governance.holdout / registry)
     — Holdout-A is NOT run here; it must be run ONCE later, frozen.

Usage
-----
    # REAL data (needs pyarrow + lightgbm; torch optional but recommended)
    python run_on_real_data.py --symbols BTC/USDT ETH/USDT SOL/USDT BNB/USDT

    # quick offline self-test (no parquet / heavy libs) — validates the wiring
    python run_on_real_data.py --synthetic
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from backtest.annualization import infer_annual_periods
from crypto.schemas import FrozenConfig, environment_hash, make_audit_id
from crypto.adapters import to_bars_schema
from etl.dataset_builder import build_market_dataset, DEFAULT_FEATURE_SET
from crypto.pipeline_1b import run_phase1b
from crypto.experiments.incremental_study import run_incremental_study, per_year_sharpe
from crypto.experiments.patchtst_ablation import run_ablation, _oof_alpha
from crypto.cv.purged_kfold import purged_embargoed_splits, default_embargo_delta
from crypto.governance.pbo import cscv_pbo
from crypto.governance.holdout import dev_holdout_split, freeze_config, load_frozen
from crypto.governance.registry import pre_register, assert_preregistered
from crypto.skills.catalog import (get_feature_row, check_data_quality, detect_regime,
                                   compute_confidence, risk_size_and_gate,
                                   compute_circuit_breaker, portfolio_risk_overlay)
from crypto.risk.portfolio import equity_risk_metrics
from crypto.live.risk_guard import CircuitBreaker

BARS_PER_YEAR_4H = infer_annual_periods(timeframe="4h", market="crypto")


# --------------------------------------------------------------------------- #
# synthetic provider — exercises the EXACT real code path without parquet
# --------------------------------------------------------------------------- #
class _Tee:
    """Duplicate stdout to a log file so the full console run is archived alongside the
    other experiment artifacts (outdir/console_log.txt). Line-buffered + flushed."""
    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "w", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def _synthetic_bars(seed: int, timeframe: str):
    rng = np.random.default_rng(seed + {"1h": 0, "4h": 1, "1d": 2}[timeframe])
    n, freq = {"1h": (24 * 320, "1h"), "4h": (6 * 320, "4h"), "1d": (320, "1D")}[timeframe]
    idx = pd.date_range("2022-01-01", periods=n, freq=freq)
    ret = rng.normal(0, 0.01, n)
    for s in range(0, n, max(n // 16, 1)):
        ret[s:s + n // 16] += rng.normal(0, 0.0008)
    close = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame({
        "open": np.r_[close[0], close[:-1]], "high": close * 1.004, "low": close * 0.996,
        "close": close, "volume": rng.lognormal(10, 0.4, n),
        "taker_buy_vol": rng.lognormal(9, 0.4, n), "net_taker_vol": rng.normal(0, 1, n),
    }, index=idx)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _xs_regime_by_year(close_panel):
    """PURE DIAGNOSTIC (no strategy/params touched). Per calendar year, quantify how
    'together' the universe moves -- a cross-sectional (relative) book needs DISPERSION
    to work, so a dispersion collapse mechanically weakens it. Three scale-aware /
    scale-free views, all on dev-period returns:
      avg_pair_corr : mean off-diagonal pairwise return correlation (higher = worse)
      xs_dispersion : mean over time of the cross-sectional std of returns, i.e. the
                      residual spread AROUND the common move that a neutral book trades
                      (lower = worse)
      pc1_var_share : share of variance in the 1st principal component / common mode
                      (higher = common mode dominates = less residual = worse)."""
    ret = close_panel.pct_change().replace([np.inf, -np.inf], np.nan)
    ret.index = pd.to_datetime(ret.index)
    out = {}
    for yr, grp in ret.groupby(ret.index.year):
        g = grp.dropna(axis=1, thresh=max(20, int(0.5 * len(grp))))  # symbols present >=half the year
        g = g.dropna(axis=0, how="any")
        if g.shape[1] < 2 or len(g) < 20:
            out[int(yr)] = {"avg_pair_corr": float("nan"), "xs_dispersion": float("nan"),
                            "pc1_var_share": float("nan"), "n_sym": g.shape[1]}
            continue
        C = g.corr().to_numpy()
        n = C.shape[0]
        avg_corr = float(np.nanmean(C[~np.eye(n, dtype=bool)]))
        xs_disp = float(g.std(axis=1).mean())
        eig = np.linalg.eigvalsh(np.nan_to_num(C, nan=0.0))
        pc1 = float(eig[-1] / eig.sum()) if eig.sum() > 1e-12 else float("nan")
        out[int(yr)] = {"avg_pair_corr": avg_corr, "xs_dispersion": xs_disp,
                        "pc1_var_share": pc1, "n_sym": n}
    return out


def _pbo_over_abcd(ds, tabular_cols, fcfg):
    """PBO (CSCV) over the four ablation configs' OOF sign-strategy returns."""
    forecast = [c for c in ds.columns if c.startswith("patchtst_forecast_")]
    emb = [c for c in ds.columns if c.startswith("patchtst_emb_")]
    cols = {"A": tabular_cols, "B": forecast, "C": tabular_cols + emb,
            "D": tabular_cols + emb + forecast}
    cols = {k: v for k, v in cols.items() if v}
    if len(cols) < 2:
        return {"pbo": float("nan"), "n_combinations": 0}
    splits = purged_embargoed_splits(ds["decision_time"], ds["entry_time"], ds["exit_time"],
                                     fcfg.cv.n_splits, default_embargo_delta(fcfg.cv),
                                     symbol=ds["symbol"])
    fwd = ds["raw_exit_return_long"].to_numpy(float)
    streams = []
    for c in cols.values():
        a = _oof_alpha(ds, c, fcfg, splits)
        streams.append(np.where(np.isfinite(a), np.sign(np.nan_to_num(a)) * fwd, 0.0))
    return cscv_pbo(np.column_stack(streams), n_blocks=8)


def _pbo_over_grid(ds, tabular_cols, fcfg, n_blocks=12):
    """CSCV PBO over a WIDER grid of model-hyperparameter configs (the textbook CSCV
    setup). More reliable than the 4-config ablation PBO, which is noisy when A/C/D are
    near-tied (all DSR~1.0 -> 'which is best' is a coin-flip -> PBO drifts to ~0.5).
    Each grid point is a perturbation of the frozen LightGBM config; we build its OOF
    sign-strategy return stream on the MARKET feature set and run CSCV over all streams.
    This does NOT change the deployed strategy -- it only stress-tests how much
    'selecting the best config' would overfit. OOF + purged/embargoed throughout."""
    if not tabular_cols:
        return {"pbo": float("nan"), "n_combinations": 0, "n_configs": 0}
    grid = [(d, n, lr)
            for d in (3, 4, 5)            # capacity (depth)
            for n in (150, 300)           # capacity (trees)
            for lr in (0.03, 0.08)]       # learning rate  -> 3*2*2 = 12 configs
    splits = purged_embargoed_splits(ds["decision_time"], ds["entry_time"], ds["exit_time"],
                                     fcfg.cv.n_splits, default_embargo_delta(fcfg.cv),
                                     symbol=ds["symbol"])
    fwd = ds["raw_exit_return_long"].to_numpy(float)
    streams = []
    for (d, n, lr) in grid:
        fc = replace(fcfg, model=replace(fcfg.model, max_depth=d, n_estimators=n,
                                         learning_rate=lr))
        a = _oof_alpha(ds, tabular_cols, fc, splits)
        streams.append(np.where(np.isfinite(a), np.sign(np.nan_to_num(a)) * fwd, 0.0))
    res = cscv_pbo(np.column_stack(streams), n_blocks=n_blocks)
    res["n_configs"] = len(grid)
    # per-config mean per-trade edge (bps) -> spread tells near-tie from genuine spread:
    # if all 12 configs have ~identical edge, the signal is hyperparam-robust and a high
    # PBO is a near-tie artifact (NOT overfitting); a wide spread + high PBO would instead
    # mean the IS-best config genuinely overfits (which would justify freezing by principle).
    edges = [float(np.mean(s)) * 1e4 for s in streams]
    res["edge_bps_min"], res["edge_bps_max"] = min(edges), max(edges)
    res["edge_bps_spread"] = max(edges) - min(edges)
    return res


def _latest_signals(ds, signals, fcfg, code_hash="experiment", close_panel=None,
                    equity_returns=None, bars_per_year=BARS_PER_YEAR_4H):
    """§1.4-style structured signal for the latest decision per symbol, reusing the
    existing Skills (regime + confidence + risk sizing) PLUS the new B1 circuit
    breaker and A2 portfolio overlay. Returns (rows, portfolio_report)."""
    merged = ds.merge(signals, on=["symbol", "decision_time"], how="left", suffixes=("", "_sig"))

    # --- B1: circuit-breaker level from the deployable equity curve ---------- #
    cb_level, cb_reason, dd = 0, "ok", 0.0
    if equity_returns is not None and len(equity_returns) > 0:
        dd, daily_loss, roll_abs = equity_risk_metrics(equity_returns, bars_per_day=6,
                                                        rolling_dd_days=90)
        out_cb = compute_circuit_breaker(CircuitBreaker(), drawdown=dd, daily_loss=daily_loss,
                                         rolling_abs_daily_returns=roll_abs)
        cb_level, cb_reason = out_cb["cb_level"], out_cb["cb_reason"]

    # --- pass 1: per-symbol intents (risk sizing with the shared cb_level) --- #
    rows, intents = [], {}
    for sym, g in merged.groupby("symbol"):
        g = g.dropna(subset=["combined_alpha"])
        if g.empty:
            continue
        row = g.sort_values("decision_time").iloc[-1]
        dt = row["decision_time"]
        feat_row = get_feature_row(merged, sym, dt)
        fusion_out = {"primary_direction": row.get("primary_direction", "flat"),
                      "combined_alpha": float(row.get("combined_alpha", 0.0)),
                      "meta_trade_prob_calibrated": float(row.get("meta_trade_prob_calibrated", np.nan))}
        dq = check_data_quality(feat_row, [c for c in ds.columns if c in feat_row])
        regime = detect_regime(feat_row)["regime"]
        conf = compute_confidence(feat_row, fusion_out, dq["data_quality_score"])["confidence"]
        risk = risk_size_and_gate(fusion_out, conf, feat_row, fcfg, cb_level=cb_level)
        rows.append({"sym": sym, "dt": dt, "fusion_out": fusion_out, "regime": regime,
                     "conf": conf, "dq": dq, "risk": risk})
        if risk["risk_approved"]:
            intents[sym] = {"target_position": risk["target_position"],
                            "direction": fusion_out["primary_direction"]}

    # --- A2: cross-symbol portfolio overlay (PIT close panel up to last bar) - #
    cp = None
    if close_panel is not None and rows:
        last_dt = max(r["dt"] for r in rows)
        cp = close_panel.loc[close_panel.index <= last_dt]
    adjusted, port_report = {}, {"note": "no_intents"}
    if intents:
        ov = portfolio_risk_overlay(intents, cp, fcfg, equity_drawdown=dd,
                                    bars_per_year=bars_per_year)
        adjusted, port_report = ov["adjusted_positions"], ov["portfolio_report"]
    port_report["circuit_breaker_level"] = cb_level
    port_report["circuit_breaker_reason"] = cb_reason

    # --- pass 2: structured output with post-overlay positions --------------- #
    out = []
    for r in rows:
        sym, dt, fusion_out, risk = r["sym"], r["dt"], r["fusion_out"], r["risk"]
        pre = float(risk["target_position"])
        final = float(adjusted.get(sym, pre if risk["risk_approved"] else 0.0))
        approved = risk["risk_approved"] and abs(final) > 1e-9
        out.append({
            "decision_time": pd.Timestamp(dt).isoformat(),
            "symbol": sym.replace("/", ""),
            "regime": r["regime"],
            "combined_alpha": round(fusion_out["combined_alpha"], 4),
            "primary_direction": fusion_out["primary_direction"],
            "meta_trade_prob_calibrated": (None if np.isnan(fusion_out["meta_trade_prob_calibrated"])
                                           else round(fusion_out["meta_trade_prob_calibrated"], 4)),
            "confidence": round(r["conf"], 4),
            "action": fusion_out["primary_direction"] if approved else "flat",
            "target_position": round(final, 4),
            "target_position_pre_portfolio": round(pre, 4),
            "vol_target_scalar": round(risk.get("vol_target_scalar", 0.0), 4),
            "stop_loss": round(risk["stop_loss"], 4),
            "take_profit": round(risk["take_profit"], 4),
            "barrier_source": "ATR20_1h_at_decision_time",
            "risk_level": risk["risk_level"],
            "circuit_breaker_level": cb_level,
            "data_quality_score": round(r["dq"]["data_quality_score"], 4),
            "reason": risk["reason"],
            "audit_id": make_audit_id(sym, dt, "experiment_oof", "processed_parquet", code_hash, fcfg),
        })
    return out, port_report


def _fmt(x, p="{:+.3f}"):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else p.format(x)


# --------------------------------------------------------------------------- #
# acceptance + report
# --------------------------------------------------------------------------- #
def _acceptance(md, ladder, ablation, pbo, diag):
    lines, ok = [], {}
    pit = md.audit["future_function_checks_passed"]
    ok["pit"] = pit
    lines.append(f"[{'PASS' if pit else 'FAIL'}] PIT no-leakage audit")
    ece = diag.get("ece_calibrated", float("nan"))
    ece_ok = np.isfinite(ece) and ece <= 0.05
    ok["ece"] = ece_ok
    lines.append(f"[{'PASS' if ece_ok else 'CHECK'}] calibration ECE<=0.05  (ece_calibrated={_fmt(ece,'{:.4f}')})")
    inc = ladder[ladder.index != "Step0_baseline_tsmom"]
    any_incr = bool(inc["included"].fillna(False).any())
    ok["increment"] = any_incr
    lines.append(f"[{'PASS' if any_incr else 'NOTE'}] at least one modality shows significant increment "
                 f"(incr_NW_t>2). If none: honest 'no-increment' result is valid (§1.3).")
    if {"A_tabular"}.issubset(ablation.index):
        a_t = ablation.loc["A_tabular", "IC_NW_t"]
        cd = [c for c in ["C_emb_plus_tabular", "D_full_fusion"] if c in ablation.index]
        best_cd = max((ablation.loc[c, "IC_NW_t"] for c in cd), default=float("nan"))
        patch_ok = np.isfinite(best_cd) and np.isfinite(a_t) and best_cd > a_t
        ok["patchtst"] = patch_ok
        lines.append(f"[{'PASS' if patch_ok else 'NOTE'}] PatchTST adds value: best(C/D) IC_NW_t "
                     f"{_fmt(best_cd)} vs A {_fmt(a_t)}")
    p = pbo.get("pbo", float("nan"))
    pbo_ok = np.isfinite(p) and p < 0.5
    ok["pbo"] = pbo_ok
    lines.append(f"[{'PASS' if pbo_ok else 'CHECK'}] PBO<0.5  (pbo={_fmt(p,'{:.3f}')})")
    if "Step0_baseline_tsmom" in ladder.index:
        base_sh = ladder.loc["Step0_baseline_tsmom", "sharpe_ann"]
        best_sh = ladder["sharpe_ann"].drop("Step0_baseline_tsmom", errors="ignore").max()
        beat = np.isfinite(best_sh) and np.isfinite(base_sh) and best_sh > base_sh
        ok["beats_tsmom"] = beat
        lines.append(f"[{'PASS' if beat else 'NOTE'}] beats TSMOM baseline (best Sharpe {_fmt(best_sh)} "
                     f"vs TSMOM {_fmt(base_sh)})")
    lines.append("")
    lines.append("Reminder: these are DEVELOPMENT-period numbers. The main grade comes from the "
                 "FROZEN Holdout-A, run exactly once, later.")
    return {"lines": lines, "ok": ok}


def _write_report(outdir, stamp, fcfg, md, ladder, ablation, pbo, diag, checks, latest, args):
    L = [f"# Market Experiment Report — {stamp}\n",
         f"- mode: **{'SYNTHETIC (wiring check, NOT a real result)' if args.synthetic else 'REAL parquet'}**",
         f"- symbols: {', '.join(args.symbols)}",
         f"- config_hash: `{fcfg.config_hash()}`  ·  env_hash: `{environment_hash()}`",
         f"- dataset rows: {len(md.dataset)}  ·  market feats: {len(md.tabular_cols)}  "
         f"·  PatchTST feats: {len(md.modality_cols['patchtst'])}",
         "\n## Acceptance checklist\n"]
    L += [f"- {ln}" for ln in checks["lines"] if ln]
    L += ["\n## Incremental ladder (Step0->Step6)\n", "```\n" + ladder.round(4).to_string() + "\n```",
          "\n## PatchTST A/B/C/D ablation\n", "```\n" + ablation.round(4).to_string() + "\n```",
          f"\n## PBO\n\n- PBO = **{_fmt(pbo['pbo'], '{:.3f}')}** over {pbo['n_combinations']} combinations",
          "\n## Phase-1b diagnostics\n"]
    L += [f"- {k}: {v}" for k, v in diag.items()]
    L += ["\n## Latest structured signals (§1.4)\n",
          "```json\n" + json.dumps(latest, indent=2, ensure_ascii=False) + "\n```"]
    (outdir / "EXPERIMENT_report.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _split_cut(ds, holdout_frac):
    dt = ds["decision_time"].sort_values().reset_index(drop=True)
    return dt.iloc[int(len(dt) * (1 - holdout_frac))]


def _run_holdout(md, cut, fcfg, outdir, args):
    """FINAL TEST — run the frozen Holdout-A exactly once.

    Train ONE alpha model on the development period (everything before `cut`),
    then predict the untouched holdout (>= cut) and score it. This consumes
    Holdout-A; per v6 §6.4 it must be run a single time, with frozen config.
    """
    from crypto.models.base_lgb import MultiClassLearner
    from crypto.benchmark.tsmom import vol_parity_tsmom_weights
    from backtest.engine import run_vector_backtest, BacktestConfig
    from crypto.eval.significance import deflated_sharpe_ratio, information_coefficient

    ds = md.dataset
    feat = md.feature_cols
    dev = ds[ds["decision_time"] < cut].reset_index(drop=True)
    hold = ds[ds["decision_time"] >= cut].reset_index(drop=True)
    if len(hold) < 20:
        print("    holdout too small — abort."); return

    m = MultiClassLearner(fcfg.model).fit(dev[feat].to_numpy(float),
                                          dev["tb_label"].to_numpy(int),
                                          dev["uniqueness_weight"].fillna(1.0).to_numpy(float))
    pdn, pne, pup = m.predict_proba_df(hold[feat].to_numpy(float))
    alpha = pup - pdn

    fee = fcfg.cost.fee_bps / 1e4
    slip = (fcfg.cost.base_slippage_bps + fcfg.cost.spread_proxy_bps) / 1e4
    cfg = BacktestConfig(fee_rate=fee, slippage_rate=slip, execution_lag=1,
                         annual_periods=BARS_PER_YEAR_4H, market="crypto", timeframe="4h")

    def _port(raw_w):   # same LOW-TURNOVER deployment as the dev ladder
        tmp = hold[["symbol", "decision_time"]].copy(); tmp["w"] = raw_w
        wp = tmp.pivot_table(index="decision_time", columns="symbol", values="w", aggfunc="last").fillna(0)
        if args.smooth_bars and args.smooth_bars > 1:
            wp = wp.ewm(span=int(args.smooth_bars), adjust=False).mean()
        cp = md.close_panel.reindex(wp.index).ffill().dropna()
        wp = wp.reindex(cp.index).fillna(0.0)
        if len(cp) < 5:
            return float("nan"), pd.Series(dtype=float), cp, float("nan"), float("nan")
        res = run_vector_backtest(cp, wp, config=cfg, strategy_name="holdout")
        r = res["returns"]; sh = r.mean() / r.std() * np.sqrt(BARS_PER_YEAR_4H) if r.std() > 0 else float("nan")
        return (sh, r, cp, float(res["turnover"].mean() * BARS_PER_YEAR_4H),
                float(res["cost"].mean() * BARS_PER_YEAR_4H))

    raw = (np.abs(alpha) >= args.deadband).astype(float) * np.sign(alpha) * fcfg.risk.max_pos_per_symbol
    sharpe, ret, cp, ann_turn, ann_cost = _port(raw)
    dsr = deflated_sharpe_ratio(sharpe / np.sqrt(BARS_PER_YEAR_4H) if np.isfinite(sharpe) else float("nan"),
                                n_obs=len(ret), n_trials=1)
    fwd = hold["raw_exit_return_long"].to_numpy(float); mk = np.isfinite(alpha) & np.isfinite(fwd)
    ic = information_coefficient(alpha[mk], fwd[mk])

    # TSMOM baseline on the SAME holdout window
    tw = vol_parity_tsmom_weights(cp, lookback_mom=90, vol_window=30, cov_window=30, bars_per_year=BARS_PER_YEAR_4H)
    tret = run_vector_backtest(cp, tw, config=cfg, strategy_name="tsmom")["returns"]
    tsh = tret.mean() / tret.std() * np.sqrt(BARS_PER_YEAR_4H) if tret.std() > 0 else float("nan")

    dirser = pd.Series(np.where(alpha > fcfg.theta_long, "long",
                                np.where(alpha < -fcfg.theta_short, "short", "flat")))
    dirdist = dirser.value_counts(normalize=True).round(4).to_dict()

    # ------------------------------------------------------------------ #
    #  FULL SAMPLE-OUT SUITE (same strategies + metrics as the dev study) #
    #  Built on the holdout window only, frozen config, ONE model fit.    #
    #  This is the confirmatory analog of the dev real-returns table so   #
    #  the report can analyze Step5 / Step7 / +stacks out-of-sample with  #
    #  the full metric set (return / risk / risk-adjusted / trading).     #
    # ------------------------------------------------------------------ #
    holdout_suite = {}
    try:
        from crypto.experiments.incremental_study import (
            _alpha_to_weight_panel, _tsmom_weight_panel, _risk_sleeve_combine, _panel_to_returns)
        from crypto.eval.decision_backtest import (
            run_decision_backtest, run_directional_decision_backtest)
        from backtest.metrics import calc_full_metrics

        # alpha panel on the holdout (the cross-sectional ML signal = Step5 core)
        hsig = hold[["symbol", "decision_time"]].copy()
        hsig["combined_alpha"] = alpha
        hsig["primary_direction"] = dirser.values
        hsig["meta_trade_prob_calibrated"] = np.nan

        def _full(rseries, turn=None, cost=None):
            r = pd.Series(rseries).dropna()
            m = calc_full_metrics(r, turnover=turn, cost=cost, annual_periods=BARS_PER_YEAR_4H)
            eq = (1.0 + r).cumprod()
            return {"ann_return": m.get("strategy_annual_return"),
                    "cum_return": float(eq.iloc[-1] - 1.0) if len(eq) else float("nan"),
                    "ann_vol": m.get("strategy_volatility"),
                    "max_drawdown": abs(m.get("strategy_max_drawdown")) if m.get("strategy_max_drawdown") is not None else None,
                    "sharpe": m.get("strategy_sharpe"), "sortino": m.get("strategy_sortino"),
                    "calmar": m.get("strategy_calmar"), "win_rate": m.get("win_rate"),
                    "profit_loss_ratio": m.get("profit_loss_ratio"),
                    "ann_turnover": m.get("annual_turnover"),
                    "ann_cost_pct": (m.get("avg_period_cost") * BARS_PER_YEAR_4H * 100
                                     if m.get("avg_period_cost") is not None else None)}

        # Step5: cross-sectional neutral book from the ML alpha
        w5 = _alpha_to_weight_panel(hold, alpha, cp, fcfg, BARS_PER_YEAR_4H,
                                    smooth_bars=args.smooth_bars, deadband=args.deadband,
                                    meta_prob=None, deploy_mode="xs_neutral",
                                    max_vol_scale=fcfg.risk.max_vol_scalar,
                                    no_trade_band=args.no_trade_band, allow_flat=True)
        r5, t5, c5 = _panel_to_returns(cp, w5, BARS_PER_YEAR_4H, fcfg)
        holdout_suite["Step5_fusion (pure signal)"] = _full(r5)

        # Step7: Step5 neutral sleeve + TSMOM sleeve, equal-risk combine
        w7 = _risk_sleeve_combine(cp, w5, tw, BARS_PER_YEAR_4H, fcfg)
        r7, t7, c7 = _panel_to_returns(cp, w7, BARS_PER_YEAR_4H, fcfg)
        holdout_suite["Step7_tsmom_fusion (MAIN deliverable)"] = _full(r7)

        # Step5 + NEUTRAL decision stack
        dz = run_decision_backtest(hold, hsig, cp, fcfg, bars_per_year=BARS_PER_YEAR_4H,
                                   apply_overlay=True, apply_circuit_breaker=True,
                                   sizing_mode="xs_neutral", smooth_bars=args.smooth_bars,
                                   deadband=args.deadband, cb_dd_l1=args.cb_dd_start,
                                   cb_dd_l2=args.cb_dd_l2, cb_dd_l3=args.cb_dd_stop,
                                   strategy_name="holdout_decision_stack")
        holdout_suite["Step5 + neutral decision stack"] = _full(
            dz["raw"]["returns"], dz["raw"].get("turnover"), dz["raw"].get("cost"))

        # Step7 + DIRECTIONAL decision stack
        dd = run_directional_decision_backtest(w7, cp, fcfg, bars_per_year=BARS_PER_YEAR_4H,
                                               net_exposure_cap=1.5, apply_circuit_breaker=True,
                                               cb_dd_l1=args.cb_dd_start, cb_dd_l2=args.cb_dd_l2,
                                               cb_dd_l3=args.cb_dd_stop,
                                               strategy_name="holdout_directional_stack")
        holdout_suite["Step7 + directional stack (full risk)"] = _full(
            dd["raw"]["returns"], dd["raw"].get("turnover"), dd["raw"].get("cost"))
        holdout_suite["_directional_info"] = dd["info"]

        # TSMOM benchmark
        holdout_suite["Step0_TSMOM (benchmark)"] = _full(tret)
    except Exception as e:
        print(f"    [holdout suite] full-suite build failed ({e}); core verdict still valid")
        import traceback; traceback.print_exc()

    print("\n" + "!" * 74)
    print(" HOLDOUT-A FINAL TEST (consumes the holdout; run ONCE)")
    print("!" * 74)
    print(f"    holdout window : {hold['decision_time'].min()} -> {hold['decision_time'].max()}  (rows={len(hold)})")
    print(f"    config_hash    : {fcfg.config_hash()}   (must equal the value you froze earlier)")
    print(f"    model IC       : {_fmt(ic)}")
    print(f"    model Sharpe   : {_fmt(sharpe)}   DSR={_fmt(dsr,'{:.3f}')}")
    print(f"    model turnover : {_fmt(ann_turn,'{:.0f}')}/yr   cost drag={_fmt(ann_cost*100,'{:.1f}')}%/yr")
    print(f"    TSMOM Sharpe   : {_fmt(tsh)}   (baseline on same window)")
    print(f"    direction dist : {dirdist}")
    verdict = ("MODEL BEATS TSMOM on holdout" if np.isfinite(sharpe) and np.isfinite(tsh) and sharpe > tsh
               else "model does NOT beat TSMOM on holdout")
    print(f"    VERDICT        : {verdict}")

    # ---- full sample-out strategy table (return / risk / risk-adjusted / trading) ----
    suite = {k: v for k, v in holdout_suite.items() if not k.startswith("_")}
    if suite:
        print("\n    HOLDOUT REAL RETURNS (sample-out)  [年化 / 累计 / 波动 / 回撤 / Sharpe / Sortino / Calmar / 换手 / 胜率]")
        print("    " + "-" * 110)
        print(f"    {'strategy':<42}{'AnnRet':>8}{'CumRet':>9}{'Vol':>8}{'MaxDD':>8}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'Turn':>7}{'Win':>7}")
        print("    " + "-" * 110)
        def _p(x, pct=False, dec=2):
            if x is None or (isinstance(x, float) and not np.isfinite(x)): return "   nan"
            return f"{x*100:.1f}%" if pct else f"{x:.{dec}f}"
        for nm, s in suite.items():
            print(f"    {nm:<42}{_p(s['ann_return'],1):>8}{_p(s['cum_return'],1):>9}"
                  f"{_p(s['ann_vol'],1):>8}{_p(s['max_drawdown'],1):>8}{_p(s['sharpe']):>8}"
                  f"{_p(s['sortino']):>9}{_p(s['calmar']):>8}{_p(s['ann_turnover'],0,0):>7}{_p(s['win_rate'],1):>7}")
        print("    " + "-" * 110)
        (outdir / "holdout_real_returns_table.json").write_text(
            json.dumps(holdout_suite, indent=2, default=str), encoding="utf-8")
        # flat CSV
        import csv as _csv
        with open(outdir / "holdout_real_returns_table.csv", "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["strategy", "ann_return", "cum_return", "ann_vol", "max_drawdown",
                        "sharpe", "sortino", "calmar", "win_rate", "profit_loss_ratio",
                        "ann_turnover", "ann_cost_pct"])
            for nm, s in suite.items():
                w.writerow([nm, s["ann_return"], s["cum_return"], s["ann_vol"], s["max_drawdown"],
                            s["sharpe"], s["sortino"], s["calmar"], s["win_rate"],
                            s["profit_loss_ratio"], s["ann_turnover"], s["ann_cost_pct"]])
        print(f"    -> holdout_real_returns_table.csv / .json")

    # live §1.4 signal for the latest bar (train on dev, predict latest)
    sigdf = hold[["symbol", "decision_time"]].copy()
    sigdf["combined_alpha"] = alpha
    sigdf["primary_direction"] = dirser.values
    sigdf["meta_trade_prob_calibrated"] = np.nan
    latest, port = _latest_signals(hold, sigdf, fcfg, code_hash="holdout_final",
                                   close_panel=cp, equity_returns=ret)
    (outdir / "signals_latest.json").write_text(json.dumps(latest, indent=2, ensure_ascii=False))
    (outdir / "portfolio_risk.json").write_text(json.dumps(port, indent=2, ensure_ascii=False, default=str))

    rep = [f"# HOLDOUT-A FINAL TEST — {datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}\n",
           "> Consumes Holdout-A. Per v6 §6.4 this is the once-only confirmatory grade.\n",
           f"- holdout window: {hold['decision_time'].min()} -> {hold['decision_time'].max()} (rows={len(hold)})",
           f"- config_hash: `{fcfg.config_hash()}`",
           f"- model: IC={_fmt(ic)}, Sharpe={_fmt(sharpe)}, DSR={_fmt(dsr,'{:.3f}')}",
           f"- TSMOM baseline Sharpe: {_fmt(tsh)}",
           f"- direction distribution: {dirdist}",
           f"- **VERDICT: {verdict}**"]
    if suite:
        rep.append("\n## Sample-out strategy table (frozen config, same metrics as dev)\n")
        rep.append("| strategy | AnnRet | CumRet | Vol | MaxDD | Sharpe | Sortino | Calmar | Turn/yr | Win |")
        rep.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        def _m(x, pct=False, dec=2):
            if x is None or (isinstance(x, float) and not np.isfinite(x)): return "—"
            return f"{x*100:.1f}%" if pct else f"{x:.{dec}f}"
        for nm, s in suite.items():
            rep.append(f"| {nm} | {_m(s['ann_return'],1)} | {_m(s['cum_return'],1)} | "
                       f"{_m(s['ann_vol'],1)} | {_m(s['max_drawdown'],1)} | {_m(s['sharpe'])} | "
                       f"{_m(s['sortino'])} | {_m(s['calmar'])} | {_m(s['ann_turnover'],False,0)} | "
                       f"{_m(s['win_rate'],1)} |")
        di = holdout_suite.get("_directional_info", {})
        if di:
            rep.append(f"\n- Step7 directional stack: avg|net|={di.get('avg_abs_net_exposure'):.2f} "
                       f"(cap {di.get('net_exposure_cap')}), target vol {di.get('target_portfolio_vol')}, "
                       f"CB-active {di.get('cb_active_frac',0)*100:.1f}%")
    (outdir / "HOLDOUT_report.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"\n    HOLDOUT_report.md + signals_latest.json -> {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+",
                    default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
    ap.add_argument("--synthetic", action="store_true",
                    help="run on synthetic bars (no parquet / heavy libs) to validate wiring")
    ap.add_argument("--outdir", default=None, help="output dir (default: data_storage/experiments/<ts>)")
    ap.add_argument("--holdout_frac", type=float, default=0.2, help="final holdout fraction (frozen)")
    ap.add_argument("--patchtst_emb_dim", type=int, default=8)
    ap.add_argument("--feature_set", default=DEFAULT_FEATURE_SET,
                    help="registry model feature set (market_core_v1 | market_plus_funding_v1 | "
                         "market_plus_onchain_v1 | market_plus_funding_onchain_v1)")
    ap.add_argument("--smooth_bars", type=int, default=24,
                    help="EMA span for position smoothing (low turnover). Default 24 (~4 days "
                         "at 4h): chosen to control transaction costs on this low-frequency book, "
                         "not to fit dev Sharpe. Use 6 for a more responsive (higher-cost) variant.")
    ap.add_argument("--deadband", type=float, default=0.05,
                    help="ignore |alpha| below this (no trade on weak signal)")
    ap.add_argument("--deploy_mode", choices=["vol_parity", "simple", "xs_neutral"], default="vol_parity",
                    help="vol_parity = inverse-vol + covariance vol-target (same engine as TSMOM); "
                         "simple = naive per-symbol conviction*max_pos (reviewer's proposal, for A/B); "
                         "xs_neutral = cross-sectional dollar-neutral (long strongest/short weakest, "
                         "removes common BTC factor -> raises breadth)")
    ap.add_argument("--max_vol_scale", type=float, default=3.0,
                    help="cap on low-vol leverage of the vol-parity engine (lower -> less turnover; "
                         "1.0 = no leverage)")
    ap.add_argument("--no_trade_band", type=float, default=0.05,
                    help="final-weight hysteresis band (default 0.05): only rebalance a symbol when its "
                         "target weight moves more than this (turnover control)")
    ap.add_argument("--no_pbo_grid", action="store_true",
                    help="skip the wider-grid PBO in [5d] (faster; the 4-config ablation PBO in [5] "
                         "is still computed)")
    ap.add_argument("--xs_allow_flat", action="store_true",
                    help="xs_neutral: let the book GROSS float with average conviction strength "
                         "(stay partly flat when signals are weak) instead of always investing to "
                         "full gross_cap. Still dollar-neutral. OOF/causal, no tuned knob.")
    ap.add_argument("--narrative_parquet", default=None,
                    help="path to narrative_features.parquet (per-symbol CryptoBERT sentiment). "
                         "If given, the narrative modality is asof-merged in (PIT-safe) and Step3 / "
                         "Step5 fusion will use it. Build it with etl.score_news + "
                         "etl.build_narrative_features.")
    ap.add_argument("--event_parquet", default="data_storage/factors/sentiment/event_features.parquet",
                    help="path to event_features.parquet (per-symbol LLM event/narrative "
                         "factors). Loaded BY DEFAULT if it exists (PIT asof-merge, registered "
                         "as the 'event' modality -> Step3b_+event). Build it with "
                         "etl.extract_events_llm + etl.build_event_features. Skipped silently if "
                         "the file is absent, or disable explicitly with --no_event.")
    ap.add_argument("--no_event", action="store_true",
                    help="disable the event modality even if event_features.parquet exists.")
    ap.add_argument("--decision_backtest", action="store_true",
                    help="also backtest the FULL agent decision stack (per-symbol risk "
                         "sizing + portfolio overlay + circuit breaker) through backtest/ "
                         "engine, via crypto.eval.decision_backtest. Saves decision_stack_*.json.")
    ap.add_argument("--xs_features", action="store_true",
                    help="add PIT-safe cross-sectional relative features (relative strength vs the "
                         "cross-section / vs BTC) as a new 'xsmom' modality + a Step1b_+xsmom ladder "
                         "step. Zero new data; most relevant for --deploy_mode xs_neutral.")
    ap.add_argument("--decision_use_meta_gate", action="store_true",
                    help="in the decision-stack backtest, gate positions on the meta probability "
                         "(meta_prob>p_threshold). Default OFF: the ladder's Step6 shows the meta "
                         "gate hurts this book, so the stack sizes by signal conviction instead.")
    ap.add_argument("--narrative_buffer_min", type=int, default=0,
                    help="extra PIT safety buffer in minutes for the narrative asof-merge "
                         "(default 0; the 4h binning already guarantees strictly-before-decision news)")
    ap.add_argument("--tsmom_lookbacks", type=int, nargs="+", default=None,
                    help="multi-scale TSMOM lookbacks in BARS (e.g. 180 540 1080 = 1/3/6 months "
                         "at 4h, per Moskowitz-Ooi-Pedersen 2012). Default None = single-scale "
                         "90-bar baseline. Affects Step0 (TSMOM benchmark) and Step7 (ML+TSMOM).")
    ap.add_argument("--p_threshold", type=float, default=None,
                    help="override meta-gate probability threshold (default 0.55 from RiskConfig); "
                         "changes config_hash")
    # model-capacity / regularization overrides (change config_hash; judge by OOF
    # IC_NW_t / DSR / PBO, NOT by dev Sharpe)
    ap.add_argument("--n_estimators", type=int, default=None)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--min_child_samples", type=int, default=None)
    ap.add_argument("--subsample", type=float, default=None)
    ap.add_argument("--colsample_bytree", type=float, default=None)
    ap.add_argument("--reg_alpha", type=float, default=None)
    ap.add_argument("--reg_lambda", type=float, default=None)
    ap.add_argument("--tp_mult", type=float, default=None,
                    help="override take-profit barrier (ATR mult). Symmetric labels: --tp_mult 1.5 --sl_mult 1.5")
    ap.add_argument("--sl_mult", type=float, default=None, help="override stop-loss barrier (ATR mult)")
    ap.add_argument("--run_holdout", action="store_true",
                    help="FINAL TEST: train on dev, evaluate the frozen holdout ONCE (consumes Holdout-A)")
    ap.add_argument("--cb_dd_start", type=float, default=None,
                    help="circuit-breaker L1 drawdown trip point (default 0.10); also moves the "
                         "smooth dd-scaler start. Loosen (e.g. 0.25) for risk-sensitivity analysis.")
    ap.add_argument("--cb_dd_l2", type=float, default=None,
                    help="circuit-breaker L2 drawdown trip point (default 0.15).")
    ap.add_argument("--cb_dd_stop", type=float, default=None,
                    help="circuit-breaker L3 drawdown trip point (default 0.20); also moves the "
                         "smooth dd-scaler stop. Loosen (e.g. 0.45 ~= 2x annual vol) for sensitivity.")
    args = ap.parse_args()

    fcfg = FrozenConfig()
    if args.tp_mult is not None or args.sl_mult is not None:   # research knob -> new config_hash (logged)
        fcfg = replace(fcfg, label=replace(fcfg.label,
                                           tp_mult=args.tp_mult if args.tp_mult is not None else fcfg.label.tp_mult,
                                           sl_mult=args.sl_mult if args.sl_mult is not None else fcfg.label.sl_mult))
    if args.p_threshold is not None:   # meta-gate threshold knob -> new config_hash (logged)
        fcfg = replace(fcfg, risk=replace(fcfg.risk, p_threshold=args.p_threshold))
    _mo = {k: getattr(args, k) for k in ("n_estimators", "max_depth", "learning_rate",
            "min_child_samples", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda")
           if getattr(args, k) is not None}
    if _mo:   # model-capacity overrides -> new config_hash (logged)
        fcfg = replace(fcfg, model=replace(fcfg.model, **_mo))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.outdir) if args.outdir else Path(config.PathConfig.EXPERIMENTS) / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, outdir / "console_log.txt")  # archive full console run

    print("=" * 74)
    print(f" v6 MARKET EXPERIMENT  (LightGBM + PatchTST)   {stamp}")
    print(f" config_hash={fcfg.config_hash()}   env={environment_hash()}")
    print(f" deployment: smooth_bars={args.smooth_bars} deadband={args.deadband} "
          f"max_pos={fcfg.risk.max_pos_per_symbol} (low-turnover)")
    print(f" feature_set={args.feature_set}")
    print(f" deploy_mode={args.deploy_mode} max_vol_scale={args.max_vol_scale} "
          f"no_trade_band={args.no_trade_band} p_threshold={fcfg.risk.p_threshold}")
    print(f" tsmom={'multiscale ' + str(args.tsmom_lookbacks) if args.tsmom_lookbacks else 'single-scale 90-bar'}")
    print(f" label barriers: tp={fcfg.label.tp_mult} sl={fcfg.label.sl_mult}"
          f"{'  [SYMMETRIC]' if fcfg.label.tp_mult == fcfg.label.sl_mult else ''}")
    print(f" mode={'SYNTHETIC (wiring check)' if args.synthetic else 'REAL parquet'}"
          f"{'  + HOLDOUT FINAL TEST' if args.run_holdout else ''}")
    print(f" outputs -> {outdir}")
    print("=" * 74)

    # ---- 1. dataset (full) ----
    provider = None
    if args.synthetic:
        seeds = {s: i + 1 for i, s in enumerate(args.symbols)}

        def provider(sym, tf):
            return to_bars_schema(_synthetic_bars(seeds[sym], tf), tf)

    md = build_market_dataset(args.symbols, fcfg, patchtst_emb_dim=args.patchtst_emb_dim,
                              bars_provider=provider, feature_set=args.feature_set,
                              synthetic=args.synthetic, xs_features=args.xs_features)
    print("\n[1] DATASET")
    for s, info in md.per_symbol.items():
        if s.startswith("_"):
            continue
        print(f"    {s}: {info}")
    print(f"    rows={len(md.dataset)}  market_feats={len(md.tabular_cols)}  "
          f"onchain_feats={len(md.modality_cols['onchain'])}  "
          f"patchtst_feats={len(md.modality_cols['patchtst'])}  total_feats={len(md.feature_cols)}")
    print(f"    market columns : {md.tabular_cols}")
    if md.modality_cols["onchain"]:
        oc = md.per_symbol.get("_onchain", {})
        print(f"    onchain columns: {md.modality_cols['onchain']}")
        print(f"    onchain coverage: {oc.get('rows_with_onchain', 0)} rows "
              f"(real DefiLlama data from {oc.get('onchain_start', 'n/a')})")

    # --- narrative modality (PREVIEW hook): asof-merge per-symbol CryptoBERT sentiment ---
    if args.narrative_parquet:
        from etl.narrative_loader import load_narrative_features, attach_narrative
        from etl.build_narrative_features import FEATURE_COLS as _NARR
        narr = load_narrative_features(args.narrative_parquet)
        attach_narrative(md.dataset, md.modality_cols, narr, buffer_min=args.narrative_buffer_min)
        cov = (md.dataset.assign(_h=(md.dataset[_NARR].abs().sum(axis=1) > 0))
               .groupby("symbol")["_h"].mean())
        print(f"    narrative columns: {md.modality_cols['narrative']}  "
              f"(PIT asof, buffer={args.narrative_buffer_min}min)")
        print("    narrative coverage (fraction of decisions with any news, by symbol): "
              + ", ".join(f"{s.split('/')[0]}={v:.2f}" for s, v in cov.items()))

    # --- LLM event modality (A1): asof-merge per-symbol structured event factors ---
    # --- LLM event modality (A1): on by default; skipped if file missing/synthetic/--no_event ---
    _ev_path = Path(args.event_parquet) if args.event_parquet else None
    if args.no_event or args.synthetic:
        if args.no_event:
            print("    event modality: disabled via --no_event")
    elif _ev_path is None or not _ev_path.exists():
        print(f"    event modality: skipped (no file at {args.event_parquet}; "
              f"build it with etl.extract_events_llm + etl.build_event_features)")
    else:
        from etl.narrative_loader import load_event_features, attach_event_features
        from etl.build_event_features import EVENT_FEATURE_COLS
        ev = load_event_features(str(_ev_path))
        attach_event_features(md.dataset, md.modality_cols, ev, buffer_min=args.narrative_buffer_min)
        evcov = (md.dataset.assign(_h=(md.dataset[EVENT_FEATURE_COLS].abs().sum(axis=1) > 0))
                 .groupby("symbol")["_h"].mean())
        print(f"    event modality: loaded {_ev_path}  cols={EVENT_FEATURE_COLS}  "
              f"(PIT asof, buffer={args.narrative_buffer_min}min)")
        print("    event coverage (fraction of decisions with any event signal, by symbol): "
              + ", ".join(f"{s.split('/')[0]}={v:.2f}" for s, v in evcov.items()))

    print("\n[2] PIT LEAKAGE AUDIT")
    print(f"    future_function_checks_passed = {md.audit['future_function_checks_passed']}  "
          f"(violations={md.audit['availability_lag_violations']}, rows={md.audit['n_rows']})")

    # ---- dev / holdout split (holdout is NEVER used during development) ----
    cut = _split_cut(md.dataset, args.holdout_frac)

    if args.run_holdout:
        _run_holdout(md, cut, fcfg, outdir, args)
        print(f"\nDONE (holdout). Report in: {outdir}")
        return

    dev = md.dataset[md.dataset["decision_time"] < cut].reset_index(drop=True)
    dev_close = md.close_panel.loc[md.close_panel.index < cut]
    n_hold = int((md.dataset["decision_time"] >= cut).sum())
    print(f"\n[*] DEV/HOLDOUT SPLIT: experiment uses DEV only "
          f"(dev rows={len(dev)}, holdout rows={n_hold} from {pd.Timestamp(cut)} are untouched)")

    # ---- 2. incremental ladder (DEV ONLY) ----
    print("\n[3] INCREMENTAL PROOF LADDER (Step0 -> Step7)  [dev only]")
    ladder = run_incremental_study(dev, dev_close, md.modality_cols, fcfg, bars_per_year=BARS_PER_YEAR_4H,
                                   smooth_bars=args.smooth_bars, deadband=args.deadband,
                                   deploy_mode=args.deploy_mode, max_vol_scale=args.max_vol_scale,
                                   no_trade_band=args.no_trade_band, allow_flat=args.xs_allow_flat,
                                   tsmom_lookbacks=args.tsmom_lookbacks)
    pd.set_option("display.width", 180, "display.max_columns", 20)
    print(ladder.round(4).to_string())
    ladder.to_csv(outdir / "incremental_ladder.csv")

    # ---- 3. A/B/C/D ablation (DEV ONLY) ----
    print("\n[4] PATCHTST A/B/C/D ABLATION  [dev only]")
    horizon_bars = max(2, int(fcfg.label.vertical_days * 6))
    ablation = run_ablation(dev, md.tabular_cols, fcfg,
                            bars_per_year=BARS_PER_YEAR_4H, max_label_horizon_bars=horizon_bars)
    print(ablation.round(4).to_string())
    ablation.to_csv(outdir / "patchtst_ablation.csv")

    # ---- 4. PBO (DEV ONLY) ----
    print("\n[5] PBO (CSCV over A/B/C/D, development period)")
    pbo = _pbo_over_abcd(dev, md.tabular_cols, fcfg)
    print(f"    PBO = {_fmt(pbo['pbo'], '{:.3f}')}  over {pbo['n_combinations']} combinations "
          f"(lower is better; <0.5 = IS-best tends to stay good OOS)")

    # ---- 5b. per-year Sharpe stability (time-robustness; clarifies a borderline PBO) ----
    print("\n[5b] PER-YEAR SHARPE STABILITY  [dev only]  (consistent positive years = "
          "regime-robust; a borderline PBO over near-tied A/B/C/D is then a ranking artifact)")
    rbs = ladder.attrs.get("returns_by_step", {})
    wps = ladder.attrs.get("weight_panels", {})       # signed panels (Step7) for directional stack
    key_steps = [s for s in ("Step0_baseline_tsmom", "Step1_market", "Step5_fusion",
                             "Step7_tsmom_fusion", "Step8_onchain_overlay") if s in rbs]
    if not key_steps:
        print("    (no deployable step returns available)")
    else:
        years = sorted({y for s in key_steps for y in per_year_sharpe(rbs[s], BARS_PER_YEAR_4H)})
        print("    " + "step".ljust(22) + "".join(f"{y:>8}" for y in years) + "    full")
        for s in key_steps:
            py = per_year_sharpe(rbs[s], BARS_PER_YEAR_4H)
            full = ladder.loc[s, "sharpe_ann"] if s in ladder.index else float("nan")
            cells = "".join(f"{py.get(y, float('nan')):>8.2f}" for y in years)
            print("    " + s.ljust(22) + cells + f"    {full:>5.2f}")

    # ---- 5c. cross-sectional regime diagnostic (pure; explains 2025 weakness) ----
    print("\n[5c] CROSS-SECTIONAL REGIME DIAGNOSTIC  [dev only]  (pure diagnostic, no "
          "strategy/param change)\n     a relative book needs DISPERSION: if a weak year "
          "shows higher corr / lower dispersion / higher PC1, that explains it")
    reg = _xs_regime_by_year(dev_close)
    s1 = per_year_sharpe(rbs["Step1_market"], BARS_PER_YEAR_4H) if "Step1_market" in rbs else {}
    print("    " + "year".ljust(8) + "avg_pair_corr".rjust(15) + "xs_dispersion".rjust(15)
          + "pc1_var_share".rjust(15) + "Step1_Sharpe".rjust(14))
    for y in sorted(reg):
        r = reg[y]
        print("    " + str(y).ljust(8)
              + f"{r['avg_pair_corr']:>15.3f}{r['xs_dispersion']:>15.4f}"
              + f"{r['pc1_var_share']:>15.3f}{s1.get(y, float('nan')):>14.2f}")

    # ---- 5d. wider-grid PBO (more reliable than the 4-config ablation PBO) ----
    if not args.no_pbo_grid:
        print("\n[5d] PBO over WIDER MODEL GRID  [dev only]  (more reliable than [5]'s "
              "4-config ablation PBO, which is noisy when A/C/D are near-tied)")
        pbo_grid = _pbo_over_grid(dev, md.tabular_cols, fcfg)
        print(f"    PBO = {_fmt(pbo_grid['pbo'], '{:.3f}')}  over {pbo_grid['n_combinations']} "
              f"combinations from {pbo_grid.get('n_configs', 0)} model configs "
              f"(depth x trees x lr; lower is better, <0.5 good)")
        print(f"    per-config per-trade edge: min={pbo_grid.get('edge_bps_min', float('nan')):.3f} "
              f"max={pbo_grid.get('edge_bps_max', float('nan')):.3f} "
              f"spread={pbo_grid.get('edge_bps_spread', float('nan')):.3f} bps  "
              f"(TIGHT spread => configs near-identical => high PBO is a near-tie artifact, "
              f"not overfitting)")
    else:
        pbo_grid = {"pbo": float("nan"), "n_configs": 0}

    # ---- 5. phase-1b signals + ECE (DEV ONLY) ----
    print("\n[6] PHASE-1B SIGNALS (two-stage meta-labelling + calibration)  [dev only]")
    signals, diag = run_phase1b(dev, md.feature_cols, fcfg)
    for k, v in diag.items():
        print(f"    {k}: {v}")
    signals.to_csv(outdir / "signals_oof.csv", index=False)

    # ---- 6. §1.4 latest structured signals (end of dev) + portfolio overlay ----
    eq_ret = rbs.get("Step5_fusion", rbs.get("Step1_market"))   # deployable dev equity curve
    latest, port = _latest_signals(dev, signals, fcfg, close_panel=dev_close, equity_returns=eq_ret)
    (outdir / "signals_latest.json").write_text(json.dumps(latest, indent=2, ensure_ascii=False))
    (outdir / "portfolio_risk.json").write_text(json.dumps(port, indent=2, ensure_ascii=False, default=str))
    print(f"    latest structured signals -> signals_latest.json ({len(latest)} symbols)")
    print(f"    portfolio overlay: gross {port.get('gross_before')} -> {port.get('gross_after')}  "
          f"cb_level={port.get('circuit_breaker_level')} ({port.get('circuit_breaker_reason')})  "
          f"steps={port.get('steps')}")

    # ---- 6b. OPTIONAL: backtest the FULL decision stack through backtest/ engine ----
    if args.decision_backtest:
        from crypto.eval.decision_backtest import run_decision_backtest
        _sizing = "xs_neutral" if args.deploy_mode == "xs_neutral" else "per_symbol"
        _cbmsg = ""
        if any(v is not None for v in (args.cb_dd_start, args.cb_dd_l2, args.cb_dd_stop)):
            _cbmsg = (f" cb_dd=L1:{args.cb_dd_start or 0.10:.0%}/"
                      f"L2:{args.cb_dd_l2 or 0.15:.0%}/L3:{args.cb_dd_stop or 0.20:.0%}")
        print(f"    [decision-stack backtest] running full risk+portfolio+CB stack "
              f"(sizing={_sizing}, meta_gate={'ON' if args.decision_use_meta_gate else 'OFF'}{_cbmsg}) through backtest engine ...")
        dbt = run_decision_backtest(dev, signals, dev_close, fcfg, bars_per_year=BARS_PER_YEAR_4H,
                                    apply_overlay=True, apply_circuit_breaker=True,
                                    use_meta_gate=args.decision_use_meta_gate, sizing_mode=_sizing,
                                    smooth_bars=args.smooth_bars, deadband=args.deadband,
                                    cb_dd_l1=args.cb_dd_start, cb_dd_l2=args.cb_dd_l2,
                                    cb_dd_l3=args.cb_dd_stop,
                                    strategy_name="decision_stack")
        m = dbt["metrics"]
        summary = {"info": dbt["info"],
                   "sharpe": m.get("strategy_sharpe"), "annual_return": m.get("strategy_annual_return"),
                   "volatility": m.get("strategy_volatility"), "max_drawdown": m.get("strategy_max_drawdown"),
                   "avg_turnover": m.get("avg_turnover"), "total_cost": m.get("total_cost")}
        (outdir / "decision_stack_metrics.json").write_text(json.dumps(m, indent=2, default=str))
        dbt["raw"]["returns"].to_frame("returns").to_csv(outdir / "decision_stack_returns.csv")
        def _f(x, scale=1.0, suffix=""):
            return "n/a" if x is None else f"{scale*x:.2f}{suffix}"
        print(f"      decision-stack: Sharpe={_f(summary['sharpe'])}  "
              f"AnnRet={_f(summary['annual_return'], 100, '%')}  "
              f"MaxDD={_f(summary['max_drawdown'], 100, '%')}  "
              f"CB-active={dbt['info'].get('cb_active_frac', 0):.2%}  -> decision_stack_metrics.json")
        if summary["sharpe"] is None:
            print("      [note] decision-stack held ~no position over the backtest (likely the "
                  "circuit breaker tripped on the dev-period equity, or all intents were gated). "
                  "Metrics are n/a; see decision_stack_metrics.json / the ladder table instead.")
        fn = dbt["info"].get("funnel", {})
        if fn:
            print(f"      [decision-stack funnel] of {fn.get('rows',0)} (symbol,bar) intents over "
                  f"{fn.get('timestamps_total',0)} bars:")
            print(f"        flat/no-meta rejected : {fn.get('flat_or_no_meta',0)}")
            print(f"        meta-gate rejected    : {fn.get('meta_gated',0)}  (use_meta_gate={args.decision_use_meta_gate})")
            print(f"        zero-size/blocked     : {fn.get('zero_size',0)}  (e.g. vol_target_scalar=0)")
            print(f"        -> approved intents   : {fn.get('approved_intents',0)}")
            print(f"        bars with >=1 intent  : {fn.get('timestamps_with_intent',0)} / {fn.get('timestamps_total',0)}")
            print(f"        bars zeroed by overlay: {fn.get('overlay_zeroed_ts',0)}")

        # decision stack (risk+portfolio+CB) vs pure-signal ladder steps -> defense table.
        # Also run a CB-OFF variant (risk sizing + portfolio overlay only): isolates what
        # the risk/portfolio machinery does to the signal, separate from the circuit
        # breaker (which can halt trading entirely on a losing dev period).
        from crypto.eval.decision_backtest import compare_decision_vs_signal, format_comparison_table
        dbt_nocb = run_decision_backtest(dev, signals, dev_close, fcfg, bars_per_year=BARS_PER_YEAR_4H,
                                         apply_overlay=True, apply_circuit_breaker=False,
                                         use_meta_gate=args.decision_use_meta_gate, sizing_mode=_sizing,
                                         smooth_bars=args.smooth_bars, deadband=args.deadband,
                                    cb_dd_l1=args.cb_dd_start, cb_dd_l2=args.cb_dd_l2,
                                         cb_dd_l3=args.cb_dd_stop,
                                         strategy_name="decision_stack_noCB")
        sig = {"decision_stack_noCB (risk+port)": dbt_nocb["raw"]["returns"],
               "Step5_fusion (pure signal)": rbs.get("Step5_fusion"),
               "Step6_meta_gate (signal+gate)": rbs.get("Step6_meta_gate"),
               "Step0_TSMOM (benchmark)": rbs.get("Step0_baseline_tsmom")}
        cmp = compare_decision_vs_signal(dbt["raw"]["returns"], sig, bars_per_year=BARS_PER_YEAR_4H)
        (outdir / "decision_vs_signal.json").write_text(json.dumps(cmp, indent=2, default=str))
        print(format_comparison_table(cmp))

        # ---- DIRECTIONAL decision stack on the MAIN deliverable Step7 ----------------
        # Step5 is a neutral book -> neutral overlay (above). Step7 (ML+TSMOM) is a
        # DIRECTIONAL book -> the directional risk paradigm (net-exposure cap + total
        # vol target + circuit breaker), NOT the neutral overlay (which would neutralize
        # TSMOM's net-direction alpha). This demonstrates "two paradigms x matched risk".
        dir_summary = None
        _w7 = wps.get("Step7_tsmom_fusion")
        if _w7 is not None and len(_w7):
            from crypto.eval.decision_backtest import run_directional_decision_backtest
            print("    [directional decision-stack] Step7 (ML+TSMOM) through the DIRECTIONAL "
                  "risk paradigm (net-exposure cap + total vol target + CB) ...")
            ddir = run_directional_decision_backtest(
                _w7, dev_close, fcfg, bars_per_year=BARS_PER_YEAR_4H, net_exposure_cap=1.5,
                apply_circuit_breaker=True, cb_dd_l1=args.cb_dd_start, cb_dd_l2=args.cb_dd_l2,
                cb_dd_l3=args.cb_dd_stop, strategy_name="directional_stack_step7")
            dm = ddir["metrics"]; di = ddir["info"]
            print(f"      directional-stack(Step7): Sharpe={dm['strategy_sharpe']:.2f}  "
                  f"AnnRet={dm['strategy_annual_return']*100:.2f}%  MaxDD={dm['strategy_max_drawdown']*100:.2f}%  "
                  f"avg|net|={di['avg_abs_net_exposure']:.2f} (cap {di['net_exposure_cap']})  "
                  f"CB-active={di.get('cb_active_frac',0)*100:.2f}%")
            dir_summary = {"info": di, "sharpe": dm["strategy_sharpe"],
                           "annual_return": dm["strategy_annual_return"],
                           "max_drawdown": dm["strategy_max_drawdown"]}
            (outdir / "directional_stack_step7.json").write_text(
                json.dumps({"metrics": {k: dm.get(k) for k in
                            ("strategy_sharpe", "strategy_annual_return", "strategy_total_return",
                             "strategy_max_drawdown", "strategy_volatility", "strategy_calmar")},
                            "info": di}, indent=2, default=str))


        # ---- REAL RETURNS TABLE: annual / cumulative / MaxDD / Sharpe, side by side ----
        # More intuitive than Sharpe alone -- shows whether the book actually compounds.
        # Computed uniformly from each strategy's own return series (same engine), so the
        # columns are directly comparable. Step7 added explicitly as the MAIN deliverable.
        def _ret_stats(r):
            r = pd.Series(r).dropna()
            if len(r) < 5:
                return None
            ann = float(r.mean() * BARS_PER_YEAR_4H)
            cum = float((1.0 + r).prod() - 1.0)
            eq = (1.0 + r).cumprod()
            mdd = float((1.0 - eq / eq.cummax()).max())
            sd = float(r.std())
            shp = float(r.mean() / sd * np.sqrt(BARS_PER_YEAR_4H)) if sd > 1e-12 else float("nan")
            return {"ann_return": ann, "cum_return": cum, "max_drawdown": mdd, "sharpe": shp}
        real_rows = {}
        real_rows["decision_stack (risk+port+CB)"] = _ret_stats(dbt["raw"]["returns"])
        real_rows["decision_stack_noCB (risk+port)"] = _ret_stats(dbt_nocb["raw"]["returns"])

        # MAIN-DELIVERABLE Step7 PUT THROUGH THE RISK CONTROL: Step7 is a directional
        # ML+TSMOM book, so it can't ride the xs_neutral cross-sectional sizing. But the
        # circuit-breaker is a return-series-level drawdown overlay, so we CAN apply it to
        # the Step7 equity curve to show "what risk control does to the main deliverable":
        # delever by the same CB drawdown scaler whenever Step7's rolling drawdown is deep.
        def _apply_cb_to_returns(r, cb_l1, cb_l2, cb_l3):
            r = pd.Series(r).dropna()
            if len(r) < 10:
                return None
            from crypto.live.risk_guard import CircuitBreaker, CBLevel
            from crypto.eval.decision_backtest import CB_POS_MULT, _dd_scaler
            cbk = {}
            if cb_l1 is not None: cbk["dd_l1"] = cb_l1
            if cb_l2 is not None: cbk["dd_l2"] = cb_l2
            if cb_l3 is not None: cbk["dd_l3"] = cb_l3
            cb = CircuitBreaker(**cbk)
            eq = (1.0 + r).cumprod()
            win = max(2, int(90 * 6))                     # rolling 90-day peak
            dd = (1.0 - eq / eq.rolling(win, min_periods=1).max())
            daily = (1.0 + r).rolling(6).apply(np.prod, raw=True) - 1.0
            dl = (-daily).clip(lower=0.0).fillna(0.0)
            mult = pd.Series(1.0, index=r.index)
            for t in r.index:
                lvl = int(cb.evaluate(drawdown=float(dd.loc[t]), daily_loss=float(dl.loc[t])))
                mult.loc[t] = CB_POS_MULT.get(lvl, 1.0) * _dd_scaler(float(dd.loc[t]), fcfg.risk)
            # decision at t-1 governs exposure earning r[t]
            return (r * mult.shift(1).fillna(1.0))
        _r7 = rbs.get("Step7_tsmom_fusion")
        if _r7 is not None:
            real_rows["Step7_tsmom_fusion (MAIN deliverable)"] = _ret_stats(_r7)
            _r7cb = _apply_cb_to_returns(_r7, args.cb_dd_start, args.cb_dd_l2, args.cb_dd_stop)
            if _r7cb is not None:
                real_rows["Step7 + circuit-breaker (deliverable+risk)"] = _ret_stats(_r7cb)
        if dir_summary is not None:
            real_rows["Step7 + directional stack (full risk)"] = _ret_stats(ddir["raw"]["returns"])
        for _nm, _key in [("Step5_fusion (pure signal)", "Step5_fusion"),
                          ("Step0_TSMOM (benchmark)", "Step0_baseline_tsmom")]:
            _r = rbs.get(_key)
            if _r is not None:
                real_rows[_nm] = _ret_stats(_r)
        real_rows = {k: v for k, v in real_rows.items() if v is not None}
        if real_rows:
            import pandas as _pd
            rr = _pd.DataFrame(real_rows).T[["ann_return", "cum_return", "max_drawdown", "sharpe"]]
            rr.to_csv(outdir / "real_returns_table.csv")
            (outdir / "real_returns_table.json").write_text(
                json.dumps({k: v for k, v in real_rows.items()}, indent=2, default=str))
            print("\n    REAL RETURNS (dev period)  [年化收益 / 累计收益 / 最大回撤 / Sharpe]")
            print("    " + "-" * 78)
            print(f"    {'strategy':<38}{'AnnRet':>9}{'CumRet':>10}{'MaxDD':>9}{'Sharpe':>9}")
            print("    " + "-" * 78)
            for nm, st in real_rows.items():
                print(f"    {nm:<38}{st['ann_return']*100:>8.1f}%{st['cum_return']*100:>9.1f}%"
                      f"{st['max_drawdown']*100:>8.1f}%{st['sharpe']:>9.2f}")
            print("    " + "-" * 78)
            print("    -> real_returns_table.csv / .json")

    # ---- 7. governance: freeze + pre-register (do NOT run holdout) ----
    print("\n[7] GOVERNANCE (freeze + pre-register; Holdout-A NOT run here)")
    fp = outdir / "frozen_config.json"
    h = freeze_config(fcfg, fp)
    load_frozen(fp)
    reg = outdir / "registry.json"
    pre_register(fcfg.to_dict(), reg, label="market_experiment_confirmatory")
    assert_preregistered(fcfg.to_dict(), reg)
    print(f"    dev rows={len(dev)}  holdout rows={n_hold} (holdout_start={pd.Timestamp(cut)})")
    print(f"    frozen config_hash={h}  pre-registration OK")
    print("    -> to grade on the holdout LATER (once): rerun the SAME command with --run_holdout")

    # ---- 8. acceptance checklist + report ----
    print("\n[8] ACCEPTANCE CHECKLIST (v6 gates)")
    checks = _acceptance(md, ladder, ablation, pbo, diag)
    for line in checks["lines"]:
        print("    " + line)

    _write_report(outdir, stamp, fcfg, md, ladder, ablation, pbo, diag, checks, latest, args)
    print(f"\nDONE. Full report + CSV/JSON in: {outdir}")
    print("Read EXPERIMENT_report.md first; signals_oof.csv = full OOF signals; "
          "signals_latest.json = §1.4 structured output.")
    print(f"Full console log archived -> {outdir / 'console_log.txt'}")


if __name__ == "__main__":
    main()
