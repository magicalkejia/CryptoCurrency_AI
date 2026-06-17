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
                                   compute_confidence, risk_size_and_gate)

BARS_PER_YEAR_4H = 2190  # 365 * 6


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


def _latest_signals(ds, signals, fcfg, code_hash="experiment"):
    """§1.4-style structured signal for the latest decision per symbol, reusing
    the existing Skills (regime + confidence + risk sizing)."""
    merged = ds.merge(signals, on=["symbol", "decision_time"], how="left", suffixes=("", "_sig"))
    out = []
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
        risk = risk_size_and_gate(fusion_out, conf, feat_row, fcfg, cb_level=0)
        out.append({
            "decision_time": pd.Timestamp(dt).isoformat(),
            "symbol": sym.replace("/", ""),
            "regime": regime,
            "combined_alpha": round(fusion_out["combined_alpha"], 4),
            "primary_direction": fusion_out["primary_direction"],
            "meta_trade_prob_calibrated": (None if np.isnan(fusion_out["meta_trade_prob_calibrated"])
                                           else round(fusion_out["meta_trade_prob_calibrated"], 4)),
            "confidence": round(conf, 4),
            "action": fusion_out["primary_direction"] if risk["risk_approved"] else "flat",
            "target_position": round(risk["target_position"], 4),
            "vol_target_scalar": round(risk.get("vol_target_scalar", 0.0), 4),
            "stop_loss": round(risk["stop_loss"], 4),
            "take_profit": round(risk["take_profit"], 4),
            "barrier_source": "ATR20_1h_at_decision_time",
            "risk_level": risk["risk_level"],
            "data_quality_score": round(dq["data_quality_score"], 4),
            "reason": risk["reason"],
            "audit_id": make_audit_id(sym, dt, "experiment_oof", "processed_parquet", code_hash, fcfg),
        })
    return out


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
    cfg = BacktestConfig(fee_rate=fee, slippage_rate=slip, execution_lag=1, annual_days=BARS_PER_YEAR_4H)

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

    # live §1.4 signal for the latest bar (train on dev, predict latest)
    sigdf = hold[["symbol", "decision_time"]].copy()
    sigdf["combined_alpha"] = alpha
    sigdf["primary_direction"] = dirser.values
    sigdf["meta_trade_prob_calibrated"] = np.nan
    latest = _latest_signals(hold, sigdf, fcfg, code_hash="holdout_final")
    (outdir / "signals_latest.json").write_text(json.dumps(latest, indent=2, ensure_ascii=False))

    rep = [f"# HOLDOUT-A FINAL TEST — {datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}\n",
           "> Consumes Holdout-A. Per v6 §6.4 this is the once-only confirmatory grade.\n",
           f"- holdout window: {hold['decision_time'].min()} -> {hold['decision_time'].max()} (rows={len(hold)})",
           f"- config_hash: `{fcfg.config_hash()}`",
           f"- model: IC={_fmt(ic)}, Sharpe={_fmt(sharpe)}, DSR={_fmt(dsr,'{:.3f}')}",
           f"- TSMOM baseline Sharpe: {_fmt(tsh)}",
           f"- direction distribution: {dirdist}",
           f"- **VERDICT: {verdict}**"]
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
    ap.add_argument("--smooth_bars", type=int, default=6,
                    help="EMA span for position smoothing (low turnover). ~1 day at 4h.")
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
    ap.add_argument("--no_trade_band", type=float, default=0.0,
                    help="final-weight hysteresis band (e.g. 0.05): only rebalance a symbol when its "
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
    ap.add_argument("--narrative_buffer_min", type=int, default=0,
                    help="extra PIT safety buffer in minutes for the narrative asof-merge "
                         "(default 0; the 4h binning already guarantees strictly-before-decision news)")
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
                              synthetic=args.synthetic)
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
                                   no_trade_band=args.no_trade_band, allow_flat=args.xs_allow_flat)
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

    # ---- 6. §1.4 latest structured signals (end of dev) ----
    latest = _latest_signals(dev, signals, fcfg)
    (outdir / "signals_latest.json").write_text(json.dumps(latest, indent=2, ensure_ascii=False))
    print(f"    latest structured signals -> signals_latest.json ({len(latest)} symbols)")

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
