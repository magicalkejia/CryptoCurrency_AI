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
  1. multi-timeframe PIT dataset  (etl.feature_builder.build_market_dataset)
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from crypto.schemas import FrozenConfig, environment_hash, make_audit_id
from crypto.adapters import to_bars_schema
from etl.feature_builder import build_market_dataset
from crypto.pipeline_1b import run_phase1b
from crypto.experiments.incremental_study import run_incremental_study
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+",
                    default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
    ap.add_argument("--synthetic", action="store_true",
                    help="run on synthetic bars (no parquet / heavy libs) to validate wiring")
    ap.add_argument("--outdir", default=None, help="output dir (default: data_storage/experiments/<ts>)")
    ap.add_argument("--holdout_frac", type=float, default=0.2, help="final holdout fraction (frozen)")
    ap.add_argument("--patchtst_emb_dim", type=int, default=8)
    args = ap.parse_args()

    fcfg = FrozenConfig()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.outdir) if args.outdir else Path(config.PathConfig.EXPERIMENTS) / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f" v6 MARKET EXPERIMENT  (LightGBM + PatchTST)   {stamp}")
    print(f" config_hash={fcfg.config_hash()}   env={environment_hash()}")
    print(f" mode={'SYNTHETIC (wiring check)' if args.synthetic else 'REAL parquet'}")
    print(f" outputs -> {outdir}")
    print("=" * 74)

    # ---- 1. dataset ----
    provider = None
    if args.synthetic:
        seeds = {s: i + 1 for i, s in enumerate(args.symbols)}

        def provider(sym, tf):
            return to_bars_schema(_synthetic_bars(seeds[sym], tf), tf)

    md = build_market_dataset(args.symbols, fcfg, patchtst_emb_dim=args.patchtst_emb_dim,
                              bars_provider=provider)
    print("\n[1] DATASET")
    for s, info in md.per_symbol.items():
        print(f"    {s}: {info}")
    print(f"    rows={len(md.dataset)}  market_feats={len(md.tabular_cols)}  "
          f"patchtst_feats={len(md.modality_cols['patchtst'])}  total_feats={len(md.feature_cols)}")
    print(f"    market columns : {md.tabular_cols}")

    print("\n[2] PIT LEAKAGE AUDIT")
    print(f"    future_function_checks_passed = {md.audit['future_function_checks_passed']}  "
          f"(violations={md.audit['availability_lag_violations']}, rows={md.audit['n_rows']})")

    # ---- 2. incremental ladder ----
    print("\n[3] INCREMENTAL PROOF LADDER (Step0 -> Step6)")
    ladder = run_incremental_study(md.dataset, md.close_panel, md.modality_cols, fcfg,
                                   bars_per_year=BARS_PER_YEAR_4H)
    pd.set_option("display.width", 180, "display.max_columns", 20)
    print(ladder.round(4).to_string())
    ladder.to_csv(outdir / "incremental_ladder.csv")

    # ---- 3. A/B/C/D ablation ----
    print("\n[4] PATCHTST A/B/C/D ABLATION")
    horizon_bars = max(2, int(fcfg.label.vertical_days * 6))
    ablation = run_ablation(md.dataset, md.tabular_cols, fcfg,
                            bars_per_year=BARS_PER_YEAR_4H, max_label_horizon_bars=horizon_bars)
    print(ablation.round(4).to_string())
    ablation.to_csv(outdir / "patchtst_ablation.csv")

    # ---- 4. PBO ----
    print("\n[5] PBO (CSCV over A/B/C/D, development period)")
    pbo = _pbo_over_abcd(md.dataset, md.tabular_cols, fcfg)
    print(f"    PBO = {_fmt(pbo['pbo'], '{:.3f}')}  over {pbo['n_combinations']} combinations "
          f"(lower is better; <0.5 = IS-best tends to stay good OOS)")

    # ---- 5. phase-1b signals + ECE ----
    print("\n[6] PHASE-1B SIGNALS (two-stage meta-labelling + calibration)")
    signals, diag = run_phase1b(md.dataset, md.feature_cols, fcfg)
    for k, v in diag.items():
        print(f"    {k}: {v}")
    signals.to_csv(outdir / "signals_oof.csv", index=False)

    # ---- 6. §1.4 latest structured signals ----
    latest = _latest_signals(md.dataset, signals, fcfg)
    (outdir / "signals_latest.json").write_text(json.dumps(latest, indent=2, ensure_ascii=False))
    print(f"    latest structured signals -> signals_latest.json ({len(latest)} symbols)")

    # ---- 7. governance: freeze + pre-register (do NOT run holdout) ----
    print("\n[7] GOVERNANCE (freeze + pre-register; Holdout-A NOT run here)")
    dt_series = md.dataset["decision_time"].sort_values().reset_index(drop=True)
    cut = dt_series.iloc[int(len(dt_series) * (1 - args.holdout_frac))]
    dev_idx, hold_idx = dev_holdout_split(md.dataset["decision_time"], cut)
    fp = outdir / "frozen_config.json"
    h = freeze_config(fcfg, fp)
    load_frozen(fp)
    reg = outdir / "registry.json"
    pre_register(fcfg.to_dict(), reg, label="market_experiment_confirmatory")
    assert_preregistered(fcfg.to_dict(), reg)
    print(f"    dev rows={len(dev_idx)}  holdout rows={len(hold_idx)} (holdout_start={pd.Timestamp(cut)})")
    print(f"    frozen config_hash={h}  pre-registration OK")

    # ---- 8. acceptance checklist + report ----
    print("\n[8] ACCEPTANCE CHECKLIST (v6 gates)")
    checks = _acceptance(md, ladder, ablation, pbo, diag)
    for line in checks["lines"]:
        print("    " + line)

    _write_report(outdir, stamp, fcfg, md, ladder, ablation, pbo, diag, checks, latest, args)
    print(f"\nDONE. Full report + CSV/JSON in: {outdir}")
    print("Read EXPERIMENT_report.md first; signals_oof.csv = full OOF signals; "
          "signals_latest.json = §1.4 structured output.")


if __name__ == "__main__":
    main()
