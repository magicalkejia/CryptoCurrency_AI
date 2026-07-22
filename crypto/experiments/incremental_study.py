"""
crypto.experiments.incremental_study
====================================
The §7.1 incremental-proof ladder, as a dedicated, scientific experiment.

It answers, one controlled step at a time, whether each added component carries
**statistically significant incremental** value over the previous step — the
thing that makes the project "like research".

Ladder (each step adds exactly one thing; everything else held fixed):
  Step0  baseline                  波动率平价 TSMOM(及格线,无 ML)
  Step1  + market/derivatives      结构化数据模型能否产生 alpha?
  Step2  + on-chain                链上数据是否提升预测能力?
  Step3  + narrative               文本情绪因子(CryptoBERT)是否有增量信息?
  Step3b + event                   LLM 事件/叙事因子相对情绪基线是否有显著增量?
  Step4  + PatchTST                时序表征是否有增量?
  Step5  fusion(多模态融合)        Agent 融合系统是否优于单模型?
  Step6  + meta-labeling 门控       元标签/概率门控是否提升风险收益比?

Comparability: EVERY step is turned into a target-weight panel and run through
the SAME existing backtest engine on the SAME close panel, so all steps yield
comparable per-period portfolio returns (no apples-to-oranges).

Incremental significance (overlap-aware): for step_k vs step_{k-1} we form the
per-period return-difference series d_t = r_k − r_{k-1} and test mean(d)=0 with a
Newey-West t-stat (lag = max label horizon in bars). A step is marked
`included=True` only if NW-t(d) > t_threshold (default 2.0). Absolute strategy
quality is reported via annualized Sharpe + Deflated Sharpe (n_trials = #steps,
penalizing the multiple comparisons). Exploratory vs Confirmatory is a flag;
Confirmatory runs should pass through governance.registry.assert_preregistered.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from crypto.cv.purged_kfold import purged_embargoed_splits, default_embargo_delta
from crypto.models.base_lgb import MultiClassLearner, BinaryLearner
from crypto.labels.meta_label import build_meta_label
from crypto.benchmark.tsmom import (vol_parity_tsmom_weights, vol_parity_weights_from_signal,
                                    vol_parity_tsmom_weights_multiscale)
from crypto.eval.significance import (newey_west_tstat, deflated_sharpe_ratio,
                                      information_coefficient, block_bootstrap_sharpe)
from backtest.engine import run_vector_backtest, BacktestConfig


RESEARCH_QUESTION = {
    "Step0_baseline_tsmom": "及格线基准(波动率平价 TSMOM,无 ML)",
    "Step1_market": "结构化数据模型能否产生 alpha?",
    "Step1b_+xsmom": "横截面相对特征(相对强弱/相对BTC)对中性book是否有增量?",
    "Step2_+onchain": "链上数据是否提升预测能力?",
    "Step3_+narrative": "文本情绪因子(CryptoBERT)是否有增量信息?",
    "Step3b_+event": "LLM 事件/叙事因子相对情绪基线是否有显著增量?",
    "Step4_+patchtst": "PatchTST 时序表征是否有增量?",
    "Step5_fusion": "多模态融合是否优于单模型?",
    "Step6_meta_gate": "元标签/概率门控是否提升风险收益比?",
    "Step7_tsmom_fusion": "ML信号与TSMOM融合(而非替代)能否跑赢单独TSMOM?",
    "Step8_onchain_overlay": "链上做总暴露择时叠加(独立sleeve)能否提升中性book?",
    "Step3c_event_overlay": "事件/新闻做总暴露择时叠加(独立sleeve)能否捕捉其共模信号?",
    "Step3d_event_risk_gate": "事件/新闻风险特征做波动率择时(高新闻风险时降仓)能否改善风险调整收益?",
}


def _oof_alpha(df: pd.DataFrame, cols: List[str], fcfg, splits) -> np.ndarray:
    """OOF combined_alpha = P(up)-P(down) from a single 3-class model on `cols`."""
    if not cols:
        return np.full(len(df), np.nan)
    X = df[cols].to_numpy(float)
    y = df["tb_label"].to_numpy(int)
    w = df["uniqueness_weight"].fillna(1.0).to_numpy(float)
    alpha = np.full(len(df), np.nan)
    for f in splits:
        if len(f.train_idx) < 30 or len(f.test_idx) == 0:
            continue
        m = MultiClassLearner(fcfg.model).fit(X[f.train_idx], y[f.train_idx], w[f.train_idx])
        pdn, pne, pup = m.predict_proba_df(X[f.test_idx])
        alpha[f.test_idx] = pup - pdn
    return alpha


def _cost_rates(fcfg):
    """Per-leg fee + slippage as fractions, read from CostConfig (was hardcoded)."""
    if fcfg is not None:
        return fcfg.cost.fee_bps / 1e4, (fcfg.cost.base_slippage_bps + fcfg.cost.spread_proxy_bps) / 1e4
    return 0.0004, 0.0004


def _apply_no_trade_band(w: pd.DataFrame, band: float) -> pd.DataFrame:
    """Turnover control: hysteresis on the final weights — only move a symbol's
    weight when the new target differs from the held weight by more than `band`.
    This cuts churn from per-bar vol/cov rescaling without abandoning risk
    management. Principled (cost-based) lever, not tuned to Sharpe."""
    arr = w.to_numpy(dtype=float)
    out = np.zeros_like(arr)
    prev = np.zeros(arr.shape[1], dtype=float)
    for i in range(len(arr)):
        tgt = arr[i]
        move = np.abs(tgt - prev) > band
        prev = np.where(move, tgt, prev)
        out[i] = prev
    return pd.DataFrame(out, index=w.index, columns=w.columns)


def _alpha_to_weight_panel(df, alpha, close_panel, fcfg, bars_per_year,
                           smooth_bars=1, deadband=0.0, meta_prob=None, p_threshold=0.55,
                           deploy_mode="vol_parity", max_vol_scale=3.0, no_trade_band=0.0,
                           allow_flat=False):
    """Deploy an OOF alpha as a target-weight panel. Sizing is by CONVICTION
    (per-symbol signed conviction in [-1, 1] that scales with |alpha|, full at
    |alpha| >= edge_cap), with a no-trade deadband, optional meta gate, and EMA
    persistence. Deployment modes (selectable for A/B comparison):
      * "vol_parity" (default): route the conviction panel through the SAME
        inverse-vol + covariance vol-target + gross-cap engine as TSMOM.
      * "simple": naive per-symbol sizing (conviction * max_pos_per_symbol), no
        vol-parity (the quant reviewer's proposal; kept for comparison).
      * "xs_neutral": cross-sectional market-neutral. Remove the common (market)
        component at each timestamp so the book bets on RELATIVE strength across
        symbols (long the strongest / short the weakest), dollar-neutral, gross
        normalized to gross_cap. This strips the dominant common (BTC) factor and
        raises effective breadth -- the structural lever for the breadth limit.
    `no_trade_band` applies a final-weight hysteresis (turnover control).
    `max_vol_scale` caps the low-vol leverage of the vol-parity engine."""
    scale = fcfg.risk.edge_cap if fcfg is not None else 0.2
    max_pos = fcfg.risk.max_pos_per_symbol if fcfg is not None else 0.25
    gross_cap = fcfg.risk.gross_cap if fcfg is not None else 1.0
    a = np.nan_to_num(alpha)
    conviction = np.sign(a) * np.clip(np.abs(a) / max(scale, 1e-9), 0.0, 1.0)   # in [-1, 1]
    conviction = np.where(np.abs(a) >= deadband, conviction, 0.0)
    if meta_prob is not None:
        conviction = conviction * (np.nan_to_num(meta_prob) > p_threshold).astype(float)
    tmp = df[["symbol", "decision_time"]].copy()
    tmp["c"] = conviction
    cpanel = tmp.pivot_table(index="decision_time", columns="symbol", values="c",
                             aggfunc="last").fillna(0.0)
    if smooth_bars and smooth_bars > 1:
        cpanel = cpanel.ewm(span=int(smooth_bars), adjust=False).mean()         # conviction persistence

    if deploy_mode == "simple":
        w = cpanel * max_pos                                                    # naive per-symbol sizing
    elif deploy_mode == "xs_neutral":
        c_xs = cpanel.sub(cpanel.mean(axis=1), axis=0)                          # remove common factor -> market neutral
        gross = c_xs.abs().sum(axis=1).replace(0.0, np.nan)
        w = c_xs.div(gross, axis=0).fillna(0.0) * gross_cap                     # dollar-neutral, gross=gross_cap
        if allow_flat:
            # FLAT CAPABILITY: instead of always investing to full gross_cap, let the
            # book's gross FLOAT with average conviction strength (mean |conviction| in
            # [0,1], full at |alpha|>=edge_cap). Weak/clustered signals -> low gross
            # (partly flat); strong dispersed signals -> full gross. Causal, OOF, and
            # parameter-free (reuses edge_cap/gross_cap); the book stays dollar-neutral.
            fullness = cpanel.abs().mean(axis=1).clip(0.0, 1.0)
            w = w.mul(fullness, axis=0)
        if smooth_bars and smooth_bars > 1:
            w = w.ewm(span=int(smooth_bars), adjust=False).mean()               # damp cross-sectional rank-flip churn (turnover control)
    else:
        close = close_panel.reindex(cpanel.index).ffill()
        w = vol_parity_weights_from_signal(close, cpanel, vol_window=30, cov_window=30,
                                            bars_per_year=bars_per_year, gross_cap=gross_cap,
                                            max_vol_scale=max_vol_scale)
    if no_trade_band and no_trade_band > 0:
        w = _apply_no_trade_band(w, no_trade_band)
    if deploy_mode == "xs_neutral":
        w = w.sub(w.mean(axis=1), axis=0)                                      # re-enforce dollar-neutrality after band
    return w


def _panel_to_returns(close_panel, w, bars_per_year, fcfg):
    """Run a target-weight panel through the existing backtest engine."""
    cp = close_panel.reindex(w.index).ffill().dropna()
    w = w.reindex(cp.index).fillna(0.0)
    if len(cp) < 5:
        return pd.Series(dtype=float), float("nan"), float("nan")
    fee, slip = _cost_rates(fcfg)
    cfg = BacktestConfig(fee_rate=fee, slippage_rate=slip, execution_lag=1,
                         annual_periods=bars_per_year, market="crypto", timeframe="4h")
    res = run_vector_backtest(cp, w, config=cfg, strategy_name="step")
    return (res["returns"], float(res["turnover"].mean() * bars_per_year),
            float(res["cost"].mean() * bars_per_year))


def _alpha_to_returns(df, alpha, close_panel, bars_per_year,
                      meta_prob=None, p_threshold=0.55, fcfg=None,
                      smooth_bars=1, deadband=0.0,
                      deploy_mode="vol_parity", max_vol_scale=3.0, no_trade_band=0.0,
                      allow_flat=False):
    """Conviction-sized deployment of an OOF alpha.
    Returns (returns, ann_turnover, ann_cost_fraction)."""
    w = _alpha_to_weight_panel(df, alpha, close_panel, fcfg, bars_per_year,
                               smooth_bars=smooth_bars, deadband=deadband,
                               meta_prob=meta_prob, p_threshold=p_threshold,
                               deploy_mode=deploy_mode, max_vol_scale=max_vol_scale,
                               no_trade_band=no_trade_band, allow_flat=allow_flat)
    return _panel_to_returns(close_panel, w, bars_per_year, fcfg)


def _tsmom_weight_panel(close_panel, bars_per_year, tsmom_lookbacks=None):
    if tsmom_lookbacks:
        return vol_parity_tsmom_weights_multiscale(
            close_panel, lookbacks=tsmom_lookbacks, vol_window=30,
            cov_window=30, bars_per_year=bars_per_year)
    return vol_parity_tsmom_weights(close_panel, lookback_mom=90, vol_window=30,
                                    cov_window=30, bars_per_year=bars_per_year)


def _tsmom_returns(close_panel, bars_per_year, fcfg=None, tsmom_lookbacks=None):
    w = _tsmom_weight_panel(close_panel, bars_per_year, tsmom_lookbacks=tsmom_lookbacks)
    return _panel_to_returns(close_panel, w, bars_per_year, fcfg)


def per_year_sharpe(returns: pd.Series, bars_per_year: int) -> dict:
    """Annualized Sharpe of a strategy's return series within each CALENDAR YEAR.
    A time-stability check: if every year is positive and of similar magnitude,
    the strategy is regime-robust (and a borderline PBO over near-tied configs is
    almost certainly a config-ranking artifact rather than genuine overfitting)."""
    if returns is None or len(returns) < 5:
        return {}
    s = returns.copy()
    s.index = pd.to_datetime(s.index)
    ann = np.sqrt(bars_per_year)
    out = {}
    for yr, grp in s.groupby(s.index.year):
        sd = grp.std()
        out[int(yr)] = float(grp.mean() / sd * ann) if (sd > 0 and len(grp) > 5) else float("nan")
    return out


def _confidence_gross_scalar(df, meta_prob, p_threshold):
    """Per-decision_time fraction of symbols whose calibrated meta-prob clears the
    threshold. Scales the GROSS of a market-neutral book by average conviction WITHOUT
    zeroing individual legs (zeroing a leg breaks dollar-neutrality). Shown in the
    ladder as the 'neutral meta-gate' diagnostic (it degrades the book -> reported as
    an honest negative result: the meta-gate is a directional concept)."""
    conf = (np.nan_to_num(meta_prob) > p_threshold).astype(float)
    tmp = df[["decision_time"]].copy()
    tmp["conf"] = conf
    return tmp.groupby("decision_time")["conf"].mean()


def _risk_sleeve_combine(close_panel, w_a, w_b, bars_per_year, fcfg, target_vol=0.15):
    """Combine two strategy sleeves at EQUAL RISK: scale each weight panel so its
    realized return vol equals `target_vol`, then sum. This is the correct way to
    add a market-neutral sleeve to a directional one (a 50/50 weight blend would
    just halve the directional book). `target_vol` is a fixed risk budget (half of
    the 0.30 portfolio target), not tuned to Sharpe. The per-sleeve vol used for
    scaling is a risk normalization (no directional lookahead)."""
    ann = np.sqrt(bars_per_year)
    ra = _panel_to_returns(close_panel, w_a, bars_per_year, fcfg)[0]
    rb = _panel_to_returns(close_panel, w_b, bars_per_year, fcfg)[0]
    sda = float(ra.std() * ann) if len(ra) else 0.0
    sdb = float(rb.std() * ann) if len(rb) else 0.0
    sa = (target_vol / sda) if sda > 1e-9 else 0.0
    sb = (target_vol / sdb) if sdb > 1e-9 else 0.0
    idx = w_a.index.union(w_b.index)
    cols = w_a.columns.union(w_b.columns)
    return (sa * w_a.reindex(index=idx, columns=cols).fillna(0.0)
            + sb * w_b.reindex(index=idx, columns=cols).fillna(0.0))


def _xs_ic_tstat(a, fwd, times, idx) -> float:
    """Per-period CROSS-SECTIONAL rank-IC t-stat: demean alpha and fwd-return ACROSS
    SYMBOLS within each decision_time, correlate, then t = mean(IC)/se(IC) over periods.
    A common-mode signal (identical across symbols) demeans to ~0 -> t~0, so this is the
    correct 'skill' metric for a dollar-neutral book. `idx` restricts to (train) rows."""
    a = np.asarray(a, float); fwd = np.asarray(fwd, float)
    m = ~(np.isnan(a) | np.isnan(fwd))
    sel = np.zeros(len(a), bool); sel[np.asarray(idx, int)] = True; m &= sel
    if m.sum() < 30:
        return 0.0
    d = pd.DataFrame({"t": pd.Series(times)[m].to_numpy(), "a": a[m], "f": fwd[m]})
    d["ra"] = d.groupby("t")["a"].rank()
    d["rf"] = d.groupby("t")["f"].rank()
    d["da"] = d["ra"] - d.groupby("t")["ra"].transform("mean")
    d["dfd"] = d["rf"] - d.groupby("t")["rf"].transform("mean")
    g = (d.assign(dadf=d["da"] * d["dfd"], da2=d["da"] ** 2, df2=d["dfd"] ** 2)
         .groupby("t")[["dadf", "da2", "df2"]].sum())
    ic = (g["dadf"] / np.sqrt(g["da2"] * g["df2"])).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ic) < 5 or ic.std(ddof=1) < 1e-12:
        return 0.0
    return float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))))


def _cm_ic_tstat(a, fwd, times, idx) -> float:
    """Common-mode / time-series IC t-stat: does the per-period MEAN alpha (the
    cross-sectional average = 'market sentiment') predict the per-period MEAN forward
    return ('market move') over time? High here WITH low _xs_ic_tstat => the signal is
    common-mode/timing (like global on-chain), not a cross-sectional ranker."""
    a = np.asarray(a, float); fwd = np.asarray(fwd, float)
    m = ~(np.isnan(a) | np.isnan(fwd))
    sel = np.zeros(len(a), bool); sel[np.asarray(idx, int)] = True; m &= sel
    if m.sum() < 30:
        return 0.0
    d = pd.DataFrame({"t": pd.Series(times)[m].to_numpy(), "a": a[m], "f": fwd[m]}).groupby("t").mean()
    if len(d) < 5 or d["a"].std() < 1e-12 or d["f"].std() < 1e-12:
        return 0.0
    r = float(np.corrcoef(d["a"], d["f"])[0, 1])
    n = len(d)
    if not np.isfinite(r) or abs(r) >= 1.0:
        return 0.0
    return float(r * np.sqrt((n - 2) / (1.0 - r ** 2)))


def _fusion_alpha(df, modality_cols, fcfg, splits, fwd=None, xs=False):
    """Leakage-free fusion of per-modality OOF alphas. Returns (fused, base_oof, weights).

    * Cross-sectional (xs_neutral) book: weight each modality by its CROSS-SECTIONAL
      skill, computed PER FOLD on the TRAIN rows only (zero look-ahead) and applied to the
      TEST rows. weight_m = max(0, train xs-IC t-stat), renormalized; if no modality has
      positive cross-sectional skill, fall back to equal weight. This (a) needs no arbitrary
      t threshold, (b) auto-downweights a weak modality (narrative) instead of letting equal
      weighting halve the strong one, and (c) gives ~0 weight to a common-mode modality
      (on-chain) whose cross-sectional skill is ~0 -- so it cannot sneak into the neutral book.
    * Directional (vol_parity / simple) book or a single modality: plain z-scored equal
      weight (unchanged), since the deliverable is the cross-sectional book.
    A *fitted* meta-combiner would need nested CV; this rule-based combiner stays a clean
    'fusion vs single-model' comparison."""
    base = {}
    for mod, cols in modality_cols.items():
        if cols:
            base[mod] = _oof_alpha(df, cols, fcfg, splits)
    if not base:
        return np.full(len(df), np.nan), {}, {}
    mods = list(base.keys())

    if not xs or fwd is None or len(mods) == 1:
        mat = []
        for m in mods:
            s = np.nan_to_num(base[m]); sd = s.std()
            mat.append(s / sd if sd > 1e-12 else s)
        fused = np.mean(np.column_stack(mat), axis=1)
        return fused, base, {m: 1.0 / len(mods) for m in mods}

    times = df["decision_time"].to_numpy()
    fused = np.full(len(df), np.nan)
    wlog = {m: [] for m in mods}
    for fo in splits:
        tr, te = fo.train_idx, fo.test_idx
        if len(tr) < 30 or len(te) == 0:
            continue
        ws, ztest = {}, {}
        for m in mods:
            a = base[m]
            sd = np.nanstd(a[tr]); sd = sd if sd > 1e-12 else 1.0
            ztest[m] = np.nan_to_num(a[te]) / sd
            ws[m] = max(0.0, _xs_ic_tstat(a, fwd, times, tr))   # TRAIN-only skill
        tot = sum(ws.values())
        if tot <= 0:                                            # no xs skill -> equal weight
            ws = {m: 1.0 for m in mods}; tot = float(len(mods))
        acc = np.zeros(len(te))
        for m in mods:
            wn = ws[m] / tot
            acc += wn * ztest[m]
            wlog[m].append(wn)
        fused[te] = acc
    weights = {m: (float(np.mean(wlog[m])) if wlog[m] else 0.0) for m in mods}
    return fused, base, weights


def _meta_gate(df, fused_alpha, base_oof, fcfg, splits):
    """Stage-2 meta gate on the fusion alpha (OOF direction -> meta-label -> OOF prob)."""
    direction = np.where(fused_alpha > fcfg.theta_long, "long",
                         np.where(fused_alpha < -fcfg.theta_short, "short", "flat"))
    meta = build_meta_label(pd.Series(direction), df["net_exit_return_long"],
                            df["net_exit_return_short"], source="oof")
    y = pd.Series(np.nan, index=df.index)
    y.loc[meta.index] = meta["meta_label"].to_numpy()
    feats = np.column_stack([np.nan_to_num(fused_alpha)] +
                            [np.nan_to_num(v) for v in base_oof.values()])
    prob = np.full(len(df), np.nan)
    yv = y.to_numpy()
    nonflat = ~np.isnan(yv)
    for f in splits:
        tr = [i for i in f.train_idx if nonflat[i]]
        te = [i for i in f.test_idx if nonflat[i]]
        if len(tr) < 30 or len(te) == 0 or len(np.unique(yv[tr])) < 2:
            continue
        b = BinaryLearner(fcfg.model).fit(feats[tr], yv[tr].astype(int))
        prob[te] = b.predict_proba(feats[te])
    return direction, prob


def run_incremental_study(
    dataset: pd.DataFrame,
    close_panel: pd.DataFrame,
    modality_cols: Dict[str, List[str]],
    fcfg,
    fwd_col: str = "raw_exit_return_long",
    bars_per_year: int = 2190,
    t_threshold: float = 2.0,
    max_pos: float = None,
    smooth_bars: int = 6,
    deadband: float = 0.05,
    deploy_mode: str = "vol_parity",
    max_vol_scale: float = 3.0,
    no_trade_band: float = 0.0,
    kind: str = "exploratory",
    allow_flat: bool = False,
    tsmom_lookbacks=None,
) -> pd.DataFrame:
    """
    modality_cols keys expected: market / onchain / narrative / patchtst (any may
    be empty -> that step is 'skipped (no features)').
    Returns a per-step table with research question, IC, Sharpe, DSR, and the
    incremental NW-t vs the previous step + an `included` decision.
    """
    df = dataset.reset_index(drop=True).copy()
    splits = purged_embargoed_splits(df["decision_time"], df["entry_time"], df["exit_time"],
                                     fcfg.cv.n_splits, default_embargo_delta(fcfg.cv),
                                     symbol=df["symbol"])
    fwd = df[fwd_col].to_numpy(float)
    horizon_bars = max(2, int(fcfg.label.vertical_days * 6))  # 4h bars in vertical window

    mk = modality_cols.get("market", [])
    oc = modality_cols.get("onchain", [])
    nv = modality_cols.get("narrative", [])      # CryptoBERT sentiment (Step3)
    ev = modality_cols.get("event", [])          # LLM event/narrative factors (Step3b)
    xm = modality_cols.get("xsmom", [])           # cross-sectional relative features (Step1b)
    pt = modality_cols.get("patchtst", [])
    # PatchTST: the forecast head is non-predictive (ablation config B) and dilutes
    # the tree; the embedding (config C) is the only part worth concatenating, and
    # C beats the full set D. Step4 therefore adds the EMBEDDING ONLY.
    pt_emb = [c for c in pt if "patchtst_emb_" in c] or pt

    if max_pos is None:
        max_pos = fcfg.risk.max_pos_per_symbol

    def deploy(a, meta=None):
        return _alpha_to_returns(df, a, close_panel, bars_per_year,
                                 meta_prob=meta, p_threshold=fcfg.risk.p_threshold, fcfg=fcfg,
                                 smooth_bars=smooth_bars, deadband=deadband,
                                 deploy_mode=deploy_mode, max_vol_scale=max_vol_scale,
                                 no_trade_band=no_trade_band, allow_flat=allow_flat)

    steps = []  # (name, alpha_or_None, returns, ann_turnover, ann_cost)
    _weight_panels = {}      # step -> signed weight panel (for the directional decision stack)
    skip_status = {}
    xs = (deploy_mode == "xs_neutral")
    r0, t0, c0 = _tsmom_returns(close_panel, bars_per_year, fcfg, tsmom_lookbacks=tsmom_lookbacks)
    steps.append(("Step0_baseline_tsmom", None, r0, t0, c0))

    cumulative = []
    side_branch = set()      # steps shown for completeness but OFF the main increment chain
    ladder_steps = [("Step1_market", mk)]
    if xm:                                            # only when --xs_features is on
        ladder_steps.append(("Step1b_+xsmom", xm))
    ladder_steps += [("Step2_+onchain", oc),
                     ("Step3_+narrative", nv), ("Step3b_+event", ev),
                     ("Step4_+patchtst", pt_emb)]
    for name, group in ladder_steps:
        new_cols = [c for c in group if c in df.columns]
        if not new_cols:                      # modality has no data -> carry previous
            steps.append((name, None, None, np.nan, np.nan))
            continue
        if xs and name == "Step2_+onchain":
            # SHOW market+onchain (diagnostic: does on-chain help per-trade?), but DO
            # NOT carry on-chain into the cumulative -> it stays out of Step4/Step5.
            # On-chain is a GLOBAL (common-mode) signal: a cross-sectional book demeans
            # it out, so it adds noise/turnover without relative value. Side-branch.
            a = _oof_alpha(df, cumulative + new_cols, fcfg, splits)
            r, tn, cs = deploy(a)
            steps.append((name, a, r, tn, cs))
            side_branch.add(name)
            continue                          # cumulative NOT updated
        cumulative = cumulative + new_cols
        a = _oof_alpha(df, cumulative, fcfg, splits)
        r, tn, cs = deploy(a)
        steps.append((name, a, r, tn, cs))

    # Step5 fusion (mean of per-modality OOF alphas). PatchTST is EXCLUDED (its
    # standalone signal is ~0 and dilutes). In xs_neutral, on-chain is ALSO excluded
    # (common-mode -> overlay), so the cross-sectional fusion = market (+ narrative).
    fusion_mods = {"market": mk, "narrative": nv, "event": ev}
    if xm:
        fusion_mods["xsmom"] = xm
    if not xs:
        fusion_mods["onchain"] = oc
    fused, base_oof, fusion_w = _fusion_alpha(df, fusion_mods, fcfg, splits, fwd=fwd, xs=xs)
    r5, t5, c5 = deploy(fused)
    steps.append(("Step5_fusion", fused, r5, t5, c5))
    if xs:
        print("    [Step5 fusion] cross-sectional skill weights (per-fold, train->test): "
              + ", ".join(f"{m}={fusion_w.get(m, 0.0):.2f}" for m in fusion_mods if fusion_mods[m]))

    # Narrative decomposition diagnostic (no tuning, no deployment change): is the news
    # signal a cross-sectional ranker, a common-mode timer (like global on-chain), or noise?
    narr_diag = None
    if nv:
        nv_alpha = base_oof.get("narrative")
        if nv_alpha is None:
            nv_alpha = _oof_alpha(df, nv, fcfg, splits)
        full = np.arange(len(df))
        narr_diag = {"xs_ic_t": _xs_ic_tstat(nv_alpha, fwd, df["decision_time"].to_numpy(), full),
                     "cm_ic_t": _cm_ic_tstat(nv_alpha, fwd, df["decision_time"].to_numpy(), full)}
        verdict = ("no signal" if abs(narr_diag["xs_ic_t"]) < 2 and abs(narr_diag["cm_ic_t"]) < 2
                   else "common-mode/timing" if abs(narr_diag["xs_ic_t"]) < 2 else "cross-sectional")
        print(f"    [narrative diagnostic] cross-sectional IC t={narr_diag['xs_ic_t']:.2f}  "
              f"vs common-mode/time-series IC t={narr_diag['cm_ic_t']:.2f}  -> {verdict}")

    # Event decomposition diagnostic (same as narrative): is the LLM event signal a
    # cross-sectional ranker, a common-mode/macro timer, or noise? Cheap, no tuning.
    event_diag = None
    if ev:
        ev_cols = [c for c in ev if c in df.columns]
        ev_alpha = base_oof.get("event")
        if ev_alpha is None and ev_cols:
            ev_alpha = _oof_alpha(df, ev_cols, fcfg, splits)
        if ev_alpha is not None:
            full = np.arange(len(df))
            event_diag = {"xs_ic_t": _xs_ic_tstat(ev_alpha, fwd, df["decision_time"].to_numpy(), full),
                          "cm_ic_t": _cm_ic_tstat(ev_alpha, fwd, df["decision_time"].to_numpy(), full)}
            verdict = ("no signal" if abs(event_diag["xs_ic_t"]) < 2 and abs(event_diag["cm_ic_t"]) < 2
                       else "common-mode/timing" if abs(event_diag["xs_ic_t"]) < 2 else "cross-sectional")
            print(f"    [event diagnostic] cross-sectional IC t={event_diag['xs_ic_t']:.2f}  "
                  f"vs common-mode/time-series IC t={event_diag['cm_ic_t']:.2f}  -> {verdict}")

    # Step6 meta gate. The meta-label scores whether a DIRECTIONAL bet pays off net of
    # cost -> it is a directional concept. In xs_neutral we SHOW its neutral-aware form
    # (scale the book GROSS by the fraction of confident names, preserving neutrality)
    # as an honest NEGATIVE result -- it degrades the book -- and keep it OFF the main
    # increment chain (side-branch). In vol_parity it is a normal main-chain step.
    if xs:
        direction, meta_prob = _meta_gate(df, fused, base_oof, fcfg, splits)
        w_neu6 = _alpha_to_weight_panel(df, fused, close_panel, fcfg, bars_per_year,
                                        smooth_bars=smooth_bars, deadband=deadband,
                                        meta_prob=None, deploy_mode="xs_neutral",
                                        max_vol_scale=max_vol_scale, no_trade_band=no_trade_band,
                                        allow_flat=allow_flat)
        g = _confidence_gross_scalar(df, meta_prob, fcfg.risk.p_threshold)
        w6 = w_neu6.mul(g.reindex(w_neu6.index).fillna(0.0), axis=0)
        r6, t6, c6 = _panel_to_returns(close_panel, w6, bars_per_year, fcfg)
        steps.append(("Step6_meta_gate", fused, r6, t6, c6))
        side_branch.add("Step6_meta_gate")
    else:
        direction, meta_prob = _meta_gate(df, fused, base_oof, fcfg, splits)
        r6, t6, c6 = deploy(fused, meta=meta_prob)
        steps.append(("Step6_meta_gate", fused, r6, t6, c6))

    # Step7: combine the ML sleeve with the TSMOM baseline.
    if xs:
        # NEUTRAL sleeve + TSMOM as two EQUAL-RISK vol-targeted sleeves, SUMMED
        # (not a 50/50 weight blend, which would just halve TSMOM). The neutral
        # book (no market beta) and TSMOM (pure beta) are ~uncorrelated -> diversify.
        w_neu7 = _alpha_to_weight_panel(df, fused, close_panel, fcfg, bars_per_year,
                                        smooth_bars=smooth_bars, deadband=deadband,
                                        meta_prob=None, deploy_mode="xs_neutral",
                                        max_vol_scale=max_vol_scale, no_trade_band=no_trade_band,
                                        allow_flat=allow_flat)
        w_ts = _tsmom_weight_panel(close_panel, bars_per_year, tsmom_lookbacks=tsmom_lookbacks)
        w7 = _risk_sleeve_combine(close_panel, w_neu7, w_ts, bars_per_year, fcfg)
        r7, t7, c7 = _panel_to_returns(close_panel, w7, bars_per_year, fcfg)
        _weight_panels["Step7_tsmom_fusion"] = w7      # directional book (for directional stack)
    else:
        # directional: COMBINE (not replace) momentum with meta-gated ML via a fixed
        # 50/50 convex blend of weight panels. beta=0.5 is a neutral prior.
        w_ml = _alpha_to_weight_panel(df, fused, close_panel, fcfg, bars_per_year,
                                      smooth_bars=smooth_bars, deadband=deadband,
                                      meta_prob=meta_prob, p_threshold=fcfg.risk.p_threshold,
                                      deploy_mode=deploy_mode, max_vol_scale=max_vol_scale,
                                      no_trade_band=no_trade_band)
        w_ts = _tsmom_weight_panel(close_panel, bars_per_year, tsmom_lookbacks=tsmom_lookbacks)
        idx = w_ts.index.union(w_ml.index)
        cols = w_ts.columns.union(w_ml.columns)
        beta = 0.5
        w_blend = (beta * w_ts.reindex(index=idx, columns=cols).fillna(0.0)
                   + (1 - beta) * w_ml.reindex(index=idx, columns=cols).fillna(0.0))
        r7, t7, c7 = _panel_to_returns(close_panel, w_blend, bars_per_year, fcfg)
    steps.append(("Step7_tsmom_fusion", fused, r7, t7, c7))

    # Step8 (xs_neutral only): on-chain GROSS/MARKET-TIMING overlay as a separate
    # equal-risk sleeve on top of the neutral book. On-chain (global) is deployed
    # DIRECTIONALLY -> a net market-exposure timer. Shown for completeness / the report
    # (it does NOT beat the pure neutral book -> honest negative). Side-branch, OOF, no
    # tuned knobs.
    if xs:
        oc_cols = [c for c in oc if c in df.columns]
        if oc_cols:
            oc_alpha = _oof_alpha(df, oc_cols, fcfg, splits)
            w_oc = _alpha_to_weight_panel(df, oc_alpha, close_panel, fcfg, bars_per_year,
                                          smooth_bars=smooth_bars, deadband=deadband,
                                          meta_prob=None, deploy_mode="vol_parity",
                                          max_vol_scale=max_vol_scale, no_trade_band=no_trade_band)
            w_neu8 = _alpha_to_weight_panel(df, fused, close_panel, fcfg, bars_per_year,
                                            smooth_bars=smooth_bars, deadband=deadband,
                                            meta_prob=None, deploy_mode="xs_neutral",
                                            max_vol_scale=max_vol_scale, no_trade_band=no_trade_band,
                                            allow_flat=allow_flat)
            w8 = _risk_sleeve_combine(close_panel, w_neu8, w_oc, bars_per_year, fcfg)
            r8, t8, c8 = _panel_to_returns(close_panel, w8, bars_per_year, fcfg)
            steps.append(("Step8_onchain_overlay", oc_alpha, r8, t8, c8))
            side_branch.add("Step8_onchain_overlay")

    # Step3c (xs_neutral only): EVENT (news/LLM) GROSS/MARKET-TIMING overlay as a
    # separate equal-risk sleeve. The ladder diagnostic shows the event factor carries
    # COMMON-MODE / time-series signal (common-mode IC t=+1.46, the strongest of any
    # modality) but ~zero CROSS-SECTIONAL signal (xs IC t=-0.02). A neutral selection
    # book discards exactly that common-mode dimension, so event shows no increment
    # there. This sleeve deploys the aggregate event_alpha DIRECTIONALLY (a net market
    # exposure timer) so the common-mode signal can actually be expressed. Side-branch,
    # OOF, no tuned knobs -- an honest test of "is news a market-timing signal?".
    if xs and ev:
        ev_cols = [c for c in ev if c in df.columns]
        if ev_cols:
            ev_alpha = _oof_alpha(df, ev_cols, fcfg, splits)
            w_ev = _alpha_to_weight_panel(df, ev_alpha, close_panel, fcfg, bars_per_year,
                                          smooth_bars=smooth_bars, deadband=deadband,
                                          meta_prob=None, deploy_mode="vol_parity",
                                          max_vol_scale=max_vol_scale, no_trade_band=no_trade_band)
            w_neu3c = _alpha_to_weight_panel(df, fused, close_panel, fcfg, bars_per_year,
                                             smooth_bars=smooth_bars, deadband=deadband,
                                             meta_prob=None, deploy_mode="xs_neutral",
                                             max_vol_scale=max_vol_scale, no_trade_band=no_trade_band,
                                             allow_flat=allow_flat)
            w3c = _risk_sleeve_combine(close_panel, w_neu3c, w_ev, bars_per_year, fcfg)
            r3c, t3c, c3c = _panel_to_returns(close_panel, w3c, bars_per_year, fcfg)
            steps.append(("Step3c_event_overlay", ev_alpha, r3c, t3c, c3c))
            side_branch.add("Step3c_event_overlay")

    # Step3d (xs_neutral only): NEWS-RISK VOLATILITY GATE. The literature's most robust
    # finding is that news predicts VOLATILITY/magnitude, NOT direction (e.g. HAR-RV news
    # models; "news improves magnitude not direction"). Our own diagnostic agrees: event
    # has ~zero cross-sectional (direction) IC but the strongest common-mode signal. So
    # instead of using event as a RETURN signal (Step3b/3c, both null), here we use the
    # event RISK features (narr_event_risk, narr_rumor_risk) as a forward-VOLATILITY
    # predictor and DELEVER the neutral book when news-risk is elevated. This tests the
    # honest hypothesis: "is news useful for RISK timing rather than return timing?".
    if xs and ev:
        risk_cols = [c for c in ("narr_event_risk", "narr_rumor_risk") if c in df.columns]
        if risk_cols:
            # (a) DIAGNOSTIC: does aggregate news-risk predict next-bar cross-sectional
            #     realized volatility? Build a market-level news-risk series and a forward
            #     realized-vol series, report the IC + Newey-West t (PIT: risk known at t,
            #     vol realized over (t, t+H]).
            nr = (df.groupby("decision_time")[risk_cols].mean().mean(axis=1)).sort_index()
            cp_ret = close_panel.sort_index().pct_change()
            fwd_vol = cp_ret.abs().mean(axis=1).rolling(6).mean().shift(-6)  # next ~1d avg |ret|
            join = pd.concat([nr.rename("nr"), fwd_vol.reindex(nr.index).rename("fv")], axis=1).dropna()
            vol_ic = float("nan"); vol_t = float("nan")
            if len(join) > 50:
                vol_ic = float(join["nr"].corr(join["fv"], method="spearman"))
                # NW t-stat on the per-bar product proxy
                x = (join["nr"] - join["nr"].mean()).values
                y = (join["fv"] - join["fv"].mean()).values
                prod = x * y
                if prod.std() > 0:
                    vol_t = float(newey_west_tstat(prod, lag=6))
            print(f"    [event RISK->vol diagnostic] news-risk predicts forward realized vol: "
                  f"spearman IC={vol_ic:.3f}  NW t={vol_t:.2f}  "
                  f"(>0 & significant => news carries VOLATILITY-timing info)")

            # (b) APPLY: gate the neutral book's gross by news-risk. When news-risk is in
            #     its upper quantile, scale exposure down (delever ahead of turbulence).
            #     Causal: the gate at t uses only news-risk up to t. Parameter-light:
            #     a smooth inverse map gross_mult = 1 - 0.5 * rank(news_risk) in [0.5, 1].
            w_neu3d = _alpha_to_weight_panel(df, fused, close_panel, fcfg, bars_per_year,
                                             smooth_bars=smooth_bars, deadband=deadband,
                                             meta_prob=None, deploy_mode="xs_neutral",
                                             max_vol_scale=max_vol_scale, no_trade_band=no_trade_band,
                                             allow_flat=allow_flat)
            nr_rank = nr.rolling(180, min_periods=30).apply(
                lambda s: (s.rank(pct=True).iloc[-1]), raw=False)   # expanding-ish PIT rank
            gross_mult = (1.0 - 0.5 * nr_rank.fillna(0.5)).clip(0.5, 1.0)
            gm = gross_mult.reindex(w_neu3d.index).ffill().fillna(1.0)
            w3d = w_neu3d.mul(gm, axis=0)
            r3d, t3d, c3d = _panel_to_returns(close_panel, w3d, bars_per_year, fcfg)
            steps.append(("Step3d_event_risk_gate", None, r3d, t3d, c3d))
            side_branch.add("Step3d_event_risk_gate")

    # assemble metrics + incremental significance vs previous MAIN-CHAIN step
    n_steps = sum(1 for _, _, r, _, _ in steps if r is not None)
    # fixed baseline = Step0 returns (for the "total increment vs TSMOM" column)
    base_ret = next((r for nm, _, r, _, _ in steps
                     if nm.startswith("Step0") and r is not None), None)
    rows, prev_ret = [], None
    returns_by_step = {}
    for name, alpha, ret, tn, cs in steps:
        if ret is None or len(ret) < 5:
            rows.append({"step": name, "research_question": RESEARCH_QUESTION.get(name, ""),
                         "status": skip_status.get(name, "skipped (no features)"), "IC": np.nan,
                         "sharpe_ann": np.nan, "deflated_sharpe": np.nan, "incr_NW_t": np.nan,
                         "incr_NW_t_base": np.nan,
                         "ann_turnover": np.nan, "ann_cost_pct": np.nan, "included": False})
            continue
        returns_by_step[name] = ret
        sd = ret.std()
        sharpe = ret.mean() / sd * np.sqrt(bars_per_year) if sd > 0 else np.nan
        dsr = deflated_sharpe_ratio(sharpe / np.sqrt(bars_per_year) if np.isfinite(sharpe) else np.nan,
                                    n_obs=len(ret), n_trials=max(n_steps, 1))
        ic = np.nan
        if alpha is not None:
            mm = np.isfinite(alpha) & np.isfinite(fwd)
            ic = information_coefficient(alpha[mm], fwd[mm])
        # (a) marginal increment vs the previous MAIN-CHAIN step (order/baseline sensitive)
        incr_t, included = np.nan, None
        if prev_ret is not None:
            common = ret.index.intersection(prev_ret.index)
            if len(common) > horizon_bars + 2:
                d = (ret.reindex(common) - prev_ret.reindex(common)).to_numpy()
                incr_t = newey_west_tstat(d, lag=horizon_bars)
                included = bool(np.isfinite(incr_t) and incr_t > t_threshold)
        else:
            included = True  # baseline always "in"
        # (b) TOTAL increment vs the FIXED Step0 TSMOM baseline (stable reference)
        incr_t_base = np.nan
        if base_ret is not None and not name.startswith("Step0"):
            cb = ret.index.intersection(base_ret.index)
            if len(cb) > horizon_bars + 2:
                db = (ret.reindex(cb) - base_ret.reindex(cb)).to_numpy()
                incr_t_base = newey_west_tstat(db, lag=horizon_bars)
        rows.append({"step": name, "research_question": RESEARCH_QUESTION.get(name, ""),
                     "status": kind, "IC": ic, "sharpe_ann": sharpe,
                     "deflated_sharpe": dsr, "incr_NW_t": incr_t, "incr_NW_t_base": incr_t_base,
                     "ann_turnover": tn, "ann_cost_pct": (cs * 100 if np.isfinite(cs) else np.nan),
                     "included": included})
        if name not in side_branch:          # side-branch steps stay OFF the main chain
            prev_ret = ret

    out = pd.DataFrame(rows).set_index("step")
    out.attrs["returns_by_step"] = returns_by_step       # for per-year stability diagnostic
    out.attrs["weight_panels"] = _weight_panels          # signed panels (Step7) for directional stack
    out.attrs["fusion_weights"] = fusion_w               # cross-sectional skill weights per modality
    if narr_diag is not None:
        out.attrs["narrative_diag"] = narr_diag          # xs-IC t vs common-mode-IC t
    if event_diag is not None:
        out.attrs["event_diag"] = event_diag             # LLM event modality decomposition
    return out
