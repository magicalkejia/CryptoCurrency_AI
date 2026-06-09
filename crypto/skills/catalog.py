"""
crypto.skills.catalog
========================
Concrete skills registered on REGISTRY, grouped by category (Data / Signal /
Narrative / Fusion / Risk / Execution / Review), wrapping the v6 modules.
These are the tools the Agents invoke (always via REGISTRY.call -> audited).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto.skills.registry import REGISTRY
from crypto.live.risk_guard import CircuitBreaker
from crypto.live.oms import Order, PaperBroker, OrderStatus


# ----------------------------- Data ---------------------------------------- #
@REGISTRY.register("get_feature_row", "data")
def get_feature_row(features: pd.DataFrame, symbol: str, decision_time) -> dict:
    m = (features["symbol"] == symbol) & (features["decision_time"] == decision_time)
    sub = features[m]
    if sub.empty:
        return {"found": False}
    row = sub.iloc[0].to_dict()
    row["found"] = True
    return row


@REGISTRY.register("check_data_quality", "data")
def check_data_quality(feat_row: dict, feature_cols: list) -> dict:
    if not feat_row.get("found", False):
        return {"data_quality_score": 0.0, "warnings": ["row_missing"]}
    vals = [feat_row.get(c) for c in feature_cols]
    n_missing = sum(1 for v in vals if v is None or (isinstance(v, float) and np.isnan(v)))
    score = 1.0 - n_missing / max(len(feature_cols), 1)
    # PIT check (detail #20: signal data must be <= decision_time)
    avail = feat_row.get("max_feature_availability_ts")
    dt = feat_row.get("decision_time")
    warnings = []
    if avail is not None and dt is not None and pd.Timestamp(avail) > pd.Timestamp(dt):
        warnings.append("pit_violation")
        score = 0.0
    if n_missing:
        warnings.append(f"missing={n_missing}")
    return {"data_quality_score": float(score), "warnings": warnings}


# ----------------------------- Narrative (LLM, pluggable) ------------------- #
@REGISTRY.register("narrative_infer", "narrative")
def narrative_infer(symbol: str, decision_time, texts: list | None = None,
                    llm_fn=None) -> dict:
    """
    LLM event/narrative -> structured factor (v6 §7.3/§10.3).
    LLM is PLUGGABLE: pass llm_fn(texts)->dict for Qwen2.5/FinBERT in production.
    Offline default = neutral stub, clearly flagged (never invents a signal).
    Output is structured-only; never a direction/position.
    """
    if llm_fn is not None and texts:
        out = llm_fn(texts)
        out["stub"] = False
        return out
    return {"narrative_alpha": 0.0, "event_risk": 0.0, "rumor_risk": 0.0,
            "dominant_narrative": "none", "source_credibility": 0.0,
            "stub": True}   # honest: no LLM/text available offline


# ----------------------------- Fusion -------------------------------------- #
@REGISTRY.register("detect_regime", "fusion")
def detect_regime(feat_row: dict) -> dict:
    """Rule-based regime (v6 §6.5) from already-computed PIT features."""
    vol = feat_row.get("vol_24")
    mom = feat_row.get("mom_z")
    funding_z = feat_row.get("funding_rate_z")
    regime = "range_bound"
    if funding_z is not None and not (isinstance(funding_z, float) and np.isnan(funding_z)) and abs(funding_z) > 2:
        regime = "funding_overheated"
    elif vol is not None and not np.isnan(vol) and mom is not None and not np.isnan(mom):
        hi_vol = vol > 0.02
        up = mom > 0.5
        dn = mom < -0.5
        if hi_vol and up:
            regime = "high_volatility_uptrend"
        elif hi_vol and dn:
            regime = "panic_selloff"
        elif up:
            regime = "low_volatility_uptrend"
        elif dn:
            regime = "low_volatility_downtrend"
    return {"regime": regime}


@REGISTRY.register("fusion_infer", "fusion")
def fusion_infer(bundle, feat_row: dict) -> dict:
    return bundle.infer(feat_row)


@REGISTRY.register("compute_confidence", "fusion")
def compute_confidence(feat_row: dict, fusion_out: dict, data_quality: float) -> dict:
    """v6 §7.7: non-return reliability, used only as a downward haircut."""
    alpha = fusion_out.get("combined_alpha", 0.0)
    # model agreement: do alpha sign and patchtst 24h forecast agree?
    pf = feat_row.get("patchtst_forecast_24h")
    agree = 1.0
    if pf is not None and not (isinstance(pf, float) and np.isnan(pf)):
        agree = 1.0 if np.sign(pf) == np.sign(alpha) or alpha == 0 else 0.5
    meta = fusion_out.get("meta_trade_prob_calibrated", 0.5)
    cal_conf = abs((meta if meta == meta else 0.5) - 0.5) * 2.0   # 0..1
    conf = float(np.mean([data_quality, agree, cal_conf]))
    return {"confidence": conf}


# ----------------------------- Risk (highest authority) -------------------- #
@REGISTRY.register("risk_size_and_gate", "risk")
def risk_size_and_gate(fusion_out: dict, confidence: float, feat_row: dict,
                       fcfg, cb_level: int = 0) -> dict:
    r = fcfg.risk
    direction = fusion_out["primary_direction"]
    meta = fusion_out.get("meta_trade_prob_calibrated", np.nan)
    dir_sign = {"long": 1, "short": -1, "flat": 0}[direction]

    if direction == "flat" or np.isnan(meta):
        return {"risk_approved": False, "target_position": 0.0, "risk_level": "none",
                "reason": "flat_or_no_meta", "stop_loss": 0.0, "take_profit": 0.0}

    edge = meta - r.p_threshold
    if edge <= 0:
        return {"risk_approved": False, "target_position": 0.0, "risk_level": "low",
                "reason": f"meta_prob {meta:.2f} <= threshold {r.p_threshold}",
                "stop_loss": 0.0, "take_profit": 0.0}

    base = r.max_pos_per_symbol * min(edge / r.edge_cap, 1.0) * dir_sign
    vol = feat_row.get("vol_24")
    realized_ann = (vol * np.sqrt(2190)) if (vol and not np.isnan(vol)) else None
    if realized_ann is None or realized_ann <= r.eps:
        vts = 0.0
    else:
        vts = min(r.target_annual_vol / realized_ann, r.max_vol_scalar)
    conf_haircut = min(1.0, confidence / 0.8)
    pos = base * vts * conf_haircut
    pos = float(np.clip(pos, -r.max_pos_per_symbol, r.max_pos_per_symbol))

    # circuit breaker authority (v6 §8.8): L3+ -> no new position
    cb_mult = 0.0 if cb_level >= 3 else (0.5 if cb_level == 2 else 1.0)
    pos *= cb_mult

    bw = feat_row.get("barrier_width_pct", 0.02) or 0.02
    sl = float(fcfg.label.sl_mult * bw)
    tp = float(fcfg.label.tp_mult * bw)
    approved = abs(pos) > 1e-9
    return {"risk_approved": approved, "target_position": pos, "vol_target_scalar": float(vts),
            "risk_level": "medium" if approved else "blocked",
            "stop_loss": sl, "take_profit": tp,
            "reason": (f"edge={edge:.2f} vts={vts:.2f} conf={conf_haircut:.2f} cb={cb_level}"
                       if approved else f"blocked cb={cb_level}")}


# ----------------------------- Execution ----------------------------------- #
@REGISTRY.register("execute_paper", "execution")
def execute_paper(broker: PaperBroker, symbol: str, target_position: float,
                  ref_price: float, available_liquidity: float = 1e9) -> dict:
    if abs(target_position) < 1e-9:
        return {"execution_status": OrderStatus.NOT_SUBMITTED.value, "filled_qty": 0.0,
                "filled_price": float("nan")}
    side = "buy" if target_position > 0 else "sell"
    o = Order(symbol=symbol, side=side, qty=abs(target_position), order_type="market")
    o = broker.submit(o, ref_price=ref_price, available_liquidity=available_liquidity)
    return {"execution_status": o.status.value, "filled_qty": o.filled_qty,
            "filled_price": o.avg_fill_price}


# ----------------------------- Review -------------------------------------- #
@REGISTRY.register("review_decision", "review")
def review_decision(state: dict, ctx_summary: dict) -> dict:
    """Compose review summary + a (stub) retrain trigger flag."""
    return {"skills_summary": ctx_summary,
            "retraining_required": False,   # wired to drift detection in production
            "review_note": f"{state.get('symbol')} {state.get('action')} "
                           f"pos={state.get('target_position'):.3f} cb={state.get('circuit_breaker_level')}"}
