"""
crypto.agents.agents
=======================
The seven agents (v6 §8): Data, SignalResearch, Narrative, Fusion, Risk,
Execution, Review.  Each agent invokes Skills through the audited REGISTRY and
reads/writes a shared TradingState.  Design principles enforced:
  * agents don't predict price (models do) — agents orchestrate/check/fuse/gate.
  * Risk agent has VETO authority (its rejection overrides everything).
  * every skill call is audited (registry records into ctx.audit_log).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from crypto.skills.registry import REGISTRY, SkillContext
import crypto.skills.catalog  # noqa: F401  (registers skills on import)


def new_state(symbol: str, decision_time) -> Dict[str, Any]:
    """TradingState (v6 §10.2), dict-based for flexibility."""
    return {
        "symbol": symbol, "decision_time": decision_time,
        "data_quality_score": None, "data_warnings": [],
        "feat_row": None, "base_model_pred": {},
        "narrative": None, "narrative_alpha": None,
        "regime": None, "combined_alpha": None, "primary_direction": None,
        "meta_trade_prob_calibrated": None, "confidence": None,
        "risk_approved": None, "risk_level": None, "target_position": 0.0,
        "vol_target_scalar": None, "stop_loss": None, "take_profit": None,
        "circuit_breaker_level": 0, "circuit_breaker_reason": "ok",
        "target_position_pre_portfolio": None,
        "action": "no_trade", "execution_status": None, "filled_price": None,
        "reason": "", "review": None,
    }


class DataAgent:
    def run(self, state, ctx: SkillContext, features, feature_cols):
        row = REGISTRY.call("get_feature_row", ctx, features=features,
                            symbol=state["symbol"], decision_time=state["decision_time"])
        state["feat_row"] = row
        q = REGISTRY.call("check_data_quality", ctx, feat_row=row, feature_cols=feature_cols)
        state["data_quality_score"] = q["data_quality_score"]
        state["data_warnings"] = q["warnings"]
        return state


class SignalResearchAgent:
    """Surfaces base-model predictions already carried in the feature row
    (market model + PatchTST forecasts) for transparency/audit."""
    def run(self, state, ctx, feature_cols):
        out = REGISTRY.call("signal_infer", ctx, feat_row=state["feat_row"] or {},
                            feature_cols=feature_cols)
        state["base_model_pred"] = out["base_model_pred"]
        return state


class NarrativeAgent:
    def run(self, state, ctx, texts=None, llm_fn=None, precomputed=None):
        out = REGISTRY.call("narrative_infer", ctx, symbol=state["symbol"],
                            decision_time=state["decision_time"], texts=texts,
                            llm_fn=llm_fn, precomputed=precomputed)
        state["narrative"] = out
        state["narrative_alpha"] = out.get("narrative_alpha")
        return state


class FusionAgent:
    def run(self, state, ctx, bundle):
        row = state["feat_row"]
        reg = REGISTRY.call("detect_regime", ctx, feat_row=row)
        state["regime"] = reg["regime"]
        fout = REGISTRY.call("fusion_infer", ctx, bundle=bundle, feat_row=row)
        state["combined_alpha"] = fout["combined_alpha"]
        state["primary_direction"] = fout["primary_direction"]
        state["meta_trade_prob_calibrated"] = fout["meta_trade_prob_calibrated"]
        conf = REGISTRY.call("compute_confidence", ctx, feat_row=row, fusion_out=fout,
                             data_quality=state["data_quality_score"])
        state["confidence"] = conf["confidence"]
        state["_fusion_out"] = fout
        return state


class RiskAgent:
    """Highest authority: can veto regardless of signal strength (v6 §8 / 原则3)."""
    def run(self, state, ctx, fcfg, cb_level=0, bars_per_year=2190):
        out = REGISTRY.call("risk_size_and_gate", ctx, fusion_out=state["_fusion_out"],
                            confidence=state["confidence"], feat_row=state["feat_row"],
                            fcfg=fcfg, cb_level=cb_level, bars_per_year=bars_per_year)
        state["risk_approved"] = out["risk_approved"]
        state["risk_level"] = out["risk_level"]
        state["target_position"] = out["target_position"]
        state["vol_target_scalar"] = out.get("vol_target_scalar")
        state["stop_loss"] = out["stop_loss"]
        state["take_profit"] = out["take_profit"]
        state["circuit_breaker_level"] = cb_level
        state["reason"] = out["reason"]
        state["action"] = (state["primary_direction"] if out["risk_approved"] else "no_trade")
        return state


class RiskGuardAgent:
    """B1: computes the live circuit-breaker LEVEL from portfolio equity state and
    writes it onto the state, so the Risk agent reacts to a real cb_level instead
    of a hardcoded 0 (v6 §8.8)."""
    def run(self, state, ctx, circuit_breaker, drawdown=0.0, daily_loss=0.0,
            rolling_abs_daily_returns=None, connection_ok=True,
            reconciliation_ok=True, kill_switch=False):
        out = REGISTRY.call("compute_circuit_breaker", ctx,
                            circuit_breaker=circuit_breaker, drawdown=drawdown,
                            daily_loss=daily_loss,
                            rolling_abs_daily_returns=rolling_abs_daily_returns,
                            connection_ok=connection_ok, reconciliation_ok=reconciliation_ok,
                            kill_switch=kill_switch)
        state["circuit_breaker_level"] = out["cb_level"]
        state["circuit_breaker_reason"] = out["cb_reason"]
        return state


class PortfolioAgent:
    """A2: applies the cross-symbol portfolio risk overlay to a BATCH of per-symbol
    states (v6 §8.2). Mutates each state's target_position in place and returns the
    portfolio report. Positions zeroed by the overlay are demoted to no_trade."""
    def run(self, states, ctx, fcfg, close_panel, equity_drawdown=0.0, bars_per_year=2190):
        intents = {s["symbol"]: {"target_position": s["target_position"],
                                 "direction": s.get("primary_direction")}
                   for s in states if s.get("risk_approved")}
        if not intents:
            return states, {"note": "no_approved_intents"}
        out = REGISTRY.call("portfolio_risk_overlay", ctx, intents=intents,
                            close_panel=close_panel, fcfg=fcfg,
                            equity_drawdown=equity_drawdown, bars_per_year=bars_per_year)
        adj = out["adjusted_positions"]
        for s in states:
            if s["symbol"] in adj:
                s["target_position_pre_portfolio"] = s["target_position"]
                s["target_position"] = adj[s["symbol"]]
                if abs(s["target_position"]) <= 1e-9:
                    s["risk_approved"] = False
                    s["action"] = "no_trade"
                    s["reason"] = (s.get("reason", "") + " | portfolio_overlay_zeroed").strip(" |")
        return states, out["portfolio_report"]


class ExecutionAgent:
    def run(self, state, ctx, broker, ref_price, available_liquidity=1e9):
        # Always call execute_paper so the Execution stage is logged in the audit
        # trail (it short-circuits to NOT_SUBMITTED when there is nothing to trade).
        tgt = state["target_position"] if state.get("risk_approved") else 0.0
        out = REGISTRY.call("execute_paper", ctx, broker=broker, symbol=state["symbol"],
                            target_position=tgt, ref_price=ref_price,
                            available_liquidity=available_liquidity)
        state["execution_status"] = out["execution_status"]
        state["filled_price"] = out["filled_price"]
        return state


class ReviewAgent:
    def run(self, state, ctx):
        out = REGISTRY.call("review_decision", ctx, state=state, ctx_summary=ctx.summary())
        state["review"] = out
        return state
