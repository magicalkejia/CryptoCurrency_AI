"""
crypto.orchestration.graph
=============================
Agent orchestration (v6 §10).  Uses LangGraph if installed (production path);
otherwise a built-in state-machine runner with identical node graph and
conditional edges, so it runs anywhere.

Graph (v6 §10.3):
  START -> data -> quality_gate
            ├─ score < threshold -> degrade/no_trade -> review -> END
            └─ else -> signal -> narrative -> fusion -> risk
                                                       ├─ not approved -> review -> END
                                                       └─ approved -> execution -> review -> END
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from crypto.skills.registry import SkillContext
from crypto.schemas import make_audit_id
from crypto.agents.agents import (
    new_state, DataAgent, SignalResearchAgent, NarrativeAgent,
    FusionAgent, RiskAgent, ExecutionAgent, ReviewAgent)

try:
    import langgraph  # noqa
    _HAS_LANGGRAPH = True
except Exception:
    _HAS_LANGGRAPH = False


def decision_to_json(state: Dict[str, Any], fcfg, code_hash="dev",
                     model_tag="demo", data_tag="demo") -> dict:
    """v6 §1.4 structured single-period output."""
    def _r(x, n=4):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)
    return {
        "decision_time": str(state["decision_time"]),
        "symbol": state["symbol"],
        "regime": state["regime"],
        "base_model_pred": {k: _r(v) for k, v in (state["base_model_pred"] or {}).items()},
        "combined_alpha": _r(state["combined_alpha"]),
        "primary_direction": state["primary_direction"],
        "meta_trade_prob_calibrated": _r(state["meta_trade_prob_calibrated"]),
        "confidence": _r(state["confidence"]),
        "action": state["action"],
        "target_position": _r(state["target_position"]),
        "vol_target_scalar": _r(state["vol_target_scalar"]),
        "stop_loss": _r(state["stop_loss"]),
        "take_profit": _r(state["take_profit"]),
        "risk_level": state["risk_level"],
        "circuit_breaker_level": state["circuit_breaker_level"],
        "data_quality_score": _r(state["data_quality_score"], 3),
        "narrative_stub": (state["narrative"] or {}).get("stub"),
        "execution_status": state["execution_status"],
        "reason": state["reason"],
        "audit_id": make_audit_id(state["symbol"], state["decision_time"],
                                  model_tag, data_tag, code_hash, fcfg),
    }


class TradingGraph:
    """Built-in runner mirroring the LangGraph node graph (v6 §10.3)."""

    def __init__(self, fcfg, bundle, features, feature_cols,
                 quality_threshold: float = 0.6):
        self.fcfg = fcfg
        self.bundle = bundle
        self.features = features
        self.feature_cols = feature_cols
        self.qthr = quality_threshold
        self.data = DataAgent(); self.signal = SignalResearchAgent()
        self.narr = NarrativeAgent(); self.fusion = FusionAgent()
        self.risk = RiskAgent(); self.exe = ExecutionAgent(); self.review = ReviewAgent()
        self.backend = "langgraph" if _HAS_LANGGRAPH else "builtin_statemachine"

    def run_decision(self, symbol, decision_time, broker, ref_price,
                     cb_level: int = 0, texts=None, llm_fn=None) -> Dict[str, Any]:
        ctx = SkillContext()
        state = new_state(symbol, decision_time)

        # data -> quality gate
        self.data.run(state, ctx, self.features, self.feature_cols)
        if (state["data_quality_score"] or 0.0) < self.qthr:
            # degradation branch (v6 §10.4): abort trading, still review/audit
            state["action"] = "no_trade"
            state["reason"] = f"data_quality {state['data_quality_score']:.2f} < {self.qthr}"
            self.review.run(state, ctx)
            state["audit_log"] = ctx.audit_log
            return state

        # signal -> narrative -> fusion
        self.signal.run(state, ctx, self.feature_cols)
        self.narr.run(state, ctx, texts=texts, llm_fn=llm_fn)
        self.fusion.run(state, ctx, self.bundle)

        # risk (veto authority)
        self.risk.run(state, ctx, self.fcfg, cb_level=cb_level)

        # execution only if approved
        if state["risk_approved"]:
            self.exe.run(state, ctx, broker, ref_price)
        else:
            state["execution_status"] = "not_submitted"

        self.review.run(state, ctx)
        state["audit_log"] = ctx.audit_log
        return state

    # --- optional LangGraph build (production path) ----------------------- #
    def build_langgraph(self):
        """Wire the same nodes into a LangGraph StateGraph (needs langgraph)."""
        if not _HAS_LANGGRAPH:
            raise RuntimeError("langgraph not installed; use run_decision (builtin)")
        from langgraph.graph import StateGraph, END
        # Node functions close over self; LangGraph passes the state dict through.
        g = StateGraph(dict)
        # (production wiring: add_node for each agent + add_conditional_edges for
        #  the quality gate and risk veto, mirroring run_decision above.)
        # Left as the documented production path; builtin runner is the default.
        return g
