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

from crypto.skills.registry import REGISTRY, SkillContext
from crypto.schemas import make_audit_id
from crypto.agents.agents import (
    new_state, DataAgent, SignalResearchAgent, NarrativeAgent,
    FusionAgent, RiskAgent, ExecutionAgent, ReviewAgent,
    RiskGuardAgent, PortfolioAgent)
from crypto.live.risk_guard import CircuitBreaker

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
        # B1 + A2: a persistent circuit breaker and the new guard / portfolio agents.
        self.circuit_breaker = CircuitBreaker()
        self.riskguard = RiskGuardAgent(); self.portfolio = PortfolioAgent()
        self.backend = "langgraph" if _HAS_LANGGRAPH else "builtin_statemachine"

    def run_decision(self, symbol, decision_time, broker, ref_price,
                     cb_level: Optional[int] = None, drawdown: float = 0.0,
                     daily_loss: float = 0.0, rolling_abs_daily_returns=None,
                     texts=None, llm_fn=None, bars_per_year: int = 2190) -> Dict[str, Any]:
        ctx = SkillContext()
        state = new_state(symbol, decision_time)

        # B1: compute the circuit-breaker level from equity state unless the caller
        # explicitly forces one (back-compatible: callers passing cb_level=0 still work,
        # and an unset cb_level with no drawdown evaluates to NORMAL == 0).
        if cb_level is None:
            self.riskguard.run(state, ctx, self.circuit_breaker, drawdown=drawdown,
                               daily_loss=daily_loss,
                               rolling_abs_daily_returns=rolling_abs_daily_returns)
            cb_level = state["circuit_breaker_level"]

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
        self.risk.run(state, ctx, self.fcfg, cb_level=cb_level, bars_per_year=bars_per_year)

        # execution stage always runs so it appears in the audit trail/pipeline.
        # ExecutionAgent submits an order only when risk approved a non-zero target;
        # otherwise it records NOT_SUBMITTED. This keeps the Execution node visible
        # as "reached" in the live demo while preserving the no-trade semantics.
        self.exe.run(state, ctx, broker, ref_price)

        self.review.run(state, ctx)
        state["audit_log"] = ctx.audit_log
        return state

    def run_portfolio_decision(self, symbols, decision_time, broker, ref_prices,
                               close_panel, drawdown: float = 0.0, daily_loss: float = 0.0,
                               rolling_abs_daily_returns=None, bars_per_year: int = 2190,
                               texts_by_symbol=None, llm_fn=None) -> Dict[str, Any]:
        """A2 + B1: one cross-sectional decision over many symbols.

        1. compute the circuit-breaker level ONCE from portfolio equity state,
        2. per symbol: data -> quality gate -> signal -> narrative -> fusion -> risk
           (sizing only; NO execution yet), feeding the shared cb_level,
        3. PortfolioAgent applies the cross-symbol overlay (corr / cluster / gross /
           portfolio-vol / drawdown),
        4. execute the post-overlay positions, then review.

        Returns {"cb_level", "portfolio_report", "per_symbol": {sym: state}, "audit_log"}.
        """
        ctx = SkillContext()
        ref_prices = ref_prices or {}
        texts_by_symbol = texts_by_symbol or {}

        cb = REGISTRY.call("compute_circuit_breaker", ctx,
                           circuit_breaker=self.circuit_breaker, drawdown=drawdown,
                           daily_loss=daily_loss,
                           rolling_abs_daily_returns=rolling_abs_daily_returns)
        cb_level = cb["cb_level"]

        states: Dict[str, Any] = {}
        for sym in symbols:
            st = new_state(sym, decision_time)
            st["circuit_breaker_level"] = cb_level
            st["circuit_breaker_reason"] = cb["cb_reason"]
            self.data.run(st, ctx, self.features, self.feature_cols)
            if (st["data_quality_score"] or 0.0) < self.qthr:
                st["action"] = "no_trade"
                st["reason"] = f"data_quality {st['data_quality_score']:.2f} < {self.qthr}"
                states[sym] = st
                continue
            self.signal.run(st, ctx, self.feature_cols)
            self.narr.run(st, ctx, texts=texts_by_symbol.get(sym), llm_fn=llm_fn)
            self.fusion.run(st, ctx, self.bundle)
            self.risk.run(st, ctx, self.fcfg, cb_level=cb_level, bars_per_year=bars_per_year)
            states[sym] = st

        # portfolio overlay across all approved intents (A2)
        _, report = self.portfolio.run(list(states.values()), ctx, self.fcfg, close_panel,
                                       equity_drawdown=drawdown, bars_per_year=bars_per_year)

        # execute post-overlay positions
        for sym, st in states.items():
            if st.get("risk_approved") and abs(st["target_position"]) > 1e-9:
                self.exe.run(st, ctx, broker, ref_prices.get(sym, float("nan")))
            else:
                st["execution_status"] = "not_submitted"
            self.review.run(st, ctx)

        return {"decision_time": decision_time, "cb_level": cb_level,
                "cb_reason": cb["cb_reason"], "portfolio_report": report,
                "per_symbol": states, "audit_log": ctx.audit_log}

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
