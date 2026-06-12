"""
tests.test_agents
====================
Tests for the Skills registry, the 7 agents and the orchestration graph:
audit logging, quality-gate degradation, Risk-agent veto authority.
"""
import numpy as np
import pandas as pd

from crypto.schemas import FrozenConfig
from crypto.skills.registry import REGISTRY, SkillContext
import crypto.skills.catalog  # noqa
from crypto.agents.agents import new_state, RiskAgent, FusionAgent
from crypto.orchestration.graph import TradingGraph, decision_to_json
from crypto.live.oms import PaperBroker
from etl.model_feature_loader import add_trading_graph_compat_columns


# ---- fake bundle for isolated agent tests --------------------------------- #
class FakeBundle:
    def __init__(self, alpha, direction, meta):
        self._a, self._d, self._m = alpha, direction, meta

    def infer(self, feat_row):
        return {"combined_alpha": self._a, "primary_direction": self._d,
                "meta_trade_prob_raw": self._m, "meta_trade_prob_calibrated": self._m}


def _feat_row():
    return {"found": True, "vol_24": 0.02, "mom_z": 0.8, "barrier_width_pct": 0.02,
            "patchtst_forecast_24h": 0.01, "decision_time": pd.Timestamp("2022-01-01 12:01"),
            "max_feature_availability_ts": pd.Timestamp("2022-01-01 12:01"),
            "funding_rate_z": 0.0}


def test_skill_registry_audits_calls():
    ctx = SkillContext()
    out = REGISTRY.call("detect_regime", ctx, feat_row=_feat_row())
    assert "regime" in out
    assert len(ctx.audit_log) == 1
    assert ctx.audit_log[0].skill == "detect_regime" and ctx.audit_log[0].ok


def test_registry_has_all_categories():
    for cat in ["data", "narrative", "fusion", "risk", "execution", "review"]:
        assert len(REGISTRY.list(cat)) >= 1


def test_risk_agent_veto_on_circuit_breaker():
    fcfg = FrozenConfig()
    ctx = SkillContext()
    st = new_state("BTC/USDT", pd.Timestamp("2022-01-01 12:01"))
    st["feat_row"] = _feat_row()
    st["confidence"] = 0.9
    st["_fusion_out"] = {"combined_alpha": 0.5, "primary_direction": "long",
                         "meta_trade_prob_calibrated": 0.95}
    st["primary_direction"] = "long"
    # without CB -> approved
    RiskAgent().run(st, ctx, fcfg, cb_level=0)
    assert st["risk_approved"] and abs(st["target_position"]) > 0
    # with CB L3 -> vetoed regardless of strong signal
    RiskAgent().run(st, ctx, fcfg, cb_level=3)
    assert (not st["risk_approved"]) and abs(st["target_position"]) < 1e-9
    assert st["action"] == "no_trade"


def test_risk_agent_rejects_below_threshold():
    fcfg = FrozenConfig()
    ctx = SkillContext()
    st = new_state("BTC/USDT", pd.Timestamp("2022-01-01 12:01"))
    st["feat_row"] = _feat_row(); st["confidence"] = 0.9; st["primary_direction"] = "long"
    st["_fusion_out"] = {"combined_alpha": 0.5, "primary_direction": "long",
                         "meta_trade_prob_calibrated": 0.50}  # below p_threshold 0.55
    RiskAgent().run(st, ctx, fcfg, cb_level=0)
    assert not st["risk_approved"]


def test_graph_quality_gate_degradation():
    fcfg = FrozenConfig()
    # feature frame with a PIT violation -> data_quality 0 -> gate aborts
    feats = pd.DataFrame({
        "symbol": ["BTC/USDT"], "decision_time": [pd.Timestamp("2022-01-01 12:01")],
        "ret_1": [0.0], "max_feature_availability_ts": [pd.Timestamp("2022-01-01 13:00")],
    })
    graph = TradingGraph(fcfg, FakeBundle(0.5, "long", 0.9), feats, ["ret_1"], quality_threshold=0.6)
    st = graph.run_decision("BTC/USDT", pd.Timestamp("2022-01-01 12:01"),
                            PaperBroker(), ref_price=100, cb_level=0)
    assert st["action"] == "no_trade"
    assert "data_quality" in st["reason"]
    # skill log should stop before fusion/execution
    skills = [r.skill for r in st["audit_log"]]
    assert "fusion_infer" not in skills and "execute_paper" not in skills


def test_graph_feature_adapter_adds_legacy_aliases():
    feats = pd.DataFrame({
        "symbol": ["BTC/USDT"],
        "decision_time": [pd.Timestamp("2022-01-01 12:01")],
        "ret_24h": [0.01],
        "vol_96h": [0.02],
        "funding_rate_z_30_events": [1.5],
        "max_feature_available_time": [pd.Timestamp("2022-01-01 12:00")],
    })
    out = add_trading_graph_compat_columns(feats)
    assert out.loc[0, "vol_24"] == feats.loc[0, "vol_96h"]
    assert out.loc[0, "mom_z"] == feats.loc[0, "ret_24h"]
    assert out.loc[0, "funding_rate_z"] == feats.loc[0, "funding_rate_z_30_events"]
    assert out.loc[0, "max_feature_availability_ts"] == feats.loc[0, "max_feature_available_time"]


def test_graph_full_flow_audit_complete():
    fcfg = FrozenConfig()
    feats = pd.DataFrame({
        "symbol": ["BTC/USDT"], "decision_time": [pd.Timestamp("2022-01-01 12:01")],
        "ret_1": [0.01], "vol_24": [0.02], "mom_z": [0.8], "barrier_width_pct": [0.02],
        "patchtst_forecast_24h": [0.01],
        "max_feature_availability_ts": [pd.Timestamp("2022-01-01 12:00")],
    })
    graph = TradingGraph(fcfg, FakeBundle(0.5, "long", 0.95),
                         feats, ["ret_1", "vol_24", "mom_z"], quality_threshold=0.6)
    st = graph.run_decision("BTC/USDT", pd.Timestamp("2022-01-01 12:01"),
                            PaperBroker(max_slippage_bps=3), ref_price=100, cb_level=0)
    skills = [r.skill for r in st["audit_log"]]
    # full chain present
    for s in ["get_feature_row", "check_data_quality", "narrative_infer", "detect_regime",
              "fusion_infer", "compute_confidence", "risk_size_and_gate", "execute_paper",
              "review_decision"]:
        assert s in skills
    js = decision_to_json(st, fcfg)
    assert js["action"] in ("long", "no_trade")
    assert "audit_id" in js and "config=" in js["audit_id"]


def test_narrative_agent_is_stub_offline():
    ctx = SkillContext()
    out = REGISTRY.call("narrative_infer", ctx, symbol="BTC/USDT",
                        decision_time=pd.Timestamp("2022-01-01"), texts=None, llm_fn=None)
    assert out["stub"] is True and out["narrative_alpha"] == 0.0
