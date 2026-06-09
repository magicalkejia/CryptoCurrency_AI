"""
crypto.schemas
=================
Frozen configuration dataclasses + deterministic hashing for the v6 blueprint.

Implements v6 §6.4 (frozen_config) and audit fixes #20 (environment hash) / #22
(config dataclasses).  Everything that influences a Confirmatory result lives in
one of these dataclasses so it can be frozen and hashed before entering Holdout-A.

No heavy third-party deps: only stdlib + (optionally) the installed package
versions for the environment hash.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, asdict, field
from typing import Literal


# --------------------------------------------------------------------------- #
# Component configs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabelConfig:
    entry_rule: Literal["next_1h_open", "next_1m_open"] = "next_1h_open"
    tp_mult: float = 2.0
    sl_mult: float = 1.0
    vertical_days: int = 5
    atr_window: int = 20                 # ATR(20) on 1h bars
    neutral_threshold_frac: float = 0.5  # neutral_threshold = frac * barrier_width_pct
    intrabar_dual_touch: Literal["ambiguous", "stop_loss_first"] = "ambiguous"


@dataclass(frozen=True)
class CostConfig:
    fee_bps: float = 4.0            # taker fee per leg, bps of notional
    spread_proxy_bps: float = 1.0   # half-spread per leg, bps
    base_slippage_bps: float = 3.0
    k_vol_bps: float = 2.0
    k_liq_bps: float = 10.0
    k_depth: float = 0.10           # depth = k_depth * past-30d same-bucket volume
    min_depth_samples: int = 10
    rounding_bps: float = 0.0
    funding_enabled: bool = True


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    embargo_days: float = 5.0       # default embargo = max_label_horizon + buffer
    data_lag_buffer_days: float = 0.5
    conservative_embargo: bool = False   # if True, embargo = max(180d, label_horizon, buffer)
    max_feature_lookback_days: float = 180.0
    multi_asset_time_block: bool = True


@dataclass(frozen=True)
class ModelConfig:
    objective: Literal["multiclass"] = "multiclass"
    random_seed: int = 42
    # learner params kept generic so the sklearn fallback and lightgbm share them
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05


@dataclass(frozen=True)
class RiskConfig:
    target_annual_vol: float = 0.30
    max_vol_scalar: float = 3.0
    p_threshold: float = 0.55
    edge_cap: float = 0.20
    max_pos_per_symbol: float = 0.25
    gross_cap: float = 1.0
    eps: float = 1e-9


@dataclass(frozen=True)
class FrozenConfig:
    """Everything that must be frozen before Holdout-A (v6 §6.4)."""
    label: LabelConfig = field(default_factory=LabelConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    theta_long: float = 0.05
    theta_short: float = 0.05
    decision_offset_minutes: int = 1

    # -- hashing ---------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "label": asdict(self.label),
            "cost": asdict(self.cost),
            "cv": {k: (str(v) if k == "embargo_days" else v) for k, v in asdict(self.cv).items()},
            "model": asdict(self.model),
            "risk": asdict(self.risk),
            "theta_long": self.theta_long,
            "theta_short": self.theta_short,
            "decision_offset_minutes": self.decision_offset_minutes,
        }

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def label_config_hash(self) -> str:
        blob = json.dumps(asdict(self.label), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def cost_model_hash(self) -> str:
        blob = json.dumps(asdict(self.cost), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def environment_hash() -> str:
    """audit #20: pin python + key package versions for reproducibility."""
    info = {"python": sys.version, "platform": platform.platform()}
    for pkg in ("numpy", "pandas", "scipy", "sklearn", "lightgbm"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[pkg] = "absent"
    blob = json.dumps(info, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def make_audit_id(symbol: str, decision_time, model_tag: str, data_tag: str,
                  code_hash: str, fcfg: FrozenConfig) -> str:
    """v6 §1.4 + audit #23: audit_id carries config & environment hashes."""
    import pandas as pd
    dt = pd.Timestamp(decision_time).strftime("%Y%m%d%H")
    return (f"{symbol.replace('/', '')}__{dt}__model={model_tag}"
            f"__data={data_tag}__code={code_hash}"
            f"__config={fcfg.config_hash()}__env={environment_hash()}")
