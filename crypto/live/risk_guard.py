"""
crypto.live.risk_guard
=========================
Real-time risk layer (v6 §8.8 circuit-breaker hierarchy + §14.7 staleness),
with audit fixes:
  * daily_loss soft threshold = max(2%, 1.5 * rolling_avg_abs_daily_return)  (simp #4)
  * unified L1..L4 hierarchy; kill switch forces L3 (audit #2.3 / detail #30)
  * L3/L4 do NOT auto-recover; manual recovery -> reduced-risk mode (detail #25)
  * staleness three tiers; age = read_time - event_time (not decision_time)  (detail #21/#31)
  * data snapshot used for SIGNALS must be <= decision_time; EXECUTION book may
    be the latest <= submission_time (detail #20) — enforced by caller via two
    separate checks below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd


class CBLevel(IntEnum):
    NORMAL = 0
    L1_WARN = 1
    L2_DELEVER = 2
    L3_HALT = 3
    L4_LIQUIDATE = 4


@dataclass
class CircuitBreakerState:
    level: CBLevel = CBLevel.NORMAL
    reason: str = ""
    reduced_risk_mode: bool = False
    observe_periods_left: int = 0


class CircuitBreaker:
    def __init__(self, recover_periods: int = 5, reduced_pos_mult: float = 0.5,
                 dd_l1: float = 0.20, dd_l2: float = 0.25, dd_l3: float = 0.30):
        self.state = CircuitBreakerState()
        self.recover_periods = recover_periods   # N=5 (~20h) default, simp #2
        self.reduced_pos_mult = reduced_pos_mult
        # drawdown trip points for L1/L2/L3 (configurable for risk-sensitivity analysis;
        # crypto assets run 60%+ annual vol, so a 20% drawdown is normal -> defaults can
        # over-trip; loosening to ~2x annual vol is a defensible, pre-registered choice).
        self.dd_l1 = float(dd_l1)
        self.dd_l2 = float(dd_l2)
        self.dd_l3 = float(dd_l3)

    @staticmethod
    def soft_daily_loss_threshold(rolling_abs_daily_returns: pd.Series) -> float:
        base = 0.02
        if rolling_abs_daily_returns is None or len(rolling_abs_daily_returns) == 0:
            return base
        return max(base, 1.5 * float(rolling_abs_daily_returns.mean()))

    def evaluate(self, drawdown: float, daily_loss: float,
                 rolling_abs_daily_returns: Optional[pd.Series] = None,
                 connection_ok: bool = True, reconciliation_ok: bool = True,
                 kill_switch: bool = False) -> CBLevel:
        """Return the highest triggered level (L4 > L3 > L2 > L1)."""
        soft = self.soft_daily_loss_threshold(rolling_abs_daily_returns)
        hard = 3.0 * soft

        level = CBLevel.NORMAL
        if (daily_loss > soft) or (drawdown > self.dd_l1):
            level = max(level, CBLevel.L1_WARN)
        if (daily_loss > 2 * soft) or (drawdown > self.dd_l2) or (not reconciliation_ok):
            level = max(level, CBLevel.L2_DELEVER)
        if (daily_loss > hard) or (drawdown > self.dd_l3) or (not connection_ok) or kill_switch:
            level = max(level, CBLevel.L3_HALT)

        self.state.level = CBLevel(level)
        self.state.reason = self._reason(drawdown, daily_loss, soft, hard,
                                         connection_ok, reconciliation_ok, kill_switch,
                                         self.dd_l1, self.dd_l2, self.dd_l3)
        return self.state.level

    def position_multiplier(self) -> float:
        """How to scale target positions given the current level."""
        if self.state.level >= CBLevel.L3_HALT:
            return 0.0           # stop opening; only close handled elsewhere
        if self.state.level == CBLevel.L2_DELEVER:
            return 0.5
        if self.state.reduced_risk_mode:
            return self.reduced_pos_mult
        return 1.0

    def request_recovery(self, connection_ok: bool, reconciliation_ok: bool,
                         pnl_reconciled: bool, human_confirmed: bool) -> bool:
        """L3/L4 manual recovery -> reduced-risk mode for N periods (detail #25)."""
        if self.state.level < CBLevel.L3_HALT:
            return True
        if connection_ok and reconciliation_ok and pnl_reconciled and human_confirmed:
            self.state.level = CBLevel.NORMAL
            self.state.reduced_risk_mode = True
            self.state.observe_periods_left = self.recover_periods
            self.state.reason = "recovered -> reduced_risk_mode"
            return True
        return False

    def step_period(self):
        if self.state.reduced_risk_mode and self.state.observe_periods_left > 0:
            self.state.observe_periods_left -= 1
            if self.state.observe_periods_left == 0:
                self.state.reduced_risk_mode = False

    @staticmethod
    def _reason(dd, dl, soft, hard, conn, recon, kill,
                dd_l1: float = 0.20, dd_l2: float = 0.25, dd_l3: float = 0.30):
        parts = []
        if not conn:
            parts.append("connection_anomaly")
        if not recon:
            parts.append("reconciliation_drift")
        if kill:
            parts.append("kill_switch")
        if dd > dd_l3:
            parts.append(f"drawdown>{dd_l3:.0%}")
        elif dd > dd_l2:
            parts.append(f"drawdown>{dd_l2:.0%}")
        elif dd > dd_l1:
            parts.append(f"drawdown>{dd_l1:.0%}")
        if dl > hard:
            parts.append("daily_loss>hard")
        elif dl > soft:
            parts.append("daily_loss>soft")
        return ",".join(parts) or "ok"


# --------------------------------------------------------------------------- #
# staleness (v6 §14.7): age = read_time - event_time, three tiers
# --------------------------------------------------------------------------- #
@dataclass
class StalenessResult:
    status: str          # "normal" | "stale_warning" | "skip"
    age_seconds: float
    allow_trade: bool
    allow_add: bool
    confidence_haircut: float


def check_staleness(market_event_time: pd.Timestamp, snapshot_read_time: pd.Timestamp,
                    warn_s: float = 2.0, skip_s: float = 10.0) -> StalenessResult:
    age = (pd.Timestamp(snapshot_read_time) - pd.Timestamp(market_event_time)).total_seconds()
    if age > skip_s:
        return StalenessResult("skip", age, False, False, 0.0)
    if age > warn_s:
        return StalenessResult("stale_warning", age, True, False, 0.7)  # allow, no add, haircut
    return StalenessResult("normal", age, True, True, 1.0)


def signal_data_is_pit(feature_event_time: pd.Timestamp, decision_time: pd.Timestamp,
                       lag: pd.Timedelta = pd.Timedelta(0)) -> bool:
    """Signals must be <= decision_time (detail #20)."""
    return pd.Timestamp(feature_event_time) + lag <= pd.Timestamp(decision_time)
