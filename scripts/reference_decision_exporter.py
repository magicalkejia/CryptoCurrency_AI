# scripts/reference_decision_exporter.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from etl.data_loader import DataLoader


STANDARD_COLUMNS = [
    "decision_time",
    "symbol",
    "agent_name",
    "action",
    "target_position",
    "signal_score",
    "confidence",
    "risk_approved",
    "reason",
    "audit_log",
    "created_at",
]


def load_close_panel(
    symbols=None,
    timeframe: str = "4h",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    读取本地 processed crypto 数据，返回 close 宽表。
    """
    loader = DataLoader()

    matrix = loader.get_crypto_matrix(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        columns=["close"],
    )

    if not matrix or "close" not in matrix:
        raise RuntimeError(
            "没有读取到 close 矩阵。请先运行 python main.py --mode crypto，并确认 processed 数据存在。"
        )

    close = matrix["close"].sort_index()
    close = close.ffill().dropna(how="all")

    if close.empty:
        raise RuntimeError("close 矩阵为空。")

    return close


def build_reference_decisions(
    close: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 60,
    max_position: float = 0.25,
    agent_name: str = "reference_sma_rule",
) -> pd.DataFrame:
    """
    用一个简单均线规则生成标准输出表。

    规则：
    - fast_ma > slow_ma: target_position = max_position
    - fast_ma <= slow_ma: target_position = 0
    """
    close = close.sort_index().copy()

    fast_ma = close.rolling(fast_window, min_periods=fast_window).mean()
    slow_ma = close.rolling(slow_window, min_periods=slow_window).mean()

    raw_score = fast_ma / slow_ma - 1
    raw_score = raw_score.replace([np.inf, -np.inf], np.nan)

    target_position = pd.DataFrame(
        0.0,
        index=close.index,
        columns=close.columns,
    )

    target_position = target_position.mask(raw_score > 0, max_position)
    target_position = target_position.mask(raw_score < 0, -max_position)
    target_position = target_position.where(slow_ma.notna(), 0.0)

    created_at = pd.Timestamp.now("UTC")

    rows = []

    for decision_time in target_position.index:
        for symbol in target_position.columns:
            pos = float(target_position.loc[decision_time, symbol])
            score = raw_score.loc[decision_time, symbol]

            if pd.isna(score):
                score = 0.0
            else:
                score = float(score)

            if pos > 0:
                action = "long"
                reason = f"SMA{fast_window} > SMA{slow_window}"
            elif pos < 0:
                action = "short"
                reason = f"SMA{fast_window} <= SMA{slow_window}"

            else:
                action = "flat"
                reason = f"SMA insufficient history"

            audit_log = [
                {
                    "skill": "reference_sma_rule",
                    "category": "signal",
                    "ok": True,
                    "detail": {
                        "fast_window": fast_window,
                        "slow_window": slow_window,
                        "max_position": max_position,
                    },
                }
            ]

            rows.append(
                {
                    "decision_time": decision_time,
                    "symbol": symbol,
                    "agent_name": agent_name,
                    "action": action,
                    "target_position": pos,
                    "signal_score": score,
                    "confidence": min(abs(score) * 100, 1.0),
                    "risk_approved": True,
                    "reason": reason,
                    "audit_log": json.dumps(audit_log, ensure_ascii=False),
                    "created_at": created_at,
                }
            )

    decisions = pd.DataFrame(rows)
    return validate_agent_decisions(decisions)


def validate_agent_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """
    校验标准输出表，防止输出格式漂移。
    """
    missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"agent decision table 缺少字段: {missing}")

    out = df[STANDARD_COLUMNS].copy()

    out["decision_time"] = pd.to_datetime(out["decision_time"])
    out["created_at"] = pd.to_datetime(out["created_at"])

    out["symbol"] = out["symbol"].astype(str)
    out["agent_name"] = out["agent_name"].astype(str)
    out["action"] = out["action"].astype(str)
    out["reason"] = out["reason"].astype(str)
    out["audit_log"] = out["audit_log"].astype(str)

    out["target_position"] = pd.to_numeric(out["target_position"], errors="coerce").fillna(0.0)
    out["signal_score"] = pd.to_numeric(out["signal_score"], errors="coerce").fillna(0.0)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0)

    out["risk_approved"] = out["risk_approved"].astype(bool)

    if out["target_position"].abs().max() > 1:
        raise ValueError("target_position 绝对值不能超过 1")

    duplicated = out.duplicated(subset=["decision_time", "symbol"], keep=False)
    if duplicated.any():
        dup = out.loc[duplicated, ["decision_time", "symbol"]]
        raise ValueError(f"存在重复 decision_time + symbol:\n{dup.head(20)}")

    out = out.sort_values(["decision_time", "symbol"]).reset_index(drop=True)
    return out


def save_agent_decisions(
    decisions: pd.DataFrame,
    output_path: str | Path | None = None,
) -> Path:
    if output_path is None:
        output_path = config.PathConfig.SIGNALS / "reference_decisions.parquet"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    decisions.to_parquet(output_path, index=False)
    decisions.to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    print(f"saved parquet: {output_path}")
    print(f"saved csv    : {output_path.with_suffix('.csv')}")

    return output_path


def main():
    # 这里只保证 signals 目录存在。
    config.PathConfig.SIGNALS.mkdir(parents=True, exist_ok=True)

    symbols = config.TargetConfig.COINS

    close = load_close_panel(
        symbols=symbols,
        timeframe="4h",
        start_date="2022-01-01",
    )

    decisions = build_reference_decisions(
        close=close,
        fast_window=20,
        slow_window=60,
        max_position=0.25,
    )

    save_agent_decisions(
        decisions,
        output_path=config.PathConfig.SIGNALS / "reference_decisions.parquet",
    )

    print(decisions.tail())


if __name__ == "__main__":
    main()