# scripts/reference_decision_exporter.py

from __future__ import annotations

import argparse
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

ALLOWED_ACTIONS = {
    "open_long",
    "hold_long",
    "close_long",
    "open_short",
    "hold_short",
    "close_short",
    "flat",
    "unknown",
}


def load_close_panel(
    symbols: list[str] | None = None,
    timeframe: str = "4h",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    读取本地 processed crypto 数据，返回 close 宽表。

    返回格式：
    - index: timestamp
    - columns: symbols, e.g. BTC/USDT
    - values: close price
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


def build_position_with_hysteresis(
    raw_score: pd.DataFrame,
    max_position: float = 0.25,
    entry_threshold: float = 0.005,
    exit_threshold: float = 0.0,
    allow_short: bool = True,
) -> pd.DataFrame:
    """
    生成带中性区间和状态机约束的目标仓位。

    设计目的：
    - 避免 raw_score 轻微变号时在多空之间反复横跳；
    - 不允许同一根 bar 直接 long <-> short 翻转；
    - 反向前必须先经过 flat，因此 records 中会出现 close_long / close_short。

    规则：
    - flat 时，score > entry_threshold 开多；
    - flat 时，score < -entry_threshold 开空；
    - 持多时，score < exit_threshold 平多；
    - 持空时，score > -exit_threshold 平空；
    - 持仓状态下不直接反向。
    """
    if max_position <= 0 or max_position > 1:
        raise ValueError("max_position 必须在 (0, 1] 内。")
    if entry_threshold < 0:
        raise ValueError("entry_threshold 不能为负。")
    if exit_threshold < 0:
        raise ValueError("exit_threshold 不能为负。")
    if exit_threshold > entry_threshold:
        raise ValueError("exit_threshold 不应大于 entry_threshold，否则滞后区间无意义。")

    target = pd.DataFrame(0.0, index=raw_score.index, columns=raw_score.columns)

    for symbol in raw_score.columns:
        pos = 0.0

        for dt in raw_score.index:
            score = raw_score.loc[dt, symbol]

            if pd.isna(score):
                pos = 0.0
                target.loc[dt, symbol] = pos
                continue

            if abs(pos) < 1e-12:
                if score > entry_threshold:
                    pos = max_position
                elif allow_short and score < -entry_threshold:
                    pos = -max_position
                else:
                    pos = 0.0

            elif pos > 0:
                # 持多时只允许继续持多或平多，不直接开空。
                if score < exit_threshold:
                    pos = 0.0

            elif pos < 0:
                # 持空时只允许继续持空或平空，不直接开多。
                if score > -exit_threshold:
                    pos = 0.0

            target.loc[dt, symbol] = pos

    return target


def classify_action(pre_pos: float, post_pos: float) -> str:
    """
    按仓位变化给标准决策表标注 action。
    注意：这是决策动作，不是交易撮合 side。
    """
    eps = 1e-12

    if abs(pre_pos) < eps and post_pos > eps:
        return "open_long"
    if pre_pos > eps and post_pos > eps:
        return "hold_long"
    if pre_pos > eps and abs(post_pos) < eps:
        return "close_long"

    if abs(pre_pos) < eps and post_pos < -eps:
        return "open_short"
    if pre_pos < -eps and post_pos < -eps:
        return "hold_short"
    if pre_pos < -eps and abs(post_pos) < eps:
        return "close_short"

    if abs(pre_pos) < eps and abs(post_pos) < eps:
        return "flat"

    # 正常情况下 build_position_with_hysteresis 不会产生 flip。
    return "unknown"


def reason_from_action(
    action: str,
    fast_window: int,
    slow_window: int,
    entry_threshold: float,
    exit_threshold: float,
) -> str:
    if action == "open_long":
        return f"SMA{fast_window}/SMA{slow_window} score crossed above +{entry_threshold:.4f}"
    if action == "open_short":
        return f"SMA{fast_window}/SMA{slow_window} score crossed below -{entry_threshold:.4f}"
    if action == "close_long":
        return f"long closed because score fell below +{exit_threshold:.4f}"
    if action == "close_short":
        return f"short closed because score rose above -{exit_threshold:.4f}"
    if action == "hold_long":
        return "holding long under hysteresis rule"
    if action == "hold_short":
        return "holding short under hysteresis rule"
    if action == "flat":
        return "flat: insufficient signal or neutral zone"
    return "unknown action"


def build_reference_decisions(
    close: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 120,
    max_position: float = 0.25,
    entry_threshold: float = 0.005,
    exit_threshold: float = 0.0,
    allow_short: bool = True,
    agent_name: str = "reference_sma_hysteresis_rule",
) -> pd.DataFrame:
    """
    用一个稳定一点的参考均线状态机生成标准输出表。

    该策略只用于验证接口：
    local close panel -> standard agent_decisions.parquet -> target_weight -> backtest。
    不应作为最终交易策略结论。
    """
    close = close.sort_index().copy()

    fast_ma = close.rolling(fast_window, min_periods=fast_window).mean()
    slow_ma = close.rolling(slow_window, min_periods=slow_window).mean()

    raw_score = fast_ma / slow_ma - 1
    raw_score = raw_score.replace([np.inf, -np.inf], np.nan)

    target_position = build_position_with_hysteresis(
        raw_score=raw_score,
        max_position=max_position,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        allow_short=allow_short,
    )

    # 均线还没形成时强制空仓。
    target_position = target_position.where(slow_ma.notna(), 0.0)
    pre_position = target_position.shift(1).fillna(0.0)

    created_at = pd.Timestamp.now(tz="UTC")
    rows: list[dict] = []

    for decision_time in target_position.index:
        for symbol in target_position.columns:
            pre_pos = float(pre_position.loc[decision_time, symbol])
            pos = float(target_position.loc[decision_time, symbol])
            score = raw_score.loc[decision_time, symbol]
            score = 0.0 if pd.isna(score) else float(score)

            action = classify_action(pre_pos, pos)
            reason = reason_from_action(
                action=action,
                fast_window=fast_window,
                slow_window=slow_window,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
            )

            audit_log = [
                {
                    "skill": "reference_sma_hysteresis_rule",
                    "category": "signal",
                    "ok": True,
                    "detail": {
                        "fast_window": fast_window,
                        "slow_window": slow_window,
                        "max_position": max_position,
                        "entry_threshold": entry_threshold,
                        "exit_threshold": exit_threshold,
                        "allow_short": allow_short,
                        "pre_position": pre_pos,
                        "post_position": pos,
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
                    "confidence": min(abs(score) / max(entry_threshold, 1e-12), 1.0),
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

    invalid_actions = sorted(set(out["action"]) - ALLOWED_ACTIONS)
    if invalid_actions:
        raise ValueError(f"非法 action: {invalid_actions}")

    out["target_position"] = pd.to_numeric(out["target_position"], errors="coerce").fillna(0.0)
    out["signal_score"] = pd.to_numeric(out["signal_score"], errors="coerce").fillna(0.0)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0)
    out["confidence"] = out["confidence"].clip(0.0, 1.0)

    out["risk_approved"] = out["risk_approved"].astype(bool)

    if out["target_position"].abs().max() > 1:
        raise ValueError("target_position 绝对值不能超过 1")

    duplicated = out.duplicated(subset=["decision_time", "symbol"], keep=False)
    if duplicated.any():
        dup = out.loc[duplicated, ["decision_time", "symbol"]]
        raise ValueError(f"存在重复 decision_time + symbol:\n{dup.head(20)}")

    return out.sort_values(["decision_time", "symbol"]).reset_index(drop=True)


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
    print("action distribution:")
    print(decisions["action"].value_counts(dropna=False).to_string())

    return output_path


def _parse_symbols(raw: list[str] | None) -> list[str]:
    if raw:
        return raw
    return list(config.TargetConfig.COINS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--fast-window", type=int, default=20)
    parser.add_argument("--slow-window", type=int, default=120)
    parser.add_argument("--max-position", type=float, default=0.25)
    parser.add_argument("--entry-threshold", type=float, default=0.005)
    parser.add_argument("--exit-threshold", type=float, default=0.0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument(
        "--output",
        default=str(config.PathConfig.SIGNALS / "reference_decisions.parquet"),
    )
    args = parser.parse_args()

    config.PathConfig.SIGNALS.mkdir(parents=True, exist_ok=True)

    close = load_close_panel(
        symbols=_parse_symbols(args.symbols),
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    decisions = build_reference_decisions(
        close=close,
        fast_window=args.fast_window,
        slow_window=args.slow_window,
        max_position=args.max_position,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        allow_short=not args.long_only,
    )

    save_agent_decisions(decisions, output_path=args.output)
    print(decisions.tail())


if __name__ == "__main__":
    main()
