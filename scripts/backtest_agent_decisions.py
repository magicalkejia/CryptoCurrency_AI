# scripts/backtest_agent_decisions.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from etl.data_loader import DataLoader
from backtest.annualization import infer_annual_periods
from backtest.quick import quick_backtest


REQUIRED_COLUMNS = ["decision_time", "symbol", "target_position"]


def _to_naive_timestamp(s: pd.Series) -> pd.Series:
    """
    将 decision_time 统一为 timezone-naive datetime，避免和本地 K 线 index 对齐失败。
    本项目 processed K 线当前通常是 naive timestamp。
    """
    dt = pd.to_datetime(s)
    try:
        if getattr(dt.dt, "tz", None) is not None:
            return dt.dt.tz_convert(None)
    except Exception:
        pass
    return dt


def load_decisions(path: str | Path, approved_only: bool = False) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"决策文件不存在: {path}")

    df = pd.read_parquet(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"决策表缺少字段: {missing}")

    df = df.copy()
    df["decision_time"] = _to_naive_timestamp(df["decision_time"])
    df["symbol"] = df["symbol"].astype(str)
    df["target_position"] = pd.to_numeric(df["target_position"], errors="coerce").fillna(0.0)

    if approved_only and "risk_approved" in df.columns:
        # 未通过风控的决策保留为 0 仓位，而不是直接删除，避免索引缺口导致持仓延续。
        df.loc[~df["risk_approved"].astype(bool), "target_position"] = 0.0

    if df["target_position"].abs().max() > 1:
        raise ValueError("target_position 绝对值不能超过 1；当前回测默认不是杠杆/保证金模型。")

    duplicated = df.duplicated(subset=["decision_time", "symbol"], keep=False)
    if duplicated.any():
        dup = df.loc[duplicated, ["decision_time", "symbol", "target_position"]]
        raise ValueError(f"存在重复 decision_time + symbol，请先去重：\n{dup.head(20)}")

    return df.sort_values(["decision_time", "symbol"]).reset_index(drop=True)


def decisions_to_target_weight(decisions: pd.DataFrame) -> pd.DataFrame:
    """
    将标准决策表转换为 backtest 需要的 target_weight 宽表。
    """
    target_weight = decisions.pivot_table(
        index="decision_time",
        columns="symbol",
        values="target_position",
        aggfunc="last",
    )

    target_weight = target_weight.sort_index().fillna(0.0)
    return target_weight


def load_close_for_decisions(
    decisions: pd.DataFrame,
    timeframe: str = "4h",
    extra_days: int = 2,
) -> pd.DataFrame:
    symbols = sorted(decisions["symbol"].unique().tolist())

    start_ts = decisions["decision_time"].min() - pd.Timedelta(days=extra_days)
    end_ts = decisions["decision_time"].max() + pd.Timedelta(days=extra_days)

    loader = DataLoader()

    matrix = loader.get_crypto_matrix(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_ts,
        end_date=end_ts,
        columns=["close"],
    )

    if not matrix or "close" not in matrix:
        raise RuntimeError("无法读取 close 矩阵。")

    close = matrix["close"].sort_index().ffill().dropna(how="all")
    return close


def align_close_and_target(
    close: pd.DataFrame,
    target_weight: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = close.sort_index()
    target_weight = target_weight.sort_index()

    common_columns = close.columns.intersection(target_weight.columns)
    if len(common_columns) == 0:
        raise RuntimeError("close 和 target_weight 没有共同 symbol。")

    # 只在决策时间点回测。若你的 target_weight index 是 4h bar index，这里应能直接对齐。
    common_index = close.index.intersection(target_weight.index)
    if len(common_index) == 0:
        raise RuntimeError(
            "close 和 target_weight 没有可对齐的时间点。"
            "请检查 decision_time 是否与 processed K 线 timestamp 口径一致。"
        )

    close = close.loc[common_index, common_columns].sort_index()
    target_weight = target_weight.loc[common_index, common_columns].sort_index().fillna(0.0)

    if close.empty or target_weight.empty:
        raise RuntimeError("close 或 target_weight 对齐后为空。")

    return close, target_weight


def infer_annual_days(timeframe: str) -> int:
    return infer_annual_periods(timeframe=timeframe, market="crypto")


def run_backtest_from_decisions(
    decision_path: str | Path,
    timeframe: str = "4h",
    strategy_name: str = "reference_decision_strategy",
    approved_only: bool = False,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0003,
    execution_lag: int = 1,
    display: bool = True,
):
    decisions = load_decisions(decision_path, approved_only=approved_only)
    target_weight = decisions_to_target_weight(decisions)
    close = load_close_for_decisions(decisions, timeframe=timeframe)
    close, target_weight = align_close_and_target(close, target_weight)

    annual_days = infer_annual_days(timeframe)

    print(f"decision rows : {len(decisions)}")
    print(f"backtest bars : {len(close)}")
    print(f"symbols       : {list(close.columns)}")
    print(f"time range    : {close.index.min()} -> {close.index.max()}")
    print("gross exposure summary:")
    print(target_weight.abs().sum(axis=1).describe().to_string())

    result = quick_backtest(
        close=close,
        target_weight=target_weight,
        strategy_name=strategy_name,
        experiment_name=strategy_name,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        execution_lag=execution_lag,
        annual_periods=annual_days,
        market="crypto",
        timeframe=timeframe,
        output_root=str(config.PathConfig.BACKTEST_RESULTS),
        save=True,
        display=display,
    )

    print(f"backtest output: {result.output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decision-path",
        default=str(config.PathConfig.SIGNALS / "reference_decisions.parquet"),
    )
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--strategy-name", default="reference_sma_hysteresis_rule")
    parser.add_argument("--approved-only", action="store_true")
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--slippage-rate", type=float, default=0.0003)
    parser.add_argument("--execution-lag", type=int, default=1)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    run_backtest_from_decisions(
        decision_path=args.decision_path,
        timeframe=args.timeframe,
        strategy_name=args.strategy_name,
        approved_only=args.approved_only,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        execution_lag=args.execution_lag,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
