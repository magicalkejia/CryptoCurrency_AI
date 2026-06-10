# scripts/backtest_agent_decisions.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from etl.data_loader import DataLoader
from backtest.quick import quick_backtest


def load_decisions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"决策文件不存在: {path}")

    df = pd.read_parquet(path)
    df["decision_time"] = pd.to_datetime(df["decision_time"])
    return df.sort_values(["decision_time", "symbol"]).reset_index(drop=True)


def decisions_to_target_weight(decisions: pd.DataFrame) -> pd.DataFrame:
    """
    将标准决策表转换为 backtest 需要的 target_weight 宽表。
    """
    required = ["decision_time", "symbol", "target_position"]
    missing = [c for c in required if c not in decisions.columns]
    if missing:
        raise ValueError(f"决策表缺少字段: {missing}")

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
) -> pd.DataFrame:
    symbols = sorted(decisions["symbol"].unique().tolist())

    start_date = decisions["decision_time"].min().strftime("%Y-%m-%d")
    end_date = decisions["decision_time"].max().strftime("%Y-%m-%d")

    loader = DataLoader()

    matrix = loader.get_crypto_matrix(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        columns=["close"],
    )

    if not matrix or "close" not in matrix:
        raise RuntimeError("无法读取 close 矩阵。")

    close = matrix["close"].sort_index().ffill()
    return close


def run_backtest_from_decisions(
    decision_path: str | Path,
    timeframe: str = "4h",
    strategy_name: str = "reference_decision_strategy",
):
    decisions = load_decisions(decision_path)

    target_weight = decisions_to_target_weight(decisions)
    close = load_close_for_decisions(decisions, timeframe=timeframe)

    # 对齐 index 和 columns
    common_index = close.index.intersection(target_weight.index)
    common_columns = close.columns.intersection(target_weight.columns)

    close = close.loc[common_index, common_columns].sort_index()
    target_weight = target_weight.loc[common_index, common_columns].sort_index()

    if close.empty:
        raise RuntimeError("close 和 target_weight 没有可对齐的时间区间。")

    annual_days = {
        "1d": 365,
        "4h": 365 * 6,
        "1h": 365 * 24,
    }.get(timeframe, 365)

    result = quick_backtest(
        close=close,
        target_weight=target_weight,
        strategy_name=strategy_name,
        experiment_name=strategy_name,
        fee_rate=0.0004,
        slippage_rate=0.0003,
        execution_lag=1,
        annual_days=annual_days,
        output_root=str(config.PathConfig.BACKTEST_RESULTS),
        save=True,
        display=True,
    )

    return result


def main():
    path = config.PathConfig.SIGNALS / "reference_decisions.parquet"

    run_backtest_from_decisions(
        decision_path=path,
        timeframe="4h",
        strategy_name="reference_sma_rule",
    )


if __name__ == "__main__":
    main()