# backtest/engine.py

from dataclasses import dataclass
import pandas as pd
import numpy as np
from backtest.annualization import resolve_annual_periods
from backtest.records import build_trade_records, build_position_records
@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000
    fee_rate: float = 0.0015
    slippage_rate: float = 0.0
    execution_lag: int = 1
    # Legacy name kept for compatibility.  It means "periods per year", not
    # necessarily calendar days.  New code can set market/timeframe or
    # annual_periods directly.
    annual_days: int | None = None
    annual_periods: int | None = None
    market: str = "stock"
    timeframe: str = "1d"
    warmup_days: int = 0
    warmup_bars: int | None = None
    normalize_weight: bool = True

    @property
    def periods_per_year(self) -> int:
        return resolve_annual_periods(
            annual_periods=self.annual_periods,
            annual_days=self.annual_days,
            market=self.market,
            timeframe=self.timeframe,
        )

    @property
    def warmup_periods(self) -> int:
        return int(self.warmup_bars if self.warmup_bars is not None else self.warmup_days)


def align_data(close: pd.DataFrame, target_weight: pd.DataFrame):
    """对齐价格矩阵和权重矩阵"""
    common_index = close.index.intersection(target_weight.index)
    common_columns = close.columns.intersection(target_weight.columns)

    close = close.loc[common_index, common_columns].sort_index()
    target_weight = target_weight.loc[common_index, common_columns].sort_index()

    return close, target_weight


def normalize_target_weight(target_weight: pd.DataFrame) -> pd.DataFrame:
    """
    对每日权重做归一化。
    如果某天总权重超过 1，则压回 1。
    如果本身小于等于 1，则保留现金仓位。
    """
    gross = target_weight.abs().sum(axis=1)
    scale = gross.where(gross > 1, 1.0)
    return target_weight.div(scale, axis=0).fillna(0)

def run_vector_backtest(
    close,
    target_weight,
    config=None,
    benchmark_returns=None,
    experiment_id=None,
    strategy_name=None,
):
    """
    向量化历史回测。
    
    输入：
    - close: 收盘价宽表，index=date, columns=symbol
    - target_weight: 目标权重宽表，index=date, columns=symbol
    
    输出：
    - dict: returns, equity_curve, weights, turnover, cost, metrics
    """
    config = config or BacktestConfig()

    close, target_weight = align_data(close, target_weight)

    if config.normalize_weight:
        target_weight = normalize_target_weight(target_weight)

    # 资产日收益
    asset_returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

    # 信号滞后，避免未来函数
    actual_weight = target_weight.shift(config.execution_lag).fillna(0)

    # 换手率：今日实际权重 - 昨日实际权重
    turnover = actual_weight.diff().abs().sum(axis=1).fillna(0)

    # 交易摩擦成本
    total_cost_rate = config.fee_rate + config.slippage_rate
    cost = turnover * total_cost_rate

    # 组合日收益
    portfolio_returns = (actual_weight * asset_returns).sum(axis=1) - cost

    if config.warmup_periods > 0:
        portfolio_returns = portfolio_returns.iloc[config.warmup_periods:]
        actual_weight = actual_weight.loc[portfolio_returns.index]
        turnover = turnover.loc[portfolio_returns.index]
        cost = cost.loc[portfolio_returns.index]

    equity_curve = (1 + portfolio_returns).cumprod() * config.initial_cash
    trades = build_trade_records(
        close=close,
        target_weight=target_weight,
        equity_curve=equity_curve,
        initial_cash=config.initial_cash,
        fee_rate=config.fee_rate,
        slippage_rate=getattr(config, "slippage_rate", 0.0),
        execution_lag=config.execution_lag,
        experiment_id=experiment_id,
        strategy_name=strategy_name,
    )

    positions = build_position_records(
        close=close,
        actual_weight=actual_weight,
        equity_curve=equity_curve,
    )

    result = {
        "returns": portfolio_returns,
        "equity_curve": equity_curve,
        "weights": actual_weight,
        "target_weight": target_weight,
        "turnover": turnover,
        "cost": cost,
        "trades": trades,
        "positions": positions,
        "asset_returns": asset_returns,
        "annual_periods": config.periods_per_year,
    }

    return result
