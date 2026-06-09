# backtest/records.py

from pathlib import Path
import numpy as np
import pandas as pd


def _to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    宽表转长表：
    index = date
    columns = code
    """
    out = df.stack().rename(value_name).reset_index()
    out.columns = ["date", "code", value_name]
    return out


def build_trade_records(
    close: pd.DataFrame,
    target_weight: pd.DataFrame,
    equity_curve: pd.Series,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.0015,
    slippage_rate: float = 0.0,
    execution_lag: int = 1,
    experiment_id: str | None = None,
    strategy_name: str | None = None,
    min_abs_weight_change: float = 1e-8,
) -> pd.DataFrame:
    """
    基于目标权重矩阵生成模拟逐笔交易记录。

    该函数与当前向量化回测逻辑一致：
    - target_weight.shift(execution_lag) 得到实际持仓权重
    - actual_weight 的变化视为调仓交易
    - 使用上一交易日 close 作为成交价格近似
    """

    close = close.sort_index()
    target_weight = target_weight.reindex(index=close.index, columns=close.columns).fillna(0)

    actual_weight = target_weight.shift(execution_lag).fillna(0)
    pre_weight = actual_weight.shift(1).fillna(0)
    delta_weight = actual_weight - pre_weight

    # 为了与 close-to-close 收益模型一致：
    # 今日持仓赚取 close[t] / close[t-1] - 1，
    # 则调仓成交价格用 close[t-1] 近似。
    execution_price = close.shift(1)

    portfolio_value_before_trade = equity_curve.shift(1).fillna(initial_cash)
    portfolio_value_before_trade = portfolio_value_before_trade.reindex(close.index).fillna(initial_cash)

    delta_long = _to_long(delta_weight, "delta_weight")
    delta_long = delta_long[delta_long["delta_weight"].abs() > min_abs_weight_change].copy()

    if delta_long.empty:
        return pd.DataFrame(columns=[
            "experiment_id",
            "strategy_name",
            "trade_date",
            "signal_date",
            "code",
            "side",
            "execution_price",
            "pre_weight",
            "post_weight",
            "delta_weight",
            "portfolio_value_before_trade",
            "trade_amount",
            "quantity",
            "fee",
            "slippage",
            "total_cost",
            "net_trade_amount",
        ])

    price_long = _to_long(execution_price, "execution_price")
    pre_weight_long = _to_long(pre_weight, "pre_weight")
    post_weight_long = _to_long(actual_weight, "post_weight")

    port_long = portfolio_value_before_trade.rename("portfolio_value_before_trade").reset_index()
    port_long.columns = ["date", "portfolio_value_before_trade"]

    trades = (
        delta_long
        .merge(price_long, on=["date", "code"], how="left")
        .merge(pre_weight_long, on=["date", "code"], how="left")
        .merge(post_weight_long, on=["date", "code"], how="left")
        .merge(port_long, on="date", how="left")
    )

    trades = trades.dropna(subset=["execution_price"])
    trades = trades[trades["execution_price"] > 0].copy()

    trades["side"] = np.where(trades["delta_weight"] > 0, "BUY", "SELL")
    trades["trade_amount"] = trades["delta_weight"] * trades["portfolio_value_before_trade"]
    trades["quantity"] = trades["trade_amount"] / trades["execution_price"]

    trades["fee"] = trades["trade_amount"].abs() * fee_rate
    trades["slippage"] = trades["trade_amount"].abs() * slippage_rate
    trades["total_cost"] = trades["fee"] + trades["slippage"]

    # 买入为正，卖出为负；成本单独记录
    trades["net_trade_amount"] = trades["trade_amount"] - np.sign(trades["trade_amount"]) * trades["total_cost"]

    trades = trades.rename(columns={"date": "trade_date"})

    # signal_date：trade_date 向前 execution_lag 个交易日
    dates = list(close.index)
    signal_date_map = {}
    for i, d in enumerate(dates):
        signal_idx = i - execution_lag
        signal_date_map[d] = dates[signal_idx] if signal_idx >= 0 else pd.NaT

    trades["signal_date"] = trades["trade_date"].map(signal_date_map)

    trades.insert(0, "strategy_name", strategy_name)
    trades.insert(0, "experiment_id", experiment_id)

    ordered_cols = [
        "experiment_id",
        "strategy_name",
        "trade_date",
        "signal_date",
        "code",
        "side",
        "execution_price",
        "pre_weight",
        "post_weight",
        "delta_weight",
        "portfolio_value_before_trade",
        "trade_amount",
        "quantity",
        "fee",
        "slippage",
        "total_cost",
        "net_trade_amount",
    ]

    return trades[ordered_cols].sort_values(["trade_date", "code"]).reset_index(drop=True)


def build_position_records(
    close: pd.DataFrame,
    actual_weight: pd.DataFrame,
    equity_curve: pd.Series,
) -> pd.DataFrame:
    """
    生成每日持仓记录。
    用于持仓归因、可视化和复盘。
    """
    close = close.sort_index()
    actual_weight = actual_weight.reindex(index=close.index, columns=close.columns).fillna(0)

    asset_returns = close.pct_change().fillna(0)
    portfolio_value = equity_curve.reindex(close.index).ffill()

    market_value = actual_weight.mul(portfolio_value, axis=0)
    pnl_contribution = actual_weight * asset_returns

    weights_long = _to_long(actual_weight, "weight")
    close_long = _to_long(close, "close")
    value_long = _to_long(market_value, "market_value")
    ret_long = _to_long(asset_returns, "asset_return")
    pnl_long = _to_long(pnl_contribution, "return_contribution")

    positions = (
        weights_long
        .merge(close_long, on=["date", "code"], how="left")
        .merge(value_long, on=["date", "code"], how="left")
        .merge(ret_long, on=["date", "code"], how="left")
        .merge(pnl_long, on=["date", "code"], how="left")
    )

    positions = positions[positions["weight"].abs() > 1e-8].copy()
    positions = positions.sort_values(["date", "code"]).reset_index(drop=True)

    return positions


def save_backtest_records(
    output_dir: str | Path,
    trades: pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
    export_csv: bool = True,
):
    """
    保存交易记录和持仓记录。
    Parquet 作为主存储，CSV 作为兼容导出。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if trades is not None:
        trades.to_parquet(output_dir / "trades.parquet", index=False)
        if export_csv:
            trades.to_csv(output_dir / "trades.csv", index=False, encoding="utf-8-sig")

    if positions is not None:
        positions.to_parquet(output_dir / "positions.parquet", index=False)
        if export_csv:
            positions.to_csv(output_dir / "positions.csv", index=False, encoding="utf-8-sig")