# backtest/metrics.py

import numpy as np
import pandas as pd


def _safe_div(a, b):
    if b is None or b == 0 or pd.isna(b):
        return np.nan
    return a / b


def align_returns(strategy_returns: pd.Series, benchmark_returns: pd.Series | None = None):
    strategy_returns = strategy_returns.dropna().astype(float)

    if benchmark_returns is None:
        return strategy_returns, None

    benchmark_returns = benchmark_returns.dropna().astype(float)

    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]

    return strategy_returns, benchmark_returns


def calc_nav(returns: pd.Series, initial_nav: float = 1.0) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod() * initial_nav


def calc_annual_return(returns: pd.Series, annual_days: int = 252) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    total_return = (1 + returns).prod() - 1
    years = len(returns) / annual_days

    if years <= 0:
        return np.nan

    return (1 + total_return) ** (1 / years) - 1


def calc_max_drawdown_info(nav: pd.Series) -> dict:
    nav = nav.dropna()
    if nav.empty:
        return {
            "max_drawdown": np.nan,
            "max_drawdown_start": None,
            "max_drawdown_end": None,
            "max_drawdown_recovery": None,
            "max_drawdown_days": np.nan,
        }

    running_max = nav.cummax()
    drawdown = nav / running_max - 1

    trough_date = drawdown.idxmin()
    max_drawdown = drawdown.loc[trough_date]

    peak_date = nav.loc[:trough_date].idxmax()
    peak_value = nav.loc[peak_date]

    after_trough = nav.loc[trough_date:]
    recovered = after_trough[after_trough >= peak_value]

    recovery_date = recovered.index[0] if not recovered.empty else None

    if recovery_date is not None:
        max_drawdown_days = (recovery_date - peak_date).days
    else:
        max_drawdown_days = (nav.index[-1] - peak_date).days

    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_start": peak_date,
        "max_drawdown_end": trough_date,
        "max_drawdown_recovery": recovery_date,
        "max_drawdown_days": max_drawdown_days,
    }


def calc_profit_loss_stats(returns: pd.Series) -> dict:
    returns = returns.dropna()

    positive = returns[returns > 0]
    negative = returns[returns < 0]

    profit_count = len(positive)
    loss_count = len(negative)

    avg_profit = positive.mean() if profit_count > 0 else np.nan
    avg_loss = negative.mean() if loss_count > 0 else np.nan

    profit_loss_ratio = _safe_div(avg_profit, abs(avg_loss)) if loss_count > 0 else np.nan

    return {
        "win_rate": _safe_div(profit_count, profit_count + loss_count),
        "profit_count": profit_count,
        "loss_count": loss_count,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_loss_ratio": profit_loss_ratio,
    }


def calc_sortino(returns: pd.Series, annual_days: int = 252) -> float:
    returns = returns.dropna()
    downside = returns[returns < 0]

    if downside.empty:
        return np.nan

    annual_return = calc_annual_return(returns, annual_days)
    downside_vol = downside.std() * np.sqrt(annual_days)

    return _safe_div(annual_return, downside_vol)


def calc_basic_metrics(
    returns: pd.Series,
    annual_days: int = 252,
    prefix: str = "strategy",
) -> dict:
    returns = returns.dropna()

    if returns.empty:
        return {}

    nav = calc_nav(returns)

    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    annual_return = calc_annual_return(returns, annual_days)
    volatility = returns.std() * np.sqrt(annual_days)

    sharpe = _safe_div(annual_return, volatility)
    sortino = calc_sortino(returns, annual_days)

    dd_info = calc_max_drawdown_info(nav)
    calmar = _safe_div(annual_return, abs(dd_info["max_drawdown"]))

    profit_loss = calc_profit_loss_stats(returns)

    metrics = {
        f"{prefix}_total_return": total_return,
        f"{prefix}_annual_return": annual_return,
        f"{prefix}_volatility": volatility,
        f"{prefix}_sharpe": sharpe,
        f"{prefix}_sortino": sortino,
        f"{prefix}_calmar": calmar,
        f"{prefix}_max_drawdown": dd_info["max_drawdown"],
        f"{prefix}_max_drawdown_start": dd_info["max_drawdown_start"],
        f"{prefix}_max_drawdown_end": dd_info["max_drawdown_end"],
        f"{prefix}_max_drawdown_recovery": dd_info["max_drawdown_recovery"],
        f"{prefix}_max_drawdown_days": dd_info["max_drawdown_days"],
    }

    metrics.update(profit_loss)

    return metrics


def calc_benchmark_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    annual_days: int = 252,
) -> dict:
    strategy_returns, benchmark_returns = align_returns(strategy_returns, benchmark_returns)

    if benchmark_returns is None or benchmark_returns.empty:
        return {}

    strategy_nav = calc_nav(strategy_returns)
    benchmark_nav = calc_nav(benchmark_returns)

    excess_returns = strategy_returns - benchmark_returns
    excess_nav = strategy_nav / benchmark_nav

    strategy_ann = calc_annual_return(strategy_returns, annual_days)
    benchmark_ann = calc_annual_return(benchmark_returns, annual_days)

    beta = _safe_div(
        strategy_returns.cov(benchmark_returns),
        benchmark_returns.var()
    )

    alpha = strategy_ann - beta * benchmark_ann if pd.notna(beta) else np.nan

    tracking_error = excess_returns.std() * np.sqrt(annual_days)
    information_ratio = _safe_div(
        excess_returns.mean() * annual_days,
        tracking_error
    )

    excess_total_return = excess_nav.iloc[-1] / excess_nav.iloc[0] - 1
    excess_annual_return = calc_annual_return(excess_returns, annual_days)

    excess_dd_info = calc_max_drawdown_info(excess_nav)

    benchmark_metrics = calc_basic_metrics(
        benchmark_returns,
        annual_days=annual_days,
        prefix="benchmark"
    )

    result = {
        "alpha": alpha,
        "beta": beta,
        "excess_total_return": excess_total_return,
        "excess_annual_return": excess_annual_return,
        "excess_daily_mean": excess_returns.mean(),
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "excess_sharpe": _safe_div(
            excess_returns.mean() * annual_days,
            excess_returns.std() * np.sqrt(annual_days)
        ),
        "excess_win_rate": (strategy_returns > benchmark_returns).mean(),
        "excess_max_drawdown": excess_dd_info["max_drawdown"],
        "excess_max_drawdown_start": excess_dd_info["max_drawdown_start"],
        "excess_max_drawdown_end": excess_dd_info["max_drawdown_end"],
        "excess_max_drawdown_recovery": excess_dd_info["max_drawdown_recovery"],
        "excess_max_drawdown_days": excess_dd_info["max_drawdown_days"],
    }

    result.update(benchmark_metrics)

    return result


def calc_trading_metrics(
    turnover: pd.Series | None = None,
    cost: pd.Series | None = None,
    weights: pd.DataFrame | None = None,
    annual_days: int = 252,
) -> dict:
    metrics = {}

    if turnover is not None:
        turnover = turnover.dropna()
        metrics["avg_turnover"] = turnover.mean()
        metrics["annual_turnover"] = turnover.mean() * annual_days
        metrics["total_turnover"] = turnover.sum()
        metrics["rebalance_count"] = (turnover > 0).sum()

    if cost is not None:
        cost = cost.dropna()
        metrics["total_cost"] = cost.sum()
        metrics["avg_daily_cost"] = cost.mean()

    if weights is not None:
        exposure = weights.abs().sum(axis=1)
        metrics["avg_exposure"] = exposure.mean()
        metrics["max_exposure"] = exposure.max()
        metrics["avg_cash_ratio"] = 1 - exposure.mean()

    return metrics


def calc_full_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    cost: pd.Series | None = None,
    weights: pd.DataFrame | None = None,
    annual_days: int = 252,
) -> dict:
    strategy_returns, benchmark_returns = align_returns(strategy_returns, benchmark_returns)

    metrics = {}
    metrics.update(calc_basic_metrics(strategy_returns, annual_days, prefix="strategy"))

    if benchmark_returns is not None:
        metrics.update(calc_benchmark_metrics(strategy_returns, benchmark_returns, annual_days))

    metrics.update(calc_trading_metrics(turnover, cost, weights, annual_days))

    return metrics