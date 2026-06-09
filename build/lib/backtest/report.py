# backtest/report.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs

from backtest.metrics import calc_max_drawdown_info


def _ensure_output_path(output_path):
    if output_path is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _align_returns(strategy_returns, benchmark_returns=None):
    strategy_returns = strategy_returns.dropna().astype(float)

    if benchmark_returns is None:
        return strategy_returns, None

    benchmark_returns = benchmark_returns.dropna().astype(float)

    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]

    return strategy_returns, benchmark_returns


def print_basic_report(returns, benchmark_returns=None):
    """
    在 notebook / terminal 中打印 quantstats 基础指标。
    """
    returns, benchmark_returns = _align_returns(returns, benchmark_returns)

    if benchmark_returns is not None:
        qs.reports.metrics(returns, benchmark=benchmark_returns, mode="basic")
    else:
        qs.reports.metrics(returns, mode="basic")


def save_html_report(
    returns,
    output_path,
    title="Backtest Report",
    benchmark_returns=None,
):
    """
    保存 quantstats HTML 报告。
    """
    output_path = _ensure_output_path(output_path)

    returns, benchmark_returns = _align_returns(returns, benchmark_returns)

    if benchmark_returns is not None:
        qs.reports.html(
            returns,
            benchmark=benchmark_returns,
            title=title,
            output=str(output_path),
        )
    else:
        qs.reports.html(
            returns,
            title=title,
            output=str(output_path),
        )

    return output_path


def plot_nav(strategy_returns, benchmark_returns=None, output_path=None):
    """
    绘制策略净值、基准净值、超额净值。
    """
    output_path = _ensure_output_path(output_path)

    strategy_returns, benchmark_returns = _align_returns(
        strategy_returns,
        benchmark_returns,
    )

    strategy_nav = (1 + strategy_returns).cumprod()

    plt.figure(figsize=(12, 5))
    plt.plot(strategy_nav.index, strategy_nav, label="Strategy NAV")

    if benchmark_returns is not None:
        benchmark_nav = (1 + benchmark_returns).cumprod()
        excess_nav = strategy_nav / benchmark_nav

        plt.plot(benchmark_nav.index, benchmark_nav, label="Benchmark NAV")
        plt.plot(excess_nav.index, excess_nav, label="Excess NAV")

    plt.title("Strategy vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)

    plt.close()


def plot_drawdown(strategy_returns, output_path=None):
    """
    绘制策略回撤，并标记最大回撤区间。
    """
    output_path = _ensure_output_path(output_path)

    strategy_returns = strategy_returns.dropna().astype(float)

    nav = (1 + strategy_returns).cumprod()
    running_max = nav.cummax()
    drawdown = nav / running_max - 1

    dd_info = calc_max_drawdown_info(nav)

    plt.figure(figsize=(12, 4))
    plt.plot(drawdown.index, drawdown, label="Drawdown")
    plt.axhline(0, linewidth=1)

    start = dd_info.get("max_drawdown_start")
    end = dd_info.get("max_drawdown_end")

    if start is not None and end is not None:
        plt.axvspan(start, end, alpha=0.2, label="Max Drawdown Period")

    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)

    plt.close()


def calc_monthly_returns(returns):
    """
    计算月度收益矩阵。
    输出：index=year, columns=month, values=monthly_return
    """
    returns = returns.dropna().astype(float)

    nav = (1 + returns).cumprod()

    # pandas 新版本更推荐 ME；如果你的版本不支持 ME，可改回 M
    monthly_nav = nav.resample("ME").last()
    monthly_returns = monthly_nav.pct_change().dropna()

    table = monthly_returns.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month

    return table.pivot(index="year", columns="month", values="return")


def calc_rolling_sharpe(returns, window=252, annual_days=252):
    """
    计算滚动夏普。
    """
    returns = returns.dropna().astype(float)

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    return rolling_mean / rolling_std * np.sqrt(annual_days)


def plot_rolling_sharpe(returns, window=252, annual_days=252, output_path=None):
    """
    绘制滚动夏普。
    """
    output_path = _ensure_output_path(output_path)

    rolling_sharpe = calc_rolling_sharpe(
        returns,
        window=window,
        annual_days=annual_days,
    )

    plt.figure(figsize=(12, 4))
    plt.plot(rolling_sharpe.index, rolling_sharpe, label=f"Rolling Sharpe ({window}D)")
    plt.axhline(0, linewidth=1)
    plt.title("Rolling Sharpe")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)

    plt.close()


def plot_weights(weights, output_path=None):
    """
    绘制持仓权重面积图。
    """
    output_path = _ensure_output_path(output_path)

    if weights is None or weights.empty:
        return

    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    weights.fillna(0).plot.area(ax=ax)

    plt.title("Portfolio Weights")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)

    plt.close()


def save_core_plots(
    returns,
    output_dir,
    benchmark_returns=None,
    weights=None,
):
    """
    一次性保存核心图表。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_nav(
        strategy_returns=returns,
        benchmark_returns=benchmark_returns,
        output_path=output_dir / "nav.png",
    )

    plot_drawdown(
        strategy_returns=returns,
        output_path=output_dir / "drawdown.png",
    )

    plot_rolling_sharpe(
        returns=returns,
        output_path=output_dir / "rolling_sharpe.png",
    )

    if weights is not None:
        plot_weights(
            weights=weights,
            output_path=output_dir / "weights.png",
        )