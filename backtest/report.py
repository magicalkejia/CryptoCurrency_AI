# backtest/report.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest.annualization import resolve_annual_periods
from backtest.metrics import calc_max_drawdown_info


_QS = None
_QS_STATS = None
_QS_ACTIVE_PERIODS_PER_YEAR = 252


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


def _load_quantstats():
    """Import QuantStats only when a QuantStats report is actually requested."""
    global _QS, _QS_STATS
    if _QS is None or _QS_STATS is None:
        import importlib

        _QS = importlib.import_module("quantstats")
        _QS_STATS = importlib.import_module("quantstats.stats")
    return _QS, _QS_STATS


def _patch_quantstats_annualization(periods_per_year: int, qs_stats=None) -> None:
    """Force QuantStats CAGR-like metrics to use bar counts, not calendar span.

    Some QuantStats releases annualize CAGR from calendar date span; newer
    releases fixed ``cagr`` but still call ``rar -> cagr(returns)`` without
    forwarding ``periods_per_year``.  This patch keeps the HTML report aligned
    with our engine metrics for both stock daily bars and 24/7 crypto bars.
    """
    global _QS_ACTIVE_PERIODS_PER_YEAR
    _QS_ACTIVE_PERIODS_PER_YEAR = int(periods_per_year)
    if qs_stats is None:
        _, qs_stats = _load_quantstats()

    if getattr(qs_stats, "_trading_system_annualization_patch", False):
        return

    def accurate_cagr(returns, rf=0.0, compounded=True, periods=None):
        qs_stats.validate_input(returns)
        periods = int(periods or _QS_ACTIVE_PERIODS_PER_YEAR)
        if periods <= 0:
            return np.nan

        prepared = qs_stats._utils._prepare_returns(returns, rf)
        if len(prepared) == 0:
            return np.nan

        if compounded:
            total = qs_stats.comp(prepared)
        else:
            total = np.sum(prepared, axis=0)

        years = len(prepared) / periods
        if years <= 0:
            return np.nan

        result = (1.0 + total) ** (1.0 / years) - 1.0
        if isinstance(returns, pd.DataFrame):
            result = pd.Series(result, index=returns.columns)
        return result

    def accurate_rar(returns, rf=0.0):
        prepared = qs_stats._utils._prepare_returns(returns, rf)
        return accurate_cagr(
            prepared,
            rf=0.0,
            compounded=True,
            periods=_QS_ACTIVE_PERIODS_PER_YEAR,
        ) / qs_stats.exposure(prepared)

    qs_stats.cagr = accurate_cagr
    qs_stats.rar = accurate_rar
    qs_stats._trading_system_annualization_patch = True


def print_basic_report(returns, benchmark_returns=None, periods_per_year: int = 252):
    """
    在 notebook / terminal 中打印 quantstats 基础指标。
    """
    returns, benchmark_returns = _align_returns(returns, benchmark_returns)
    qs, qs_stats = _load_quantstats()
    _patch_quantstats_annualization(periods_per_year, qs_stats=qs_stats)

    if benchmark_returns is not None:
        qs.reports.metrics(
            returns,
            benchmark=benchmark_returns,
            mode="basic",
            periods_per_year=periods_per_year,
            match_dates=False,
        )
    else:
        qs.reports.metrics(
            returns,
            mode="basic",
            periods_per_year=periods_per_year,
            match_dates=False,
        )


def save_html_report(
    returns,
    output_path,
    title="Backtest Report",
    benchmark_returns=None,
    periods_per_year: int = 252,
):
    """
    保存 quantstats HTML 报告。
    """
    output_path = _ensure_output_path(output_path)

    returns, benchmark_returns = _align_returns(returns, benchmark_returns)
    qs, qs_stats = _load_quantstats()
    _patch_quantstats_annualization(periods_per_year, qs_stats=qs_stats)

    if benchmark_returns is not None:
        qs.reports.html(
            returns,
            benchmark=benchmark_returns,
            title=title,
            output=str(output_path),
            periods_per_year=periods_per_year,
            match_dates=False,
        )
    else:
        qs.reports.html(
            returns,
            title=title,
            output=str(output_path),
            periods_per_year=periods_per_year,
            match_dates=False,
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


def calc_rolling_sharpe(
    returns,
    window=None,
    annual_days=None,
    *,
    annual_periods=None,
    market="stock",
    timeframe="1d",
):
    """
    计算滚动夏普。
    """
    periods = resolve_annual_periods(
        annual_periods=annual_periods,
        annual_days=annual_days,
        market=market,
        timeframe=timeframe,
    )
    if window is None:
        window = periods

    returns = returns.dropna().astype(float)

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    return rolling_mean / rolling_std * np.sqrt(periods)


def plot_rolling_sharpe(
    returns,
    window=None,
    annual_days=None,
    output_path=None,
    *,
    annual_periods=None,
    market="stock",
    timeframe="1d",
):
    """
    绘制滚动夏普。
    """
    output_path = _ensure_output_path(output_path)
    periods = resolve_annual_periods(
        annual_periods=annual_periods,
        annual_days=annual_days,
        market=market,
        timeframe=timeframe,
    )
    if window is None:
        window = periods

    rolling_sharpe = calc_rolling_sharpe(
        returns,
        window=window,
        annual_periods=periods,
    )

    plt.figure(figsize=(12, 4))
    plt.plot(rolling_sharpe.index, rolling_sharpe, label=f"Rolling Sharpe ({window} bars)")
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
