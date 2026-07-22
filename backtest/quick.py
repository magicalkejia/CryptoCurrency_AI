# backtest/quick.py

from pathlib import Path
from datetime import datetime
import json
import pandas as pd

from backtest.engine import run_vector_backtest, BacktestConfig
from backtest.annualization import resolve_annual_periods
from backtest.metrics import calc_full_metrics
from backtest.records import save_backtest_records
from backtest.report import (
    print_basic_report,
    save_html_report,
    plot_nav,
    plot_drawdown,
    plot_weights,
    calc_monthly_returns,
    calc_rolling_sharpe,
)


class BacktestResult:
    def __init__(self, experiment_id, output_dir, raw_result, metrics):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.raw_result = raw_result
        self.metrics = metrics

        self.returns = raw_result["returns"]
        self.equity_curve = raw_result["equity_curve"]
        self.weights = raw_result["weights"]
        self.trades = raw_result.get("trades")
        self.positions = raw_result.get("positions")

    def show_metrics(self):
        try:
            from IPython.display import display
            display(pd.DataFrame([self.metrics]).T.rename(columns={0: "value"}))
        except Exception:
            print(self.metrics)

    def show(self):
        self.show_metrics()
        print(f"Experiment saved to: {self.output_dir}")


def make_experiment_id(strategy_name, experiment_name=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        return f"{experiment_name}_{ts}"
    return f"{strategy_name}_{ts}"


def quick_backtest(
    close: pd.DataFrame,
    target_weight: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    experiment_name: str | None = None,
    strategy_name: str = "research_strategy",
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.0015,
    slippage_rate: float = 0.0,
    execution_lag: int = 1,
    annual_days: int | None = None,
    annual_periods: int | None = None,
    market: str = "stock",
    timeframe: str = "1d",
    output_root: str = "data_storage/backtest_results",
    save: bool = True,
    display: bool = True,
):
    experiment_id = make_experiment_id(strategy_name, experiment_name)
    output_dir = Path(output_root) / experiment_id
    periods_per_year = resolve_annual_periods(
        annual_periods=annual_periods,
        annual_days=annual_days,
        market=market,
        timeframe=timeframe,
    )

    config = BacktestConfig(
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        execution_lag=execution_lag,
        annual_days=annual_days,
        annual_periods=periods_per_year,
        market=market,
        timeframe=timeframe,
    )

    raw_result = run_vector_backtest(
        close=close,
        target_weight=target_weight,
        config=config,
        benchmark_returns=benchmark_returns,
        experiment_id=experiment_id,
        strategy_name=strategy_name,
    )

    metrics = calc_full_metrics(
        strategy_returns=raw_result["returns"],
        benchmark_returns=benchmark_returns,
        turnover=raw_result["turnover"],
        cost=raw_result["cost"],
        weights=raw_result["weights"],
        annual_periods=periods_per_year,
    )

    result = BacktestResult(
        experiment_id=experiment_id,
        output_dir=output_dir,
        raw_result=raw_result,
        metrics=metrics,
    )

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_result["returns"].to_frame("returns").to_parquet(output_dir / "returns.parquet")
        raw_result["equity_curve"].to_frame("equity").to_parquet(output_dir / "equity_curve.parquet")
        raw_result["weights"].to_parquet(output_dir / "weights.parquet")
        raw_result["target_weight"].to_parquet(output_dir / "target_weight.parquet")
        raw_result["turnover"].to_frame("turnover").to_parquet(output_dir / "turnover.parquet")
        raw_result["cost"].to_frame("cost").to_parquet(output_dir / "cost.parquet")

        if benchmark_returns is not None:
            benchmark_returns.to_frame("benchmark_returns").to_parquet(
                output_dir / "benchmark_returns.parquet"
            )

        save_backtest_records(
            output_dir=output_dir,
            trades=raw_result.get("trades"),
            positions=raw_result.get("positions"),
            export_csv=True,
        )

        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

        config_dict = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "strategy_name": strategy_name,
            "initial_cash": initial_cash,
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "execution_lag": execution_lag,
            "annual_days": periods_per_year,
            "annual_periods": periods_per_year,
            "market": market,
            "timeframe": timeframe,
            "created_at": datetime.now().isoformat(),
        }

        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

        save_html_report(
            raw_result["returns"],
            benchmark_returns=benchmark_returns,
            output_path=output_dir / "quantstats_report.html",
            title=strategy_name,
            periods_per_year=periods_per_year,
        )

        plot_nav(
            raw_result["returns"],
            benchmark_returns=benchmark_returns,
            output_path=output_dir / "nav.png",
        )

        plot_drawdown(
            raw_result["returns"],
            output_path=output_dir / "drawdown.png",
        )

    if display:
        result.show()

    return result
