# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# This table is intentionally kept in code, not only in README.
# It lets model/Agent developers inspect what every feature means without
# reading each builder function. The values are descriptive metadata only;
# actual calculations are implemented in the builder functions below.
import pandas as pd
FEATURE_DEFINITIONS: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------
    # Identity / time grid
    # ------------------------------------------------------------------
    "symbol": {
        "group": "identity",
        "definition": "Trading pair, e.g. BTC/USDT.",
        "calculation": "Copied from config.TargetConfig.COINS / source files.",
        "source": "config / processed market files",
        "usage": "Primary entity key. Not a numeric model feature.",
    },
    "ts_open": {
        "group": "time_grid",
        "definition": "Open timestamp of the 4h decision bar.",
        "calculation": "processed 4h bar timestamp. With label='left', this is the left edge of the 4h window.",
        "source": "processed/{SYMBOL}_4h.parquet.timestamp",
        "usage": "Audit / explainability. Not a model feature by default.",
    },
    "ts_close": {
        "group": "time_grid",
        "definition": "Close timestamp of the 4h decision bar.",
        "calculation": "ts_open + 4h.",
        "source": "derived from ts_open",
        "usage": "Defines when the 4h bar has completed.",
    },
    "decision_time": {
        "group": "time_grid",
        "definition": "Time at which the strategy/Agent is allowed to make a decision.",
        "calculation": "ts_close + cfg.market_latency; default = ts_close + 1 minute.",
        "source": "derived",
        "usage": "Primary PIT merge key. All features must satisfy available_time <= decision_time.",
    },

    # ------------------------------------------------------------------
    # 4h market features: main LLM/model input
    # ------------------------------------------------------------------
    "open_4h": {
        "group": "market_4h",
        "definition": "Open price of the completed 4h bar.",
        "calculation": "first 1m open within the 4h resample window.",
        "source": "processed/{SYMBOL}_4h.parquet.open",
        "usage": "4h price structure.",
    },
    "high_4h": {
        "group": "market_4h",
        "definition": "High price of the completed 4h bar.",
        "calculation": "max 1m high within the 4h resample window.",
        "source": "processed/{SYMBOL}_4h.parquet.high",
        "usage": "4h price range / volatility proxy.",
    },
    "low_4h": {
        "group": "market_4h",
        "definition": "Low price of the completed 4h bar.",
        "calculation": "min 1m low within the 4h resample window.",
        "source": "processed/{SYMBOL}_4h.parquet.low",
        "usage": "4h price range / volatility proxy.",
    },
    "close_4h": {
        "group": "market_4h",
        "definition": "Close price of the completed 4h bar.",
        "calculation": "last 1m close within the 4h resample window.",
        "source": "processed/{SYMBOL}_4h.parquet.close",
        "usage": "Main price anchor for returns and downstream backtest alignment.",
    },
    "volume_4h": {
        "group": "market_4h",
        "definition": "Total traded volume in the completed 4h bar.",
        "calculation": "sum of 1m volume within the 4h resample window.",
        "source": "processed/{SYMBOL}_4h.parquet.volume",
        "usage": "Liquidity / activity state.",
    },
    "ret_4h": {
        "group": "market_4h",
        "definition": "Most recent 4h return.",
        "calculation": "close_4h.pct_change(1).",
        "source": "derived from close_4h",
        "usage": "Short momentum / immediate price change.",
    },
    "ret_24h": {
        "group": "market_4h",
        "definition": "Most recent 24h return based on 4h bars.",
        "calculation": "close_4h.pct_change(6). 6 bars * 4h = 24h.",
        "source": "derived from close_4h",
        "usage": "One-day momentum.",
    },
    "ret_96h": {
        "group": "market_4h",
        "definition": "Most recent 96h / 4-day return based on 4h bars.",
        "calculation": "close_4h.pct_change(24). 24 bars * 4h = 96h.",
        "source": "derived from close_4h",
        "usage": "Medium-horizon momentum.",
    },
    "vol_96h": {
        "group": "market_4h",
        "definition": "Rolling 96h volatility based on 4h returns.",
        "calculation": "close_4h.pct_change().rolling(24, min_periods=cfg.min_periods_4h).std().",
        "source": "derived from close_4h",
        "usage": "Volatility / risk scale for model and TP/SL profile design.",
    },
    "range_4h": {
        "group": "market_4h",
        "definition": "Normalized high-low range of the 4h bar.",
        "calculation": "(high_4h - low_4h) / close_4h.",
        "source": "derived from high_4h, low_4h, close_4h",
        "usage": "Intrabar volatility / expansion.",
    },
    "close_position_in_4h_range": {
        "group": "market_4h",
        "definition": "Where the close sits inside the 4h high-low range.",
        "calculation": "(close_4h - low_4h) / (high_4h - low_4h).",
        "source": "derived from high_4h, low_4h, close_4h",
        "usage": "Candle pressure. Near 1 = close near high; near 0 = close near low.",
    },
    "volume_z_96h": {
        "group": "market_4h",
        "definition": "4h volume z-score against the recent 96h window.",
        "calculation": "zscore(volume_4h, rolling window=24 4h bars).",
        "source": "derived from volume_4h",
        "usage": "Volume abnormality / activity spike.",
    },
    "net_taker_vol_4h": {
        "group": "market_4h_flow",
        "definition": "Taker buy volume minus taker sell volume in the 4h bar.",
        "calculation": "taker_buy_vol_4h - (volume_4h - taker_buy_vol_4h).",
        "source": "processed/{SYMBOL}_4h.parquet.net_taker_vol",
        "usage": "Proxy for aggressive buy/sell pressure.",
    },
    "net_taker_vol_z_96h": {
        "group": "market_4h_flow",
        "definition": "Rolling z-score of 4h net taker volume.",
        "calculation": "zscore(net_taker_vol_4h, rolling window=24 4h bars).",
        "source": "derived from net_taker_vol_4h",
        "usage": "Abnormal aggressive order-flow pressure.",
    },

    # ------------------------------------------------------------------
    # 1h short-term features: asof merged into 4h grid
    # ------------------------------------------------------------------
    "ret_1h": {"group": "market_1h", "definition": "Most recent 1h return.", "calculation": "close_1h.pct_change(1).", "source": "processed/{SYMBOL}_1h.parquet.close", "usage": "Very short-term momentum."},
    "ret_3h": {"group": "market_1h", "definition": "Most recent 3h return.", "calculation": "close_1h.pct_change(3).", "source": "processed/{SYMBOL}_1h.parquet.close", "usage": "Short-term momentum."},
    "ret_6h": {"group": "market_1h", "definition": "Most recent 6h return.", "calculation": "close_1h.pct_change(6).", "source": "processed/{SYMBOL}_1h.parquet.close", "usage": "Short-term continuation/reversal state."},
    "ret_12h": {"group": "market_1h", "definition": "Most recent 12h return.", "calculation": "close_1h.pct_change(12).", "source": "processed/{SYMBOL}_1h.parquet.close", "usage": "Half-day momentum."},
    "ret_24h_from_1h": {"group": "market_1h", "definition": "Most recent 24h return calculated from 1h bars.", "calculation": "close_1h.pct_change(24).", "source": "processed/{SYMBOL}_1h.parquet.close", "usage": "One-day momentum with 1h granularity."},
    "vol_24h_from_1h": {"group": "market_1h", "definition": "Rolling 24h volatility from 1h returns.", "calculation": "close_1h.pct_change().rolling(24, min_periods=cfg.min_periods_1h).std().", "source": "processed/{SYMBOL}_1h.parquet.close", "usage": "Short-term volatility state."},
    "volume_z_24h_from_1h": {"group": "market_1h", "definition": "1h volume z-score against a 24h window.", "calculation": "zscore(volume_1h, rolling window=24 1h bars).", "source": "processed/{SYMBOL}_1h.parquet.volume", "usage": "Short-term volume anomaly."},
    "range_6h_from_1h": {"group": "market_1h", "definition": "Average normalized 1h range over the past 6h.", "calculation": "mean over 6 bars of (high_1h - low_1h) / close_1h.", "source": "processed/{SYMBOL}_1h.parquet high/low/close", "usage": "Short-term realized range."},
    "net_taker_vol_1h": {"group": "market_1h_flow", "definition": "Most recent 1h net taker volume.", "calculation": "processed 1h net_taker_vol.", "source": "processed/{SYMBOL}_1h.parquet.net_taker_vol", "usage": "Latest aggressive order-flow pressure."},
    "net_taker_vol_6h": {"group": "market_1h_flow", "definition": "Rolling 6h sum of net taker volume.", "calculation": "net_taker_vol_1h.rolling(6, min_periods=2).sum().", "source": "derived from processed 1h net_taker_vol", "usage": "Accumulated short-term aggressive pressure."},
    "net_taker_vol_z_24h_from_1h": {"group": "market_1h_flow", "definition": "1h net taker volume z-score against a 24h window.", "calculation": "zscore(net_taker_vol_1h, rolling window=24 1h bars).", "source": "derived from processed 1h net_taker_vol", "usage": "Order-flow abnormality."},

    # ------------------------------------------------------------------
    # 1d regime features
    # ------------------------------------------------------------------
    "ret_1d": {"group": "market_1d", "definition": "Most recent 1d return.", "calculation": "close_1d.pct_change(1).", "source": "processed/{SYMBOL}_1d.parquet.close", "usage": "Daily price change."},
    "ret_3d": {"group": "market_1d", "definition": "Most recent 3d return.", "calculation": "close_1d.pct_change(3).", "source": "processed/{SYMBOL}_1d.parquet.close", "usage": "Short daily momentum."},
    "ret_7d": {"group": "market_1d", "definition": "Most recent 7d return.", "calculation": "close_1d.pct_change(7).", "source": "processed/{SYMBOL}_1d.parquet.close", "usage": "Weekly momentum."},
    "ret_30d": {"group": "market_1d", "definition": "Most recent 30d return.", "calculation": "close_1d.pct_change(30).", "source": "processed/{SYMBOL}_1d.parquet.close", "usage": "Monthly momentum."},
    "vol_30d": {"group": "market_1d", "definition": "Rolling 30d volatility from daily returns.", "calculation": "close_1d.pct_change().rolling(30, min_periods=cfg.min_periods_1d).std().", "source": "processed/{SYMBOL}_1d.parquet.close", "usage": "Daily regime risk."},
    "daily_ma_gap_7_30": {"group": "market_1d", "definition": "Relative gap between 7d and 30d moving averages.", "calculation": "MA7(close_1d) / MA30(close_1d) - 1.", "source": "derived from close_1d", "usage": "Daily trend slope / regime state."},
    "daily_trend_up": {"group": "market_1d", "definition": "Daily trend flag.", "calculation": "1.0 if close_1d > MA30(close_1d), else 0.0.", "source": "derived from close_1d", "usage": "Simple regime filter."},
    "drawdown_from_30d_high": {"group": "market_1d", "definition": "Current close relative to rolling 30d high.", "calculation": "close_1d / rolling_max_30d(close_1d) - 1.", "source": "derived from close_1d", "usage": "Daily drawdown / trend damage."},

    # ------------------------------------------------------------------
    # Derivatives and optional sources
    # ------------------------------------------------------------------
    "funding_rate": {"group": "derivatives_funding", "definition": "Settled funding rate at funding timestamp.", "calculation": "Loaded from processed/derivatives/funding.parquet and asof-merged by funding_available_time <= decision_time.", "source": "processed funding long table", "usage": "Crowding / perp cost state."},
    "funding_interval_hours": {"group": "derivatives_funding", "definition": "Rounded funding event interval in hours.", "calculation": "round(diff(funding timestamp).total_seconds()/3600) per symbol.", "source": "processed funding long table", "usage": "Detect 1h/4h/8h interval regime; not usually fed alone."},
    "funding_rate_8h_equiv": {"group": "derivatives_funding", "definition": "Funding rate normalized to an 8h equivalent.", "calculation": "funding_rate * 8 / funding_interval_hours.", "source": "processed funding long table", "usage": "Cross-symbol comparison when funding intervals differ."},
    "funding_rate_chg": {"group": "derivatives_funding", "definition": "Change versus previous funding event.", "calculation": "funding_rate.diff() per symbol.", "source": "processed funding long table", "usage": "Rising/falling funding pressure."},
    "funding_rate_z_30_events": {"group": "derivatives_funding", "definition": "Funding rate z-score over the past 30 funding events.", "calculation": "zscore(funding_rate, rolling window=30 events). For 8h funding, 30 events ≈ 10 days.", "source": "processed funding long table", "usage": "Crowding abnormality."},
    "open_interest": {"group": "derivatives_oi", "definition": "Open interest level.", "calculation": "Loaded from processed/derivatives/oi.parquet and asof-merged.", "source": "processed OI long table", "usage": "Leverage / positioning state."},
    "open_interest_value": {"group": "derivatives_oi", "definition": "Open interest notional value if provided by source.", "calculation": "Loaded from processed/derivatives/oi.parquet.", "source": "processed OI long table", "usage": "Notional leverage proxy."},
    "oi_chg_1": {"group": "derivatives_oi", "definition": "One-period percentage change in OI.", "calculation": "open_interest.pct_change(1) per symbol.", "source": "derived from OI table", "usage": "Leverage expansion/contraction."},
    "oi_z_30": {"group": "derivatives_oi", "definition": "OI rolling z-score over 30 observations.", "calculation": "zscore(open_interest, rolling window=30 observations).", "source": "derived from OI table", "usage": "Abnormal leverage state."},
    "spot_cvd_proxy_4h": {"group": "flow", "definition": "Cumulative proxy CVD from 4h net taker volume.", "calculation": "cumsum(net_taker_vol_4h.fillna(0)) when include_cvd_proxy=True.", "source": "derived from market net_taker_vol_4h", "usage": "Proxy for cumulative aggressive buy/sell pressure; not true tick-level CVD."},
    "spot_cvd_proxy_chg_4h": {"group": "flow", "definition": "Change in proxy CVD from previous 4h decision bar.", "calculation": "spot_cvd_proxy_4h.diff().", "source": "derived from spot_cvd_proxy_4h", "usage": "Incremental flow pressure."},
    "sentiment_xxx": {"group": "sentiment", "definition": "Any numeric sentiment feature loaded from processed/sentiment/x_sentiment_4h.parquet.", "calculation": "Existing numeric columns are carried through, prefixed if needed, and asof-merged by sentiment_available_time <= decision_time.", "source": "processed sentiment table", "usage": "Narrative / attention / event-risk feature."},
    "onchain_xxx": {"group": "onchain", "definition": "Any numeric on-chain feature loaded from processed/onchain/onchain_daily.parquet.", "calculation": "Existing numeric columns are prefixed with onchain_ if needed and asof-merged with cfg.onchain_availability_lag, default 1d.", "source": "processed on-chain daily table", "usage": "Slow-moving chain activity / flow regime."},

    # ------------------------------------------------------------------
    # PIT audit columns
    # ------------------------------------------------------------------
    "market_4h_available_time": {"group": "pit_audit", "definition": "When completed 4h market features become available.", "calculation": "ts_close + cfg.market_latency.", "source": "derived", "usage": "PIT audit only."},
    "market_1h_available_time": {"group": "pit_audit", "definition": "When 1h summary row becomes available.", "calculation": "1h bar timestamp + 1h + cfg.market_latency.", "source": "derived", "usage": "PIT audit only."},
    "market_1d_available_time": {"group": "pit_audit", "definition": "When 1d regime row becomes available.", "calculation": "1d bar timestamp + 1d + cfg.market_latency.", "source": "derived", "usage": "PIT audit only."},
    "funding_available_time": {"group": "pit_audit", "definition": "When settled funding event is allowed to be used.", "calculation": "funding timestamp + cfg.derivatives_latency.", "source": "derived from processed funding timestamp", "usage": "PIT audit only."},
    "oi_available_time": {"group": "pit_audit", "definition": "When OI observation is allowed to be used.", "calculation": "OI timestamp + cfg.derivatives_latency.", "source": "derived from processed OI timestamp", "usage": "PIT audit only."},
    "sentiment_available_time": {"group": "pit_audit", "definition": "When sentiment aggregate is allowed to be used.", "calculation": "sentiment timestamp + cfg.sentiment_latency.", "source": "derived", "usage": "PIT audit only."},
    "onchain_available_time": {"group": "pit_audit", "definition": "When daily on-chain aggregate is allowed to be used.", "calculation": "on-chain day/timestamp + cfg.onchain_availability_lag; default lag = 1 day.", "source": "derived", "usage": "PIT audit only."},
    "max_feature_available_time": {"group": "pit_audit", "definition": "Latest available_time among all merged sources for the row.", "calculation": "row-wise max of all *_available_time columns.", "source": "derived", "usage": "Must be <= decision_time, otherwise there is look-ahead leakage."},
    "feature_version": {"group": "metadata", "definition": "Feature builder version string.", "calculation": "Constant assigned by build_crypto_features().", "source": "derived", "usage": "Dataset lineage / reproducibility."},
}



def get_feature_definitions() -> pd.DataFrame:
    """Return a DataFrame documenting feature meaning, calculation and source.

    Example:
        from etl.feature_builder import get_feature_definitions
        display(get_feature_definitions())
    """
    rows = []
    for name, meta in FEATURE_DEFINITIONS.items():
        row = {"feature": name}
        row.update(meta)
        rows.append(row)
    return pd.DataFrame(rows)[["feature", "group", "definition", "calculation", "source", "usage"]]
