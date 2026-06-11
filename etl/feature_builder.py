import pandas as pd
import numpy as np

from etl.data_loader import DataLoader

def build_market_funding_features(symbol: str, timeframe: str = "4h"):
    loader = DataLoader()

    market = loader.get_crypto_kline_data(
        symbol=symbol,
        timeframe=timeframe,
    )

    if market is None or market.empty:
        raise RuntimeError(f"market data empty: {symbol}")

    funding = loader.get_funding_rate_data(symbol)

    if funding is None or funding.empty:
        funding = pd.DataFrame(
            index=market.index,
            columns=[
                "funding_rate",
                "funding_interval_hours",
                "funding_rate_8h_equiv",
            ],
        )

    # market features
    feat = market.copy()
    feat["symbol"] = symbol
    feat["decision_time"] = feat.index

    feat["ret_1"] = feat["close"].pct_change()
    feat["ret_6"] = feat["close"].pct_change(6)
    feat["ret_24"] = feat["close"].pct_change(24)
    feat["vol_24"] = feat["close"].pct_change().rolling(24).std()

    # CVD proxy：基于 K 线 taker_buy_vol 近似，不是真正逐笔 CVD
    if "net_taker_vol" in feat.columns:
        feat["spot_cvd_proxy"] = feat["net_taker_vol"].fillna(0).cumsum()
        feat["spot_cvd_proxy_chg"] = feat["spot_cvd_proxy"].diff()
        feat["net_taker_vol_z"] = (
            feat["net_taker_vol"] - feat["net_taker_vol"].rolling(60).mean()
        ) / (feat["net_taker_vol"].rolling(60).std() + 1e-9)

    # funding asof merge
    feat_reset = feat.reset_index(drop=True).sort_values("decision_time")

    funding_reset = funding.reset_index().rename(
        columns={"timestamp": "funding_time"}
    )

    if "funding_time" not in funding_reset.columns:
        funding_reset = funding_reset.rename(columns={funding_reset.columns[0]: "funding_time"})

    funding_reset = funding_reset.sort_values("funding_time")

    merged = pd.merge_asof(
        feat_reset,
        funding_reset[
            [
                "funding_time",
                "funding_rate",
                "funding_interval_hours",
                "funding_rate_8h_equiv",
            ]
        ],
        left_on="decision_time",
        right_on="funding_time",
        direction="backward",
        allow_exact_matches=True,
    )

    merged["funding_rate_z_30"] = (
        merged["funding_rate"] - merged["funding_rate"].rolling(30).mean()
    ) / (merged["funding_rate"].rolling(30).std() + 1e-9)

    merged["funding_rate_chg"] = merged["funding_rate"].diff()

    # 审计字段：证明特征没有用未来
    merged["max_feature_availability_ts"] = merged[[
        "decision_time",
        "funding_time",
    ]].max(axis=1)

    return merged