from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import config


MODEL_EXCLUDE_COLS = [
    "symbol",
    "ts_open",
    "ts_close",
    "decision_time",
    "market_4h_available_time",
    "market_1h_available_time",
    "market_1d_available_time",
    "funding_available_time",
    "oi_available_time",
    "cvd_available_time",
    "sentiment_available_time",
    "onchain_available_time",
    "max_feature_availability_ts",
    "max_feature_available_time",
    "feature_version",
]


def load_crypto_feature_table(
    feature_path: str | Path | None = None,
    symbols: Sequence[str] | None = None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    path = Path(feature_path) if feature_path else config.PathConfig.FACTORS / "crypto_features.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_parquet(path)
    df["decision_time"] = pd.to_datetime(df["decision_time"])

    if symbols is not None:
        df = df[df["symbol"].isin(list(symbols))]

    if start_date is not None:
        df = df[df["decision_time"] >= pd.to_datetime(start_date)]

    if end_date is not None:
        df = df[df["decision_time"] <= pd.to_datetime(end_date)]

    return df.sort_values(["decision_time", "symbol"]).reset_index(drop=True)


def split_model_inputs(
    features: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
):
    if feature_columns is None:
        feature_columns = [
            c for c in features.columns
            if c not in MODEL_EXCLUDE_COLS
            and pd.api.types.is_numeric_dtype(features[c])
        ]

    X = features[list(feature_columns)].replace([np.inf, -np.inf], np.nan)
    meta = features[["symbol", "decision_time"]].copy()

    return X, meta, list(feature_columns)


def add_trading_graph_compat_columns(features: pd.DataFrame) -> pd.DataFrame:
    """
    Add in-memory compatibility columns expected by crypto.orchestration.graph.

    The graph/skills layer currently looks for legacy names such as vol_24,
    mom_z, funding_rate_z, and max_feature_availability_ts. The multimodal
    feature builder may produce semantically similar but differently named
    columns. This adapter keeps that contract out of the partner-owned crypto
    package and does not write these aliases back to parquet.
    """
    out = features.copy()

    if "max_feature_availability_ts" not in out.columns and "max_feature_available_time" in out.columns:
        out["max_feature_availability_ts"] = out["max_feature_available_time"]

    _fill_alias(out, "vol_24", ["vol_96h", "vol_24h_from_1h", "vol_30d"])
    _fill_alias(out, "mom_z", ["ret_24h", "ret_96h", "ret_24h_from_1h"])
    _fill_alias(out, "funding_rate_z", ["funding_rate_z_30_events"])

    return out


def _fill_alias(df: pd.DataFrame, alias: str, candidates: Sequence[str]) -> None:
    if alias in df.columns:
        return
    for col in candidates:
        if col in df.columns:
            df[alias] = df[col]
            return


def load_trading_graph_inputs(
    feature_path: str | Path | None = None,
    symbols: Sequence[str] | None = None,
    start_date=None,
    end_date=None,
    feature_columns: Sequence[str] | None = None,
    dropna: bool = False,
    graph_compat: bool = True,
):
    """
    Load the final PIT feature table for TradingGraph inference.

    TradingGraph itself does not read parquet. It expects:
        features: DataFrame with symbol + decision_time + audit columns
        feature_cols: numeric model columns used by the fitted ModelBundle
    """
    features = load_crypto_feature_table(
        feature_path=feature_path,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )
    X, _, feature_cols = split_model_inputs(features, feature_columns=feature_columns)
    if dropna:
        keep = X.notna().all(axis=1)
        features = features.loc[keep].reset_index(drop=True)
    if graph_compat:
        features = add_trading_graph_compat_columns(features)
    return features, feature_cols
