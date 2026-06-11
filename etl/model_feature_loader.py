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