"""
etl.feature_builder
===================

Build a point-in-time crypto feature table for 4h decision frequency.

Pipeline:
    market + derivatives + onchain + sentiment
        -> PIT asof merge on (symbol, decision_time)
        -> data_storage/factors/crypto_features.parquet

Design principles:
    1. Market 4h bars define the decision grid.
    2. 1h bars are summarized as short-term features.
    3. 1d bars are summarized as regime/context features.
    4. Funding / OI / CVD / sentiment / on-chain are optional and merged by
       available_time <= decision_time, never by raw timestamp equality.
    5. Missing optional data sources do not break the build; they are skipped.
    6. A PIT audit column max_feature_available_time is generated and checked.

Expected existing files:
    processed/{SYMBOL}_1h.parquet
    processed/{SYMBOL}_4h.parquet
    processed/{SYMBOL}_1d.parquet
    processed/derivatives/funding.parquet                    optional
    processed/derivatives/oi.parquet                         optional
    processed/flow/spot_cvd_4h.parquet                       optional
    processed/sentiment/x_sentiment_4h.parquet               optional
    factors/onchain_features.parquet                        optional
    processed/onchain/onchain_daily.parquet                  fallback only

Output:
    factors/crypto_features.parquet

Notes:
    Your current resampling uses label='left', closed='left'. Therefore a 4h row
    timestamped at 08:00 covers [08:00, 12:00), and should only be available at
    ts_close = 12:00 plus a small latency. This builder explicitly creates:
        ts_open, ts_close, decision_time, market_4h_available_time
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

import config
from etl.feature_registry import get_feature_definitions





# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FeatureBuilderConfig:
    """Runtime configuration for PIT feature building."""

    decision_timeframe: str = "4h"
    output_name: str = "crypto_features.parquet"

    # Small latency after bar close / event time. Prevents treating a bar close
    # as available exactly at the same instant in downstream merge logic.
    market_latency: pd.Timedelta = pd.Timedelta(minutes=1)
    derivatives_latency: pd.Timedelta = pd.Timedelta(minutes=1)
    sentiment_latency: pd.Timedelta = pd.Timedelta(minutes=1)
    onchain_availability_lag: pd.Timedelta = pd.Timedelta(days=1)

    # Optional source switches.
    include_1h_features: bool = True
    include_1d_features: bool = True
    include_funding: bool = True
    include_oi: bool = False
    include_cvd_proxy: bool = False
    include_sentiment: bool = False
    include_onchain: bool = True

    # Rolling windows.
    vol_window_1h: int = 24
    volume_z_window_1h: int = 24
    vol_window_4h: int = 24       # 24 bars * 4h = 96h
    volume_z_window_4h: int = 24
    vol_window_1d: int = 30

    # Minimum periods for rolling statistics.
    min_periods_1h: int = 6
    min_periods_4h: int = 6
    min_periods_1d: int = 7

    # If true, raise on PIT violation. If false, only print warnings.
    strict_pit_check: bool = True

    # Columns that should not be carried through from optional source files.
    audit_drop_columns: set[str] = field(default_factory=lambda: {"created_at", "source"})


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _path_attr(name: str, fallback: Path) -> Path:
    return Path(getattr(config.PathConfig, name, fallback))


def data_root() -> Path:
    return Path(config.PathConfig.DATA_ROOT)


def processed_dir() -> Path:
    return Path(config.PathConfig.PROCESSED)


def factors_dir() -> Path:
    return Path(config.PathConfig.FACTORS)


def processed_derivatives_dir() -> Path:
    return _path_attr("PROCESSED_DERIVATIVES", processed_dir() / "derivatives")


def processed_flow_dir() -> Path:
    return _path_attr("PROCESSED_FLOW", processed_dir() / "flow")


def processed_sentiment_dir() -> Path:
    return _path_attr("PROCESSED_SENTIMENT", processed_dir() / "sentiment")


def processed_onchain_dir() -> Path:
    return _path_attr("PROCESSED_ONCHAIN", processed_dir() / "onchain")


def onchain_factors_path() -> Path:
    return factors_dir() / "onchain_features.parquet"


def symbol_key(symbol: str) -> str:
    return symbol.replace("/", "")


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    tf = timeframe.strip().lower()
    if tf.endswith("min"):
        return pd.Timedelta(minutes=int(tf[:-3]))
    if tf.endswith("m"):
        return pd.Timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h"):
        return pd.Timedelta(hours=int(tf[:-1]))
    if tf.endswith("d"):
        return pd.Timedelta(days=int(tf[:-1]))
    raise ValueError(f"Unsupported timeframe: {timeframe}")


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------


def _read_parquet_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def _safe_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / (std + 1e-12)


def _close_position_in_range(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    return (close - low) / ((high - low).replace(0, np.nan))


def _normalize_symbol_col(df: pd.DataFrame, default_symbol: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    if "symbol" not in df.columns:
        if default_symbol is None:
            raise ValueError("DataFrame missing symbol column and no default_symbol was provided")
        df["symbol"] = default_symbol
    df["symbol"] = df["symbol"].astype(str)
    return df


def _to_datetime_ns(series: pd.Series) -> pd.Series:
    """Normalize datetime-like values to timezone-naive datetime64[ns].

    Pandas/pyarrow can read parquet timestamps as datetime64[us], while
    arithmetic-created columns are often datetime64[ns]. `merge_asof` requires
    the left and right merge keys to have exactly the same dtype, so every time
    key used in PIT merges is normalized through this helper.
    """
    s = pd.to_datetime(series, errors="coerce")

    # If a tz-aware dtype appears, drop the timezone after converting to UTC.
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    except AttributeError:
        pass

    return s.astype("datetime64[ns]")


def _ensure_datetime_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        raise ValueError(f"DataFrame missing {col!r} column")
    df[col] = _to_datetime_ns(df[col])
    return df.dropna(subset=[col])


def _ensure_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    return _ensure_datetime_col(df, col)


def _asof_merge_by_symbol(
    base: pd.DataFrame,
    feature: pd.DataFrame,
    feature_time_col: str,
    source_name: str,
) -> pd.DataFrame:
    """
    PIT merge feature rows into base by symbol where feature_time <= decision_time.

    base requires: symbol, decision_time.
    feature requires: symbol, feature_time_col.
    """
    if feature is None or feature.empty:
        return base

    left = base.copy()
    right = feature.copy()

    left = _ensure_datetime_col(left, "decision_time")
    right = _ensure_datetime_col(right, feature_time_col)

    left = left.sort_values(["symbol", "decision_time"]).reset_index(drop=True)
    right = right.sort_values(["symbol", feature_time_col]).reset_index(drop=True)

    # pandas merge_asof requires the merge key to be globally sorted, not only
    # lexicographically by [symbol, time], on some versions. Groupby is safer.
    merged_frames = []
    for sym, left_g in left.groupby("symbol", sort=False):
        right_g = right[right["symbol"] == sym]
        if right_g.empty:
            merged_frames.append(left_g)
            continue

        m = pd.merge_asof(
            left_g.sort_values("decision_time"),
            right_g.sort_values(feature_time_col),
            left_on="decision_time",
            right_on=feature_time_col,
            direction="backward",
            suffixes=("", f"_{source_name}"),
        )

        # merge_asof duplicates the symbol column as symbol_<source> if present.
        dup_symbol = f"symbol_{source_name}"
        if dup_symbol in m.columns:
            m = m.drop(columns=[dup_symbol])
        merged_frames.append(m)

    out = pd.concat(merged_frames, ignore_index=True)
    return out.sort_values(["symbol", "decision_time"]).reset_index(drop=True)


def _asof_merge_global(
    base: pd.DataFrame,
    feature: pd.DataFrame,
    feature_time_col: str,
    source_name: str,
) -> pd.DataFrame:
    """Merge a global non-symbol feature table into every symbol by time."""
    if feature is None or feature.empty:
        return base

    base = _ensure_datetime_col(base, "decision_time")
    right = _ensure_datetime_col(feature, feature_time_col)

    merged_frames = []
    right = right.sort_values(feature_time_col).copy()
    for _, left_g in base.groupby("symbol", sort=False):
        m = pd.merge_asof(
            left_g.sort_values("decision_time"),
            right,
            left_on="decision_time",
            right_on=feature_time_col,
            direction="backward",
            suffixes=("", f"_{source_name}"),
        )
        merged_frames.append(m)
    out = pd.concat(merged_frames, ignore_index=True)
    return out.sort_values(["symbol", "decision_time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Market feature builders
# ---------------------------------------------------------------------------


def load_processed_kline(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    path = processed_dir() / f"{symbol_key(symbol)}_{timeframe}.parquet"
    df = _read_parquet_if_exists(path)
    if df is None:
        print(f"[WARN] Missing processed kline: {path}")
        return None
    if df.empty:
        return df
    df = _ensure_timestamp(df, "timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["symbol"] = symbol
    return df


def build_market_4h_features(
    symbol: str,
    cfg: FeatureBuilderConfig,
) -> Optional[pd.DataFrame]:
    """Build the 4h decision grid and primary market features.

    Implementation summary:
    - processed 4h timestamp is treated as ts_open because resample uses
      label='left', closed='left'.
    - ts_close = ts_open + 4h.
    - decision_time = ts_close + market_latency.
    - Returns use close.pct_change(n), where n is the number of 4h bars.
    - Rolling volatility uses std of 4h returns.
    """
    df = load_processed_kline(symbol, cfg.decision_timeframe)
    if df is None or df.empty:
        return None

    delta = timeframe_to_timedelta(cfg.decision_timeframe)
    out = pd.DataFrame({
        "symbol": symbol,
        "ts_open": df["timestamp"],
        "ts_close": df["timestamp"] + delta,
    })
    out["decision_time"] = out["ts_close"] + cfg.market_latency
    out["market_4h_available_time"] = out["decision_time"]

    for col in ["open", "high", "low", "close", "volume", "taker_buy_vol", "net_taker_vol"]:
        if col in df.columns:
            out[f"{col}_4h"] = pd.to_numeric(df[col], errors="coerce")

    close = out["close_4h"]
    high = out.get("high_4h")
    low = out.get("low_4h")
    volume = out.get("volume_4h")

    out["ret_4h"] = close.pct_change(1)
    out["ret_24h"] = close.pct_change(6)
    out["ret_96h"] = close.pct_change(24)
    out["vol_96h"] = close.pct_change().rolling(
        cfg.vol_window_4h,
        min_periods=cfg.min_periods_4h,
    ).std()

    if high is not None and low is not None:
        out["range_4h"] = (high - low) / close.replace(0, np.nan)
        out["close_position_in_4h_range"] = _close_position_in_range(close, high, low)

    if volume is not None:
        out["volume_z_96h"] = _safe_zscore(volume, cfg.volume_z_window_4h, cfg.min_periods_4h)

    if "net_taker_vol_4h" in out.columns:
        out["net_taker_vol_z_96h"] = _safe_zscore(
            out["net_taker_vol_4h"],
            cfg.volume_z_window_4h,
            cfg.min_periods_4h,
        )
        if cfg.include_cvd_proxy:
            out["spot_cvd_proxy_4h"] = out["net_taker_vol_4h"].fillna(0).cumsum()
            out["spot_cvd_proxy_chg_4h"] = out["spot_cvd_proxy_4h"].diff()

    return out.sort_values("decision_time").reset_index(drop=True)


def build_1h_summary_features(
    symbol: str,
    cfg: FeatureBuilderConfig,
) -> Optional[pd.DataFrame]:
    """Build 1h-derived short-term features for asof merge.

    These features do not define the decision grid. They are summarized from
    completed 1h bars and merged backward into each 4h decision_time.
    """
    df = load_processed_kline(symbol, "1h")
    if df is None or df.empty:
        return None

    delta = pd.Timedelta(hours=1)
    out = pd.DataFrame({
        "symbol": symbol,
        "market_1h_available_time": df["timestamp"] + delta + cfg.market_latency,
    })

    close = pd.to_numeric(df["close"], errors="coerce")
    out["ret_1h"] = close.pct_change(1)
    out["ret_3h"] = close.pct_change(3)
    out["ret_6h"] = close.pct_change(6)
    out["ret_12h"] = close.pct_change(12)
    out["ret_24h_from_1h"] = close.pct_change(24)
    out["vol_24h_from_1h"] = close.pct_change().rolling(
        cfg.vol_window_1h,
        min_periods=cfg.min_periods_1h,
    ).std()

    if "volume" in df.columns:
        volume = pd.to_numeric(df["volume"], errors="coerce")
        out["volume_z_24h_from_1h"] = _safe_zscore(volume, cfg.volume_z_window_1h, cfg.min_periods_1h)

    if {"high", "low"}.issubset(df.columns):
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        out["range_6h_from_1h"] = ((high - low) / close.replace(0, np.nan)).rolling(
            6,
            min_periods=max(2, min(cfg.min_periods_1h, 6)),
        ).mean()

    if "net_taker_vol" in df.columns:
        ntv = pd.to_numeric(df["net_taker_vol"], errors="coerce")
        out["net_taker_vol_1h"] = ntv
        out["net_taker_vol_6h"] = ntv.rolling(6, min_periods=2).sum()
        out["net_taker_vol_z_24h_from_1h"] = _safe_zscore(ntv, cfg.volume_z_window_1h, cfg.min_periods_1h)

    return out.sort_values("market_1h_available_time").reset_index(drop=True)


def build_1d_regime_features(
    symbol: str,
    cfg: FeatureBuilderConfig,
) -> Optional[pd.DataFrame]:
    """Build 1d-derived regime/context features for asof merge.

    Daily features are slow regime/context variables. A daily bar timestamped at
    day D is available only after D + 1 day + market_latency.
    """
    df = load_processed_kline(symbol, "1d")
    if df is None or df.empty:
        return None

    delta = pd.Timedelta(days=1)
    out = pd.DataFrame({
        "symbol": symbol,
        "market_1d_available_time": df["timestamp"] + delta + cfg.market_latency,
    })

    close = pd.to_numeric(df["close"], errors="coerce")
    out["ret_1d"] = close.pct_change(1)
    out["ret_3d"] = close.pct_change(3)
    out["ret_7d"] = close.pct_change(7)
    out["ret_30d"] = close.pct_change(30)
    out["vol_30d"] = close.pct_change().rolling(
        cfg.vol_window_1d,
        min_periods=cfg.min_periods_1d,
    ).std()

    ma_7 = close.rolling(7, min_periods=cfg.min_periods_1d).mean()
    ma_30 = close.rolling(30, min_periods=cfg.min_periods_1d).mean()
    rolling_high_30 = close.rolling(30, min_periods=cfg.min_periods_1d).max()

    out["daily_ma_gap_7_30"] = ma_7 / ma_30 - 1
    out["daily_trend_up"] = (close > ma_30).astype("float")
    out["drawdown_from_30d_high"] = close / rolling_high_30 - 1

    return out.sort_values("market_1d_available_time").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Optional source loaders
# ---------------------------------------------------------------------------


def load_funding_features(cfg: FeatureBuilderConfig) -> Optional[pd.DataFrame]:
    path = processed_derivatives_dir() / "funding.parquet"
    df = _read_parquet_if_exists(path)
    if df is None:
        print(f"[WARN] Funding feature source missing, skipped: {path}")
        return None
    if df.empty:
        return df

    df = _ensure_timestamp(_normalize_symbol_col(df), "timestamp")
    keep = [
        "symbol",
        "timestamp",
        "funding_rate",
        "funding_interval_hours",
        "funding_rate_8h_equiv",
        "funding_rate_chg",
    ]
    z_cols = [c for c in df.columns if c.startswith("funding_rate_z_")]
    keep += z_cols
    keep = [c for c in keep if c in df.columns]

    out = df[keep].copy()
    out["funding_available_time"] = out["timestamp"] + cfg.derivatives_latency
    out = out.drop(columns=["timestamp"])
    return out.sort_values(["symbol", "funding_available_time"]).reset_index(drop=True)


def load_oi_features(cfg: FeatureBuilderConfig) -> Optional[pd.DataFrame]:
    path = processed_derivatives_dir() / "oi.parquet"
    df = _read_parquet_if_exists(path)
    if df is None:
        print(f"[WARN] OI feature source missing, skipped: {path}")
        return None
    if df.empty:
        return df

    df = _ensure_timestamp(_normalize_symbol_col(df), "timestamp")
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    for col in ["open_interest", "open_interest_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    frames = []
    for symbol, g in df.groupby("symbol", sort=False):
        g = g.copy().sort_values("timestamp")
        if "open_interest" in g.columns:
            g["oi_chg_1"] = g["open_interest"].pct_change(1)
            g["oi_z_30"] = _safe_zscore(g["open_interest"], 30, 10)
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    out["oi_available_time"] = out["timestamp"] + cfg.derivatives_latency

    keep = [
        "symbol",
        "oi_available_time",
        "open_interest",
        "open_interest_value",
        "oi_chg_1",
        "oi_z_30",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].sort_values(["symbol", "oi_available_time"]).reset_index(drop=True)


def load_cvd_features(cfg: FeatureBuilderConfig) -> Optional[pd.DataFrame]:
    path = processed_flow_dir() / "spot_cvd_4h.parquet"
    df = _read_parquet_if_exists(path)
    if df is None:
        # spot_cvd_proxy_4h may already be derived from market net_taker_vol.
        print(f"[WARN] CVD source missing, skipped: {path}")
        return None
    if df.empty:
        return df

    df = _ensure_timestamp(_normalize_symbol_col(df), "timestamp")
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df["cvd_available_time"] = df["timestamp"] + timeframe_to_timedelta(cfg.decision_timeframe) + cfg.market_latency

    # Keep all numeric feature columns except timestamp/source audit columns.
    keep = ["symbol", "cvd_available_time"]
    for c in df.columns:
        if c in {"symbol", "timestamp"} or c in cfg.audit_drop_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            keep.append(c if c.startswith("cvd") or c.startswith("spot_cvd") else f"cvd_{c}")

    out = df[["symbol", "cvd_available_time"]].copy()
    for c in df.columns:
        if c in {"symbol", "timestamp"} or c in cfg.audit_drop_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            new_c = c if c.startswith("cvd") or c.startswith("spot_cvd") else f"cvd_{c}"
            out[new_c] = df[c]

    return out.sort_values(["symbol", "cvd_available_time"]).reset_index(drop=True)


def load_sentiment_features(cfg: FeatureBuilderConfig) -> Optional[pd.DataFrame]:
    path = processed_sentiment_dir() / "x_sentiment_4h.parquet"
    df = _read_parquet_if_exists(path)
    if df is None:
        print(f"[WARN] Sentiment source missing, skipped: {path}")
        return None
    if df.empty:
        return df

    df = _ensure_timestamp(df, "timestamp")
    has_symbol = "symbol" in df.columns
    if has_symbol:
        df = _normalize_symbol_col(df)

    df["sentiment_available_time"] = df["timestamp"] + cfg.sentiment_latency

    out_cols = ["sentiment_available_time"] + (["symbol"] if has_symbol else [])
    out = df[out_cols].copy()
    for c in df.columns:
        if c in {"timestamp", "symbol", "sentiment_available_time"} or c in cfg.audit_drop_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out[c if c.startswith("sentiment") or c.startswith("x_") else f"sentiment_{c}"] = df[c]

    sort_cols = (["symbol"] if has_symbol else []) + ["sentiment_available_time"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def load_onchain_features(cfg: FeatureBuilderConfig) -> Optional[pd.DataFrame]:
    path = onchain_factors_path()
    if not path.exists():
        fallback = processed_onchain_dir() / "onchain_daily.parquet"
        if fallback.exists():
            print(f"[WARN] On-chain factors missing, using processed fallback: {fallback}")
            path = fallback

    df = _read_parquet_if_exists(path)
    if df is None:
        print(f"[WARN] On-chain source missing, skipped: {path}")
        return None
    if df.empty:
        return df

    time_col = "timestamp" if "timestamp" in df.columns else "day" if "day" in df.columns else None
    if time_col is None:
        print(f"[WARN] On-chain source missing timestamp/day column, skipped: {path}")
        return None

    df = _ensure_timestamp(df.rename(columns={time_col: "timestamp"}), "timestamp")
    has_symbol = "symbol" in df.columns
    if has_symbol:
        df = _normalize_symbol_col(df)

    df["onchain_available_time"] = df["timestamp"] + cfg.onchain_availability_lag

    out_cols = ["onchain_available_time"] + (["symbol"] if has_symbol else [])
    out = df[out_cols].copy()
    for c in df.columns:
        if c in {"timestamp", "day", "symbol", "onchain_available_time"} or c in cfg.audit_drop_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out[c if c.startswith("onchain_") else f"onchain_{c}"] = df[c]

    sort_cols = (["symbol"] if has_symbol else []) + ["onchain_available_time"]
    return out.sort_values(sort_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# PIT build and validation
# ---------------------------------------------------------------------------


def _collect_availability_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_available_time")]


def validate_pit_features(features: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Validate uniqueness and point-in-time availability."""
    required = {"symbol", "decision_time"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature table missing required columns: {missing}")

    duplicated = features.duplicated(subset=["symbol", "decision_time"], keep=False)
    if duplicated.any():
        sample = features.loc[duplicated, ["symbol", "decision_time"]].head(20)
        raise ValueError(f"Duplicate symbol + decision_time found:\n{sample}")

    availability_cols = _collect_availability_columns(features)
    if availability_cols:
        features = features.copy()
        features["decision_time"] = _to_datetime_ns(features["decision_time"])
        for c in availability_cols:
            features[c] = _to_datetime_ns(features[c])
        features["max_feature_available_time"] = features[availability_cols].max(axis=1)
        violation = features["max_feature_available_time"] > features["decision_time"]
        if violation.any():
            sample = features.loc[
                violation,
                ["symbol", "decision_time", "max_feature_available_time"] + availability_cols,
            ].head(20)
            msg = f"PIT violation: feature available_time exceeds decision_time:\n{sample}"
            if strict:
                raise ValueError(msg)
            print(f"[WARN] {msg}")
    else:
        features = features.copy()
        features["max_feature_available_time"] = pd.NaT

    return features.sort_values(["symbol", "decision_time"]).reset_index(drop=True)


def build_crypto_features(
    symbols: Optional[Sequence[str]] = None,
    start_date=None,
    end_date=None,
    cfg: Optional[FeatureBuilderConfig] = None,
    save: bool = True,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Build a PIT crypto feature table for 4h decisions.

    Parameters
    ----------
    symbols:
        Symbols such as ["BTC/USDT", "ETH/USDT"]. Defaults to TargetConfig.COINS.
    start_date, end_date:
        Optional filters on decision_time.
    cfg:
        FeatureBuilderConfig.
    save:
        Save output parquet if True.
    output_path:
        Optional explicit output path. Defaults to FACTORS / cfg.output_name.

    Returns
    -------
    DataFrame:
        Long feature table where one row is one symbol at one decision_time.
    """
    cfg = cfg or FeatureBuilderConfig()
    symbols = list(symbols or config.TargetConfig.COINS)

    # 1. Build 4h market decision grid.
    frames = []
    for symbol in symbols:
        df_4h = build_market_4h_features(symbol, cfg)
        if df_4h is not None and not df_4h.empty:
            frames.append(df_4h)

    if not frames:
        raise FileNotFoundError("No 4h market data found. Build processed 4h K-lines first.")

    features = pd.concat(frames, ignore_index=True)
    features["decision_time"] = _to_datetime_ns(features["decision_time"])
    features = features.sort_values(["symbol", "decision_time"]).reset_index(drop=True)

    if start_date is not None:
        features = features[features["decision_time"] >= _to_datetime_ns(pd.Series([start_date])).iloc[0]]
    if end_date is not None:
        features = features[features["decision_time"] <= _to_datetime_ns(pd.Series([end_date])).iloc[0]]

    # 2. Merge 1h short-term features.
    if cfg.include_1h_features:
        frames_1h = []
        for symbol in symbols:
            f = build_1h_summary_features(symbol, cfg)
            if f is not None and not f.empty:
                frames_1h.append(f)
        if frames_1h:
            f1h = pd.concat(frames_1h, ignore_index=True)
            features = _asof_merge_by_symbol(features, f1h, "market_1h_available_time", "1h")

    # 3. Merge 1d regime features.
    if cfg.include_1d_features:
        frames_1d = []
        for symbol in symbols:
            f = build_1d_regime_features(symbol, cfg)
            if f is not None and not f.empty:
                frames_1d.append(f)
        if frames_1d:
            f1d = pd.concat(frames_1d, ignore_index=True)
            features = _asof_merge_by_symbol(features, f1d, "market_1d_available_time", "1d")

    # 4. Merge derivatives and optional sources.
    if cfg.include_funding:
        funding = load_funding_features(cfg)
        if funding is not None and not funding.empty:
            features = _asof_merge_by_symbol(features, funding, "funding_available_time", "funding")

    if cfg.include_oi:
        oi = load_oi_features(cfg)
        if oi is not None and not oi.empty:
            features = _asof_merge_by_symbol(features, oi, "oi_available_time", "oi")

    if cfg.include_cvd_proxy:
        cvd = load_cvd_features(cfg)
        if cvd is not None and not cvd.empty:
            features = _asof_merge_by_symbol(features, cvd, "cvd_available_time", "cvd")

    if cfg.include_sentiment:
        sentiment = load_sentiment_features(cfg)
        if sentiment is not None and not sentiment.empty:
            if "symbol" in sentiment.columns:
                features = _asof_merge_by_symbol(features, sentiment, "sentiment_available_time", "sentiment")
            else:
                features = _asof_merge_global(features, sentiment, "sentiment_available_time", "sentiment")

    if cfg.include_onchain:
        onchain = load_onchain_features(cfg)
        if onchain is not None and not onchain.empty:
            if "symbol" in onchain.columns:
                features = _asof_merge_by_symbol(features, onchain, "onchain_available_time", "onchain")
            else:
                features = _asof_merge_global(features, onchain, "onchain_available_time", "onchain")

    # 5. Final audit and save.
    features = validate_pit_features(features, strict=cfg.strict_pit_check)
    features["feature_version"] = "feature_builder_v1"

    if save:
        out_path = Path(output_path) if output_path is not None else factors_dir() / cfg.output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
        print(f"saved crypto PIT features: {out_path} ({len(features)} rows, {len(features.columns)} columns)")

    return features


def build_feature_view(
    feature_path: Optional[str | Path] = None,
    columns: Optional[Iterable[str]] = None,
    symbols: Optional[Sequence[str]] = None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """
    Read a saved crypto_features.parquet and return a smaller feature view.

    This is useful for Agent input or model experiments where you do not want to
    load every optional feature column.
    """
    path = Path(feature_path) if feature_path is not None else factors_dir() / "crypto_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    base_cols = ["symbol", "decision_time"]
    if columns is None:
        df = pd.read_parquet(path)
    else:
        cols = list(dict.fromkeys(base_cols + list(columns)))
        df = pd.read_parquet(path, columns=cols)

    df["decision_time"] = _to_datetime_ns(df["decision_time"])

    if symbols is not None:
        df = df[df["symbol"].isin(list(symbols))]
    if start_date is not None:
        df = df[df["decision_time"] >= _to_datetime_ns(pd.Series([start_date])).iloc[0]]
    if end_date is not None:
        df = df[df["decision_time"] <= _to_datetime_ns(pd.Series([end_date])).iloc[0]]

    return df.sort_values(["symbol", "decision_time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Common feature views for LLM / model input
# ---------------------------------------------------------------------------


_4H_MARKET_COLUMNS = [
    "open_4h",
    "high_4h",
    "low_4h",
    "close_4h",
    "volume_4h",
    "ret_4h",
    "ret_24h",
    "ret_96h",
    "vol_96h",
    "range_4h",
    "close_position_in_4h_range",
    "volume_z_96h",
]

_4H_MULTITIMEFRAME_COLUMNS = _4H_MARKET_COLUMNS + [
    "ret_1h",
    "ret_3h",
    "ret_6h",
    "ret_12h",
    "ret_24h_from_1h",
    "vol_24h_from_1h",
    "volume_z_24h_from_1h",
    "range_6h_from_1h",
    "ret_1d",
    "ret_3d",
    "ret_7d",
    "ret_30d",
    "vol_30d",
    "daily_ma_gap_7_30",
    "daily_trend_up",
    "drawdown_from_30d_high",
]

_4H_FULL_OPTIONAL_COLUMNS = _4H_MULTITIMEFRAME_COLUMNS + [
    "funding_rate",
    "funding_interval_hours",
    "funding_rate_8h_equiv",
    "funding_rate_chg",
    "funding_rate_z_30_events",
    "open_interest",
    "open_interest_value",
    "oi_chg_1",
    "oi_z_30",
    "spot_cvd_proxy_4h",
    "spot_cvd_proxy_chg_4h",
]


if __name__ == "__main__":
    cfg = FeatureBuilderConfig()
    build_crypto_features(cfg=cfg, save=True)
