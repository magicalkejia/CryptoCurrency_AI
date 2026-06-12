"""
etl.onchain_feature_builder
===========================

Build on-chain / DeFi factor tables from processed daily source data.

Input:
    data_storage/processed/onchain/onchain_daily.parquet

Output:
    data_storage/factors/onchain_features.parquet

Layering rule:
    onchain_processor.py cleans and merges source data only.
    This module creates derived features used by models.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import config


def _path_attr(name: str, fallback: Path) -> Path:
    return Path(getattr(config.PathConfig, name, fallback))


def processed_onchain_root() -> Path:
    return _path_attr("PROCESSED_ONCHAIN", Path(config.PathConfig.PROCESSED) / "onchain")


def factors_root() -> Path:
    return Path(config.PathConfig.FACTORS)


def _ensure_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        raise ValueError(f"missing timestamp column: {col}")
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.dropna(subset=[col])


def _safe_pct_change(s: pd.Series, periods: int) -> pd.Series:
    return s.replace(0, np.nan).pct_change(periods)


def _safe_zscore(s: pd.Series, window: int = 30, min_periods: int = 10) -> pd.Series:
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    return (s - mean) / (std + 1e-12)


def build_onchain_factors(
    processed_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    include_levels: bool = True,
    pct_change_periods: Sequence[int] = (1, 7),
    zscore_windows: Sequence[int] = (30,),
    zscore_min_periods: int = 10,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build model-ready on-chain factors from processed daily data.

    Level features are kept by default because they describe slow regime state.
    Derived features improve stationarity and comparability:
        *_chg_1d = x_t / x_{t-1} - 1
        *_chg_7d = x_t / x_{t-7} - 1
        *_z_30d  = (x_t - rolling_mean_30d) / rolling_std_30d
    """
    in_path = Path(processed_path) if processed_path is not None else processed_onchain_root() / "onchain_daily.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Processed on-chain data not found: {in_path}")

    base = pd.read_parquet(in_path)
    if base.empty:
        out = pd.DataFrame()
    else:
        base = _ensure_timestamp(base).sort_values("timestamp").reset_index(drop=True)
        numeric_cols = [
            c for c in base.columns
            if c != "timestamp" and pd.api.types.is_numeric_dtype(base[c])
        ]

        out = base[["timestamp"]].copy()
        if include_levels:
            for col in numeric_cols:
                out[col] = pd.to_numeric(base[col], errors="coerce")

        for col in numeric_cols:
            s = pd.to_numeric(base[col], errors="coerce")
            for period in pct_change_periods:
                out[f"{col}_chg_{period}d"] = _safe_pct_change(s, period)
            for window in zscore_windows:
                out[f"{col}_z_{window}d"] = _safe_zscore(
                    s,
                    window=window,
                    min_periods=min(zscore_min_periods, window),
                )

        out = out.replace([np.inf, -np.inf], np.nan)

    if save:
        out_path = Path(output_path) if output_path is not None else factors_root() / "onchain_features.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
        print(f"saved on-chain factors: {out_path} ({len(out):,} rows, {len(out.columns)} columns)")

    return out


def load_onchain_factors(
    start_date=None,
    end_date=None,
    columns: Optional[Sequence[str]] = None,
    path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Read saved on-chain factors with optional date and column filters."""
    factor_path = Path(path) if path is not None else factors_root() / "onchain_features.parquet"
    if not factor_path.exists():
        raise FileNotFoundError(f"On-chain factor table not found: {factor_path}")

    base_cols = ["timestamp"]
    if columns is not None:
        read_cols = list(dict.fromkeys(base_cols + list(columns)))
        df = pd.read_parquet(factor_path, columns=read_cols)
    else:
        df = pd.read_parquet(factor_path)

    df = _ensure_timestamp(df)
    if start_date is not None:
        df = df[df["timestamp"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["timestamp"] <= pd.to_datetime(end_date)]
    return df.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build on-chain factor table from processed daily on-chain data.")
    parser.add_argument("--input", default=None, help="Processed input parquet path")
    parser.add_argument("--output", default=None, help="Factor output parquet path")
    parser.add_argument("--no-levels", action="store_true", help="Only write derived features, not base levels")
    args = parser.parse_args()

    build_onchain_factors(
        processed_path=args.input,
        output_path=args.output,
        include_levels=not args.no_levels,
        save=True,
    )


if __name__ == "__main__":
    main()
