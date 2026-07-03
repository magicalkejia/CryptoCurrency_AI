import pandas as pd
import numpy as np
import os
import config  

def final_quality_check(df, file_name):
    """
    Final Quality Check before saving.
    """
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        return False, f" NaN values found in columns: {null_cols}"

    if np.isinf(df.select_dtypes(include=np.number)).any().any():
        return False, " Infinity values found"

    if (df['close'] <= 0).any():
        return False, " Price <= 0 found"

    if (df['high'] < df['low']).any():
        return False, " High < Low found"

    if not df['timestamp'].is_unique:
        return False, " Non-unique timestamps"
    if not df['timestamp'].is_monotonic_increasing:
        return False, " Timestamps not monotonic"

    return True, "✅ Passed"

def clean_single_file(symbol):
    """
    Reads RAW data -> Cleans it -> Saves to PROCESSED -> Returns DataFrame
    """
    # Construct paths using Config
    symbol_clean = symbol.replace('/', '')
    file_name = f"{symbol_clean}_{config.TargetConfig.TIMEFRAMES['base']}.parquet"
    
    input_path = config.PathConfig.RAW / file_name
    output_path = config.PathConfig.PROCESSED / file_name

    if not os.path.exists(input_path):
        print(f"⚠️ Raw file not found: {input_path}")
        return None

    print(f"\n🧹 [Cleaning] {symbol}...")
    
    try:
        df = pd.read_parquet(input_path, engine='pyarrow')
        if df.empty:
            print(f"⚠️ Empty raw file: {input_path}")
            return None
        # --- A. Deduplication ---
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # --- B. Gap Filling ---
        df = df.sort_values('timestamp')
        df = df.set_index('timestamp')
        
        if len(df) > 0:
            full_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='1min')
            if len(full_range) != len(df):
                missing_count = len(full_range) - len(df)
                print(f"   🔧 Filling gaps: {missing_count} min")
                df = df.reindex(full_range)
        
        # --- C. Imputation ---
        df['close'] = df['close'].ffill()
        
        # Handle empty head
        if df['close'].iloc[0] is None or np.isnan(df['close'].iloc[0]):
             df['close'] = df['close'].bfill()

        df['open'] = df['open'].fillna(df['close'])
        df['high'] = df['high'].fillna(df['close'])
        df['low']  = df['low'].fillna(df['close'])
        
        df['volume'] = df['volume'].fillna(0)
        df['taker_buy_vol'] = df['taker_buy_vol'].fillna(0)
        
        # --- D. Sanity Check Fixes ---
        mask = df['high'] < df['low']
        if mask.any():
            df.loc[mask, ['high', 'low']] = df.loc[mask, ['low', 'high']].values
            
        mask_zero = df['close'] <= 0
        if mask_zero.any():
            df.loc[mask_zero, ['open','high','low','close']] = np.nan
            df['close'] = df['close'].ffill()
            df['open'] = df['open'].fillna(df['close'])
            df['high'] = df['high'].fillna(df['close'])
            df['low']  = df['low'].fillna(df['close'])

        # --- E. Derived Metrics ---
        df['net_taker_vol'] = df['taker_buy_vol'] - (df['volume'] - df['taker_buy_vol'])

        # --- F. Quality Check & Save ---
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        
        passed, msg = final_quality_check(df, file_name)
        if not passed:
            print(f"   ⛔ {msg}")
            return None
            
        # Ensure directory exists
        ##os.makedirs(output_path.parent, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, engine='pyarrow', compression='zstd', index=False)
        print(f"   💾 Saved Processed Data")
        
        return df  # RETURN THE DATAFRAME for resampling

    except Exception as e:
        print(f"   ❌ Error cleaning {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def resample_data(df_1m, symbol,drop_last_incomplete=True):
    """
    Resamples 1m data to 1h, 4h, 1d based on Config
    """
    if df_1m is None or df_1m.empty: return

    print(f"🔨 [Resampling] Generating multi-timeframe data for {symbol}...")
    df_1m = df_1m.set_index('timestamp')

    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'taker_buy_vol': 'sum'
    }

    for timeframe in config.TargetConfig.TIMEFRAMES['resample']: # e.g. ['1h', '4h']
        try:
            # Pandas freq conversion
            freq = timeframe.lower().replace('m', 'min').replace('d', 'D') 
            
            # Resample
            df_resampled = df_1m.resample(freq, label='left', closed='left').agg(agg_rules)
            
            # Re-calculate derived metric
            df_resampled['net_taker_vol'] = df_resampled['taker_buy_vol'] - (df_resampled['volume'] - df_resampled['taker_buy_vol'])
            
            # Drop NaN (usually at the start)
            df_resampled = df_resampled.dropna()
            if drop_last_incomplete and len(df_resampled) > 0:
                 df_resampled = df_resampled.iloc[:-1]

            # Save
            symbol_clean = symbol.replace('/', '')
            save_name = f"{symbol_clean}_{timeframe}.parquet"
            save_path = config.PathConfig.PROCESSED / save_name
            
            df_resampled = df_resampled.reset_index()
            df_resampled.to_parquet(
                save_path,
                engine="pyarrow",
                compression="zstd",
                index=False,
            )
            print(f"   -> Generated: {timeframe} ({len(df_resampled)} rows)")
            
        except Exception as e:
            print(f"  ❌ Error resampling {timeframe}: {e}")

# =====================================================================
# Crypto 衍生品数据处理：funding rate
# =====================================================================
def funding_long_quality_check(df: pd.DataFrame, table_name: str = "funding"):
    """
    Funding processed long table 数据质量检查。

    long table 允许不同 symbol 共享同一个 timestamp，
    因此唯一性必须按 [symbol, timestamp] 检查，而不是 timestamp 全局唯一。
    """
    required_cols = [
        "timestamp",
        "symbol",
        "funding_rate",
        "source",
        "created_at",
        "funding_interval_hours_raw",
        "funding_interval_hours",
        "funding_rate_8h_equiv",
        "funding_rate_chg",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"

    if df.empty:
        return False, "Empty funding dataframe"

    not_null_cols = ["timestamp", "symbol", "funding_rate", "source"]
    null_cols = [c for c in not_null_cols if df[c].isnull().any()]
    if null_cols:
        return False, f"Null values found in required columns: {null_cols}"

    duplicated = df.duplicated(subset=["symbol", "timestamp"], keep=False)
    if duplicated.any():
        dup = df.loc[duplicated, ["symbol", "timestamp"]].head(20)
        return False, f"Duplicate symbol + timestamp found:\n{dup}"

    for symbol, g in df.sort_values(["symbol", "timestamp"]).groupby("symbol"):
        if not g["timestamp"].is_monotonic_increasing:
            return False, f"Timestamps not monotonic for {symbol}"

    numeric_cols = df.select_dtypes(include=np.number).columns
    if np.isinf(df[numeric_cols]).any().any():
        return False, "Infinity values found"

    return True, "✅ Passed"


def _funding_raw_paths(symbol: str):
    """新路径优先，兼容旧版本 raw/{SYMBOL}_funding.parquet。"""
    symbol_clean = symbol.replace("/", "")
    new_path = config.PathConfig.RAW_FUNDING / f"{symbol_clean}.parquet"
    legacy_path = config.PathConfig.RAW / f"{symbol_clean}_funding.parquet"
    return new_path, legacy_path


def process_single_funding_rate(
    symbol: str,
    z_window: int = 30,
) -> pd.DataFrame | None:
    """
    处理单个 symbol 的 RAW funding history，返回单币种 processed DataFrame。

    RAW 输入：
        data_storage/raw/derivatives/funding/{SYMBOL}.parquet

    兼容旧路径：
        data_storage/raw/{SYMBOL}_funding.parquet

    注意：
    - 本函数只返回 DataFrame，不保存 per-symbol processed 文件。
    - 最终保存由 process_funding_rates() 统一生成 long table。
    """
    new_path, legacy_path = _funding_raw_paths(symbol)
    input_path = new_path if os.path.exists(new_path) else legacy_path

    if not os.path.exists(input_path):
        print(f"⚠️ Funding raw file not found: {new_path} or {legacy_path}")
        return None

    try:
        df = pd.read_parquet(input_path, engine="pyarrow")
        if df.empty:
            print(f"⚠️ Empty funding raw file: {input_path}")
            return None

        # 兼容旧版本 raw 文件。
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        else:
            df["symbol"] = df["symbol"].fillna(symbol)
        if "source" not in df.columns:
            df["source"] = "binance_usdm"
        else:
            df["source"] = df["source"].fillna("binance_usdm")
        if "created_at" not in df.columns:
            df["created_at"] = pd.NaT

        required_cols = ["timestamp", "symbol", "funding_rate", "source", "created_at"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Funding raw file missing required columns: {missing_cols}")

        df = df[required_cols].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
        df["symbol"] = df["symbol"].astype(str)
        df["source"] = df["source"].astype(str)

        df = df.dropna(subset=["timestamp", "symbol", "funding_rate"])
        df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        # 单币种内部计算 funding interval / change / rolling z-score。
        interval_raw = df["timestamp"].diff().dt.total_seconds() / 3600
        df["funding_interval_hours_raw"] = interval_raw
        df["funding_interval_hours"] = interval_raw.round().astype("Int64")

        df["funding_rate_8h_equiv"] = np.where(
            df["funding_interval_hours"].notna() & (df["funding_interval_hours"] > 0),
            df["funding_rate"] * 8 / df["funding_interval_hours"].astype(float),
            np.nan,
        )

        df["funding_rate_chg"] = df["funding_rate"].diff()

        min_periods = max(5, z_window // 3)
        rolling_mean = df["funding_rate"].rolling(
            z_window,
            min_periods=min_periods,
        ).mean()
        rolling_std = df["funding_rate"].rolling(
            z_window,
            min_periods=min_periods,
        ).std()
        df[f"funding_rate_z_{z_window}_events"] = (
            (df["funding_rate"] - rolling_mean) / (rolling_std + 1e-12)
        )

        return df

    except Exception as e:
        print(f"   ❌ Error processing funding {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_funding_rates(
    symbols=None,
    z_window: int = 30,
    save: bool = True,
) -> pd.DataFrame | None:
    """
    处理所有 symbol 的 RAW funding history，生成 PROCESSED long table。

    PROCESSED 输出：
        data_storage/processed/derivatives/funding.parquet

    输出结构：
        一行 = 一个 symbol 的一条 funding event。
        主键 = [symbol, timestamp]
    """
    if symbols is None:
        symbols = config.TargetConfig.COINS

    frames = []
    for symbol in symbols:
        df = process_single_funding_rate(symbol=symbol, z_window=z_window)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        print("⚠️ No funding data processed")
        return None

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    passed, msg = funding_long_quality_check(out, "funding.parquet")
    if not passed:
        print(f"   ⛔ Funding long table quality check failed: {msg}")
        return None

    if save:
        output_path = config.PathConfig.PROCESSED_DERIVATIVES / "funding.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)
        print(f"   💾 Saved processed funding long table: {output_path} ({len(out)} rows)")

    return out


def process_funding_rate(
    symbol: str,
    z_window: int = 30,
    save: bool = False,
):
    """
    向后兼容旧调用。

    不再保存 data_storage/processed/{SYMBOL}_funding.parquet。
    新流程请使用 process_funding_rates(symbols)，统一输出 long table：
        data_storage/processed/derivatives/funding.parquet
    """
    if save:
        print("⚠️ process_funding_rate(save=True) 已废弃；请使用 process_funding_rates() 生成 long table。")
    return process_single_funding_rate(symbol=symbol, z_window=z_window)


# =====================================================================
# Spot data, spot/perp basis and taker-CVD processing
# =====================================================================
def _timeframe_to_freq(timeframe: str) -> str:
    tf = timeframe.lower()
    return tf.replace("m", "min").replace("d", "D")


def _timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    tf = timeframe.lower()
    if tf.endswith("m"):
        return pd.Timedelta(minutes=int(tf[:-1]))
    if tf.endswith("h"):
        return pd.Timedelta(hours=int(tf[:-1]))
    if tf.endswith("d"):
        return pd.Timedelta(days=int(tf[:-1]))
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _rolling_zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or max(5, window // 3)
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    return (s - mean) / (std + 1e-12)


def _spot_raw_path(symbol: str, timeframe: str = "1m"):
    symbol_clean = symbol.replace("/", "")
    return config.PathConfig.RAW_SPOT / f"{symbol_clean}_{timeframe}.parquet"


def _load_clean_spot_1m(symbol: str) -> pd.DataFrame | None:
    input_path = _spot_raw_path(symbol, timeframe=config.TargetConfig.TIMEFRAMES["base"])
    if not os.path.exists(input_path):
        print(f"[WARN] Spot raw file not found: {input_path}")
        return None

    try:
        df = pd.read_parquet(input_path, engine="pyarrow")
        if df.empty:
            print(f"[WARN] Empty spot raw file: {input_path}")
            return None

        if "symbol" not in df.columns:
            df["symbol"] = symbol
        else:
            df["symbol"] = df["symbol"].fillna(symbol)
        if "source" not in df.columns:
            df["source"] = "binance_spot"
        if "created_at" not in df.columns:
            df["created_at"] = pd.NaT

        required = [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "taker_buy_vol",
            "source",
            "created_at",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Spot raw file missing columns: {missing}")

        df = df[required].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume", "taker_buy_vol"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["timestamp"])
        df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        df = df.set_index("timestamp")
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1min")
        if len(full_range) != len(df):
            missing_count = len(full_range) - len(df)
            print(f"   Filling spot gaps for {symbol}: {missing_count} min")
            df = df.reindex(full_range)

        df["symbol"] = symbol
        df["source"] = df["source"].ffill().bfill().fillna("binance_spot")
        df["created_at"] = df["created_at"].ffill().bfill()
        df["close"] = df["close"].ffill().bfill()
        df["open"] = df["open"].fillna(df["close"])
        df["high"] = df["high"].fillna(df["close"])
        df["low"] = df["low"].fillna(df["close"])
        df["volume"] = df["volume"].fillna(0)
        df["taker_buy_vol"] = df["taker_buy_vol"].fillna(0)

        mask = df["high"] < df["low"]
        if mask.any():
            df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values

        mask_zero = df["close"] <= 0
        if mask_zero.any():
            df.loc[mask_zero, ["open", "high", "low", "close"]] = np.nan
            df["close"] = df["close"].ffill().bfill()
            df["open"] = df["open"].fillna(df["close"])
            df["high"] = df["high"].fillna(df["close"])
            df["low"] = df["low"].fillna(df["close"])

        df["net_taker_vol"] = df["taker_buy_vol"] - (df["volume"] - df["taker_buy_vol"])
        return df.reset_index().rename(columns={"index": "timestamp"})

    except Exception as e:
        print(f"[ERROR] Failed to process spot raw {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _resample_spot_ohlcv(
    df_1m: pd.DataFrame,
    timeframe: str,
    drop_last_incomplete: bool = True,
) -> pd.DataFrame:
    df_1m = df_1m.copy()
    if timeframe == "1m":
        out = df_1m.copy()
    else:
        symbol = str(df_1m["symbol"].iloc[0])
        source = str(df_1m["source"].dropna().iloc[-1]) if df_1m["source"].notna().any() else "binance_spot"
        g = df_1m.set_index("timestamp").sort_index()
        out = g.resample(
            _timeframe_to_freq(timeframe),
            label="left",
            closed="left",
        ).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "taker_buy_vol": "sum",
            "net_taker_vol": "sum",
        })
        out = out.dropna(subset=["open", "high", "low", "close"])
        if drop_last_incomplete and len(out) > 0:
            out = out.iloc[:-1]
        out = out.reset_index()
        out["symbol"] = symbol
        out["source"] = source
        out["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)

    out["timeframe"] = timeframe
    keep = [
        "timestamp",
        "symbol",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "taker_buy_vol",
        "net_taker_vol",
        "source",
        "created_at",
    ]
    return out[keep].sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def process_spot_klines(
    symbols=None,
    timeframes: tuple[str, ...] | list[str] | str = ("4h",),
    save: bool = True,
    drop_last_incomplete: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Process Binance spot raw K-lines into long tables.

    Outputs:
        data_storage/processed/spot/spot_klines_{timeframe}.parquet
    """
    if symbols is None:
        symbols = config.TargetConfig.DIVERSIFIED_10_COINS
    if isinstance(timeframes, str):
        timeframes = (timeframes,)

    frames_by_tf: dict[str, list[pd.DataFrame]] = {tf: [] for tf in timeframes}
    for symbol in symbols:
        df_1m = _load_clean_spot_1m(symbol)
        if df_1m is None or df_1m.empty:
            continue
        for tf in timeframes:
            frames_by_tf[tf].append(
                _resample_spot_ohlcv(
                    df_1m,
                    timeframe=tf,
                    drop_last_incomplete=drop_last_incomplete,
                )
            )

    outputs: dict[str, pd.DataFrame] = {}
    for tf, frames in frames_by_tf.items():
        if not frames:
            print(f"[WARN] No spot data processed for timeframe={tf}")
            continue
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset=["symbol", "timeframe", "timestamp"], keep="last")
        out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        outputs[tf] = out

        if save:
            output_path = config.PathConfig.PROCESSED_SPOT / f"spot_klines_{tf}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)
            print(f"   Saved processed spot table: {output_path} ({len(out)} rows)")

    return outputs


def _load_spot_table_or_build(symbols, timeframe: str) -> pd.DataFrame | None:
    path = config.PathConfig.PROCESSED_SPOT / f"spot_klines_{timeframe}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
    else:
        built = process_spot_klines(symbols=symbols, timeframes=(timeframe,), save=True)
        df = built.get(timeframe)

    if df is None or df.empty:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp", "symbol"]).sort_values(["symbol", "timestamp"])


def process_spot_perp_basis(
    symbols=None,
    timeframe: str = "4h",
    z_window: int = 30,
    save: bool = True,
) -> pd.DataFrame | None:
    """
    Build a long table of spot/perpetual premium.

    Output:
        data_storage/processed/derivatives/spot_perp_basis.parquet

    Positive perp_premium means perp close > spot close.
    """
    if symbols is None:
        symbols = config.TargetConfig.DIVERSIFIED_10_COINS

    spot = _load_spot_table_or_build(symbols=symbols, timeframe=timeframe)
    if spot is None or spot.empty:
        print("[WARN] No processed spot table available for basis calculation")
        return None

    perp_frames = []
    for symbol in symbols:
        symbol_clean = symbol.replace("/", "")
        p = config.PathConfig.PROCESSED / f"{symbol_clean}_{timeframe}.parquet"
        if not os.path.exists(p):
            print(f"[WARN] Perp processed file not found for basis: {p}")
            continue
        df = pd.read_parquet(p, columns=["timestamp", "close"])
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["symbol"] = symbol
        df = df.rename(columns={"close": "perp_close"})
        perp_frames.append(df[["timestamp", "symbol", "perp_close"]])

    if not perp_frames:
        print("[WARN] No perp data available for basis calculation")
        return None

    perp = pd.concat(perp_frames, ignore_index=True)
    merged = pd.merge(
        spot[["timestamp", "symbol", "close"]].rename(columns={"close": "spot_close"}),
        perp,
        on=["symbol", "timestamp"],
        how="inner",
    )
    if merged.empty:
        print("[WARN] Spot/perp basis merge produced no rows")
        return None

    out = merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    out["timeframe"] = timeframe
    out["perp_premium"] = out["perp_close"] / out["spot_close"] - 1
    out["spot_premium"] = out["spot_close"] / out["perp_close"] - 1
    out["perp_premium_bps"] = out["perp_premium"] * 10000
    out["perp_premium_chg_1"] = out.groupby("symbol")["perp_premium"].diff()
    out[f"perp_premium_z_{z_window}"] = out.groupby("symbol")["perp_premium"].transform(
        lambda s: _rolling_zscore(s, z_window)
    )
    out["basis_available_time"] = out["timestamp"] + _timeframe_to_timedelta(timeframe)
    out["source"] = "binance_spot_vs_usdm_perp"

    keep = [
        "timestamp",
        "symbol",
        "timeframe",
        "spot_close",
        "perp_close",
        "perp_premium",
        "spot_premium",
        "perp_premium_bps",
        "perp_premium_chg_1",
        f"perp_premium_z_{z_window}",
        "basis_available_time",
        "source",
    ]
    out = out[keep].sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    if save:
        output_path = config.PathConfig.PROCESSED_DERIVATIVES / "spot_perp_basis.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)
        print(f"   Saved processed spot/perp basis table: {output_path} ({len(out)} rows)")

    return out


def process_cvd(
    symbols=None,
    timeframe: str = "4h",
    z_window: int = 30,
    save: bool = True,
    drop_last_incomplete: bool = True,
) -> pd.DataFrame | None:
    """
    Build taker-CVD from processed USD-M futures K-lines.

    Output:
        data_storage/processed/flow/cvd.parquet

    This is kline-level taker CVD, not tick-level true CVD.
    """
    if symbols is None:
        symbols = config.TargetConfig.DIVERSIFIED_10_COINS

    frames = []
    for symbol in symbols:
        symbol_clean = symbol.replace("/", "")
        p = config.PathConfig.PROCESSED / f"{symbol_clean}_1m.parquet"
        if not os.path.exists(p):
            print(f"[WARN] Perp 1m processed file not found for CVD: {p}")
            continue

        df = pd.read_parquet(p)
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        for col in ["volume", "taker_buy_vol"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if "net_taker_vol" not in df.columns:
            df["net_taker_vol"] = df["taker_buy_vol"] - (df["volume"] - df["taker_buy_vol"])
        else:
            df["net_taker_vol"] = pd.to_numeric(df["net_taker_vol"], errors="coerce").fillna(0)

        g = df.set_index("timestamp")
        res = g.resample(
            _timeframe_to_freq(timeframe),
            label="left",
            closed="left",
        ).agg({
            "volume": "sum",
            "taker_buy_vol": "sum",
            "net_taker_vol": "sum",
        })
        res = res.dropna()
        if drop_last_incomplete and len(res) > 0:
            res = res.iloc[:-1]
        if res.empty:
            continue

        res = res.reset_index()
        res["symbol"] = symbol
        res["timeframe"] = timeframe
        res["taker_sell_vol"] = res["volume"] - res["taker_buy_vol"]
        res["cvd_delta"] = res["net_taker_vol"]
        res["cvd"] = res["cvd_delta"].cumsum()
        res["cvd_delta_chg_1"] = res["cvd_delta"].diff()
        res[f"cvd_delta_z_{z_window}"] = _rolling_zscore(res["cvd_delta"], z_window)
        res["cvd_available_time"] = res["timestamp"] + _timeframe_to_timedelta(timeframe)
        res["source"] = "binance_usdm_kline_taker_flow"
        frames.append(res)

    if not frames:
        print("[WARN] No CVD data processed")
        return None

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["symbol", "timeframe", "timestamp"], keep="last")
    keep = [
        "timestamp",
        "symbol",
        "timeframe",
        "volume",
        "taker_buy_vol",
        "taker_sell_vol",
        "cvd_delta",
        "cvd",
        "cvd_delta_chg_1",
        f"cvd_delta_z_{z_window}",
        "cvd_available_time",
        "source",
    ]
    out = out[keep].sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    if save:
        output_path = config.PathConfig.PROCESSED_FLOW / "cvd.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)
        print(f"   Saved processed CVD table: {output_path} ({len(out)} rows)")

    return out
