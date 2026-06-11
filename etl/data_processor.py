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
