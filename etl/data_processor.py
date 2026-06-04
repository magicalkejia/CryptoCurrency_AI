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