import time
import config
import data_updater
import data_processor

def run_pipeline():
    #显式检测和初始化本地存储的文件夹
    config.init_directories()
    print("========================================")
    print(" QUANT DATA PIPELINE STARTED")
    print("========================================")
    total_start = time.time()
    
    # Iterate through target coins defined in config.py
    for symbol in config.TargetConfig.COINS:
        print(f"\n----------------------------------------")
        print(f"Target: {symbol}")
        print(f"----------------------------------------")
        
        # --- Step 1: Collection / Incremental Update ---
        # Returns True if new data was downloaded, False if already up to date
        has_new_data = data_updater.fetch_data(symbol)
        
        # --- Step 2: Cleaning ---
        # We perform cleaning if there is new data OR if we want to ensure consistency.
        # Since cleaning is fast with Parquet, running it every time ensures
        # we always have the latest 'processed' version ready for resampling.
        df_clean = data_processor.clean_single_file(symbol)
        
        # --- Step 3: Resampling ---
        # Only proceed if cleaning was successful
        if df_clean is not None:
            data_processor.resample_data(df_clean, symbol)
        else:
            print(f"⚠️ Skipping resampling for {symbol} due to cleaning failure.")

    total_elapsed = time.time() - total_start
    print("\n========================================")
    print(f"✅ PIPELINE FINISHED in {total_elapsed/60:.2f} minutes")
    print("========================================")

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n🛑 Pipeline stopped by user.")
    except Exception as e:
        print(f"\n❌ Critical Pipeline Error: {e}")