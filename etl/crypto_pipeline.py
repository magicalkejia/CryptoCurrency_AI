# etl/crypto_pipeline.py
import time
import config
import etl.data_updater as data_updater
import etl.data_processor as data_processor
import pandas as pd
def run_crypto_pipeline():
    results = []

    for symbol in config.TargetConfig.COINS:
        ok_fetch = data_updater.fetch_data(symbol)
        df_clean = data_processor.clean_single_file(symbol)

        ok_clean = df_clean is not None
        ok_resample = False

        if ok_clean:
            data_processor.resample_data(df_clean, symbol)
            ok_resample = True

        results.append({
            "symbol": symbol,
            "fetch": ok_fetch,
            "clean": ok_clean,
            "resample": ok_resample,
        })

    return pd.DataFrame(results)