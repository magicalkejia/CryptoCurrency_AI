# etl/crypto_pipeline.py
import time
import config
import etl.data_updater as data_updater
import etl.data_processor as data_processor
import pandas as pd
def run_crypto_pipeline(fetch_derivatives: bool = False):
    results = []

    for symbol in config.TargetConfig.COINS:
        ok_fetch = data_updater.fetch_data(symbol)
        df_clean = data_processor.clean_single_file(symbol)

        ok_clean = df_clean is not None
        ok_resample = False

        if ok_clean:
            data_processor.resample_data(df_clean, symbol)
            ok_resample = True

        # v6 新增：可选抓取衍生品 (funding / open interest)，默认关闭
        ok_funding = ok_oi = None
        if fetch_derivatives:
            ok_funding = data_updater.fetch_funding_rate(symbol)
            ok_oi = data_updater.fetch_open_interest(symbol, period='1h')

        results.append({
            "symbol": symbol,
            "fetch": ok_fetch,
            "clean": ok_clean,
            "resample": ok_resample,
            "funding": ok_funding,
            "oi": ok_oi,
        })

    # FIX (v6): these three lines were indented INSIDE the for-loop, so the
    # pipeline returned after the FIRST coin (only BTC) was ever processed.
    # De-indented to run for all coins in TargetConfig.COINS.
    df_result = pd.DataFrame(results)
    print(df_result)
    return df_result