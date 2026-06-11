# etl/crypto_pipeline.py
import pandas as pd

import config
import etl.data_processor as data_processor
import etl.data_updater as data_updater


def run_crypto_pipeline(fetch_derivatives: bool = False, fetch_oi: bool = False):
    """
    Crypto 数据主流程。

    基础行情：
    - 每个 symbol 拉取 raw 1m K 线；
    - 清洗为 processed 1m；
    - 重采样为 1h / 4h / 1d。

    衍生品数据：
    - raw 层按 symbol + source 分目录增量抓取，便于失败重跑；
    - processed 层统一在所有 symbol 抓取完成后生成 long table。

    当前 processed long table 已实现：
    - funding: data_storage/processed/derivatives/funding.parquet

    OI 当前只抓 raw：
    - data_storage/raw/derivatives/oi/{SYMBOL}.parquet
    后续再补 processed/derivatives/oi.parquet。
    """
    results = []

    for symbol in config.TargetConfig.COINS:
        ok_fetch = data_updater.fetch_data(symbol)
        df_clean = data_processor.clean_single_file(symbol)

        ok_clean = df_clean is not None
        ok_resample = False

        if ok_clean:
            data_processor.resample_data(df_clean, symbol)
            ok_resample = True

        ok_funding = None
        ok_oi = None
        if fetch_derivatives:
            ok_funding = data_updater.fetch_funding_rate(symbol)
            if fetch_oi:
                ok_oi = data_updater.fetch_open_interest(symbol, period="1h")

        results.append({
            "symbol": symbol,
            "fetch": ok_fetch,
            "clean": ok_clean,
            "resample": ok_resample,
            "funding_raw": ok_funding,
            "oi_raw": ok_oi,
        })

    funding_rows = None
    if fetch_derivatives:
        funding_long = data_processor.process_funding_rates(config.TargetConfig.COINS)
        funding_rows = None if funding_long is None else len(funding_long)

    df_result = pd.DataFrame(results)
    if funding_rows is not None:
        df_result.attrs["processed_funding_rows"] = funding_rows
        print(f"processed funding long table rows: {funding_rows}")

    print(df_result)
    return df_result
