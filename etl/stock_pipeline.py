# etl/stock_pipeline.py
import time
import etl.data_updater as data_updater
import etl.factor_builder as factor_builder

def run_stock_pipeline(update_history=False, build_factors=True):
    print("========================================")
    print(" STOCK DATA PIPELINE STARTED")
    print("========================================")

    total_start = time.time()

    data_updater.update_instrument_master()
    # data_updater.update_daily_cross_section()

    if update_history:
        data_updater.update_all_history_data()

    if build_factors:
        factor_builder.build_all_factors()

    total_elapsed = time.time() - total_start
    print(f"\n✅ STOCK PIPELINE FINISHED in {total_elapsed/60:.2f} minutes")