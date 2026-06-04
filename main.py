# main.py
import argparse
import config
from etl.crypto_pipeline import run_crypto_pipeline
from etl.stock_pipeline import run_stock_pipeline

def run_pipeline(mode: str):
    config.init_directories()

    if mode == "crypto":
        run_crypto_pipeline()
    elif mode == "stock":
        run_stock_pipeline()
    elif mode == "all":
        run_stock_pipeline()
        run_crypto_pipeline()
    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["crypto", "stock", "all"],
        default="crypto"
    )
    args = parser.parse_args()

    run_pipeline(args.mode)