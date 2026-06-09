# main.py
import argparse
import config
from etl.crypto_pipeline import run_crypto_pipeline
# from etl.stock_pipeline import run_stock_pipeline

def run_pipeline(mode: str):
    config.init_directories()

    if mode == "crypto":
        result = run_crypto_pipeline()
        print(result)
        return result

    # elif mode == "stock":
    #     # result = run_stock_pipeline()
    #     return result

    # elif mode == "all":
    #     stock_result = run_stock_pipeline()
    #     crypto_result = run_crypto_pipeline()
    #     print(crypto_result)
    #     return {
    #         "stock": stock_result,
    #         "crypto": crypto_result,
    #     }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["crypto"],
        default="crypto"
    )
    args = parser.parse_args()

    run_pipeline(args.mode)