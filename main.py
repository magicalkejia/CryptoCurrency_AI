# main.py
import argparse
import config
from etl.crypto_pipeline import run_crypto_pipeline


def run_pipeline(mode: str, fetch_derivatives: bool = False):
    config.init_directories()

    if mode == "crypto":
        result = run_crypto_pipeline(fetch_derivatives=fetch_derivatives)
        print(result)
        return result

    raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["crypto"],
        default="crypto",
    )
    parser.add_argument(
        "--fetch-derivatives",
        action="store_true",
        help="Fetch and process derivatives data, e.g. funding rate.",
    )
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        fetch_derivatives=args.fetch_derivatives,
    )
