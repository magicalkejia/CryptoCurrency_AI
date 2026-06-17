# main.py
import argparse
import config
from etl.crypto_pipeline import run_crypto_pipeline


def run_pipeline(
    mode: str,
    fetch_derivatives: bool = False,
    fetch_oi: bool = False,
    skip_market: bool = False,
    fetch_onchain: bool = False,
    process_onchain: bool = False,
    build_onchain_factors: bool = False,
    fetch_sentiment: bool = False,
    sentiment_start: str | None = None,
    sentiment_end: str | None = None,
    build_features: bool = False,
    feature_builder: str = "none",
):
    config.init_directories()

    if mode == "crypto":
        result = run_crypto_pipeline(
            fetch_derivatives=fetch_derivatives,
            fetch_oi=fetch_oi,
            fetch_market=not skip_market,
            process_market=not skip_market,
            fetch_onchain=fetch_onchain,
            process_onchain=process_onchain,
            build_onchain_factors=build_onchain_factors,
            fetch_sentiment=fetch_sentiment,
            sentiment_start=sentiment_start,
            sentiment_end=sentiment_end,
            build_model_features=build_features,
            feature_builder=feature_builder,
        )
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
    parser.add_argument("--fetch-oi", action="store_true", help="Also fetch open-interest raw data.")
    parser.add_argument("--skip-market", action="store_true", help="Skip market K-line fetch/process steps.")
    parser.add_argument("--fetch-onchain", action="store_true", help="Fetch DefiLlama on-chain raw data.")
    parser.add_argument("--process-onchain", action="store_true", help="Build processed on-chain daily tables.")
    parser.add_argument("--build-onchain-factors", action="store_true", help="Build factors/onchain_features.parquet.")
    parser.add_argument("--fetch-sentiment", action="store_true", help="Fetch GDELT raw news/sentiment articles.")
    parser.add_argument("--sentiment-start", default=None, help="Start datetime for sentiment fetch, e.g. 2024-01-01.")
    parser.add_argument("--sentiment-end", default=None, help="End datetime for sentiment fetch, e.g. 2024-01-08.")
    parser.add_argument("--build-features", action="store_true", help="Build final crypto feature table.")
    parser.add_argument(
        "--feature-builder",
        choices=["none", "multimodal"],
        default="none",
        help="Feature builder used with --build-features.",
    )
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        fetch_derivatives=args.fetch_derivatives,
        fetch_oi=args.fetch_oi,
        skip_market=args.skip_market,
        fetch_onchain=args.fetch_onchain,
        process_onchain=args.process_onchain,
        build_onchain_factors=args.build_onchain_factors,
        fetch_sentiment=args.fetch_sentiment,
        sentiment_start=args.sentiment_start,
        sentiment_end=args.sentiment_end,
        build_features=args.build_features,
        feature_builder=args.feature_builder,
    )
