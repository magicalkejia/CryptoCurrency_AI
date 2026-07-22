# main.py
import argparse
import config
from etl.crypto_pipeline import run_crypto_pipeline


def run_pipeline(
    mode: str,
    fetch_derivatives: bool = False,
    fetch_oi: bool = False,
    skip_market: bool = False,
    fetch_spot: bool = False,
    process_spot: bool = False,
    build_basis: bool = False,
    build_cvd: bool = False,
    flow_timeframe: str = "4h",
    fetch_onchain: bool = False,
    process_onchain: bool = False,
    build_onchain_factors: bool = False,
    fetch_sentiment: bool = False,
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
            fetch_spot=fetch_spot,
            process_spot=process_spot,
            build_basis=build_basis,
            build_cvd=build_cvd,
            flow_timeframe=flow_timeframe,
            fetch_onchain=fetch_onchain,
            process_onchain=process_onchain,
            build_onchain_factors=build_onchain_factors,
            fetch_sentiment=fetch_sentiment,
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
    parser.add_argument("--fetch-spot", action="store_true", help="Fetch Binance spot K-line raw data.")
    parser.add_argument("--process-spot", action="store_true", help="Build processed spot long table.")
    parser.add_argument("--build-basis", action="store_true", help="Build processed spot/perp basis table.")
    parser.add_argument("--build-cvd", action="store_true", help="Build processed taker-CVD table.")
    parser.add_argument("--flow-timeframe", default="4h", help="Timeframe for spot/basis/CVD processed tables.")
    parser.add_argument("--fetch-onchain", action="store_true", help="Fetch DefiLlama on-chain raw data.")
    parser.add_argument("--process-onchain", action="store_true", help="Build processed on-chain daily tables.")
    parser.add_argument("--build-onchain-factors", action="store_true", help="Build factors/onchain_features.parquet.")
    parser.add_argument("--fetch-sentiment", action="store_true", help="Fetch raw news/sentiment articles.")
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
        fetch_spot=args.fetch_spot,
        process_spot=args.process_spot,
        build_basis=args.build_basis,
        build_cvd=args.build_cvd,
        flow_timeframe=args.flow_timeframe,
        fetch_onchain=args.fetch_onchain,
        process_onchain=args.process_onchain,
        build_onchain_factors=args.build_onchain_factors,
        fetch_sentiment=args.fetch_sentiment,
        build_features=args.build_features,
        feature_builder=args.feature_builder,
    )
