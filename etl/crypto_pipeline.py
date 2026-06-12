"""
etl.crypto_pipeline
===================

Top-level crypto data orchestration.

This module should not contain source-specific business logic. It coordinates:
    raw fetchers      -> data_updater.py / onchain_updater.py
    processors        -> data_processor.py / onchain_processor.py
    factor builders   -> onchain_feature_builder.py / feature_builder_H.py

The collaborator's current market-only path remains the default. Optional
on-chain and multimodal feature steps are opt-in so the draft can still run fast.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Sequence

import pandas as pd

import config
import etl.data_processor as data_processor
import etl.data_updater as data_updater


FeatureBuilderName = Literal["none", "multimodal"]


@dataclass
class CryptoPipelineConfig:
    symbols: Sequence[str] = field(default_factory=lambda: list(config.TargetConfig.COINS))
    #K线数据
    fetch_market: bool = True
    process_market: bool = True

    #衍生数据
    fetch_derivatives: bool = False
    fetch_oi: bool = False
    process_derivatives: bool = True
    
    #链上数据
    fetch_onchain: bool = False
    process_onchain: bool = False
    build_onchain_factors: bool = False
    onchain_chains: Sequence[str] = field(default_factory=lambda: list(config.OnchainConfig.DEFILLAMA_CHAINS))
    include_onchain_snapshots: bool = False

    build_model_features: bool = False
    feature_builder: FeatureBuilderName = "none"
    feature_output_path: str | None = None


def run_crypto_pipeline(
    fetch_derivatives: bool = False,
    fetch_oi: bool = False,
    *,
    cfg: CryptoPipelineConfig | None = None,
    symbols: Sequence[str] | None = None,
    fetch_market: bool | None = None,
    process_market: bool | None = None,
    fetch_onchain: bool = False,
    process_onchain: bool = False,
    build_onchain_factors: bool = False,
    build_model_features: bool = False,
    feature_builder: FeatureBuilderName = "none",
    onchain_chains: Sequence[str] | None = None,
    include_onchain_snapshots: bool = False,
) -> dict:
    """
    Run the crypto data pipeline.

    Backward-compatible default:
        run_crypto_pipeline(fetch_derivatives=False)

    Enhanced path:
        run_crypto_pipeline(
            fetch_onchain=True,
            process_onchain=True,
            build_onchain_factors=True,
            build_model_features=True,
            feature_builder="multimodal",
        )
    """
    if cfg is None:
        cfg = CryptoPipelineConfig(
            symbols=list(symbols or config.TargetConfig.COINS),
            fetch_market=True if fetch_market is None else fetch_market,
            process_market=True if process_market is None else process_market,
            fetch_derivatives=fetch_derivatives,
            fetch_oi=fetch_oi,
            fetch_onchain=fetch_onchain,
            process_onchain=process_onchain,
            build_onchain_factors=build_onchain_factors,
            onchain_chains=list(onchain_chains or config.OnchainConfig.DEFILLAMA_CHAINS),
            include_onchain_snapshots=include_onchain_snapshots,
            build_model_features=build_model_features,
            feature_builder=feature_builder,
        )

    result: dict = {
        "config": asdict(cfg),
        "market": None,
        "derivatives": {},
        "onchain": {},
        "features": {},
    }

    if cfg.fetch_market or cfg.process_market or cfg.fetch_derivatives:
        result["market"] = _run_market_and_derivatives(cfg)

    if cfg.fetch_onchain:
        result["onchain"]["raw"] = _fetch_onchain(cfg)

    if cfg.process_onchain:
        result["onchain"]["processed"] = _process_onchain(cfg)

    if cfg.build_onchain_factors:
        result["onchain"]["factors"] = _build_onchain_factors()

    if cfg.build_model_features:
        result["features"] = _build_model_features(cfg)

    return result


def _run_market_and_derivatives(cfg: CryptoPipelineConfig) -> pd.DataFrame:
    rows = []

    for symbol in cfg.symbols:
        ok_fetch = None
        ok_clean = None
        ok_resample = None

        if cfg.fetch_market:
            ok_fetch = data_updater.fetch_data(symbol)

        df_clean = None
        if cfg.process_market:
            df_clean = data_processor.clean_single_file(symbol)
            ok_clean = df_clean is not None
            ok_resample = False
            if ok_clean:
                data_processor.resample_data(df_clean, symbol)
                ok_resample = True

        ok_funding = None
        ok_oi = None
        if cfg.fetch_derivatives:
            ok_funding = data_updater.fetch_funding_rate(symbol)
            if cfg.fetch_oi:
                ok_oi = data_updater.fetch_open_interest(symbol, period="1h")

        rows.append({
            "symbol": symbol,
            "fetch_market": ok_fetch,
            "clean": ok_clean,
            "resample": ok_resample,
            "funding_raw": ok_funding,
            "oi_raw": ok_oi,
        })

    out = pd.DataFrame(rows)

    if cfg.fetch_derivatives and cfg.process_derivatives:
        funding_long = data_processor.process_funding_rates(list(cfg.symbols))
        out.attrs["processed_funding_rows"] = None if funding_long is None else len(funding_long)
        print(f"processed funding long table rows: {out.attrs['processed_funding_rows']}")

    print(out)
    return out


def _fetch_onchain(cfg: CryptoPipelineConfig) -> dict:
    from etl.onchain_updater import DefiLlamaConfig, fetch_defillama_all

    ocfg = DefiLlamaConfig(
        chains=tuple(cfg.onchain_chains),
        include_snapshots=cfg.include_onchain_snapshots,
    )
    frames = fetch_defillama_all(
        chains=list(cfg.onchain_chains),
        cfg=ocfg,
        include_snapshots=cfg.include_onchain_snapshots,
    )
    return {name: len(df) for name, df in frames.items()}


def _process_onchain(cfg: CryptoPipelineConfig) -> dict:
    from etl.onchain_processor import build_onchain_daily, process_defillama_daily

    defillama = process_defillama_daily(chains=list(cfg.onchain_chains), save=True)
    canonical = build_onchain_daily(save=True)
    return {
        "defillama_daily_rows": len(defillama),
        "defillama_daily_cols": len(defillama.columns),
        "onchain_daily_rows": len(canonical),
        "onchain_daily_cols": len(canonical.columns),
    }


def _build_onchain_factors() -> dict:
    from etl.onchain_feature_builder import build_onchain_factors

    factors = build_onchain_factors(save=True)
    return {
        "onchain_factor_rows": len(factors),
        "onchain_factor_cols": len(factors.columns),
    }


def _build_model_features(cfg: CryptoPipelineConfig) -> dict:
    if cfg.feature_builder == "none":
        return {"status": "skipped", "reason": "feature_builder='none'"}

    if cfg.feature_builder != "multimodal":
        raise ValueError(f"Unsupported feature_builder: {cfg.feature_builder}")

    from etl.feature_builder_H import FeatureBuilderConfig, build_crypto_features

    fcfg = FeatureBuilderConfig()
    features = build_crypto_features(
        symbols=list(cfg.symbols),
        cfg=fcfg,
        save=True,
        output_path=cfg.feature_output_path,
    )
    return {
        "builder": "feature_builder_H",
        "rows": len(features),
        "cols": len(features.columns),
        "output_path": cfg.feature_output_path or str(config.PathConfig.FACTORS / fcfg.output_name),
    }
