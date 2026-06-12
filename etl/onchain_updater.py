"""
etl.defillama_loader
=====================

Fetch and persist free DefiLlama data for crypto on-chain / DeFi regime features.

Why this file is separate from data_updater.py:
    - data_updater.py is exchange market-data oriented: Binance K-lines, funding, OI.
    - DefiLlama is an aggregated DeFi/on-chain metrics source.
    - Keeping it here follows the same pattern as dune_loader.py: one external data
      family per loader.

Data flow:
    DefiLlama Free API
        -> data_storage/raw/onchain/defillama/*.parquet
        -> onchain_processor.py
        -> data_storage/processed/onchain/defillama_daily.parquet
        -> data_storage/processed/onchain/onchain_daily.parquet
        -> onchain_feature_builder.py
        -> data_storage/factors/onchain_features.parquet

No extra dependencies: uses urllib from Python stdlib.

Free API base URL:
    https://api.llama.fi

Core endpoints used here:
    /v2/historicalChainTvl/{chain}
    https://stablecoins.llama.fi/stablecoincharts/all
    /overview/dexs
    /overview/fees
    /v2/chains
    /protocols
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Configuration and path helpers
# ---------------------------------------------------------------------------


@dataclass
class DefiLlamaConfig:
    """Runtime config for DefiLlama free API ingestion."""

    base_url: str = "https://api.llama.fi"
    timeout: int = 60
    max_retries: int = 3
    retry_sleep: float = 1.5

    # Chain names should match DefiLlama endpoint names. These map to the
    # current trading universe as DeFi ecosystem proxies:
    # BTC/USDT -> Bitcoin, ETH/USDT -> Ethereum, SOL/USDT -> Solana,
    # BNB/USDT -> BSC. Bitcoin TVL is not native BTC chain activity; it is a
    # Bitcoin ecosystem DeFi TVL proxy and should be documented as such.
    chains: tuple[str, ...] = tuple(config.OnchainConfig.DEFILLAMA_CHAINS)

    # Snapshot tables are useful for exploration but are not directly PIT-ready.
    include_snapshots: bool = True

    # For raw API response audit. This is not a market feature and should not
    # enter feature_builder output.
    source: str = "defillama"



def _path_attr(name: str, fallback: Path) -> Path:
    return Path(getattr(config.PathConfig, name, fallback))


def data_root() -> Path:
    return Path(config.PathConfig.DATA_ROOT)


def raw_onchain_root() -> Path:
    return _path_attr("RAW_ONCHAIN", Path(config.PathConfig.RAW) / "onchain")


def processed_onchain_root() -> Path:
    return _path_attr("PROCESSED_ONCHAIN", Path(config.PathConfig.PROCESSED) / "onchain")


def defillama_raw_dir() -> Path:
    return raw_onchain_root() / "defillama"


def defillama_manifest_path() -> Path:
    return defillama_raw_dir() / "_manifest.json"


def _slug(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _build_url(base_url: str, path: str, params: Optional[dict[str, Any]] = None) -> str:
    if not path.startswith("/"):
        path = "/" + path
    url = base_url.rstrip("/") + path
    if params:
        clean_params = {k: v for k, v in params.items() if v is not None}
        if clean_params:
            url += "?" + urllib.parse.urlencode(clean_params)
    return url


def _http_get_json(
    path: str,
    params: Optional[dict[str, Any]] = None,
    cfg: Optional[DefiLlamaConfig] = None,
) -> Any:
    """GET JSON with light retry. Raises after final failure."""
    cfg = cfg or DefiLlamaConfig()
    url = _build_url(cfg.base_url, path, params)
    last_error: Exception | None = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "quant-system-defillama-loader/0.1",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - deliberate top-level retry wrapper
            last_error = exc
            if attempt >= cfg.max_retries:
                break
            time.sleep(cfg.retry_sleep * attempt)

    raise RuntimeError(f"DefiLlama GET failed: {url} | error={last_error}")


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _unix_to_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(ts):
        return pd.NaT
    # DefiLlama date fields are usually seconds. If ms appears, handle it.
    unit = "ms" if ts > 10_000_000_000 else "s"
    return pd.to_datetime(ts, unit=unit, utc=True).tz_localize(None)


def _timestamp_from_row(row: dict[str, Any]) -> pd.Timestamp:
    for key in ("date", "timestamp", "time", "day"):
        if key in row:
            return _unix_to_timestamp(row[key])
    return pd.NaT


def _extract_numeric(value: Any) -> float:
    """Extract a numeric scalar from a nested DefiLlama value shape."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, dict):
        # Stablecoin endpoints often use {"peggedUSD": value, ...}.
        for preferred_key in ("peggedUSD", "USD", "usd", "value", "total"):
            if preferred_key in value:
                return _extract_numeric(value[preferred_key])
        nums = [_extract_numeric(v) for v in value.values()]
        nums = [v for v in nums if pd.notna(v)]
        return float(np.nansum(nums)) if nums else np.nan
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def _chart_to_frame(
    payload: Any,
    key: str,
    value_col: str,
    source: str,
) -> pd.DataFrame:
    """
    Convert DefiLlama chart arrays to DataFrame.

    Supported shapes:
      payload[key] = [[timestamp, value], ...]
      payload[key] = [{"date": timestamp, "value": x}, ...]
    """
    data = payload.get(key) if isinstance(payload, dict) else None
    if not data:
        return pd.DataFrame(columns=["timestamp", value_col, "source", "fetched_at"])

    rows: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            ts = _unix_to_timestamp(item[0])
            value = _extract_numeric(item[1])
        elif isinstance(item, dict):
            ts = _timestamp_from_row(item)
            if "value" in item:
                value = _extract_numeric(item.get("value"))
            elif "total" in item:
                value = _extract_numeric(item.get("total"))
            else:
                value = _extract_numeric({k: v for k, v in item.items() if k not in {"date", "timestamp", "time", "day"}})
        else:
            continue

        if pd.notna(ts):
            rows.append({"timestamp": ts, value_col: value})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["source"] = source
    out["fetched_at"] = pd.Timestamp.utcnow().tz_localize(None)
    return out.sort_values("timestamp").reset_index(drop=True)


def _save_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


def _write_manifest(records: list[dict[str, Any]]) -> Path:
    path = defillama_manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "defillama",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "tables": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fetchers: RAW layer
# ---------------------------------------------------------------------------


def fetch_chain_tvl(
    chain: str,
    cfg: Optional[DefiLlamaConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical DeFi TVL for one chain.

    Endpoint:
        /v2/historicalChainTvl/{chain}

    Output columns:
        timestamp, chain, tvl_usd, source, fetched_at
    """
    cfg = cfg or DefiLlamaConfig()
    payload = _http_get_json(f"/v2/historicalChainTvl/{urllib.parse.quote(chain)}", cfg=cfg)

    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "timestamp": _timestamp_from_row(item),
                    "chain": chain,
                    "tvl_usd": _extract_numeric(item.get("tvl")),
                    "source": cfg.source,
                    "fetched_at": pd.Timestamp.utcnow().tz_localize(None),
                }
            )
    else:
        raise ValueError(f"Unexpected historicalChainTvl payload for chain={chain}: {type(payload)}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["timestamp"]).drop_duplicates(["chain", "timestamp"], keep="last")
        df = df.sort_values(["chain", "timestamp"]).reset_index(drop=True)

    if save:
        path = defillama_raw_dir() / f"chain_tvl_{_slug(chain)}.parquet"
        _save_parquet(df, path)
        print(f"saved {len(df):,} rows -> {path}")

    return df


def fetch_chain_tvls(
    chains: Optional[Sequence[str]] = None,
    cfg: Optional[DefiLlamaConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Fetch historical chain TVL for multiple chains and save one file per chain."""
    cfg = cfg or DefiLlamaConfig()
    chains = list(chains or cfg.chains)
    frames = []
    for chain in chains:
        try:
            df = fetch_chain_tvl(chain=chain, cfg=cfg, save=save)
            if not df.empty:
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] DefiLlama chain TVL skipped: {chain} | {exc}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["chain", "timestamp"]).reset_index(drop=True)


def fetch_stablecoins_all(
    cfg: Optional[DefiLlamaConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical aggregate stablecoin market cap.

    Endpoint:
        /stablecoincharts/all

    Output columns:
        timestamp, stablecoin_mcap_usd, source, fetched_at
    """
    cfg = cfg or DefiLlamaConfig()
    stable_cfg = replace(cfg, base_url="https://stablecoins.llama.fi")
    payload = _http_get_json("/stablecoincharts/all", cfg=stable_cfg)

    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        iterable = payload
    elif isinstance(payload, dict):
        # Defensive fallback if the API wraps chart data in a key.
        iterable = payload.get("data") or payload.get("chart") or []
    else:
        iterable = []

    for item in iterable:
        if not isinstance(item, dict):
            continue
        value = np.nan
        for key in ("totalCirculatingUSD", "totalCirculating", "mcap", "marketCap", "total"):
            if key in item:
                value = _extract_numeric(item[key])
                break
        rows.append(
            {
                "timestamp": _timestamp_from_row(item),
                "stablecoin_mcap_usd": value,
                "source": cfg.source,
                "fetched_at": pd.Timestamp.utcnow().tz_localize(None),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["timestamp"]).drop_duplicates(["timestamp"], keep="last")
        df = df.sort_values("timestamp").reset_index(drop=True)

    if save:
        path = defillama_raw_dir() / "stablecoins_all.parquet"
        _save_parquet(df, path)
        print(f"saved {len(df):,} rows -> {path}")

    return df


def fetch_dex_overview(
    cfg: Optional[DefiLlamaConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch global DEX daily volume overview.

    Endpoint:
        /overview/dexs

    Output columns:
        timestamp, dex_volume_usd, source, fetched_at
    """
    cfg = cfg or DefiLlamaConfig()
    payload = _http_get_json(
        "/overview/dexs",
        params={"excludeTotalDataChartBreakdown": "true", "dataType": "dailyVolume"},
        cfg=cfg,
    )
    df = _chart_to_frame(payload, "totalDataChart", "dex_volume_usd", cfg.source)

    if save:
        path = defillama_raw_dir() / "dex_volume_global.parquet"
        _save_parquet(df, path)
        print(f"saved {len(df):,} rows -> {path}")

    return df


def fetch_fees_overview(
    cfg: Optional[DefiLlamaConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch global fees and revenue overview.

    Endpoint:
        /overview/fees

    Output columns vary by API availability, typically:
        timestamp, fees_usd, revenue_usd, source, fetched_at
    """
    cfg = cfg or DefiLlamaConfig()
    payload = _http_get_json(
        "/overview/fees",
        params={"excludeTotalDataChartBreakdown": "true", "dataType": "dailyFees"},
        cfg=cfg,
    )

    fees = _chart_to_frame(payload, "totalDataChart", "fees_usd", cfg.source)
    revenue = _chart_to_frame(payload, "totalRevenueChart", "revenue_usd", cfg.source)

    if fees.empty and revenue.empty:
        out = pd.DataFrame(columns=["timestamp", "fees_usd", "revenue_usd", "source", "fetched_at"])
    elif fees.empty:
        out = revenue
    elif revenue.empty:
        out = fees
    else:
        out = fees.merge(
            revenue[["timestamp", "revenue_usd"]],
            on="timestamp",
            how="outer",
        ).sort_values("timestamp")
        out["source"] = cfg.source
        out["fetched_at"] = pd.Timestamp.utcnow().tz_localize(None)

    if save:
        path = defillama_raw_dir() / "fees_revenue_global.parquet"
        _save_parquet(out, path)
        print(f"saved {len(out):,} rows -> {path}")

    return out


def fetch_chains_snapshot(
    cfg: Optional[DefiLlamaConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch current TVL of all chains.

    Endpoint:
        /v2/chains

    Snapshot is not directly PIT for historical backtests. Use it for metadata,
    exploration, and data-quality checks.
    """
    cfg = cfg or DefiLlamaConfig()
    payload = _http_get_json("/v2/chains", cfg=cfg)
    df = pd.DataFrame(payload if isinstance(payload, list) else [])
    if not df.empty:
        df["source"] = cfg.source
        df["fetched_at"] = pd.Timestamp.utcnow().tz_localize(None)

    if save:
        path = defillama_raw_dir() / "chains_snapshot.parquet"
        _save_parquet(df, path)
        print(f"saved {len(df):,} rows -> {path}")

    return df


def fetch_protocols_snapshot(
    cfg: Optional[DefiLlamaConfig] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch current protocols list and TVL summary.

    Endpoint:
        /protocols

    Snapshot is not directly PIT for historical backtests. Use it as metadata.
    """
    cfg = cfg or DefiLlamaConfig()
    payload = _http_get_json("/protocols", cfg=cfg)
    df = pd.DataFrame(payload if isinstance(payload, list) else [])
    if not df.empty:
        df["source"] = cfg.source
        df["fetched_at"] = pd.Timestamp.utcnow().tz_localize(None)

    if save:
        path = defillama_raw_dir() / "protocols_snapshot.parquet"
        _save_parquet(df, path)
        print(f"saved {len(df):,} rows -> {path}")

    return df


def fetch_defillama_all(
    chains: Optional[Sequence[str]] = None,
    cfg: Optional[DefiLlamaConfig] = None,
    include_snapshots: Optional[bool] = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch the recommended DefiLlama raw dataset for this project.

    Recommended first version:
        - selected chain historical TVL
        - global stablecoin market cap history
        - global DEX volume history
        - global fees / revenue history
        - optional current snapshots: chains and protocols
    """
    cfg = cfg or DefiLlamaConfig()
    include_snapshots = cfg.include_snapshots if include_snapshots is None else include_snapshots

    results: dict[str, pd.DataFrame] = {}
    manifest: list[dict[str, Any]] = []

    chain_tvls = fetch_chain_tvls(chains=chains or cfg.chains, cfg=cfg, save=True)
    results["chain_tvls"] = chain_tvls
    manifest.append({"name": "chain_tvls", "rows": len(chain_tvls), "path_pattern": "chain_tvl_{chain}.parquet"})

    for name, fn in [
        ("stablecoins_all", fetch_stablecoins_all),
        ("dex_volume_global", fetch_dex_overview),
        ("fees_revenue_global", fetch_fees_overview),
    ]:
        try:
            df = fn(cfg=cfg, save=True)
            results[name] = df
            manifest.append({"name": name, "rows": len(df), "path": f"{name}.parquet"})
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] DefiLlama fetch skipped: {name} | {exc}")

    if include_snapshots:
        for name, fn in [
            ("chains_snapshot", fetch_chains_snapshot),
            ("protocols_snapshot", fetch_protocols_snapshot),
        ]:
            try:
                df = fn(cfg=cfg, save=True)
                results[name] = df
                manifest.append({"name": name, "rows": len(df), "path": f"{name}.parquet"})
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] DefiLlama snapshot skipped: {name} | {exc}")

    mpath = _write_manifest(manifest)
    print(f"saved manifest -> {mpath}")
    return results


# ---------------------------------------------------------------------------
# Read / list / archive helpers for raw DefiLlama tables
# ---------------------------------------------------------------------------


def list_defillama_raw_tables() -> pd.DataFrame:
    """List raw DefiLlama parquet tables."""
    root = defillama_raw_dir()
    rows = []
    for p in sorted(root.glob("*.parquet")):
        rows.append(
            {
                "name": p.stem,
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "modified_at": pd.Timestamp.fromtimestamp(p.stat().st_mtime),
            }
        )
    return pd.DataFrame(rows)


def load_defillama_raw_table(name: str) -> pd.DataFrame:
    """Read one raw DefiLlama table by stem name, e.g. 'stablecoins_all'."""
    path = defillama_raw_dir() / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Raw DefiLlama table not found: {path}")
    return pd.read_parquet(path)


def archive_defillama_raw_table(name: str, archive_root: Optional[str | Path] = None) -> Path:
    """
    Archive one raw table instead of deleting it.

    Raw data should generally be immutable. If a file is bad, move it to archive
    and refetch rather than mutating rows in place.
    """
    src = defillama_raw_dir() / f"{name}.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Raw DefiLlama table not found: {src}")

    archive_dir = Path(archive_root) if archive_root is not None else data_root() / "_archive" / "raw" / "onchain" / "defillama"
    archive_dir.mkdir(parents=True, exist_ok=True)

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    dst = archive_dir / f"{src.stem}_{ts}{src.suffix}"
    src.rename(dst)
    return dst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_chains(raw: Optional[list[str]]) -> Optional[list[str]]:
    if not raw:
        return None
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch DefiLlama free API data to local raw parquet files.")
    parser.add_argument("--chains", nargs="+", default=None, help="DefiLlama chain names, e.g. Ethereum Solana BSC Base")
    parser.add_argument("--no-snapshots", action="store_true", help="Skip current chains/protocols snapshots")
    parser.add_argument("--list", action="store_true", help="List saved raw DefiLlama tables and exit")
    args = parser.parse_args()

    if args.list:
        print(list_defillama_raw_tables().to_string(index=False))
        return

    cfg = DefiLlamaConfig(include_snapshots=not args.no_snapshots)
    fetch_defillama_all(chains=_parse_chains(args.chains), cfg=cfg)


if __name__ == "__main__":
    main()
