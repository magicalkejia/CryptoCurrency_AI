"""
etl.onchain_processor
=====================

Normalize raw on-chain / DeFi source data into processed daily data tables.

Current sources:
    - DefiLlama raw tables saved by etl.defillama_loader

Outputs:
    data_storage/processed/onchain/defillama_daily.parquet
    data_storage/processed/onchain/onchain_daily.parquet

This module only cleans and merges source data. Derived features such as
percentage changes and rolling z-scores are built by onchain_feature_builder.py
and stored under data_storage/factors/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

import config


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


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
# Basic utilities
# ---------------------------------------------------------------------------


def _read_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def _ensure_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        raise ValueError(f"missing timestamp column: {col}")
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.dropna(subset=[col])


def _daily_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce").dt.floor("D")


def _merge_daily_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="timestamp", how="outer")
    return out.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# DefiLlama processing
# ---------------------------------------------------------------------------


def _process_chain_tvl_files(chains: Optional[Sequence[str]] = None) -> pd.DataFrame:
    root = defillama_raw_dir()
    paths = sorted(root.glob("chain_tvl_*.parquet"))
    if chains is not None:
        wanted = {_slug(c) for c in chains}
        paths = [p for p in paths if p.stem.replace("chain_tvl_", "") in wanted]

    frames = []
    for path in paths:
        df = _read_if_exists(path)
        if df is None or df.empty:
            continue
        df = _ensure_timestamp(df)
        chain = str(df["chain"].dropna().iloc[0]) if "chain" in df.columns and df["chain"].notna().any() else path.stem.replace("chain_tvl_", "")
        chain_slug = _slug(chain)
        value_col = f"onchain_defillama_tvl_{chain_slug}_usd"
        g = (
            df.assign(timestamp=_daily_timestamp(df))
            .groupby("timestamp", as_index=False)["tvl_usd"]
            .last()
            .rename(columns={"tvl_usd": value_col})
        )
        frames.append(g)

    if not frames:
        return pd.DataFrame()

    out = _merge_daily_frames(frames)
    tvl_cols = [c for c in out.columns if c.startswith("onchain_defillama_tvl_") and c.endswith("_usd")]
    if tvl_cols:
        out["onchain_defillama_selected_chains_tvl_usd"] = out[tvl_cols].sum(axis=1, min_count=1)
    return out


def _process_single_metric_file(
    filename: str,
    source_col: str,
    out_col: str,
) -> pd.DataFrame:
    path = defillama_raw_dir() / filename
    df = _read_if_exists(path)
    if df is None or df.empty or source_col not in df.columns:
        return pd.DataFrame()
    df = _ensure_timestamp(df)
    out = (
        df.assign(timestamp=_daily_timestamp(df))
        .groupby("timestamp", as_index=False)[source_col]
        .last()
        .rename(columns={source_col: out_col})
    )
    return out


def process_defillama_daily(
    chains: Optional[Sequence[str]] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Build processed DefiLlama daily data table.

    Output schema:
        timestamp
        onchain_defillama_tvl_{chain}_usd
        onchain_defillama_selected_chains_tvl_usd
        onchain_defillama_stablecoin_mcap_usd
        onchain_defillama_dex_volume_usd
        onchain_defillama_fees_usd
        onchain_defillama_revenue_usd
    """
    frames = []

    chain_tvl = _process_chain_tvl_files(chains=chains)
    if not chain_tvl.empty:
        frames.append(chain_tvl)

    metric_specs = [
        ("stablecoins_all.parquet", "stablecoin_mcap_usd", "onchain_defillama_stablecoin_mcap_usd"),
        ("dex_volume_global.parquet", "dex_volume_usd", "onchain_defillama_dex_volume_usd"),
        ("fees_revenue_global.parquet", "fees_usd", "onchain_defillama_fees_usd"),
        ("fees_revenue_global.parquet", "revenue_usd", "onchain_defillama_revenue_usd"),
    ]
    for filename, source_col, out_col in metric_specs:
        f = _process_single_metric_file(filename, source_col, out_col)
        if not f.empty:
            frames.append(f)

    out = _merge_daily_frames(frames)
    if out.empty:
        print("[WARN] No DefiLlama raw data found to process.")
        return out

    out = out.sort_values("timestamp").reset_index(drop=True)

    if save:
        path = processed_onchain_root() / "defillama_daily.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        print(f"saved processed DefiLlama daily data: {path} ({len(out):,} rows, {len(out.columns)} columns)")

    return out


def build_onchain_daily(
    include_defillama: bool = True,
    include_dune: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """
    Combine processed on-chain source tables into the canonical data table.

    The output is global daily data. It intentionally has no symbol column.
    onchain_feature_builder.py turns this base data into factor tables.
    """
    frames: list[pd.DataFrame] = []

    if include_defillama:
        p = processed_onchain_root() / "defillama_daily.parquet"
        df = _read_if_exists(p)
        if df is not None and not df.empty:
            frames.append(_ensure_timestamp(df))

    if include_dune:
        # Optional future table. If users later process Dune into dune_daily.parquet,
        # this merge will pick it up automatically.
        p = processed_onchain_root() / "dune_daily.parquet"
        df = _read_if_exists(p)
        if df is not None and not df.empty:
            frames.append(_ensure_timestamp(df))

    out = _merge_daily_frames(frames)
    if out.empty:
        print("[WARN] No processed on-chain source tables found.")
        return out

    out = out.sort_values("timestamp").reset_index(drop=True)

    if save:
        path = processed_onchain_root() / "onchain_daily.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        print(f"saved canonical onchain daily: {path} ({len(out):,} rows, {len(out.columns)} columns)")

    return out


# ---------------------------------------------------------------------------
# Read / list / archive helpers
# ---------------------------------------------------------------------------


def load_onchain_daily(
    start_date=None,
    end_date=None,
    columns: Optional[Sequence[str]] = None,
    table: str = "onchain_daily.parquet",
) -> pd.DataFrame:
    """Read processed on-chain daily table with optional filters."""
    path = processed_onchain_root() / table
    if not path.exists():
        raise FileNotFoundError(f"Processed on-chain table not found: {path}")

    base_cols = ["timestamp"]
    if columns is not None:
        read_cols = list(dict.fromkeys(base_cols + list(columns)))
        df = pd.read_parquet(path, columns=read_cols)
    else:
        df = pd.read_parquet(path)

    df = _ensure_timestamp(df)

    if start_date is not None:
        df = df[df["timestamp"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["timestamp"] <= pd.to_datetime(end_date)]

    return df.sort_values("timestamp").reset_index(drop=True)


def list_onchain_tables(layer: str = "processed") -> pd.DataFrame:
    """List on-chain parquet tables in raw or processed layer."""
    if layer not in {"raw", "processed"}:
        raise ValueError("layer must be 'raw' or 'processed'")
    root = raw_onchain_root() if layer == "raw" else processed_onchain_root()
    rows = []
    for p in sorted(root.rglob("*.parquet")):
        rows.append(
            {
                "layer": layer,
                "name": p.stem,
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "modified_at": pd.Timestamp.fromtimestamp(p.stat().st_mtime),
            }
        )
    return pd.DataFrame(rows)


def archive_onchain_table(
    table: str,
    layer: str = "processed",
    source: Optional[str] = None,
) -> Path:
    """Archive an on-chain table instead of deleting it."""
    if layer not in {"raw", "processed"}:
        raise ValueError("layer must be 'raw' or 'processed'")

    if layer == "raw":
        root = raw_onchain_root() / source if source else raw_onchain_root()
    else:
        root = processed_onchain_root()

    src = root / table
    if src.suffix != ".parquet":
        src = src.with_suffix(".parquet")
    if not src.exists():
        raise FileNotFoundError(f"on-chain table not found: {src}")

    archive_root = data_root() / "_archive" / layer / "onchain" / (source or "")
    archive_root.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    dst = archive_root / f"{src.stem}_{ts}{src.suffix}"
    src.rename(dst)
    return dst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Process raw on-chain data into PIT-ready daily tables.")
    parser.add_argument("--chains", nargs="+", default=None, help="Restrict DefiLlama chain TVL processing")
    parser.add_argument("--skip-onchain-merge", action="store_true", help="Only build defillama_daily.parquet")
    parser.add_argument("--list", choices=["raw", "processed"], default=None, help="List tables and exit")
    args = parser.parse_args()

    if args.list:
        print(list_onchain_tables(args.list).to_string(index=False))
        return

    process_defillama_daily(chains=args.chains, save=True)
    if not args.skip_onchain_merge:
        build_onchain_daily(save=True)


if __name__ == "__main__":
    main()
