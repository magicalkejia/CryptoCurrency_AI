"""
etl.dataset_builder  (formerly etl.feature_builder)
====================================================
Assemble the SUPERVISED LEARNING dataset for the incremental-ladder / ablation
experiments. After the migration to the partner's production PIT feature
pipeline this module is the *consumer* side:

  features  ──>  built by etl.feature_builder (partner) into
                 data_storage/factors/crypto_features.parquet
                 (market + derivatives + REAL on-chain, PIT asof-merged)

  this file ──>  reads that table through etl.model_feature_loader, selects an
                 explicit, registry-approved feature set
                 (etl.feature_registry.get_model_feature_columns), and adds the
                 supervised layer that lives on *our* side:
                   * triple-barrier labels        (needs raw 1h bars)
                   * PatchTST OOF temporal feats   (needs raw 1h bars)
                   * average-uniqueness weights
                   * make_supervised_dataset (PIT-guarded merge)

The market/derivatives/on-chain *features* are no longer rebuilt here — they are
queried by name from the registry so the model input is explicit and auditable
(the partner's expert recommendation: never feed "all numeric columns", and never
feed raw price/volume levels or audit columns).

Modalities exposed to the ladder:
    market    -> MARKET_CORE_FEATURES (+ FUNDING_EXTENSION if in the set)   = Step 1
    onchain   -> ONCHAIN_EXTENSION_FEATURES (REAL DefiLlama data)           = Step 2
    narrative -> []  (sentiment/Twitter not wired yet — deferred)           = Step 3
    patchtst  -> patchtst_* OOF features                                    = Step 4

`--synthetic` keeps a fully offline wiring self-test (no pyarrow / no parquet):
a deterministic synthetic feature frame with the registry columns is generated so
the merge + ladder mechanics can be validated without the real data. It is gated
behind `synthetic=True` and is NEVER used on a real run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import config
from crypto.adapters import to_bars_schema, decision_time_grid
from crypto.labels.triple_barrier import compute_triple_barrier
from crypto.features.uniqueness import average_uniqueness
from crypto.models.patchtst import run_patchtst
from crypto.pit import make_supervised_dataset, audit_lookahead

from etl.feature_registry import (
    get_model_feature_columns,
    MARKET_CORE_FEATURES,
    FUNDING_EXTENSION_FEATURES,
    ONCHAIN_EXTENSION_FEATURES,
)
from etl.model_feature_loader import load_crypto_feature_table

# Default model feature set for the experiments: the expert-recommended compact
# set (24 market core + 3 funding + 6 on-chain = 33 columns), which is exactly
# registry feature_set "market_plus_funding_onchain_v1" and sits inside the
# recommended 20–35 range for ~10k 4h samples.
DEFAULT_FEATURE_SET = "market_plus_funding_onchain_v1"

# Partner table audit column that carries the per-row max feature availability
# time; our PIT guard (crypto.pit) looks for "max_feature_availability_ts".
_PARTNER_AVAIL_COL = "max_feature_available_time"
_PIT_AVAIL_COL = "max_feature_availability_ts"

_BAR_TD = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


# --------------------------------------------------------------------------- #
# raw bar / funding loaders  (still needed: labels, PatchTST and the close panel
# require true OHLC prices, which the feature table deliberately does NOT carry)
# --------------------------------------------------------------------------- #
def _read_parquet_bars(processed_dir, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    sym = symbol.replace("/", "")
    p = Path(processed_dir) / f"{sym}_{timeframe}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"])).sort_index()
        df = df.loc[:, [c for c in df.columns if c != "timestamp"]]
    return df


def _load_funding_series(processed_dir, symbol: str) -> Optional[pd.Series]:
    """Funding-rate Series indexed by settlement time, PIT-clean. Used only as a
    holding COST inside the triple-barrier net return (not as a model feature —
    funding *features* now come from the registry feature set). Supports the
    consolidated `processed/derivatives/funding.parquet` (with a `symbol` column)
    and the legacy per-symbol `{SYMBOL}_funding.parquet`. None if absent."""
    sym = symbol.replace("/", "")
    p_new = Path(processed_dir) / "derivatives" / "funding.parquet"
    if p_new.exists():
        try:
            df = pd.read_parquet(p_new)
            if "symbol" in df.columns:
                df = df[df["symbol"].astype(str).str.replace("/", "") == sym]
            if not df.empty and {"timestamp", "funding_rate"}.issubset(df.columns):
                return df.set_index(pd.to_datetime(df["timestamp"]))["funding_rate"].sort_index()
        except Exception:
            pass
    p_old = Path(processed_dir) / f"{sym}_funding.parquet"
    if p_old.exists():
        df = pd.read_parquet(p_old)
        return df.set_index("timestamp")["funding_rate"].sort_index()
    return None


def load_symbol_bars(symbol: str, timeframe: str, processed_dir, loader=None) -> Optional[pd.DataFrame]:
    """v6-schema bar frame (ts_close + availability_ts attached). Tries the
    project's DataLoader (duckdb) first, then a direct pandas read."""
    raw = None
    if loader is not None:
        try:
            raw = loader.get_crypto_kline_data(symbol=symbol, timeframe=timeframe)
        except Exception:
            raw = None
    if raw is None or getattr(raw, "empty", True):
        raw = _read_parquet_bars(processed_dir, symbol, timeframe)
    if raw is None or raw.empty:
        return None
    return to_bars_schema(raw, timeframe)


# --------------------------------------------------------------------------- #
# feature-table consumption (real path) + offline synthetic frame (self-test)
# --------------------------------------------------------------------------- #
def _select_feature_frame(symbols: List[str], feature_set: str,
                          feature_path=None) -> tuple[pd.DataFrame, List[str]]:
    """Read the partner PIT feature table and keep only id + registry-approved
    feature columns + the PIT availability column (aliased for crypto.pit)."""
    table = load_crypto_feature_table(feature_path=feature_path, symbols=symbols)
    cols = get_model_feature_columns(feature_set)
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise KeyError(
            f"crypto_features.parquet is missing {len(missing)} columns required by "
            f"feature_set={feature_set!r}: {missing}. Rebuild features with "
            f"etl/feature_builder.py (partner pipeline) first.")

    keep = ["symbol", "decision_time"] + cols
    frame = table[keep].copy()
    frame["decision_time"] = pd.to_datetime(frame["decision_time"])
    if _PARTNER_AVAIL_COL in table.columns:
        frame[_PIT_AVAIL_COL] = pd.to_datetime(table[_PARTNER_AVAIL_COL])
    return frame, cols


def _synthetic_feature_frame(bars_4h: pd.DataFrame, dts: pd.DatetimeIndex,
                             symbol: str, feature_set: str, seed: int) -> pd.DataFrame:
    """OFFLINE self-test only: deterministic finite values for every registry
    column so the merge + ladder run without the real parquet. Not used on real
    runs (guarded by synthetic=True)."""
    cols = get_model_feature_columns(feature_set)
    n = len(dts)
    rng = np.random.default_rng(1000 + seed)
    close = pd.Series(bars_4h["close"].values[:n], dtype=float)
    ret = close.pct_change().fillna(0.0).to_numpy()
    frame = pd.DataFrame({"symbol": symbol, "decision_time": pd.DatetimeIndex(dts)})
    for i, c in enumerate(cols):
        # market columns track returns; on-chain/funding are slow neutral series
        if c.startswith("onchain_"):
            frame[c] = np.cumsum(rng.standard_normal(n)) * 1e-3
        elif c.startswith("funding_"):
            frame[c] = rng.standard_normal(n) * 1e-3
        else:
            frame[c] = ret * (1.0 + 0.05 * i) + rng.standard_normal(n) * 1e-3
    frame[_PIT_AVAIL_COL] = pd.DatetimeIndex(dts)   # available exactly at decision time
    return frame


# --------------------------------------------------------------------------- #
# result container + top-level builder
# --------------------------------------------------------------------------- #
@dataclass
class MarketDataset:
    dataset: pd.DataFrame
    feature_cols: List[str]
    modality_cols: Dict[str, List[str]]
    tabular_cols: List[str]        # market/derivatives only (the "A" config in A/B/C/D)
    close_panel: pd.DataFrame      # 4h close, index = decision_time
    audit: dict
    per_symbol: Dict[str, dict] = field(default_factory=dict)


def build_market_dataset(symbols: List[str], fcfg, processed_dir=None, loader=None,
                         patchtst_lookback: int = 96, patchtst_emb_dim: int = 8,
                         bars_provider: Optional[Callable[[str, str], Optional[pd.DataFrame]]] = None,
                         feature_set: str = DEFAULT_FEATURE_SET,
                         feature_path=None, synthetic: bool = False,
                         xs_features: bool = False
                         ) -> MarketDataset:
    """
    Build the full multi-symbol supervised dataset.

    Features come from the partner PIT table crypto_features.parquet (real
    market + derivatives + on-chain), selected by `feature_set` from the
    registry. Labels / PatchTST / weights are produced here from the raw bars.

    bars_provider(symbol, timeframe) -> v6-schema bars OR None (default reads the
    processed parquet). `synthetic=True` builds an offline feature frame instead
    of reading crypto_features.parquet (wiring self-test only).
    """
    processed_dir = processed_dir or config.PathConfig.PROCESSED
    if bars_provider is None:
        def bars_provider(sym, tf):
            return load_symbol_bars(sym, tf, processed_dir, loader=loader)

    all_labels, all_patch, synth_frames, close_map, per_symbol = [], [], [], {}, {}
    for i, sym in enumerate(symbols):
        bars_1h = bars_provider(sym, "1h")
        bars_4h = bars_provider(sym, "4h")
        if bars_1h is None or bars_4h is None:
            per_symbol[sym] = {"status": "missing 1h or 4h parquet — skipped"}
            continue

        dts = decision_time_grid(bars_4h, fcfg.decision_offset_minutes)
        funding = _load_funding_series(processed_dir, sym) if (processed_dir and not synthetic) else None

        labels = compute_triple_barrier(
            bars_1h, dts, sym, fcfg.label, fcfg.cost, funding=funding,
            label_config_hash=fcfg.label_config_hash(), cost_model_hash=fcfg.cost_model_hash())

        pt = run_patchtst(bars_1h, dts, sym, fcfg, lookback=patchtst_lookback, emb_dim=patchtst_emb_dim)

        all_labels.append(labels)
        if not pt.empty:
            all_patch.append(pt)
        if synthetic:
            synth_frames.append(_synthetic_feature_frame(bars_4h, dts, sym, feature_set, seed=i + 1))
        close_map[sym] = pd.Series(bars_4h["close"].values, index=dts)   # index=decision_time
        per_symbol[sym] = {"status": "ok", "bars_1h": len(bars_1h), "bars_4h": len(bars_4h),
                           "decisions": len(dts), "labels": len(labels),
                           "funding": "present" if funding is not None else "absent (hook)"}

    if not all_labels:
        raise RuntimeError("no symbol produced data — check parquet paths / filenames")

    # ---- features: registry-approved columns from the partner PIT table ----
    if synthetic:
        feats = pd.concat(synth_frames, ignore_index=True)
        feat_cols = get_model_feature_columns(feature_set)
    else:
        feats, feat_cols = _select_feature_frame(symbols, feature_set, feature_path=feature_path)

    # ---- merge our PatchTST OOF temporal features onto the feature frame ----
    if all_patch:
        patch = pd.concat(all_patch, ignore_index=True)
        patch["decision_time"] = pd.to_datetime(patch["decision_time"])
        feats = feats.merge(patch, on=["symbol", "decision_time"], how="left")

    # ---- labels + average-uniqueness weights ----
    labels = pd.concat(all_labels, ignore_index=True).dropna(subset=["entry_time", "exit_time"])
    labels["decision_time"] = pd.to_datetime(labels["decision_time"])
    labels["uniqueness_weight"] = average_uniqueness(
        labels["entry_time"], labels["exit_time"], scope="pooled", symbol=labels["symbol"]).values

    # ---- PIT audit + guarded supervised merge ----
    audit = audit_lookahead(feats)
    ds = make_supervised_dataset(feats, labels, require_pit=True)

    # ---- resolve modality / feature columns from the registry ----
    market_core = [c for c in MARKET_CORE_FEATURES if c in ds.columns]
    funding_cols = [c for c in FUNDING_EXTENSION_FEATURES if c in ds.columns]
    onchain_cols = [c for c in ONCHAIN_EXTENSION_FEATURES if c in ds.columns]
    patch_cols = [c for c in ds.columns if c.startswith("patchtst_")]
    market_cols = market_core + funding_cols                 # the "A" tabular config / Step 1
    feature_cols = market_cols + onchain_cols + patch_cols

    # On-chain is daily/slow: rows before on-chain coverage get a NEUTRAL 0
    # (z-scores / pct-changes are centered at 0) so the market sample is NOT
    # shrunk by on-chain history; market + PatchTST columns must be complete.
    if onchain_cols:
        ds[onchain_cols] = ds[onchain_cols].fillna(0.0)
    drop_subset = [c for c in (market_cols + patch_cols) if c in ds.columns]
    ds = ds.dropna(subset=drop_subset).reset_index(drop=True)

    modality_cols = {"market": market_cols, "onchain": onchain_cols,
                     "narrative": [], "patchtst": patch_cols}

    # ---- OPTIONAL: cross-sectional relative features (own modality 'xsmom') ----
    # PIT-safe, zero new data: relative strength vs the cross-section / vs BTC.
    if xs_features:
        from etl.cross_sectional_features import add_cross_sectional_features
        ds, xs_cols = add_cross_sectional_features(ds)
        if xs_cols:
            modality_cols["xsmom"] = xs_cols
            feature_cols = feature_cols + xs_cols

    close_panel = pd.DataFrame({s: close_map[s] for s in close_map})

    # coverage diagnostics (so the on-chain impact is visible in the run header)
    if onchain_cols and len(ds):
        nonzero = (ds[onchain_cols].abs().sum(axis=1) > 0)
        per_symbol["_onchain"] = {
            "feature_set": feature_set,
            "onchain_cols": len(onchain_cols),
            "rows_with_onchain": int(nonzero.sum()),
            "onchain_start": (str(ds.loc[nonzero, "decision_time"].min())
                              if nonzero.any() else "none"),
        }

    return MarketDataset(dataset=ds, feature_cols=feature_cols, modality_cols=modality_cols,
                         tabular_cols=market_cols, close_panel=close_panel,
                         audit=audit, per_symbol=per_symbol)
