"""
etl/narrative_loader.py
=======================
Stage 3 of the narrative modality: attach the per-symbol narrative features to the
experiment dataset via a point-in-time (PIT) asof-backward merge, and register them as the
`narrative` modality so Step3 (+narrative) and Step5 (fusion) light up automatically.

PIT safety: narrative_features are on a 4h right-edge grid where the value at ts=T was built
only from news strictly before T. For a decision at T+1min we asof-merge backward, so the
decision uses the grid point <= decision_time, i.e. news strictly before the decision. No
future leakage. Symbols with no news (e.g. TRX) or pre-first-news rows get 0 (neutral).

This is a self-contained PREVIEW hook: it does not touch the colleague's feature_builder /
feature_registry / dataset_builder. When the production sentiment pipeline lands (narrative
columns inside crypto_features.parquet + the feature registry), this hook can be removed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from etl.build_narrative_features import FEATURE_COLS, _read_any


def _base_ticker(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC'; 'BTCUSDT' -> 'BTC'; 'BTC' -> 'BTC'."""
    s = str(symbol).upper().replace("/", "")
    return s[:-4] if s.endswith("USDT") else s


def load_narrative_features(path: str) -> pd.DataFrame:
    feats = _read_any(path)
    feats["ts"] = pd.to_datetime(feats["ts"], utc=True, errors="coerce")
    return feats.dropna(subset=["ts"]).sort_values(["symbol", "ts"]).reset_index(drop=True)


def attach_narrative(dataset: pd.DataFrame, modality_cols: dict, narr_feats: pd.DataFrame,
                     time_col: str = "decision_time", symbol_col: str = "symbol",
                     buffer_min: int = 0) -> tuple[pd.DataFrame, dict]:
    """Asof-merge narrative features onto `dataset` per symbol (PIT-safe) and set
    modality_cols['narrative'] = FEATURE_COLS. Mutates `dataset` in place (adds columns)
    and `modality_cols` in place. Returns them for convenience."""
    # Build asof keys WITHOUT mutating the dataset's own decision_time dtype (the CV/splits
    # code elsewhere expects it unchanged). Normalize both sides to tz-naive UTC instants.
    dt = pd.to_datetime(dataset[time_col])
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    dataset["_tkr"] = dataset[symbol_col].map(_base_ticker)
    # cast asof keys to a single common resolution (ns); merge_asof requires identical dtype
    dataset["_asof"] = (dt - pd.Timedelta(minutes=int(buffer_min))).astype("datetime64[ns]")
    dataset["_ridx"] = np.arange(len(dataset))               # explicit row id (merge_asof resets index)

    narr = narr_feats.copy()
    narr["symbol"] = narr["symbol"].map(_base_ticker)
    nts = pd.to_datetime(narr["ts"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    narr["ts"] = nts.astype("datetime64[ns]")

    pieces = []
    for tkr, left in dataset.groupby("_tkr", sort=False):
        left = left.sort_values("_asof")
        right = narr[narr["symbol"] == tkr][["ts"] + FEATURE_COLS].sort_values("ts")
        if right.empty:
            merged = left[["_ridx"]].copy()
            for c in FEATURE_COLS:
                merged[c] = 0.0                              # no news for this symbol -> neutral
        else:
            merged = pd.merge_asof(left[["_ridx", "_asof"]], right, left_on="_asof",
                                   right_on="ts", direction="backward", allow_exact_matches=True)
            merged[FEATURE_COLS] = merged[FEATURE_COLS].fillna(0.0)  # pre-first-news -> neutral
        pieces.append(merged[["_ridx"] + FEATURE_COLS])

    out = pd.concat(pieces).set_index("_ridx").sort_index()  # restore original row order via row id
    for c in FEATURE_COLS:                                   # write columns back onto dataset
        dataset[c] = out[c].to_numpy()
    dataset.drop(columns=["_tkr", "_asof", "_ridx"], inplace=True, errors="ignore")

    modality_cols["narrative"] = list(FEATURE_COLS)
    return dataset, modality_cols
