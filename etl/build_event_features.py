"""
etl/build_event_features.py
===========================
Stage 2 of the LLM narrative modality (v6 §7.3/§5): turn the per-article
structured extractions (events_llm.parquet) into PIT-safe, PER-SYMBOL factors on
the same regular 4h grid the CryptoBERT narrative features use, written to
data_storage/factors/sentiment/event_features.parquet.

PIT discipline (identical to build_narrative_features.py):
  * articles are attributed to symbols via the LLM `entities` list,
  * binned to 4h RIGHT-edges: a bin labelled T contains articles published in
    [T-4h, T) (strictly before T); combined with the 1-min decision offset and the
    backward asof-merge in narrative_loader, this is leak-free,
  * all features are causal (EWMA / trailing windows), no full-sample stats.

Factors per (symbol, 4h):
  narr_event_alpha     EWMA(hl 3d) of mean[ sentiment*severity*credibility*consistency,
                       zeroed for rumors ] -> a directional, confidence-weighted view.
  narr_event_risk      EWMA(hl 2d) of the per-bin MAX severity among RISK events
                       (hack/delisting/regulation) -> a one-sided danger gauge.
  narr_rumor_risk      EWMA(hl 2d) of the per-bin fraction of rumor articles.
  narr_theme_<t>       EWMA(hl 5d) of per-bin presence of theme t (etf/regulation/defi/l2/ai).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from etl.llm_deepseek import RISK_EVENT_TYPES

BAR = "4h"
THEME_FEATURES = ["etf", "regulation", "defi", "l2", "ai"]
EVENT_FEATURE_COLS = (["narr_event_alpha", "narr_event_risk", "narr_rumor_risk"]
                      + [f"narr_theme_{t}" for t in THEME_FEATURES])


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(path) if p.suffix.lower() == ".csv" else pd.read_parquet(path)


def _write_any(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if Path(path).suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def _as_list(x):
    if isinstance(x, list):
        return x
    try:
        return json.loads(x) if isinstance(x, str) else []
    except Exception:
        return []


def build_event_features(events: pd.DataFrame,
                         alpha_hl_days: float = 3.0, risk_hl_days: float = 2.0,
                         theme_hl_days: float = 5.0, bar: str = BAR) -> pd.DataFrame:
    """events: DataFrame with [datetime_utc, entities, themes, event_type, sentiment,
    severity, is_rumor, source_credibility, consistency]. Returns long DataFrame
    [symbol, ts] + EVENT_FEATURE_COLS on a regular 4h grid per symbol."""
    df = events.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_utc"])
    if df.empty:
        return pd.DataFrame(columns=["symbol", "ts"] + EVENT_FEATURE_COLS)
    for c in ("sentiment", "severity", "source_credibility", "consistency"):
        df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0)
    df["is_rumor"] = df.get("is_rumor", False).astype(bool)
    df["entities"] = df["entities"].apply(_as_list)
    df["themes"] = df["themes"].apply(_as_list)

    # per-article contributions
    df["c_alpha"] = (df["sentiment"] * df["severity"] * df["source_credibility"]
                     * df["consistency"].clip(lower=0.0) * (~df["is_rumor"]).astype(float))
    df["c_risk"] = np.where(df["event_type"].isin(RISK_EVENT_TYPES), df["severity"], 0.0)
    df["c_rumor"] = df["is_rumor"].astype(float)
    for t in THEME_FEATURES:
        df[f"t_{t}"] = df["themes"].apply(lambda ts: 1.0 if t in ts else 0.0)

    # explode to one row per (article, ticker) — only universe tickers survive
    df = df.explode("entities").dropna(subset=["entities"])
    df = df[df["entities"].astype(str).str.len() > 0]
    if df.empty:
        return pd.DataFrame(columns=["symbol", "ts"] + EVENT_FEATURE_COLS)

    bar_td = pd.Timedelta(bar)
    bpd = pd.Timedelta("1D") / bar_td                       # 6 bars/day at 4h
    df["ts"] = df["datetime_utc"].dt.floor(bar) + bar_td     # RIGHT edge

    theme_cols = [f"t_{t}" for t in THEME_FEATURES]
    out = []
    for tkr, g in df.groupby("entities"):
        binned = g.groupby("ts").agg(
            alpha_mean=("c_alpha", "mean"), risk_max=("c_risk", "max"),
            rumor_mean=("c_rumor", "mean"),
            **{tc: (tc, "mean") for tc in theme_cols})
        full = pd.date_range(binned.index.min(), binned.index.max(), freq=bar, tz="UTC")
        binned = binned.reindex(full).fillna(0.0)
        feat = pd.DataFrame(index=full)
        feat["narr_event_alpha"] = binned["alpha_mean"].ewm(
            halflife=float(alpha_hl_days * bpd), adjust=False).mean()
        feat["narr_event_risk"] = binned["risk_max"].ewm(
            halflife=float(risk_hl_days * bpd), adjust=False).mean()
        feat["narr_rumor_risk"] = binned["rumor_mean"].ewm(
            halflife=float(risk_hl_days * bpd), adjust=False).mean()
        for t in THEME_FEATURES:
            feat[f"narr_theme_{t}"] = binned[f"t_{t}"].ewm(
                halflife=float(theme_hl_days * bpd), adjust=False).mean()
        feat = feat.reset_index().rename(columns={"index": "ts"})
        feat["symbol"] = str(tkr).upper()
        out.append(feat)

    res = pd.concat(out, ignore_index=True)
    return res[["symbol", "ts"] + EVENT_FEATURE_COLS].sort_values(
        ["symbol", "ts"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Build PIT-safe per-symbol LLM event features.")
    ap.add_argument("--events", default="data_storage/factors/sentiment/events_llm.parquet")
    ap.add_argument("--out", default="data_storage/factors/sentiment/event_features.parquet")
    args = ap.parse_args()
    events = _read_any(args.events)
    feats = build_event_features(events)
    _write_any(feats, args.out)
    print(f"wrote {len(feats)} rows for {feats['symbol'].nunique()} symbols -> {args.out}")
    print("columns:", EVENT_FEATURE_COLS)
    print(feats.groupby("symbol")["ts"].agg(["count", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
