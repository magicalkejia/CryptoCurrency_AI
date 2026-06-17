"""
etl/score_news.py
=================
Stage 1 of the narrative modality: score every news TITLE with a deterministic,
pre-trained crypto sentiment model (CryptoBERT, ElKulako/cryptobert) and write the
per-title scores to data_storage/factors/sentiment/news_scored.parquet.

Why deterministic (not an LLM API): the model is run in eval mode with fixed weights and
no sampling, so the same title always yields the same score -> fully reproducible, which is
required for a frozen config_hash. Scores are cached by a hash of the title so re-runs are
near-instant and never re-score a title differently.

CryptoBERT label order (id2label): 0=Bearish, 1=Neutral, 2=Bullish.
sentiment = P(Bullish) - P(Bearish)  in [-1, 1].

Run (on the GPU machine; needs torch + transformers + internet on first run to fetch weights):
    python -m etl.score_news
    python -m etl.score_news --raw data_storage/CryptoDataSet_v1.xlsx \
                             --out data_storage/factors/sentiment/news_scored.parquet

This file is NOT runnable in the offline research sandbox (no torch). It is meant for the
user's machine. The downstream stages (build_narrative_features.py, narrative_loader.py)
consume only its output and are tested independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "ElKulako/cryptobert"
DEFAULT_RAW = "data_storage/CryptoDataSet_v1.xlsx"
DEFAULT_OUT = "data_storage/factors/sentiment/news_scored.parquet"
DEFAULT_CACHE = "data_storage/factors/sentiment/_cryptobert_cache.json"


def _title_hash(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()


def load_raw(path: str) -> pd.DataFrame:
    """Read the raw news table and clean it. Accepts .xlsx (current) or .parquet (the
    colleague's future format). Expected columns: URL, Title, Date Time, Coin Type."""
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_excel(path)
    df = df.rename(columns={"URL": "url", "Title": "title",
                            "Date Time": "datetime_utc", "Coin Type": "coin"})
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["title", "datetime_utc"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() > 0]
    df = df.drop_duplicates(subset=["title", "datetime_utc"]).reset_index(drop=True)
    return df


def score_titles(titles, batch_size: int = 128, device: str | None = None,
                 cache_path: str | None = DEFAULT_CACHE) -> pd.DataFrame:
    """Score a list of titles with CryptoBERT. Returns a DataFrame with columns
    [p_bear, p_neutral, p_bull, sentiment] aligned to `titles`. Uses an on-disk cache
    keyed by title hash so repeated/duplicate titles are scored once."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"missing dependency '{e.name}'. Install the narrative extras:\n"
            f"    pip install -e \".[narrative]\"\n"
            f"(for a CUDA build of torch, install it first from pytorch.org, then the line above)"
        )

    cache = {}
    if cache_path and Path(cache_path).exists():
        try:
            cache = json.loads(Path(cache_path).read_text())
        except Exception:
            cache = {}

    uniq = sorted({t for t in titles})
    todo = [t for t in uniq if _title_hash(t) not in cache]

    if todo:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()
        # Map class indices to polarity by LABEL NAME (robust to label ordering).
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        bull_ix = next((i for i, l in id2label.items() if "bull" in l or "positive" in l), 2)
        bear_ix = next((i for i, l in id2label.items() if "bear" in l or "negative" in l), 0)
        neut_ix = next((i for i, l in id2label.items() if "neut" in l), 1)
        with torch.no_grad():
            for i in range(0, len(todo), batch_size):
                batch = todo[i:i + batch_size]
                enc = tok(batch, padding=True, truncation=True, max_length=128,
                          return_tensors="pt").to(device)
                probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
                for t, pr in zip(batch, probs):
                    cache[_title_hash(t)] = [float(pr[bear_ix]), float(pr[neut_ix]), float(pr[bull_ix])]
                print(f"  scored {min(i + batch_size, len(todo))}/{len(todo)} unique titles", flush=True)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps(cache))

    rows = []
    for t in titles:
        pb, pn, pu = cache[_title_hash(t)]
        rows.append((pb, pn, pu, pu - pb))
    return pd.DataFrame(rows, columns=["p_bear", "p_neutral", "p_bull", "sentiment"])


def main():
    ap = argparse.ArgumentParser(description="Score crypto news titles with CryptoBERT.")
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    print(f"[1/3] reading raw news from {args.raw}")
    df = load_raw(args.raw)
    print(f"      {len(df)} clean unique news rows  ({df['datetime_utc'].min()} -> {df['datetime_utc'].max()})")

    print(f"[2/3] scoring titles with {MODEL_NAME} (cached at {args.cache})")
    scores = score_titles(df["title"].tolist(), batch_size=args.batch_size,
                          device=args.device, cache_path=args.cache)
    out = pd.concat([df[["url", "title", "datetime_utc", "coin"]].reset_index(drop=True),
                     scores], axis=1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[3/3] wrote {len(out)} scored rows -> {args.out}")
    print(out[["datetime_utc", "coin", "sentiment"]].describe(include="all").to_string())


if __name__ == "__main__":
    main()
