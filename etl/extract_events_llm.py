"""
etl/extract_events_llm.py
=========================
Stage 1.5 of the narrative modality (v6 §7.3): run the LLM event/narrative
extractor over the CoinDesk article corpus and persist ONE structured row per
article to data_storage/factors/sentiment/events_llm.parquet.

This is the LLM analogue of score_news.py (which only does CryptoBERT sentiment).
It is run OFFLINE on the user's machine; downstream stages
(build_event_features.py, narrative_loader.attach_event_features) consume only its
output, so the trading path stays deterministic.

Input (single merged file, joined upstream by a teammate; replaces the old
coindesk_article_details/bodies pair, and now also includes cryptoslate):
  data_storage/processed/sentiment/merged_news_for_llm.parquet
  columns: source, url, title, published_at (ms epoch), published_date, section,
           author, description, asset_type, sentiment_label, article_text

PIT: the event time is `published_at` (the publish instant). Because the corpus
was scraped retrospectively, every row is also tagged `llm_mode` so the audit
report can flag capability look-ahead (v6 §4.2); for the once-only live grade you
would re-extract with the model available at decision time.

Run:
    # real model (needs `pip install openai` and DEEPSEEK_API_KEY in .env):
    python -m etl.extract_events_llm
    # quick offline self-test (deterministic heuristic, no API):
    python -m etl.extract_events_llm --offline --limit 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from pathlib import Path

import pandas as pd

from etl.llm_deepseek import DeepSeekExtractor

DEF_MERGED = "data_storage/processed/sentiment/merged_news_for_llm.parquet"
DEF_OUT = "data_storage/factors/sentiment/events_llm.parquet"
DEF_CACHE = "data_storage/factors/sentiment/_events_llm_cache.json"

OUT_COLS = ["url", "source", "title", "datetime_utc", "event_type", "entities", "themes",
            "sentiment", "severity", "is_rumor", "source_credibility", "horizon",
            "confidence", "consistency", "llm_mode"]


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(path) if p.suffix.lower() == ".csv" else pd.read_parquet(path)


def _write_any(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if Path(path).suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def _content_hash(title: str, body: str, body_hash=None) -> str:
    if isinstance(body_hash, str) and body_hash:
        return body_hash
    return hashlib.sha1(f"{title}\n{body}".encode("utf-8", "ignore")).hexdigest()


# Bump if SYSTEM_PROMPT / schema change -> old cached extractions are invalidated.
# v2: tightened `entities` to PRIMARY-SUBJECT coins only (no passing-mention attribution).
PROMPT_VERSION = "v2"


def _cache_tag(extractor) -> str:
    """Namespace the cache so OFFLINE heuristic results and per-model/per-prompt LLM
    results never collide. Prevents an earlier `--offline` run from silently feeding
    heuristic rows into a paid DeepSeek run, and invalidates stale results when the
    model or prompt changes."""
    return "offline" if extractor.offline else f"{extractor.model}|{PROMPT_VERSION}"


def _flush_cache(cache: dict, path: str) -> None:
    """Atomic cache write (tmp + os.replace) so a kill mid-write cannot corrupt the
    cache file and wipe out previously-paid extractions."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    Path(tmp).write_text(json.dumps(cache))
    os.replace(tmp, path)


def extract_missing(extractor, missing: dict, cache: dict, cache_path: str,
                    workers: int = 1, flush_every: int = 20) -> None:
    """Extract only the cache-miss keys, persisting incrementally. Resumable: the
    cache is flushed every `flush_every` items AND in a finally-block, so a crash or
    Ctrl-C never loses more than `flush_every` calls. workers>1 fans out concurrently
    (the OpenAI client is thread-safe; cache writes are lock-guarded)."""
    keys = list(missing)
    done = {"n": 0}
    lock = threading.Lock()

    def _record(k, ev):
        with lock:
            cache[k] = ev
            done["n"] += 1
            if done["n"] % flush_every == 0:
                _flush_cache(cache, cache_path)
                print(f"      extracted {done['n']}/{len(keys)} new ...", flush=True)

    try:
        if workers <= 1:
            for k in keys:
                title, body = missing[k]
                _record(k, extractor.extract(title, body))
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def work(k):
                title, body = missing[k]
                return k, extractor.extract(title, body)

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(work, k) for k in keys]
                for fut in as_completed(futs):
                    k, ev = fut.result()
                    _record(k, ev)
    finally:
        _flush_cache(cache, cache_path)


def load_corpus(merged_path: str) -> pd.DataFrame:
    """Load the merged news corpus (coindesk + cryptoslate) produced upstream.

    Expected columns: source, url, title, published_at (ms epoch), article_text.
    `article_text` -> `body_text`; `published_at` -> PIT `datetime_utc`. There is no
    body_hash in the merged file, so caching falls back to a content hash."""
    df = _read_any(merged_path)
    need = {"url", "title", "published_at", "article_text"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"merged news file missing columns {sorted(missing)}; "
                         f"got {list(df.columns)}")
    df = df.rename(columns={"article_text": "body_text"})
    if "source" not in df.columns:
        df["source"] = "unknown"
    # published_at is a ms epoch -> the PIT event time
    df["datetime_utc"] = pd.to_datetime(df["published_at"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_utc", "title"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["body_text"] = df["body_text"].fillna("").astype(str)
    df["source"] = df["source"].fillna("unknown").astype(str)
    df["body_hash"] = None                          # no precomputed hash in merged file
    return df.drop_duplicates(subset=["url"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="LLM event extraction over the merged news corpus.")
    ap.add_argument("--merged", default=DEF_MERGED,
                    help="merged news parquet (coindesk + cryptoslate): columns "
                         "source,url,title,published_at,article_text,...")
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--cache", default=DEF_CACHE)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--offline", action="store_true", help="use deterministic heuristic (no API)")
    ap.add_argument("--consistency_runs", type=int, default=1,
                    help="repeat each extraction N times for an agreement score (real LLM only)")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent API requests (default 1 = sequential). 4-8 speeds up a "
                         "large corpus; keep modest to respect DeepSeek rate limits.")
    ap.add_argument("--limit", type=int, default=None, help="cap number of articles (debug)")
    args = ap.parse_args()

    print(f"[1/3] loading merged corpus  {args.merged}")
    corpus = load_corpus(args.merged)
    if args.limit:
        corpus = corpus.head(args.limit)
    print(f"      {len(corpus)} articles  ({corpus['datetime_utc'].min()} -> {corpus['datetime_utc'].max()})")
    print("      sources: " + ", ".join(f"{k}={v}" for k, v in corpus["source"].value_counts().items()))

    cache = {}
    if Path(args.cache).exists():
        try:
            cache = json.loads(Path(args.cache).read_text())
        except Exception:
            print(f"[warn] cache at {args.cache} unreadable; starting fresh.")
            cache = {}

    extractor = DeepSeekExtractor(model=args.model, offline=args.offline,
                                  consistency_runs=args.consistency_runs)
    mode = "offline_heuristic" if extractor.offline else f"deepseek:{args.model}"
    tag = _cache_tag(extractor)

    # Resolve every article to its NAMESPACED cache key (offline vs model|prompt).
    keyed = []
    for r in corpus.itertuples(index=False):
        k = f"{tag}:{_content_hash(r.title, r.body_text, getattr(r, 'body_hash', None))}"
        keyed.append((k, r))

    # Cost preview: how many will actually hit the API.
    missing = {}
    for k, r in keyed:
        if k not in cache and k not in missing:
            missing[k] = (r.title, r.body_text)
    print(f"[2/3] extracting events  mode={mode}  cache={args.cache}")
    print(f"      {len(cache)} entries cached | {len(missing)} new to extract"
          + (f" via {args.workers} workers" if args.workers > 1 else "")
          + (f"  (x{args.consistency_runs} consistency runs)" if args.consistency_runs > 1 else ""))

    if missing:
        extract_missing(extractor, missing, cache, args.cache, workers=max(1, args.workers))

    # Assemble output rows from the (now fully populated) cache.
    rows = []
    for k, r in keyed:
        ev = cache[k]
        rows.append({
            "url": r.url, "source": getattr(r, "source", "unknown"),
            "title": r.title, "datetime_utc": r.datetime_utc,
            "event_type": ev["event_type"], "entities": json.dumps(ev["entities"]),
            "themes": json.dumps(ev["themes"]), "sentiment": ev["sentiment"],
            "severity": ev["severity"], "is_rumor": bool(ev["is_rumor"]),
            "source_credibility": ev["source_credibility"], "horizon": ev["horizon"],
            "confidence": ev["confidence"], "consistency": ev.get("consistency", 1.0),
            "llm_mode": "offline" if ev.get("_offline") else "retrospective",
        })

    out = pd.DataFrame(rows)[OUT_COLS]
    _write_any(out, args.out)
    print(f"[3/3] wrote {len(out)} rows ({len(missing)} newly extracted) -> {args.out}")
    print(out["event_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
