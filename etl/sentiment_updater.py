"""
etl.sentiment_updater
=====================

Raw sentiment/news data fetchers.

This module keeps source-specific fetch logic for news/social sentiment in one
place. GDELT is the first implemented source; future X/Reddit fetchers can live
here without turning data_updater.py into a mixed market/news module.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

import config


@dataclass(frozen=True)
class GDELTDocConfig:
    endpoint: str = config.SentimentConfig.GDELT_DOC_ENDPOINT
    max_records: int = config.SentimentConfig.GDELT_MAX_RECORDS
    window_hours: int = config.SentimentConfig.GDELT_WINDOW_HOURS
    sleep_seconds: float = config.SentimentConfig.GDELT_SLEEP_SECONDS
    retries: int = config.SentimentConfig.GDELT_RETRIES
    backoff_seconds: float = config.SentimentConfig.GDELT_BACKOFF_SECONDS
    use_proxy: bool = config.SentimentConfig.GDELT_USE_PROXY
    sort: str = "datedesc"


GDELT_RAW_COLUMNS = [
    "query_name",
    "symbol",
    "query",
    "article_url",
    "title",
    "seendate",
    "domain",
    "language",
    "source_country",
    "tone",
    "social_image",
    "source",
    "fetched_at",
]


def _default_start_end(start_date=None, end_date=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.to_datetime(end_date) if end_date is not None else pd.Timestamp.utcnow().tz_localize(None)
    if start_date is None:
        start = end - pd.Timedelta(days=config.SentimentConfig.GDELT_DEFAULT_LOOKBACK_DAYS)
    else:
        start = pd.to_datetime(start_date)
    return _naive_ts(start), _naive_ts(end)


def _naive_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _gdelt_datetime(value: pd.Timestamp) -> str:
    return _naive_ts(value).strftime("%Y%m%d%H%M%S")


def _iter_windows(start: pd.Timestamp, end: pd.Timestamp, hours: int):
    current = start
    step = pd.Timedelta(hours=hours)
    while current < end:
        nxt = min(current + step, end)
        yield current, nxt
        current = nxt


def _proxy_opener(use_proxy: bool):
    proxies = config.SpiderConfig.PROXY if use_proxy else None
    if not proxies:
        return urllib.request.build_opener()
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies))


def _fetch_json(url: str, cfg: GDELTDocConfig, timeout_seconds: int = 30) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TradingSystem/0.1 sentiment updater",
            "Accept": "application/json",
        },
    )
    last_error: Exception | None = None
    for attempt in range(cfg.retries + 1):
        try:
            with _proxy_opener(cfg.use_proxy).open(req, timeout=timeout_seconds) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except HTTPError as exc:
            last_error = exc
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt >= cfg.retries:
                raise
            wait = cfg.backoff_seconds * (attempt + 1)
            print(f"[WARN] GDELT HTTP {exc.code}; retrying in {wait:.1f}s")
            time.sleep(wait)
        except URLError as exc:
            last_error = exc
            if attempt >= cfg.retries:
                raise
            wait = cfg.backoff_seconds * (attempt + 1)
            print(f"[WARN] GDELT network error; retrying in {wait:.1f}s: {exc}")
            time.sleep(wait)
    raise RuntimeError(f"GDELT request failed: {last_error}")


def _build_gdelt_url(
    query: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: GDELTDocConfig,
) -> str:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": _gdelt_datetime(start),
        "enddatetime": _gdelt_datetime(end),
        "maxrecords": str(cfg.max_records),
        "sort": cfg.sort,
    }
    return f"{cfg.endpoint}?{urllib.parse.urlencode(params)}"


def _normalize_gdelt_articles(
    articles: list[dict],
    *,
    query_name: str,
    symbol: str | None,
    query: str,
    fetched_at: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for item in articles:
        rows.append({
            "query_name": query_name,
            "symbol": symbol,
            "query": query,
            "article_url": item.get("url"),
            "title": item.get("title"),
            "seendate": pd.to_datetime(item.get("seendate"), errors="coerce"),
            "domain": item.get("domain"),
            "language": item.get("language"),
            "source_country": item.get("sourcecountry"),
            "tone": pd.to_numeric(item.get("tone"), errors="coerce"),
            "social_image": item.get("socialimage"),
            "source": "gdelt_doc2",
            "fetched_at": fetched_at,
        })
    if not rows:
        return pd.DataFrame(columns=GDELT_RAW_COLUMNS)
    return pd.DataFrame(rows, columns=GDELT_RAW_COLUMNS)


def fetch_gdelt_query(
    query_name: str,
    query: str,
    *,
    symbol: str | None = None,
    start_date=None,
    end_date=None,
    cfg: GDELTDocConfig | None = None,
) -> pd.DataFrame:
    """Fetch raw GDELT DOC 2.0 article-list rows for one query."""
    cfg = cfg or GDELTDocConfig()
    start, end = _default_start_end(start_date, end_date)
    fetched_at = pd.Timestamp.utcnow().tz_localize(None)
    frames = []

    for win_start, win_end in _iter_windows(start, end, cfg.window_hours):
        url = _build_gdelt_url(query, win_start, win_end, cfg)
        data = _fetch_json(url, cfg=cfg, timeout_seconds=max(30, config.SpiderConfig.TIMEOUT // 1000))
        articles = data.get("articles", [])
        frames.append(
            _normalize_gdelt_articles(
                articles,
                query_name=query_name,
                symbol=symbol,
                query=query,
                fetched_at=fetched_at,
            )
        )
        if cfg.sleep_seconds > 0:
            time.sleep(cfg.sleep_seconds)

    if not frames:
        return pd.DataFrame(columns=GDELT_RAW_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return _dedupe_gdelt(out)


def fetch_gdelt_crypto_news(
    symbols: Sequence[str] | None = None,
    *,
    include_global: bool = True,
    start_date=None,
    end_date=None,
    output_path: str | Path | None = None,
    append: bool = True,
    cfg: GDELTDocConfig | None = None,
) -> pd.DataFrame:
    """
    Fetch GDELT crypto/news articles and save a raw parquet table.

    Default behavior fetches the last SentimentConfig.GDELT_DEFAULT_LOOKBACK_DAYS
    only. For historical backfills, pass explicit start_date/end_date and expect
    many API calls because GDELT article-list results are windowed.
    """
    cfg = cfg or GDELTDocConfig()
    symbol_queries = config.SentimentConfig.GDELT_SYMBOL_QUERIES
    selected_symbols = list(symbols or symbol_queries.keys())

    query_items: list[tuple[str, str | None, str]] = []
    for symbol in selected_symbols:
        query = symbol_queries.get(symbol)
        if query is None:
            print(f"[WARN] No GDELT query configured for {symbol}; skipped.")
            continue
        query_items.append((symbol.replace("/", ""), symbol, query))

    if include_global:
        for name, query in config.SentimentConfig.GDELT_GLOBAL_QUERIES.items():
            query_items.append((name, None, query))

    frames = []
    for query_name, symbol, query in query_items:
        print(f"Fetching GDELT articles: {query_name} | {query}")
        frame = fetch_gdelt_query(
            query_name=query_name,
            query=query,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            cfg=cfg,
        )
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GDELT_RAW_COLUMNS)
    out = _dedupe_gdelt(out)

    path = Path(output_path) if output_path else config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.GDELT_OUTPUT_NAME
    if append and path.exists():
        old = pd.read_parquet(path)
        out = _dedupe_gdelt(pd.concat([old, out], ignore_index=True))

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    print(f"saved GDELT raw articles: {len(out)} rows -> {path}")
    return out


def _dedupe_gdelt(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=GDELT_RAW_COLUMNS)
    out = df.copy()
    for col in GDELT_RAW_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["seendate"] = pd.to_datetime(out["seendate"], errors="coerce")
    out["fetched_at"] = pd.to_datetime(out["fetched_at"], errors="coerce")
    out = out.drop_duplicates(subset=["query_name", "article_url"], keep="last")
    return out[GDELT_RAW_COLUMNS].sort_values(["query_name", "seendate", "article_url"]).reset_index(drop=True)


def available_gdelt_queries() -> pd.DataFrame:
    """Return configured GDELT query names for notebook inspection."""
    rows = [
        {"query_name": sym.replace("/", ""), "symbol": sym, "query": query}
        for sym, query in config.SentimentConfig.GDELT_SYMBOL_QUERIES.items()
    ]
    rows.extend(
        {"query_name": name, "symbol": None, "query": query}
        for name, query in config.SentimentConfig.GDELT_GLOBAL_QUERIES.items()
    )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fetch_gdelt_crypto_news()
