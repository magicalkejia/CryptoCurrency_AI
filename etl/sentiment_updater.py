"""
etl.sentiment_updater
=====================

Raw news/sentiment data fetchers.

The current implementation keeps two lightweight public-data paths:
    - RSS feeds for recent incremental news.
    - CoinDesk public archive index pages for title-level historical articles.
    - Optional CoinDesk article-page metadata fetches with a persistent browser
      profile for logged-in access.

Article-detail fetches store metadata only, not full article body text.
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError

import pandas as pd

import config


@dataclass(frozen=True)
class RSSConfig:
    retries: int = config.SentimentConfig.RSS_RETRIES
    backoff_seconds: float = config.SentimentConfig.RSS_BACKOFF_SECONDS
    use_proxy: bool = config.SentimentConfig.RSS_USE_PROXY


RSS_RAW_COLUMNS = [
    "feed_name",
    "article_url",
    "title",
    "published_at",
    "summary",
    "domain",
    "source",
    "fetched_at",
]


COINDESK_ARCHIVE_COLUMNS = [
    "url",
    "title",
    "published_date",
    "source",
    "archive_year",
    "archive_page",
    "archive_page_url",
    "fetched_at",
]


COINDESK_ARTICLE_COLUMNS = [
    "url",
    "title",
    "published_date",
    "published_at",
    "section",
    "author",
    "description",
    "source",
    "fetch_status",
    "fetched_at",
]


COINDESK_ARTICLE_BODY_COLUMNS = [
    "url",
    "body_text",
    "body_char_count",
    "body_word_count",
    "body_hash",
    "source",
    "fetch_status",
    "fetched_at",
]


COINDESK_MAX_ARCHIVE_PAGES_PER_YEAR = 100
COINDESK_TERMINAL_DETAIL_STATUSES = {"ok", "http_404"}


def _proxy_opener(use_proxy: bool):
    proxies = None
    if use_proxy:
        proxy_url = getattr(config.SentimentConfig, "RSS_PROXY_URL", None) or getattr(config.SpiderConfig, "_proxy_url", None)
        if proxy_url:
            proxies = {"http": proxy_url, "https": proxy_url}
        else:
            proxies = config.SpiderConfig.PROXY
    if not proxies:
        return urllib.request.build_opener()
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies))


def _fetch_bytes(
    url: str,
    *,
    retries: int,
    backoff_seconds: float,
    use_proxy: bool,
    timeout_seconds: int = 30,
) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TradingSystem/0.1 sentiment updater",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with _proxy_opener(use_proxy).open(req, timeout=timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt >= retries:
                raise
            wait = backoff_seconds * (attempt + 1)
            print(f"[WARN] HTTP {exc.code}; retrying in {wait:.1f}s: {url}")
            time.sleep(wait)
        except URLError as exc:
            last_error = exc
            if attempt >= retries:
                raise
            wait = backoff_seconds * (attempt + 1)
            print(f"[WARN] Network error; retrying in {wait:.1f}s: {exc}")
            time.sleep(wait)
    raise RuntimeError(f"Request failed: {last_error}")


def _domain_from_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        return urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return None


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(node: ET.Element, names: set[str]) -> str | None:
    for child in list(node):
        if _strip_namespace(child.tag) in names:
            text = "".join(child.itertext()).strip()
            return text or None
    return None


def _entry_link(node: ET.Element) -> str | None:
    link = _child_text(node, {"link"})
    if link:
        return link
    for child in list(node):
        if _strip_namespace(child.tag) == "link":
            href = child.attrib.get("href")
            if href:
                return href
    return None


def _parse_rss_datetime(value: str | None) -> pd.Timestamp:
    if not value:
        return pd.NaT
    return pd.to_datetime(value, errors="coerce", utc=True).tz_convert(None)


def _parse_feed(payload: bytes, *, feed_name: str, source_url: str, fetched_at: pd.Timestamp) -> pd.DataFrame:
    root = ET.fromstring(payload)
    rows = []

    for node in root.iter():
        tag = _strip_namespace(node.tag)
        if tag not in {"item", "entry"}:
            continue

        url = _entry_link(node)
        published = (
            _child_text(node, {"pubdate", "published", "updated", "dc:date"})
            or _child_text(node, {"date"})
        )
        rows.append({
            "feed_name": feed_name,
            "article_url": url,
            "title": _child_text(node, {"title"}),
            "published_at": _parse_rss_datetime(published),
            "summary": _child_text(node, {"description", "summary", "content", "encoded"}),
            "domain": _domain_from_url(url) or _domain_from_url(source_url),
            "source": "rss",
            "fetched_at": fetched_at,
        })

    if not rows:
        return pd.DataFrame(columns=RSS_RAW_COLUMNS)
    return pd.DataFrame(rows, columns=RSS_RAW_COLUMNS)


def fetch_rss_crypto_news(
    feeds: dict[str, str] | None = None,
    *,
    output_path: str | Path | None = None,
    append: bool = True,
    cfg: RSSConfig | None = None,
) -> pd.DataFrame:
    """Fetch configured crypto/news RSS feeds and save a raw parquet table."""
    cfg = cfg or RSSConfig()
    feed_map = feeds or config.SentimentConfig.RSS_FEEDS
    fetched_at = pd.Timestamp.utcnow().tz_localize(None)
    frames = []

    for feed_name, url in feed_map.items():
        print(f"Fetching RSS articles: {feed_name} | {url}")
        try:
            payload = _fetch_bytes(
                url,
                retries=cfg.retries,
                backoff_seconds=cfg.backoff_seconds,
                use_proxy=cfg.use_proxy,
                timeout_seconds=max(30, config.SpiderConfig.TIMEOUT // 1000),
            )
            frames.append(_parse_feed(payload, feed_name=feed_name, source_url=url, fetched_at=fetched_at))
        except Exception as exc:
            print(f"[WARN] RSS feed skipped: {feed_name} | {exc}")

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RSS_RAW_COLUMNS)
    out = _dedupe_rss(out)

    path = Path(output_path) if output_path else config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.RSS_OUTPUT_NAME
    if append and path.exists():
        old = pd.read_parquet(path)
        out = _dedupe_rss(pd.concat([old, out], ignore_index=True))

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    print(f"saved RSS raw articles: {len(out)} rows -> {path}")
    return out


def _dedupe_rss(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=RSS_RAW_COLUMNS)
    out = df.copy()
    for col in RSS_RAW_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce")
    out["fetched_at"] = pd.to_datetime(out["fetched_at"], errors="coerce")
    out = out.drop_duplicates(subset=["feed_name", "article_url"], keep="last")
    return out[RSS_RAW_COLUMNS].sort_values(["feed_name", "published_at", "article_url"]).reset_index(drop=True)


def _strip_html(value: str) -> str:
    text = re.sub(r"<script\b.*?</script>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value if v)
    text = _strip_html(str(value))
    return text or None


def _parse_timestamp(value) -> pd.Timestamp:
    if value is None or value == "":
        return pd.NaT
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(None)


def _json_objects_from_ld(html_text: str) -> list[dict]:
    pattern = re.compile(
        r"<script\b[^>]*type\s*=\s*['\"]application/ld\+json['\"][^>]*>(?P<body>.*?)</script>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    objects: list[dict] = []

    def add_obj(obj) -> None:
        if isinstance(obj, dict):
            objects.append(obj)
            graph = obj.get("@graph")
            if isinstance(graph, list):
                for child in graph:
                    add_obj(child)
        elif isinstance(obj, list):
            for child in obj:
                add_obj(child)

    for match in pattern.finditer(html_text):
        body = html.unescape(match.group("body")).strip()
        if not body:
            continue
        try:
            add_obj(json.loads(body))
        except json.JSONDecodeError:
            continue
    return objects


def _meta_content(html_text: str, keys: set[str]) -> str | None:
    pattern = re.compile(r"<meta\b(?P<attrs>[^>]*)>", flags=re.IGNORECASE | re.DOTALL)
    attr_pattern = re.compile(
        r"""(?P<name>name|property|itemprop)\s*=\s*['\"](?P<key>[^'\"]+)['\"]|"""
        r"""(?P<cname>content)\s*=\s*['\"](?P<content>[^'\"]*)['\"]""",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(html_text):
        attrs = match.group("attrs")
        found_key = None
        content = None
        for attr in attr_pattern.finditer(attrs):
            if attr.group("key"):
                found_key = attr.group("key").lower()
            if attr.group("content") is not None:
                content = html.unescape(attr.group("content"))
        if found_key in keys and content:
            return content
    return None


def _coindesk_section_from_url(url: str) -> str | None:
    parts = [p for p in urllib.parse.urlparse(url).path.split("/") if p]
    if parts and not re.fullmatch(r"20\d{2}", parts[0]):
        return parts[0]
    return None


def _json_text(obj: dict, *keys: str) -> str | None:
    for key in keys:
        if key not in obj:
            continue
        value = obj.get(key)
        if isinstance(value, dict):
            value = value.get("name") or value.get("@id")
        elif isinstance(value, list):
            names = []
            for item in value:
                if isinstance(item, dict):
                    names.append(item.get("name") or item.get("@id"))
                else:
                    names.append(item)
            value = [v for v in names if v]
        text = _clean_text(value)
        if text:
            return text
    return None


def _extract_tag_blocks(html_text: str, tag: str) -> list[str]:
    pattern = re.compile(
        rf"<{tag}\b[^>]*>(?P<body>.*?)</{tag}>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return [m.group("body") for m in pattern.finditer(html_text)]


def _paragraph_texts(html_text: str) -> list[str]:
    texts = []
    for tag in ("p", "li"):
        for block in _extract_tag_blocks(html_text, tag):
            text = _clean_text(block)
            if not text:
                continue
            low = text.lower()
            if any(marker in low for marker in ("subscribe", "sign up", "cookie", "advertisement")):
                continue
            if len(text) < 25:
                continue
            texts.append(text)
    return texts


def extract_coindesk_article_body(html_text: str) -> str | None:
    """Extract article body text from a CoinDesk article page."""
    json_ld = _json_objects_from_ld(html_text)
    for obj in json_ld:
        body = _json_text(obj, "articleBody")
        if body and len(body) >= 100:
            return body

    article_blocks = _extract_tag_blocks(html_text, "article")
    for block in article_blocks:
        paragraphs = _paragraph_texts(block)
        if paragraphs:
            return "\n\n".join(paragraphs)

    paragraphs = _paragraph_texts(html_text)
    if paragraphs:
        return "\n\n".join(paragraphs)
    return None


def _article_body_record(
    html_text: str,
    *,
    url: str,
    fetched_at: pd.Timestamp,
    fetch_status: str,
) -> dict:
    body = extract_coindesk_article_body(html_text) if fetch_status == "ok" else None
    if body:
        body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        status = "ok"
    else:
        body_hash = None
        status = "missing_body" if fetch_status == "ok" else fetch_status

    return {
        "url": url,
        "body_text": body,
        "body_char_count": len(body) if body else 0,
        "body_word_count": len(body.split()) if body else 0,
        "body_hash": body_hash,
        "source": "coindesk",
        "fetch_status": status,
        "fetched_at": fetched_at,
    }


def _empty_article_body_record(row, *, fetched_at: pd.Timestamp, error: Exception) -> dict:
    status = _browser_error_fetch_status(error)
    return {
        "url": getattr(row, "url", None),
        "body_text": None,
        "body_char_count": 0,
        "body_word_count": 0,
        "body_hash": None,
        "source": "coindesk",
        "fetch_status": status,
        "fetched_at": fetched_at,
    }


def _coindesk_page_fetch_status(html_text: str) -> str:
    text = _strip_html(html_text).lower()
    if "hmm, that's weird" in text or "could've sworn the page was around here somewhere" in text:
        return "http_404"
    if "404" in text and "page not found" in text:
        return "http_404"
    if "redirected you too many times" in text or "redirect too many times" in text:
        return "cookie_redirect_loop"
    if "delete your cookies" in text and "redirect" in text:
        return "cookie_redirect_loop"
    if "access denied" in text or "forbidden" in text:
        return "access_denied"
    return "ok"


def _browser_error_fetch_status(error: Exception) -> str:
    text = str(error).lower()
    if "err_too_many_redirects" in text or "too many redirects" in text:
        return "cookie_redirect_loop"
    if "timeout" in text:
        return "error: timeout"
    if "target page, context or browser has been closed" in text:
        return "error: target_closed"
    if "net::err" in text:
        return "error: network"
    return f"error: {type(error).__name__}"


def parse_coindesk_article_metadata(
    html_text: str,
    *,
    url: str,
    fallback_title: str | None = None,
    fallback_published_date=None,
    fetched_at: pd.Timestamp | None = None,
    fetch_status: str = "ok",
) -> dict:
    """
    Extract article-level metadata from a CoinDesk article page.

    The parser intentionally stores metadata only. It does not preserve article
    body text; downstream sentiment builders can derive scores separately if
    that later becomes necessary.
    """
    fetched_at = fetched_at or pd.Timestamp.utcnow().tz_localize(None)
    json_ld = _json_objects_from_ld(html_text)
    article_objs = [
        obj for obj in json_ld
        if str(obj.get("@type", "")).lower() in {"newsarticle", "article", "blogposting"}
    ]
    candidates = article_objs or json_ld

    title = None
    published_at = pd.NaT
    section = None
    author = None
    description = None

    for obj in candidates:
        title = title or _json_text(obj, "headline", "name")
        published_at = published_at if not pd.isna(published_at) else _parse_timestamp(obj.get("datePublished"))
        section = section or _json_text(obj, "articleSection", "section")
        author = author or _json_text(obj, "author", "creator")
        description = description or _json_text(obj, "description")

    title = title or _clean_text(_meta_content(html_text, {"og:title", "twitter:title"}))
    title = title or _clean_text(fallback_title)

    if pd.isna(published_at):
        published_at = _parse_timestamp(_meta_content(html_text, {"article:published_time", "date", "pubdate"}))

    section = (
        section
        or _clean_text(_meta_content(html_text, {"article:section"}))
        or _coindesk_section_from_url(url)
    )
    author = author or _clean_text(_meta_content(html_text, {"author"}))
    description = description or _clean_text(_meta_content(html_text, {"description", "og:description"}))

    published_date = pd.NaT
    if not pd.isna(published_at):
        published_date = published_at.normalize()
    elif fallback_published_date is not None:
        published_date = pd.to_datetime(fallback_published_date, errors="coerce")
    if pd.isna(published_date):
        published_date = _published_date_from_url(url)

    status = fetch_status
    if status == "ok" and pd.isna(published_at):
        status = "missing_published_at"

    return {
        "url": url,
        "title": title,
        "published_date": published_date,
        "published_at": published_at,
        "section": section,
        "author": author,
        "description": description,
        "source": "coindesk",
        "fetch_status": status,
        "fetched_at": fetched_at,
    }


def _coindesk_archive_url(year: int, page: int | None = None) -> str:
    base = f"{config.SentimentConfig.COINDESK_ARCHIVE_BASE_URL}/{year}"
    if page is None or int(page) <= 1:
        return base
    return f"{base}/{int(page)}"


def _coindesk_archive_page_count(payload: bytes, *, year: int) -> int:
    html_text = payload.decode("utf-8", errors="replace")
    counts = [1]

    text = _strip_html(html_text)
    page_text_match = re.search(r"\bPage\s+\d+\s+of\s+(\d+)\b", text, flags=re.IGNORECASE)
    if page_text_match:
        counts.append(int(page_text_match.group(1)))

    href_pattern = re.compile(
        rf"""href\s*=\s*["'][^"']*/sitemap/archive/{int(year)}/(?P<page>\d+)(?:[/?#"']|$)""",
        flags=re.IGNORECASE,
    )
    counts.extend(int(m.group("page")) for m in href_pattern.finditer(html_text))
    return max(counts)


def _coindesk_archive_declared_count(payload: bytes) -> int | None:
    text = _strip_html(payload.decode("utf-8", errors="replace"))
    match = re.search(r"\b([\d,]+)\s+articles\s+published\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def _published_date_from_url(url: str) -> pd.Timestamp:
    match = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|$)", url)
    if not match:
        return pd.NaT
    year, month, day = match.groups()
    return pd.to_datetime(f"{year}-{int(month):02d}-{int(day):02d}", errors="coerce")


def _published_date_from_context(context: str) -> pd.Timestamp:
    text = _strip_html(context)
    patterns = [
        r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b",
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+(\d{1,2}),\s+(20\d{2})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        if match.group(1).isdigit():
            year, month, day = match.groups()
            return pd.to_datetime(f"{year}-{int(month):02d}-{int(day):02d}", errors="coerce")
        return pd.to_datetime(match.group(0), errors="coerce")
    return pd.NaT


def _is_coindesk_article_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if host and not host.endswith("coindesk.com"):
        return False
    if not path or path in {"/", "/sitemap"}:
        return False
    blocked_prefixes = (
        "/sitemap",
        "/arc/outboundfeeds",
        "/cdn-cgi",
        "/author/",
        "/tag/",
        "/category/",
        "/markets/sitemap",
    )
    return not path.startswith(blocked_prefixes)


def _parse_coindesk_archive_html(
    payload: bytes,
    *,
    source_url: str,
    archive_year: int | None = None,
    archive_page: int | None = None,
    fetched_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    html_text = payload.decode("utf-8", errors="replace")
    fetched_at = fetched_at or pd.Timestamp.utcnow().tz_localize(None)
    rows = []
    anchor_pattern = re.compile(
        r"<a\b(?P<attrs>[^>]*)>(?P<body>.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    href_pattern = re.compile(r"""href\s*=\s*["'](?P<href>[^"']+)["']""", flags=re.IGNORECASE)

    for match in anchor_pattern.finditer(html_text):
        href_match = href_pattern.search(match.group("attrs"))
        if not href_match:
            continue

        url = urllib.parse.urljoin(source_url, html.unescape(href_match.group("href")))
        url = url.split("#", 1)[0]
        if not _is_coindesk_article_url(url):
            continue

        title = _strip_html(match.group("body"))
        if not title:
            continue

        context = html_text[max(0, match.start() - 500): min(len(html_text), match.end() + 500)]
        published_date = _published_date_from_url(url)
        if pd.isna(published_date):
            published_date = _published_date_from_context(context)

        rows.append({
            "url": url,
            "title": title,
            "published_date": published_date,
            "source": "coindesk",
            "archive_year": archive_year,
            "archive_page": archive_page,
            "archive_page_url": source_url,
            "fetched_at": fetched_at,
        })

    if not rows:
        return pd.DataFrame(columns=COINDESK_ARCHIVE_COLUMNS)
    return _dedupe_coindesk_archive(pd.DataFrame(rows, columns=COINDESK_ARCHIVE_COLUMNS))


def fetch_coindesk_archive_index(
    start_year: int = 2021,
    end_year: int | None = None,
    *,
    output_path: str | Path | None = None,
    append: bool = False,
    cfg: RSSConfig | None = None,
) -> pd.DataFrame:
    """
    Fetch CoinDesk archive index pages into a compact title-level raw table.

    This intentionally does not visit article pages or use logged-in state. It
    only parses public archive pages and stores url/title/published_date/source.
    """
    cfg = cfg or RSSConfig()
    end_year = end_year or pd.Timestamp.utcnow().year
    path = Path(output_path) if output_path else config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.COINDESK_ARCHIVE_OUTPUT_NAME
    existing = pd.read_parquet(path) if append and path.exists() else pd.DataFrame(columns=COINDESK_ARCHIVE_COLUMNS)
    frames = []

    for year in range(int(start_year), int(end_year) + 1):
        fetched_at = pd.Timestamp.utcnow().tz_localize(None)
        first_url = _coindesk_archive_url(year)
        print(f"Fetching CoinDesk archive index: {year} | {first_url}")
        first_payload = _fetch_bytes(
            first_url,
            retries=cfg.retries,
            backoff_seconds=cfg.backoff_seconds,
            use_proxy=cfg.use_proxy,
            timeout_seconds=max(30, config.SpiderConfig.TIMEOUT // 1000),
        )
        page_count = _coindesk_archive_page_count(first_payload, year=year)
        if page_count > COINDESK_MAX_ARCHIVE_PAGES_PER_YEAR:
            raise RuntimeError(
                f"Unreasonable CoinDesk archive page_count={page_count} for {year}. "
                "Pagination parsing likely failed; stopping before scanning bogus pages."
            )
        declared_count = _coindesk_archive_declared_count(first_payload)
        year_frames = [_parse_coindesk_archive_html(
            first_payload,
            source_url=first_url,
            archive_year=year,
            archive_page=1,
            fetched_at=fetched_at,
        )]
        _write_coindesk_archive_index(path, existing, frames + year_frames)

        for page in range(2, page_count + 1):
            page_url = _coindesk_archive_url(year, page=page)
            print(f"Fetching CoinDesk archive index: {year} page {page}/{page_count} | {page_url}")
            payload = _fetch_bytes(
                page_url,
                retries=cfg.retries,
                backoff_seconds=cfg.backoff_seconds,
                use_proxy=cfg.use_proxy,
                timeout_seconds=max(30, config.SpiderConfig.TIMEOUT // 1000),
            )
            page_frame = _parse_coindesk_archive_html(
                payload,
                source_url=page_url,
                archive_year=year,
                archive_page=page,
                fetched_at=pd.Timestamp.utcnow().tz_localize(None),
            )
            year_frames.append(page_frame)
            _write_coindesk_archive_index(path, existing, frames + year_frames)

        year_out = _dedupe_coindesk_archive(pd.concat(year_frames, ignore_index=True))
        if declared_count is None:
            print(f"CoinDesk archive year {year}: parsed {len(year_out)} article URLs across {page_count} pages")
        else:
            print(
                f"CoinDesk archive year {year}: parsed {len(year_out)} / declared {declared_count} "
                f"article URLs across {page_count} pages"
            )
        frames.append(year_out)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COINDESK_ARCHIVE_COLUMNS)
    out = _dedupe_coindesk_archive(out)
    out = _write_coindesk_archive_index(path, existing, [out])
    print(f"saved CoinDesk archive index: {len(out)} rows -> {path}")
    return out


def fetch_coindesk_archive_index_with_browser(
    start_year: int = 2021,
    end_year: int | None = None,
    *,
    output_path: str | Path | None = None,
    append: bool = False,
    cdp_url: str = "http://127.0.0.1:9222",
    page_settle_seconds: float = 0.5,
    wait_for_network_idle: bool = False,
    network_idle_timeout_ms: int = 1500,
    block_heavy_resources: bool = True,
    retries: int = 3,
    retry_sleep_seconds: float = 3.0,
) -> pd.DataFrame:
    """
    Fetch CoinDesk archive index pages through an existing Chrome CDP session.

    This is useful when urllib/Python proxy settings cannot reach CoinDesk but
    the user's browser can browse it through Clash/system proxy.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "Playwright is required for browser-backed CoinDesk archive fetching. "
            "Run `pip install -e .` first."
        ) from exc

    end_year = end_year or pd.Timestamp.utcnow().year
    path = Path(output_path) if output_path else config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.COINDESK_ARCHIVE_OUTPUT_NAME
    existing = pd.read_parquet(path) if append and path.exists() else pd.DataFrame(columns=COINDESK_ARCHIVE_COLUMNS)
    existing = _dedupe_coindesk_archive(existing)
    completed_pages = _completed_coindesk_archive_pages(existing)
    frames = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(cdp_url)
        except Exception as exc:
            raise RuntimeError(
                "Could not connect to Chrome over CDP. Start Chrome with "
                "`task chrome_cdp`, then verify "
                "`http://127.0.0.1:9222/json/version` opens in a browser."
            ) from exc

        context = browser.contexts[0] if browser.contexts else browser.new_context()
        page = context.new_page()
        if block_heavy_resources:
            _block_heavy_browser_resources(page)

        try:
            for year in range(int(start_year), int(end_year) + 1):
                first_url = _coindesk_archive_url(year)
                print(f"Fetching CoinDesk archive index via browser: {year} | {first_url}")
                first_payload = _load_browser_page_bytes(
                    page,
                    first_url,
                    page_settle_seconds=page_settle_seconds,
                    wait_for_network_idle=wait_for_network_idle,
                    network_idle_timeout_ms=network_idle_timeout_ms,
                    retries=retries,
                    retry_sleep_seconds=retry_sleep_seconds,
                )
                page_count = _coindesk_archive_page_count(first_payload, year=year)
                if page_count > COINDESK_MAX_ARCHIVE_PAGES_PER_YEAR:
                    raise RuntimeError(
                        f"Unreasonable CoinDesk archive page_count={page_count} for {year}. "
                        "Pagination parsing likely failed; stopping before scanning bogus pages."
                    )
                declared_count = _coindesk_archive_declared_count(first_payload)
                year_frames = [_parse_coindesk_archive_html(
                    first_payload,
                    source_url=first_url,
                    archive_year=year,
                    archive_page=1,
                    fetched_at=pd.Timestamp.utcnow().tz_localize(None),
                )]
                _write_coindesk_archive_index(path, existing, frames + year_frames)

                for archive_page in range(2, page_count + 1):
                    page_url = _coindesk_archive_url(year, page=archive_page)
                    if (year, archive_page) in completed_pages:
                        print(f"Skipping CoinDesk archive index via browser: {year} page {archive_page}/{page_count} already in local index")
                        page_frame = existing[
                            (existing["archive_year"].eq(year))
                            & (existing["archive_page"].eq(archive_page))
                        ].copy()
                        year_frames.append(page_frame)
                        continue

                    print(
                        f"Fetching CoinDesk archive index via browser: "
                        f"{year} page {archive_page}/{page_count} | {page_url}"
                    )
                    payload = _load_browser_page_bytes(
                        page,
                        page_url,
                        page_settle_seconds=page_settle_seconds,
                        wait_for_network_idle=wait_for_network_idle,
                        network_idle_timeout_ms=network_idle_timeout_ms,
                        retries=retries,
                        retry_sleep_seconds=retry_sleep_seconds,
                    )
                    page_frame = _parse_coindesk_archive_html(
                        payload,
                        source_url=page_url,
                        archive_year=year,
                        archive_page=archive_page,
                        fetched_at=pd.Timestamp.utcnow().tz_localize(None),
                    )
                    year_frames.append(page_frame)
                    _write_coindesk_archive_index(path, existing, frames + year_frames)

                year_out = _dedupe_coindesk_archive(pd.concat(year_frames, ignore_index=True))
                if declared_count is None:
                    print(f"CoinDesk archive year {year}: parsed {len(year_out)} article URLs across {page_count} pages")
                else:
                    print(
                        f"CoinDesk archive year {year}: parsed {len(year_out)} / declared {declared_count} "
                        f"article URLs across {page_count} pages"
                    )
                frames.append(year_out)
        finally:
            page.close()
            browser.close()

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COINDESK_ARCHIVE_COLUMNS)
    out = _dedupe_coindesk_archive(out)
    out = _write_coindesk_archive_index(path, existing, [out])
    print(f"saved CoinDesk archive index: {len(out)} rows -> {path}")
    return out


def fetch_coindesk_article_details(
    archive_index: pd.DataFrame | None = None,
    *,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    body_output_path: str | Path | None = None,
    start_date=None,
    end_date=None,
    limit: int | None = 300,
    append: bool = True,
    save_article_body: bool = True,
    use_browser: bool = False,
    use_existing_browser_cdp: bool = False,
    cdp_url: str = "http://127.0.0.1:9222",
    browser_user_data_dir: str | Path | None = None,
    browser_headless: bool = False,
    browser_channel: str | None = "chrome",
    login_first: bool = False,
    sleep_seconds: float = 0.05,
    page_settle_seconds: float = 0.1,
    wait_for_network_idle: bool = False,
    network_idle_timeout_ms: int = 1500,
    block_heavy_resources: bool = True,
    browser_page_workers: int = 2,
    progress_save_every: int = 25,
    max_consecutive_non_ok: int = 10,
    use_checkpoint_parts: bool = True,
    compact_output_at_end: bool = False,
    cfg: RSSConfig | None = None,
) -> pd.DataFrame:
    """
    Visit CoinDesk article pages and save metadata with exact publish time.

    use_browser=True uses Playwright with a persistent profile so the user can
    log in once and reuse the session. The function stores metadata only, not
    article body text.
    """
    cfg = cfg or RSSConfig()
    fetched_at = pd.Timestamp.utcnow().tz_localize(None)
    if archive_index is None:
        in_path = Path(input_path) if input_path else (
            config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.COINDESK_ARCHIVE_OUTPUT_NAME
        )
        if not in_path.exists():
            raise FileNotFoundError(f"CoinDesk archive index not found: {in_path}")
        archive_index = pd.read_parquet(in_path)

    if archive_index is None or archive_index.empty:
        out = pd.DataFrame(columns=COINDESK_ARTICLE_COLUMNS)
        return out

    rows_df = archive_index.copy()
    rows_df = rows_df.dropna(subset=["url"]).drop_duplicates(subset=["url"], keep="last")
    rows_df["published_date"] = pd.to_datetime(rows_df.get("published_date"), errors="coerce")
    if start_date is not None:
        rows_df = rows_df[rows_df["published_date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        rows_df = rows_df[rows_df["published_date"] <= pd.to_datetime(end_date)]
    rows_df = rows_df.sort_values(["published_date", "url"], na_position="last")
    print(
        "CoinDesk article detail date filter: "
        f"start={start_date or 'min'}, end={end_date or 'max'}, candidate_rows={len(rows_df)}"
    )

    out_path = Path(output_path) if output_path else (
        config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.COINDESK_ARTICLE_DETAIL_OUTPUT_NAME
    )
    body_path = Path(body_output_path) if body_output_path else (
        config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.COINDESK_ARTICLE_BODY_OUTPUT_NAME
    )
    details_parts_dir = _coindesk_parts_dir(out_path)
    body_parts_dir = _coindesk_parts_dir(body_path)
    if append:
        existing = _load_coindesk_table_with_parts(
            out_path,
            details_parts_dir,
            COINDESK_ARTICLE_COLUMNS,
            _dedupe_coindesk_article_details,
        )
    else:
        existing = pd.DataFrame(columns=COINDESK_ARTICLE_COLUMNS)

    if save_article_body and append:
        existing_bodies = _load_coindesk_table_with_parts(
            body_path,
            body_parts_dir,
            COINDESK_ARTICLE_BODY_COLUMNS,
            _dedupe_coindesk_article_bodies,
        )
    else:
        existing_bodies = pd.DataFrame(columns=COINDESK_ARTICLE_BODY_COLUMNS)

    if append:
        done_urls = _completed_coindesk_article_urls(existing, existing_bodies, save_article_body=save_article_body)
        rows_df = rows_df[~rows_df["url"].isin(done_urls)]

    if limit is not None:
        rows_df = rows_df.head(int(limit))

    if rows_df.empty:
        print("CoinDesk article details already up to date for selected rows.")
        return _dedupe_coindesk_article_details(existing)

    records: list[dict] = []
    body_records: list[dict] = []
    consecutive_non_ok = 0
    last_non_ok_warning_at = 0

    def record_sink(record: dict, body_record: dict | None = None) -> None:
        nonlocal consecutive_non_ok, last_non_ok_warning_at
        records.append(record)
        if save_article_body and body_record is not None:
            body_records.append(body_record)
        if progress_save_every and len(records) % int(progress_save_every) == 0:
            batch_count = len(records)
            if use_checkpoint_parts:
                _write_coindesk_part(details_parts_dir, records, COINDESK_ARTICLE_COLUMNS, "details")
            else:
                _write_coindesk_article_details(out_path, existing, records)
            if save_article_body:
                if use_checkpoint_parts:
                    _write_coindesk_part(body_parts_dir, body_records, COINDESK_ARTICLE_BODY_COLUMNS, "bodies")
                else:
                    _write_coindesk_article_bodies(body_path, existing_bodies, body_records)
            records.clear()
            body_records.clear()
            print(f"progress saved CoinDesk article details: +{batch_count} rows -> {out_path}")

        if record.get("fetch_status") == "ok":
            consecutive_non_ok = 0
        else:
            consecutive_non_ok += 1
            if max_consecutive_non_ok and consecutive_non_ok >= int(max_consecutive_non_ok):
                if consecutive_non_ok >= last_non_ok_warning_at + int(max_consecutive_non_ok):
                    last_non_ok_warning_at = consecutive_non_ok
                    print(
                        f"[WARN] {consecutive_non_ok} consecutive non-ok CoinDesk pages. "
                        "Continuing because unstable proxy/page loads can recover on retry. "
                        "Progress will be saved at the next checkpoint."
                    )

    if use_existing_browser_cdp:
        _fetch_coindesk_details_with_cdp(
            rows_df,
            fetched_at=fetched_at,
            cdp_url=cdp_url,
            sleep_seconds=sleep_seconds,
            page_settle_seconds=page_settle_seconds,
            wait_for_network_idle=wait_for_network_idle,
            network_idle_timeout_ms=network_idle_timeout_ms,
            block_heavy_resources=block_heavy_resources,
            page_workers=browser_page_workers,
            on_record=record_sink,
            save_article_body=save_article_body,
        )
    elif use_browser:
        _fetch_coindesk_details_with_browser(
            rows_df,
            fetched_at=fetched_at,
            user_data_dir=browser_user_data_dir,
            headless=browser_headless,
            channel=browser_channel,
            login_first=login_first,
            sleep_seconds=sleep_seconds,
            page_settle_seconds=page_settle_seconds,
            wait_for_network_idle=wait_for_network_idle,
            network_idle_timeout_ms=network_idle_timeout_ms,
            block_heavy_resources=block_heavy_resources,
            on_record=record_sink,
            save_article_body=save_article_body,
        )
    else:
        for i, row in enumerate(rows_df.itertuples(index=False), start=1):
            url = str(getattr(row, "url"))
            print(f"Fetching CoinDesk article metadata {i}/{len(rows_df)} | {url}")
            try:
                payload = _fetch_bytes(
                    url,
                    retries=cfg.retries,
                    backoff_seconds=cfg.backoff_seconds,
                    use_proxy=cfg.use_proxy,
                    timeout_seconds=max(30, config.SpiderConfig.TIMEOUT // 1000),
                )
                html_text = payload.decode("utf-8", errors="replace")
                fetch_status = _coindesk_page_fetch_status(html_text)
                meta = parse_coindesk_article_metadata(
                    html_text,
                    url=url,
                    fallback_title=getattr(row, "title", None),
                    fallback_published_date=getattr(row, "published_date", None),
                    fetched_at=fetched_at,
                    fetch_status=fetch_status,
                )
                body = _article_body_record(
                    html_text,
                    url=url,
                    fetched_at=fetched_at,
                    fetch_status=fetch_status,
                ) if save_article_body else None
                record_sink(meta, body)
            except Exception as exc:
                print(f"[WARN] CoinDesk article skipped: {url} | {exc}")
                body = _empty_article_body_record(row, fetched_at=fetched_at, error=exc) if save_article_body else None
                record_sink(_coindesk_article_error_row(row, fetched_at=fetched_at, error=exc), body)
            if sleep_seconds:
                _polite_sleep(sleep_seconds)

    if use_checkpoint_parts:
        if records:
            _write_coindesk_part(details_parts_dir, records, COINDESK_ARTICLE_COLUMNS, "details")
        if save_article_body and body_records:
            _write_coindesk_part(body_parts_dir, body_records, COINDESK_ARTICLE_BODY_COLUMNS, "bodies")
        if compact_output_at_end:
            out = compact_coindesk_article_outputs(
                details_path=out_path,
                body_path=body_path if save_article_body else None,
            )
        else:
            out = _load_coindesk_table_with_parts(
                out_path,
                details_parts_dir,
                COINDESK_ARTICLE_COLUMNS,
                _dedupe_coindesk_article_details,
            )
    else:
        out = _write_coindesk_article_details(out_path, existing, records)
    if save_article_body:
        if use_checkpoint_parts and not compact_output_at_end:
            body_out = _load_coindesk_table_with_parts(
                body_path,
                body_parts_dir,
                COINDESK_ARTICLE_BODY_COLUMNS,
                _dedupe_coindesk_article_bodies,
            )
        elif not use_checkpoint_parts:
            body_out = _write_coindesk_article_bodies(body_path, existing_bodies, body_records)
        else:
            body_out = _load_coindesk_table_with_parts(
                body_path,
                body_parts_dir,
                COINDESK_ARTICLE_BODY_COLUMNS,
                _dedupe_coindesk_article_bodies,
            )
        print(f"saved CoinDesk article bodies/checkpoints: {len(body_out)} rows -> {body_path}")
    print(f"saved CoinDesk article details: {len(out)} rows -> {out_path}")
    return out


def _fetch_coindesk_details_with_cdp(
    rows_df: pd.DataFrame,
    *,
    fetched_at: pd.Timestamp,
    cdp_url: str,
    sleep_seconds: float,
    page_settle_seconds: float,
    wait_for_network_idle: bool,
    network_idle_timeout_ms: int,
    block_heavy_resources: bool,
    page_workers: int,
    on_record,
    save_article_body: bool,
) -> None:
    if page_workers and int(page_workers) > 1:
        asyncio.run(_fetch_coindesk_details_with_cdp_async(
            rows_df,
            fetched_at=fetched_at,
            cdp_url=cdp_url,
            sleep_seconds=sleep_seconds,
            page_settle_seconds=page_settle_seconds,
            wait_for_network_idle=wait_for_network_idle,
            network_idle_timeout_ms=network_idle_timeout_ms,
            block_heavy_resources=block_heavy_resources,
            page_workers=int(page_workers),
            on_record=on_record,
            save_article_body=save_article_body,
        ))
        return

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "Playwright is required for Chrome CDP fetching. "
            "Run `pip install -e .` first."
        ) from exc

    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(cdp_url)
        except Exception as exc:
            raise RuntimeError(
                "Could not connect to Chrome over CDP. Start Chrome with "
                "`--remote-debugging-port=9222` first, then verify "
                "`http://127.0.0.1:9222/json/version` opens in a browser. "
                "If Chrome was already running, close all Chrome windows and "
                "background processes before launching it with the debugging port."
            ) from exc
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        page = context.new_page()
        if block_heavy_resources:
            _block_heavy_browser_resources(page)

        for i, row in enumerate(rows_df.itertuples(index=False), start=1):
            url = str(getattr(row, "url"))
            print(f"Fetching CoinDesk article metadata {i}/{len(rows_df)} | {url}")
            try:
                html_text = _load_browser_page_bytes(
                    page,
                    url,
                    page_settle_seconds=page_settle_seconds,
                    wait_for_network_idle=wait_for_network_idle,
                    network_idle_timeout_ms=network_idle_timeout_ms,
                    retries=3,
                    retry_sleep_seconds=3.0,
                ).decode("utf-8", errors="replace")
                fetch_status = _coindesk_page_fetch_status(html_text)
                meta = parse_coindesk_article_metadata(
                    html_text,
                    url=url,
                    fallback_title=getattr(row, "title", None),
                    fallback_published_date=getattr(row, "published_date", None),
                    fetched_at=fetched_at,
                    fetch_status=fetch_status,
                )
                body = _article_body_record(
                    html_text,
                    url=url,
                    fetched_at=fetched_at,
                    fetch_status=fetch_status,
                ) if save_article_body else None
                on_record(meta, body)
            except Exception as exc:
                print(f"[WARN] CoinDesk article skipped: {url} | {exc}")
                body = _empty_article_body_record(row, fetched_at=fetched_at, error=exc) if save_article_body else None
                on_record(_coindesk_article_error_row(row, fetched_at=fetched_at, error=exc), body)
            if sleep_seconds:
                _polite_sleep(sleep_seconds)

        page.close()
        browser.close()


async def _fetch_coindesk_details_with_cdp_async(
    rows_df: pd.DataFrame,
    *,
    fetched_at: pd.Timestamp,
    cdp_url: str,
    sleep_seconds: float,
    page_settle_seconds: float,
    wait_for_network_idle: bool,
    network_idle_timeout_ms: int,
    block_heavy_resources: bool,
    page_workers: int,
    on_record,
    save_article_body: bool,
) -> None:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise ImportError(
            "Playwright is required for Chrome CDP fetching. "
            "Run `pip install -e .` first."
        ) from exc

    items = list(enumerate(rows_df.itertuples(index=False), start=1))
    total = len(items)
    queue: asyncio.Queue = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)

    async with async_playwright() as p:
        try:
            browser = await p.chromium.connect_over_cdp(cdp_url)
        except Exception as exc:
            raise RuntimeError(
                "Could not connect to Chrome over CDP. Start Chrome with "
                "`--remote-debugging-port=9222` first, then verify "
                "`http://127.0.0.1:9222/json/version` opens in a browser. "
                "If Chrome was already running, close all Chrome windows and "
                "background processes before launching it with the debugging port."
            ) from exc

        context = browser.contexts[0] if browser.contexts else await browser.new_context()

        async def worker(worker_id: int) -> None:
            page = await context.new_page()
            if block_heavy_resources:
                await _block_heavy_browser_resources_async(page)
            try:
                while True:
                    try:
                        i, row = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    url = str(getattr(row, "url"))
                    print(f"Fetching CoinDesk article metadata {i}/{total} [page {worker_id}] | {url}")
                    try:
                        html_text = (
                            await _load_browser_page_bytes_async(
                                page,
                                url,
                                page_settle_seconds=page_settle_seconds,
                                wait_for_network_idle=wait_for_network_idle,
                                network_idle_timeout_ms=network_idle_timeout_ms,
                                retries=3,
                                retry_sleep_seconds=3.0,
                            )
                        ).decode("utf-8", errors="replace")
                        fetch_status = _coindesk_page_fetch_status(html_text)
                        meta = parse_coindesk_article_metadata(
                            html_text,
                            url=url,
                            fallback_title=getattr(row, "title", None),
                            fallback_published_date=getattr(row, "published_date", None),
                            fetched_at=fetched_at,
                            fetch_status=fetch_status,
                        )
                        body = _article_body_record(
                            html_text,
                            url=url,
                            fetched_at=fetched_at,
                            fetch_status=fetch_status,
                        ) if save_article_body else None
                        on_record(meta, body)
                    except Exception as exc:
                        print(f"[WARN] CoinDesk article skipped: {url} | {exc}")
                        body = _empty_article_body_record(row, fetched_at=fetched_at, error=exc) if save_article_body else None
                        on_record(_coindesk_article_error_row(row, fetched_at=fetched_at, error=exc), body)

                    if sleep_seconds:
                        await _polite_sleep_async(sleep_seconds)
                    queue.task_done()
            finally:
                await page.close()

        workers = [
            asyncio.create_task(worker(worker_id))
            for worker_id in range(1, max(1, int(page_workers)) + 1)
        ]
        try:
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            await browser.close()


def _block_heavy_browser_resources(page) -> None:
    blocked_types = {"image", "font", "media", "stylesheet"}

    def route_handler(route):
        try:
            if route.request.resource_type in blocked_types:
                route.abort()
            else:
                route.continue_()
        except Exception:
            route.continue_()

    page.route("**/*", route_handler)


def _completed_coindesk_archive_pages(df: pd.DataFrame) -> set[tuple[int, int]]:
    if df is None or df.empty:
        return set()
    if "archive_year" not in df.columns or "archive_page" not in df.columns:
        return set()
    tmp = df.dropna(subset=["url", "archive_year", "archive_page"]).copy()
    if tmp.empty:
        return set()
    tmp["archive_year"] = pd.to_numeric(tmp["archive_year"], errors="coerce")
    tmp["archive_page"] = pd.to_numeric(tmp["archive_page"], errors="coerce")
    tmp = tmp.dropna(subset=["archive_year", "archive_page"])
    return {
        (int(row.archive_year), int(row.archive_page))
        for row in tmp[["archive_year", "archive_page"]].drop_duplicates().itertuples(index=False)
    }


def _completed_coindesk_article_urls(
    details: pd.DataFrame,
    bodies: pd.DataFrame,
    *,
    save_article_body: bool,
) -> set[str]:
    if details is None or details.empty or "fetch_status" not in details.columns:
        return set()
    terminal_details = details[
        details["fetch_status"].isin(COINDESK_TERMINAL_DETAIL_STATUSES)
    ][["url", "fetch_status"]].dropna(subset=["url"])
    if terminal_details.empty:
        return set()

    http_404_urls = set(terminal_details.loc[terminal_details["fetch_status"].eq("http_404"), "url"])
    ok_detail_urls = set(terminal_details.loc[terminal_details["fetch_status"].eq("ok"), "url"])
    if not save_article_body:
        return http_404_urls | ok_detail_urls

    if bodies is None or bodies.empty or "fetch_status" not in bodies.columns:
        return http_404_urls
    ok_body_urls = set(bodies.loc[bodies["fetch_status"].eq("ok"), "url"].dropna())
    return http_404_urls | (ok_detail_urls & ok_body_urls)


def _jittered_delay(seconds: float) -> float:
    base = max(0.0, float(seconds))
    if base == 0:
        return 0.0
    return base * random.uniform(0.8, 1.3)


def _polite_sleep(seconds: float) -> None:
    delay = _jittered_delay(seconds)
    if delay > 0:
        time.sleep(delay)


async def _polite_sleep_async(seconds: float) -> None:
    delay = _jittered_delay(seconds)
    if delay > 0:
        await asyncio.sleep(delay)


def _load_browser_page_bytes(
    page,
    url: str,
    *,
    page_settle_seconds: float,
    wait_for_network_idle: bool,
    network_idle_timeout_ms: int,
    retries: int,
    retry_sleep_seconds: float,
) -> bytes:
    last_error: Exception | None = None
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            _settle_article_page(
                page,
                page_settle_seconds=page_settle_seconds,
                wait_for_network_idle=wait_for_network_idle,
                network_idle_timeout_ms=network_idle_timeout_ms,
            )
            return page.content().encode("utf-8")
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Browser page load failed {attempt}/{attempts}: {url} | {exc}")
            if _browser_error_fetch_status(exc) == "cookie_redirect_loop":
                raise
            if attempt < attempts:
                time.sleep(retry_sleep_seconds * attempt)
                try:
                    page.goto("about:blank", wait_until="domcontentloaded", timeout=10000)
                except Exception:
                    pass
    raise RuntimeError(f"Browser page load failed after {attempts} attempts: {url}") from last_error


async def _load_browser_page_bytes_async(
    page,
    url: str,
    *,
    page_settle_seconds: float,
    wait_for_network_idle: bool,
    network_idle_timeout_ms: int,
    retries: int,
    retry_sleep_seconds: float,
) -> bytes:
    last_error: Exception | None = None
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await _settle_article_page_async(
                page,
                page_settle_seconds=page_settle_seconds,
                wait_for_network_idle=wait_for_network_idle,
                network_idle_timeout_ms=network_idle_timeout_ms,
            )
            return (await page.content()).encode("utf-8")
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Browser page load failed {attempt}/{attempts}: {url} | {exc}")
            if _browser_error_fetch_status(exc) == "cookie_redirect_loop":
                raise
            if attempt < attempts:
                await asyncio.sleep(retry_sleep_seconds * attempt)
                try:
                    await page.goto("about:blank", wait_until="domcontentloaded", timeout=10000)
                except Exception:
                    pass
    raise RuntimeError(f"Browser page load failed after {attempts} attempts: {url}") from last_error


async def _block_heavy_browser_resources_async(page) -> None:
    blocked_types = {"image", "font", "media", "stylesheet"}

    async def route_handler(route):
        try:
            if route.request.resource_type in blocked_types:
                await route.abort()
            else:
                await route.continue_()
        except Exception:
            await route.continue_()

    await page.route("**/*", route_handler)


def _settle_article_page(
    page,
    *,
    page_settle_seconds: float,
    wait_for_network_idle: bool,
    network_idle_timeout_ms: int,
) -> None:
    if page_settle_seconds and page_settle_seconds > 0:
        page.wait_for_timeout(int(page_settle_seconds * 1000))
    if wait_for_network_idle:
        try:
            page.wait_for_load_state("networkidle", timeout=int(network_idle_timeout_ms))
        except Exception:
            pass


async def _settle_article_page_async(
    page,
    *,
    page_settle_seconds: float,
    wait_for_network_idle: bool,
    network_idle_timeout_ms: int,
) -> None:
    if page_settle_seconds and page_settle_seconds > 0:
        await page.wait_for_timeout(int(page_settle_seconds * 1000))
    if wait_for_network_idle:
        try:
            await page.wait_for_load_state("networkidle", timeout=int(network_idle_timeout_ms))
        except Exception:
            pass


def _fetch_coindesk_details_with_browser(
    rows_df: pd.DataFrame,
    *,
    fetched_at: pd.Timestamp,
    user_data_dir: str | Path | None,
    headless: bool,
    channel: str | None,
    login_first: bool,
    sleep_seconds: float,
    page_settle_seconds: float,
    wait_for_network_idle: bool,
    network_idle_timeout_ms: int,
    block_heavy_resources: bool,
    on_record,
    save_article_body: bool,
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "Playwright is required for logged-in CoinDesk fetching. "
            "Run `pip install -e .` and `python -m playwright install chromium`, "
            "or set USE_LOGGED_IN_BROWSER = False in scripts/fetch_coindesk_archive.py."
        ) from exc

    profile_dir = Path(user_data_dir) if user_data_dir else (
        config.PathConfig.DATA_ROOT / "browser_profiles" / "coindesk"
    )
    profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        launch_kwargs = {
            "headless": headless,
            "viewport": {"width": 1400, "height": 1000},
        }
        if channel:
            launch_kwargs["channel"] = channel

        context = p.chromium.launch_persistent_context(str(profile_dir), **launch_kwargs)
        page = context.new_page()
        if block_heavy_resources:
            _block_heavy_browser_resources(page)

        if login_first:
            page.goto("https://www.coindesk.com/", wait_until="domcontentloaded", timeout=60000)
            input("Log in to CoinDesk in the opened browser, then press Enter here to continue...")

        for i, row in enumerate(rows_df.itertuples(index=False), start=1):
            url = str(getattr(row, "url"))
            print(f"Fetching CoinDesk article metadata {i}/{len(rows_df)} | {url}")
            try:
                html_text = _load_browser_page_bytes(
                    page,
                    url,
                    page_settle_seconds=page_settle_seconds,
                    wait_for_network_idle=wait_for_network_idle,
                    network_idle_timeout_ms=network_idle_timeout_ms,
                    retries=3,
                    retry_sleep_seconds=3.0,
                ).decode("utf-8", errors="replace")
                fetch_status = _coindesk_page_fetch_status(html_text)
                meta = parse_coindesk_article_metadata(
                    html_text,
                    url=url,
                    fallback_title=getattr(row, "title", None),
                    fallback_published_date=getattr(row, "published_date", None),
                    fetched_at=fetched_at,
                    fetch_status=fetch_status,
                )
                body = _article_body_record(
                    html_text,
                    url=url,
                    fetched_at=fetched_at,
                    fetch_status=fetch_status,
                ) if save_article_body else None
                on_record(meta, body)
            except Exception as exc:
                print(f"[WARN] CoinDesk article skipped: {url} | {exc}")
                body = _empty_article_body_record(row, fetched_at=fetched_at, error=exc) if save_article_body else None
                on_record(_coindesk_article_error_row(row, fetched_at=fetched_at, error=exc), body)
            if sleep_seconds:
                _polite_sleep(sleep_seconds)

        context.close()


def _coindesk_article_error_row(row, *, fetched_at: pd.Timestamp, error: Exception) -> dict:
    published_date = pd.to_datetime(getattr(row, "published_date", pd.NaT), errors="coerce")
    status = _browser_error_fetch_status(error)
    return {
        "url": getattr(row, "url", None),
        "title": getattr(row, "title", None),
        "published_date": published_date,
        "published_at": pd.NaT,
        "section": _coindesk_section_from_url(str(getattr(row, "url", ""))),
        "author": None,
        "description": None,
        "source": "coindesk",
        "fetch_status": status,
        "fetched_at": fetched_at,
    }


def _write_coindesk_article_details(
    path: Path,
    existing: pd.DataFrame,
    records: list[dict],
) -> pd.DataFrame:
    new = pd.DataFrame(records, columns=COINDESK_ARTICLE_COLUMNS)
    out = _dedupe_coindesk_article_details(pd.concat([existing, new], ignore_index=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    out.to_parquet(tmp_path, engine="pyarrow", compression="zstd", index=False)
    tmp_path.replace(path)
    return out


def _write_coindesk_archive_index(
    path: Path,
    existing: pd.DataFrame,
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    parts = [existing] + [frame for frame in frames if frame is not None and not frame.empty]
    out = _dedupe_coindesk_archive(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame(columns=COINDESK_ARCHIVE_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    out.to_parquet(tmp_path, engine="pyarrow", compression="zstd", index=False)
    tmp_path.replace(path)
    return out


def _write_coindesk_article_bodies(
    path: Path,
    existing: pd.DataFrame,
    records: list[dict],
) -> pd.DataFrame:
    new = pd.DataFrame(records, columns=COINDESK_ARTICLE_BODY_COLUMNS)
    out = _dedupe_coindesk_article_bodies(pd.concat([existing, new], ignore_index=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    out.to_parquet(tmp_path, engine="pyarrow", compression="zstd", index=False)
    tmp_path.replace(path)
    return out


def _coindesk_parts_dir(path: Path) -> Path:
    return path.parent / f"_{path.stem}_parts"


def _write_coindesk_part(
    parts_dir: Path,
    records: list[dict],
    columns: list[str],
    prefix: str,
) -> Path | None:
    if not records:
        return None
    parts_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S%f")
    path = parts_dir / f"{prefix}_{ts}_{len(records)}.parquet"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(records, columns=columns).to_parquet(
        tmp_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    tmp_path.replace(path)
    return path


def _load_coindesk_table_with_parts(
    main_path: Path,
    parts_dir: Path,
    columns: list[str],
    dedupe_fn,
) -> pd.DataFrame:
    frames = []
    if main_path.exists():
        frames.append(pd.read_parquet(main_path))
    if parts_dir.exists():
        frames.extend(pd.read_parquet(p) for p in sorted(parts_dir.glob("*.parquet")))
    if not frames:
        return pd.DataFrame(columns=columns)
    return dedupe_fn(pd.concat(frames, ignore_index=True))


def compact_coindesk_article_outputs(
    *,
    details_path: str | Path | None = None,
    body_path: str | Path | None = None,
    delete_parts: bool = False,
) -> pd.DataFrame:
    details = Path(details_path) if details_path else (
        config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.COINDESK_ARTICLE_DETAIL_OUTPUT_NAME
    )
    detail_parts = _coindesk_parts_dir(details)
    detail_out = _load_coindesk_table_with_parts(
        details,
        detail_parts,
        COINDESK_ARTICLE_COLUMNS,
        _dedupe_coindesk_article_details,
    )
    _write_parquet_atomic(details, detail_out)
    if delete_parts:
        _delete_parquet_parts(detail_parts)

    if body_path is not None:
        body = Path(body_path)
        body_parts = _coindesk_parts_dir(body)
        body_out = _load_coindesk_table_with_parts(
            body,
            body_parts,
            COINDESK_ARTICLE_BODY_COLUMNS,
            _dedupe_coindesk_article_bodies,
        )
        _write_parquet_atomic(body, body_out)
        if delete_parts:
            _delete_parquet_parts(body_parts)

    return detail_out


def _delete_parquet_parts(parts_dir: Path) -> None:
    if not parts_dir.exists():
        return
    for part in parts_dir.glob("*.parquet"):
        part.unlink()


def _write_parquet_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, engine="pyarrow", compression="zstd", index=False)
    tmp_path.replace(path)


def _dedupe_coindesk_article_details(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=COINDESK_ARTICLE_COLUMNS)
    out = df.copy()
    for col in COINDESK_ARTICLE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["published_date"] = pd.to_datetime(out["published_date"], errors="coerce")
    out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce")
    out["fetched_at"] = pd.to_datetime(out["fetched_at"], errors="coerce")
    out = out.drop_duplicates(subset=["url"], keep="last")
    return out[COINDESK_ARTICLE_COLUMNS].sort_values(["published_at", "published_date", "url"], na_position="last").reset_index(drop=True)


def _dedupe_coindesk_article_bodies(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=COINDESK_ARTICLE_BODY_COLUMNS)
    out = df.copy()
    for col in COINDESK_ARTICLE_BODY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["fetched_at"] = pd.to_datetime(out["fetched_at"], errors="coerce")
    out["body_char_count"] = pd.to_numeric(out["body_char_count"], errors="coerce").fillna(0).astype(int)
    out["body_word_count"] = pd.to_numeric(out["body_word_count"], errors="coerce").fillna(0).astype(int)
    out = out.drop_duplicates(subset=["url"], keep="last")
    return out[COINDESK_ARTICLE_BODY_COLUMNS].sort_values(["fetched_at", "url"], na_position="last").reset_index(drop=True)


def _dedupe_coindesk_archive(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=COINDESK_ARCHIVE_COLUMNS)
    out = df.copy()
    for col in COINDESK_ARCHIVE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["published_date"] = pd.to_datetime(out["published_date"], errors="coerce")
    out["archive_year"] = pd.to_numeric(out["archive_year"], errors="coerce").astype("Int64")
    out["archive_page"] = pd.to_numeric(out["archive_page"], errors="coerce").astype("Int64")
    out["fetched_at"] = pd.to_datetime(out["fetched_at"], errors="coerce")
    out = out.drop_duplicates(subset=["url"], keep="last")
    return out[COINDESK_ARCHIVE_COLUMNS].sort_values(
        ["archive_year", "archive_page", "published_date", "url"],
        na_position="last",
    ).reset_index(drop=True)


def available_rss_feeds() -> pd.DataFrame:
    """Return configured RSS feeds for notebook inspection."""
    rows = [
        {"feed_name": name, "url": url}
        for name, url in config.SentimentConfig.RSS_FEEDS.items()
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fetch_rss_crypto_news()
