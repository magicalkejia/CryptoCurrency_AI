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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


CRYPTOSLATE_ARCHIVE_COLUMNS = [
    "url",
    "title",
    "published_date",
    "source",
    "section",
    "description",
    "archive_page",
    "archive_page_url",
    "fetched_at",
]


CRYPTOSLATE_ARTICLE_COLUMNS = [
    "url",
    "title",
    "published_date",
    "published_at",
    "section",
    "author",
    "description",
    "asset_tags",
    "sentiment_label",
    "source",
    "fetch_status",
    "fetched_at",
]


CRYPTOSLATE_ARTICLE_BODY_COLUMNS = [
    "url",
    "body_text",
    "body_char_count",
    "body_word_count",
    "body_hash",
    "source",
    "fetch_status",
    "fetched_at",
]


HTML_ACCEPT_HEADER = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
COINDESK_MAX_ARCHIVE_PAGES_PER_YEAR = 100
COINDESK_TERMINAL_DETAIL_STATUSES = {"ok", "http_404"}
CRYPTOSLATE_TERMINAL_DETAIL_STATUSES = {"ok", "http_404", "out_of_range"}
CRYPTOSLATE_PROGRESS_HIDDEN_STATUSES = {"http_404", "out_of_range"}


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
    accept: str = "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "TradingSystem/0.1 sentiment updater",
            "Accept": accept,
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


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "--:--:--"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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


def _delete_parquet_files(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


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


def _cryptoslate_news_url(page: int | None = None) -> str:
    base = config.SentimentConfig.CRYPTOSLATE_NEWS_BASE_URL.rstrip("/")
    if page is None or int(page) <= 1:
        return f"{base}/"
    return f"{base}/page/{int(page)}/"


def _cryptoslate_page_count(payload: bytes) -> int:
    html_text = payload.decode("utf-8", errors="replace")
    counts = [1]
    text = _strip_html(html_text)
    page_text_match = re.search(r"\bPage\s+\d+\s+of\s+(\d+)\b", text, flags=re.IGNORECASE)
    if page_text_match:
        counts.append(int(page_text_match.group(1)))
    for match in re.finditer(r"/news/page/(\d+)/?", html_text, flags=re.IGNORECASE):
        counts.append(int(match.group(1)))
    return max(counts)


def _absolute_url(url: str, *, base_url: str) -> str:
    return urllib.parse.urljoin(base_url, html.unescape(url).strip())


def _is_cryptoslate_article_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    host = parsed.netloc.lower()
    if host and not host.endswith("cryptoslate.com"):
        return False
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) != 1:
        return False
    slug = parts[0].lower()
    if not slug or "." in slug or "-" not in slug:
        return False
    excluded = {
        "about",
        "advertise",
        "companies",
        "contact",
        "cryptoslate-alpha",
        "events",
        "glossary",
        "markets",
        "news",
        "newsletter",
        "podcasts",
        "press-releases",
        "price",
        "research",
        "sitemap",
    }
    return slug not in excluded


def _first_link_text(block: str, *, base_url: str) -> tuple[str | None, str | None]:
    link_pattern = re.compile(
        r"<a\b(?P<attrs>[^>]*)>(?P<body>.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    href_pattern = re.compile(r"""href\s*=\s*['\"](?P<href>[^'\"]+)['\"]""", flags=re.IGNORECASE)
    for match in link_pattern.finditer(block):
        href_match = href_pattern.search(match.group("attrs"))
        if not href_match:
            continue
        url = _absolute_url(href_match.group("href"), base_url=base_url)
        if not _is_cryptoslate_article_url(url):
            continue
        text = _clean_text(match.group("body"))
        return url, text
    return None, None


def _first_heading_text(block: str) -> str | None:
    for tag in ("h1", "h2", "h3", "h4"):
        blocks = _extract_tag_blocks(block, tag)
        for heading in blocks:
            text = _clean_text(heading)
            if text:
                return text
    return None


def _first_paragraph_text(block: str) -> str | None:
    for paragraph in _extract_tag_blocks(block, "p"):
        text = _clean_text(paragraph)
        if text and len(text) >= 25:
            return text
    return None


def _html_title_text(html_text: str) -> str | None:
    blocks = _extract_tag_blocks(html_text, "title")
    if not blocks:
        return None
    return _clean_text(blocks[0])


def _first_datetime_from_html(block: str) -> pd.Timestamp:
    attr_patterns = [
        r"""<time\b[^>]*datetime\s*=\s*['\"](?P<value>[^'\"]+)['\"]""",
        r"""datePublished['\"]?\s*[:=]\s*['\"](?P<value>[^'\"]+)['\"]""",
        r"""published_time['\"]?\s*[:=]\s*['\"](?P<value>[^'\"]+)['\"]""",
    ]
    for pattern in attr_patterns:
        match = re.search(pattern, block, flags=re.IGNORECASE | re.DOTALL)
        if match:
            ts = _parse_timestamp(match.group("value"))
            if not pd.isna(ts):
                return ts

    visible = _strip_html(block)
    month_pattern = (
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
        r"\.?\s+\d{1,2},\s+20\d{2}"
        r"(?:\s+at\s+\d{1,2}:\d{2}\s*(?:am|pm)?\s*(?:GMT|UTC)?)?"
    )
    match = re.search(month_pattern, visible, flags=re.IGNORECASE)
    if match:
        value = match.group(0).replace("Sept.", "Sep").replace(".", "")
        value = re.sub(r"\s+at\s+", " ", value, flags=re.IGNORECASE)
        ts = _parse_timestamp(value)
        if not pd.isna(ts):
            return ts
    return pd.NaT


def _published_date_from_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(ts):
        return pd.NaT
    return pd.to_datetime(ts, errors="coerce").normalize()


def _parse_cryptoslate_archive_html(
    payload: bytes,
    *,
    archive_page: int,
    page_url: str,
    fetched_at: pd.Timestamp,
) -> pd.DataFrame:
    html_text = payload.decode("utf-8", errors="replace")
    rows_by_url: dict[str, dict] = {}

    blocks = _extract_tag_blocks(html_text, "article")
    if not blocks:
        blocks = re.findall(
            r"<div\b[^>]*>(?P<body>.*?</a>.*?</div>)",
            html_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

    for block in blocks:
        url, link_text = _first_link_text(block, base_url=page_url)
        if not url:
            continue
        published_at = _first_datetime_from_html(block)
        row = {
            "url": url,
            "title": _first_heading_text(block) or link_text,
            "published_date": _published_date_from_timestamp(published_at),
            "source": "cryptoslate",
            "section": _cryptoslate_section_from_text(block),
            "description": _first_paragraph_text(block),
            "archive_page": archive_page,
            "archive_page_url": page_url,
            "fetched_at": fetched_at,
        }
        rows_by_url[url] = row

    if not rows_by_url:
        link_pattern = re.compile(
            r"<a\b(?P<attrs>[^>]*)>(?P<body>.*?)</a>",
            flags=re.IGNORECASE | re.DOTALL,
        )
        href_pattern = re.compile(r"""href\s*=\s*['\"](?P<href>[^'\"]+)['\"]""", flags=re.IGNORECASE)
        for match in link_pattern.finditer(html_text):
            href_match = href_pattern.search(match.group("attrs"))
            if not href_match:
                continue
            url = _absolute_url(href_match.group("href"), base_url=page_url)
            if not _is_cryptoslate_article_url(url) or url in rows_by_url:
                continue
            rows_by_url[url] = {
                "url": url,
                "title": _clean_text(match.group("body")),
                "published_date": pd.NaT,
                "source": "cryptoslate",
                "section": None,
                "description": None,
                "archive_page": archive_page,
                "archive_page_url": page_url,
                "fetched_at": fetched_at,
            }

    if not rows_by_url:
        return pd.DataFrame(columns=CRYPTOSLATE_ARCHIVE_COLUMNS)
    return _dedupe_cryptoslate_archive(pd.DataFrame(rows_by_url.values(), columns=CRYPTOSLATE_ARCHIVE_COLUMNS))


def _cryptoslate_section_from_text(value: str) -> str | None:
    text = _strip_html(value)
    known = [
        "Bitcoin",
        "Ethereum",
        "DeFi",
        "Regulation",
        "Markets",
        "Macro",
        "Trading",
        "ETF",
        "Mining",
        "Stablecoins",
        "NFT",
        "AI",
        "Security",
        "Policy",
    ]
    for item in known:
        if re.search(rf"\b{re.escape(item)}\b", text, flags=re.IGNORECASE):
            return item
    return None


def fetch_cryptoslate_archive_index(
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    max_pages: int | None = None,
    append: bool = True,
    output_path: str | Path | None = None,
    stop_after_older_pages: int = 3,
    stop_after_empty_pages: int = 3,
    sleep_seconds: float = 0.5,
    cfg: RSSConfig | None = None,
) -> pd.DataFrame:
    """Fetch CryptoSlate public news listing pages and save URL-level index rows."""
    cfg = cfg or RSSConfig()
    start_ts = _parse_timestamp(start_date) if start_date is not None else pd.NaT
    end_ts = _parse_timestamp(end_date) if end_date is not None else pd.NaT
    path = Path(output_path) if output_path else (
        config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.CRYPTOSLATE_ARCHIVE_OUTPUT_NAME
    )
    existing = pd.read_parquet(path) if append and path.exists() else pd.DataFrame(columns=CRYPTOSLATE_ARCHIVE_COLUMNS)
    existing = _dedupe_cryptoslate_archive(existing)
    completed_pages = _completed_cryptoslate_archive_pages(existing)

    first_url = _cryptoslate_news_url(1)
    fetched_at = pd.Timestamp.utcnow().tz_localize(None)
    first_payload = _fetch_bytes(
        first_url,
        retries=cfg.retries,
        backoff_seconds=cfg.backoff_seconds,
        use_proxy=cfg.use_proxy,
        accept=HTML_ACCEPT_HEADER,
    )
    page_count = _cryptoslate_page_count(first_payload)
    page_limit = int(max_pages) if max_pages else page_count
    print(
        "CryptoSlate index page plan: "
        f"detected_pages={page_count}, configured_limit={max_pages or '-'}, crawl_limit={page_limit}"
    )

    frames: list[pd.DataFrame] = []
    older_pages = 0
    empty_pages = 0

    for page in range(1, page_limit + 1):
        page_url = _cryptoslate_news_url(page)
        if page in completed_pages:
            print(f"Skipping CryptoSlate index page already saved: {page}/{page_limit}")
            continue

        print(f"Fetching CryptoSlate index page: {page}/{page_limit} | {page_url}")
        try:
            payload = first_payload if page == 1 else _fetch_bytes(
                page_url,
                retries=cfg.retries,
                backoff_seconds=cfg.backoff_seconds,
                use_proxy=cfg.use_proxy,
                accept=HTML_ACCEPT_HEADER,
            )
            page_frame = _parse_cryptoslate_archive_html(
                payload,
                archive_page=page,
                page_url=page_url,
                fetched_at=fetched_at,
            )
        except Exception as exc:
            print(f"[WARN] CryptoSlate index page skipped: {page_url} | {exc}")
            _polite_sleep(sleep_seconds)
            continue

        if not page_frame.empty:
            frames.append(page_frame)
            _write_cryptoslate_archive_index(path, existing, frames)
            empty_pages = 0
        else:
            empty_pages += 1
            print(f"[WARN] CryptoSlate index page had no article URLs: {page_url}")
            if empty_pages >= max(1, int(stop_after_empty_pages)):
                print(
                    "Stopping CryptoSlate index fetch: "
                    f"{empty_pages} consecutive empty pages."
                )
                break

        page_dates = pd.to_datetime(page_frame.get("published_date"), errors="coerce").dropna()
        if not pd.isna(start_ts) and not page_dates.empty and page_dates.max() < start_ts.normalize():
            older_pages += 1
            if older_pages >= max(1, int(stop_after_older_pages)):
                print(
                    "Stopping CryptoSlate index fetch: "
                    f"{older_pages} consecutive pages are older than {start_ts.date()}."
                )
                break
        elif not pd.isna(end_ts) and not page_dates.empty and page_dates.min() > end_ts.normalize():
            pass
        else:
            older_pages = 0
        _polite_sleep(sleep_seconds)

    out = _write_cryptoslate_archive_index(path, existing, frames)
    print(f"saved CryptoSlate archive index: {len(out)} rows -> {path}")
    return out


def _cryptoslate_page_fetch_status(html_text: str) -> str:
    text = _strip_html(html_text).lower()
    title = (_html_title_text(html_text) or "").lower()
    gateway_markers = (
        "cloudflare",
        "akamai",
        "fastly",
        "nginx",
        "waf",
    )

    not_found_markers = ("page not found", "not found", "404")
    has_not_found_marker = any(marker in text for marker in not_found_markers)
    title_has_not_found_marker = any(marker in title for marker in not_found_markers)
    not_found_error_context = any(
        marker in text for marker in ("http 404", "error 404", "status code 404")
    ) or any(marker in text for marker in gateway_markers)
    if title_has_not_found_marker or (len(text) < 1500 and has_not_found_marker) or (
        has_not_found_marker and not_found_error_context
    ):
        return "http_404"

    access_markers = ("access denied", "forbidden")
    has_access_marker = any(marker in text for marker in access_markers)
    title_has_access_marker = any(marker in title for marker in access_markers)
    access_error_context = any(
        marker in text
        for marker in ("http 401", "http 403", "error 401", "error 403", "status code 401", "status code 403")
    ) or any(marker in text for marker in gateway_markers)
    if title_has_access_marker or (len(text) < 1500 and has_access_marker) or (
        has_access_marker and access_error_context
    ):
        return "access_denied"

    rate_limit_markers = ("too many requests", "rate limited", "rate limit exceeded")
    has_rate_limit_marker = any(marker in text for marker in rate_limit_markers)
    title_has_rate_limit_marker = any(marker in title for marker in rate_limit_markers)
    short_error_page = len(text) < 1500 and has_rate_limit_marker
    gateway_context = any(
        marker in text for marker in ("http 429", "error 429", "status code 429")
    ) or any(marker in text for marker in gateway_markers)
    if title_has_rate_limit_marker or short_error_page or (has_rate_limit_marker and gateway_context):
        return "rate_limited"
    if "checking your browser" in text and "cloudflare" in text:
        return "cloudflare_challenge"
    return "ok"


def _extract_cryptoslate_asset_tags(*values: str | None) -> str | None:
    text = " ".join(v for v in values if v)
    if not text:
        return None
    symbol_aliases = {
        "BTC": ["BTC", "Bitcoin"],
        "ETH": ["ETH", "Ether", "Ethereum"],
        "SOL": ["SOL", "Solana"],
        "BNB": ["BNB", "Binance Coin", "BNB Chain"],
        "XRP": ["XRP", "Ripple"],
        "DOGE": ["DOGE", "Dogecoin"],
        "LTC": ["LTC", "Litecoin"],
        "LINK": ["LINK", "Chainlink"],
        "TRX": ["TRX", "Tron", "TRON"],
        "ADA": ["ADA", "Cardano"],
        "HYPE": ["HYPE", "Hyperliquid"],
        "XMR": ["XMR", "Monero"],
        "ZEC": ["ZEC", "Zcash"],
        "USDT": ["USDT", "Tether"],
    }
    found = []
    for symbol, aliases in symbol_aliases.items():
        for alias in aliases:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", text, flags=re.IGNORECASE):
                found.append(symbol)
                break
    return json.dumps(found, ensure_ascii=True) if found else None


def _extract_cryptoslate_sentiment_label(*values: str | None) -> str | None:
    text = " ".join(v for v in values if v)
    for label in ("Bullish", "Bearish", "Neutral"):
        if re.search(rf"\b{label}\b", text, flags=re.IGNORECASE):
            return label.lower()
    return None


def parse_cryptoslate_article_metadata(
    html_text: str,
    *,
    url: str,
    fallback_title: str | None = None,
    fallback_published_date=None,
    fetched_at: pd.Timestamp | None = None,
    fetch_status: str = "ok",
) -> dict:
    """Extract CryptoSlate article metadata from a public article page."""
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
    keywords = None

    for obj in candidates:
        title = title or _json_text(obj, "headline", "name")
        published_at = published_at if not pd.isna(published_at) else _parse_timestamp(obj.get("datePublished"))
        section = section or _json_text(obj, "articleSection", "section")
        author = author or _json_text(obj, "author", "creator")
        description = description or _json_text(obj, "description")
        keywords = keywords or _json_text(obj, "keywords")

    title = title or _clean_text(_meta_content(html_text, {"og:title", "twitter:title"}))
    title = title or _clean_text(fallback_title)
    description = description or _clean_text(_meta_content(html_text, {"description", "og:description"}))
    author = author or _clean_text(_meta_content(html_text, {"author"}))
    section = section or _clean_text(_meta_content(html_text, {"article:section"}))
    keywords = keywords or _clean_text(_meta_content(html_text, {"keywords", "news_keywords"}))

    if pd.isna(published_at):
        published_at = _parse_timestamp(_meta_content(html_text, {"article:published_time", "date", "pubdate"}))
    if pd.isna(published_at):
        published_at = _first_datetime_from_html(html_text)

    published_date = _published_date_from_timestamp(published_at)
    if pd.isna(published_date) and fallback_published_date is not None:
        published_date = pd.to_datetime(fallback_published_date, errors="coerce")

    visible_head = _strip_html(html_text[:12000])
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
        "asset_tags": _extract_cryptoslate_asset_tags(title, description, section, keywords, fallback_title),
        "sentiment_label": _extract_cryptoslate_sentiment_label(title, description, keywords, visible_head[:3000]),
        "source": "cryptoslate",
        "fetch_status": status,
        "fetched_at": fetched_at,
    }


def extract_cryptoslate_article_body(html_text: str) -> str | None:
    """Extract CryptoSlate article body text for local offline scoring."""
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


def _cryptoslate_article_body_record(
    html_text: str,
    *,
    url: str,
    fetched_at: pd.Timestamp,
    fetch_status: str,
) -> dict:
    body = extract_cryptoslate_article_body(html_text) if fetch_status == "ok" else None
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
        "source": "cryptoslate",
        "fetch_status": status,
        "fetched_at": fetched_at,
    }


def _request_error_fetch_status(error: Exception) -> str:
    if isinstance(error, HTTPError):
        if error.code == 404:
            return "http_404"
        if error.code == 429:
            return "rate_limited"
        if error.code in {401, 403}:
            return "access_denied"
        return f"http_{error.code}"
    if isinstance(error, URLError):
        return "error: network"
    return _browser_error_fetch_status(error)


def _cryptoslate_article_error_row(row: dict, *, fetched_at: pd.Timestamp, error: Exception) -> dict:
    published_date = pd.to_datetime(row.get("published_date", pd.NaT), errors="coerce")
    return {
        "url": row.get("url"),
        "title": row.get("title"),
        "published_date": published_date,
        "published_at": pd.NaT,
        "section": row.get("section"),
        "author": None,
        "description": row.get("description"),
        "asset_tags": None,
        "sentiment_label": None,
        "source": "cryptoslate",
        "fetch_status": _request_error_fetch_status(error),
        "fetched_at": fetched_at,
    }


def _within_date_range(meta: dict, *, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    ts = pd.to_datetime(meta.get("published_at"), errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(meta.get("published_date"), errors="coerce")
    if pd.isna(ts):
        return True
    if not pd.isna(start_ts) and ts < start_ts:
        return False
    if not pd.isna(end_ts) and ts > end_ts:
        return False
    return True


def _fetch_one_cryptoslate_article(
    row: dict,
    *,
    cfg: RSSConfig,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    save_article_body: bool,
    sleep_seconds: float,
    timeout_seconds: int,
) -> tuple[dict, dict | None]:
    fetched_at = pd.Timestamp.utcnow().tz_localize(None)
    url = str(row.get("url") or "")
    _polite_sleep(sleep_seconds)
    payload = _fetch_bytes(
        url,
        retries=cfg.retries,
        backoff_seconds=cfg.backoff_seconds,
        use_proxy=cfg.use_proxy,
        accept=HTML_ACCEPT_HEADER,
        timeout_seconds=timeout_seconds,
    )
    html_text = payload.decode("utf-8", errors="replace")
    fetch_status = _cryptoslate_page_fetch_status(html_text)
    meta = parse_cryptoslate_article_metadata(
        html_text,
        url=url,
        fallback_title=row.get("title"),
        fallback_published_date=row.get("published_date"),
        fetched_at=fetched_at,
        fetch_status=fetch_status,
    )
    if not _within_date_range(meta, start_ts=start_ts, end_ts=end_ts):
        meta["fetch_status"] = "out_of_range"
        return meta, None
    body = _cryptoslate_article_body_record(
        html_text,
        url=url,
        fetched_at=fetched_at,
        fetch_status=fetch_status,
    ) if save_article_body else None
    return meta, body


def fetch_cryptoslate_article_details(
    archive_index: pd.DataFrame | None = None,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    limit: int | None = None,
    append: bool = True,
    details_path: str | Path | None = None,
    body_path: str | Path | None = None,
    workers: int = 4,
    progress_save_every: int = 100,
    save_article_body: bool = True,
    sleep_seconds: float = 0.2,
    rate_limit_threshold: int = 2,
    rate_limit_sleep_multiplier: float = 1.8,
    max_sleep_seconds: float = 5.0,
    article_timeout_seconds: int = 90,
    cfg: RSSConfig | None = None,
) -> pd.DataFrame:
    """Fetch CryptoSlate article-page metadata/body text over public HTTP."""
    cfg = cfg or RSSConfig()
    start_ts = _parse_timestamp(start_date) if start_date is not None else pd.NaT
    end_ts = _parse_timestamp(end_date) if end_date is not None else pd.NaT
    if archive_index is None:
        archive_path = config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.CRYPTOSLATE_ARCHIVE_OUTPUT_NAME
        if not archive_path.exists():
            raise FileNotFoundError(f"CryptoSlate archive index not found: {archive_path}")
        archive_index = pd.read_parquet(archive_path)
    archive_index = _dedupe_cryptoslate_archive(archive_index)

    out_path = Path(details_path) if details_path else (
        config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.CRYPTOSLATE_ARTICLE_DETAIL_OUTPUT_NAME
    )
    body_out_path = Path(body_path) if body_path else (
        config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.CRYPTOSLATE_ARTICLE_BODY_OUTPUT_NAME
    )
    detail_parts_dir = _coindesk_parts_dir(out_path)
    body_parts_dir = _coindesk_parts_dir(body_out_path)

    if append:
        existing = _load_coindesk_table_with_parts(
            out_path,
            detail_parts_dir,
            CRYPTOSLATE_ARTICLE_COLUMNS,
            _dedupe_cryptoslate_article_details,
        )
        existing_bodies = _load_coindesk_table_with_parts(
            body_out_path,
            body_parts_dir,
            CRYPTOSLATE_ARTICLE_BODY_COLUMNS,
            _dedupe_cryptoslate_article_bodies,
        ) if save_article_body else pd.DataFrame(columns=CRYPTOSLATE_ARTICLE_BODY_COLUMNS)
    else:
        existing = pd.DataFrame(columns=CRYPTOSLATE_ARTICLE_COLUMNS)
        existing_bodies = pd.DataFrame(columns=CRYPTOSLATE_ARTICLE_BODY_COLUMNS)

    done_urls = _completed_cryptoslate_article_urls(existing, existing_bodies, save_article_body=save_article_body)
    existing_rows = len(existing)
    completed_rows = len(done_urls)
    retry_existing_urls: set[str] = set()
    existing_status_by_url: dict[str, str] = {}
    if existing is not None and not existing.empty and "url" in existing.columns:
        retry_existing_urls = set(existing.loc[~existing["url"].isin(done_urls), "url"].dropna())
        if "fetch_status" in existing.columns:
            existing_status_by_url = {
                str(row.url): str(row.fetch_status)
                for row in existing[["url", "fetch_status"]].dropna(subset=["url"]).itertuples(index=False)
            }
    rows_df = archive_index.dropna(subset=["url"]).copy()
    if "published_date" in rows_df.columns:
        row_dates = pd.to_datetime(rows_df["published_date"], errors="coerce")
        if not pd.isna(start_ts):
            rows_df = rows_df[row_dates.isna() | (row_dates >= start_ts.normalize())].copy()
            row_dates = pd.to_datetime(rows_df["published_date"], errors="coerce")
        if not pd.isna(end_ts):
            rows_df = rows_df[row_dates.isna() | (row_dates <= end_ts.normalize())].copy()
    if done_urls:
        rows_df = rows_df[~rows_df["url"].isin(done_urls)].copy()
    if limit is not None:
        rows_df = rows_df.head(int(limit)).copy()

    print(
        "CryptoSlate article detail candidates: "
        f"{len(rows_df)} | existing_rows={existing_rows} | completed_urls={completed_rows} | "
        f"retry_urls={len(retry_existing_urls)} | limit={limit if limit is not None else '-'}"
    )
    if rows_df.empty:
        return _dedupe_cryptoslate_article_details(existing)

    records: list[dict] = []
    body_records: list[dict] = []
    pending_since_write = 0
    rows = rows_df.to_dict("records")
    max_workers = max(1, int(workers))
    base_sleep_seconds = max(0.0, float(sleep_seconds))
    adaptive_sleep_seconds = base_sleep_seconds
    consecutive_rate_limited = 0
    clean_batches = 0
    completed_count = 0
    status_counts: dict[str, int] = {}
    latest_status_by_url = dict(existing_status_by_url)
    started_at = time.monotonic()
    progress_log_every = 50

    def flush_parts() -> None:
        nonlocal pending_since_write
        if not records:
            return
        detail_count = len(records)
        body_count = len(body_records)
        detail_part = _write_coindesk_part(detail_parts_dir, records, CRYPTOSLATE_ARTICLE_COLUMNS, "details")
        body_part = None
        if save_article_body and body_records:
            body_part = _write_coindesk_part(body_parts_dir, body_records, CRYPTOSLATE_ARTICLE_BODY_COLUMNS, "bodies")
        print(
            "CryptoSlate checkpoint saved: "
            f"details={detail_count} -> {detail_part.name if detail_part else '-'} | "
            f"bodies={body_count} -> {body_part.name if body_part else '-'}"
        )
        records.clear()
        body_records.clear()
        pending_since_write = 0

    def log_progress(reason: str) -> None:
        elapsed = max(0.001, time.monotonic() - started_at)
        rate_per_min = completed_count / elapsed * 60.0
        remaining = max(0, len(rows) - completed_count)
        eta_seconds = remaining / (rate_per_min / 60.0) if rate_per_min > 0 else None
        status_summary = ", ".join(
            f"{status}={count}"
            for status, count in sorted(status_counts.items(), key=lambda item: (-item[1], item[0]))
            if status not in CRYPTOSLATE_PROGRESS_HIDDEN_STATUSES
        ) or "-"
        total_status_counts: dict[str, int] = {}
        for status in latest_status_by_url.values():
            total_status_counts[status] = total_status_counts.get(status, 0) + 1
        total_status_summary = ", ".join(
            f"{status}={count}"
            for status, count in sorted(total_status_counts.items(), key=lambda item: (-item[1], item[0]))
            if status not in CRYPTOSLATE_PROGRESS_HIDDEN_STATUSES
        ) or "-"
        retry_remaining = sum(
            1
            for url in retry_existing_urls
            if latest_status_by_url.get(url) not in CRYPTOSLATE_TERMINAL_DETAIL_STATUSES
        )
        print(
            f"CryptoSlate progress ({reason}): {completed_count}/{len(rows)} "
            f"({completed_count / max(1, len(rows)):.1%}) | "
            f"elapsed={_format_duration(elapsed)} | eta={_format_duration(eta_seconds)} | "
            f"rate={rate_per_min:.1f}/min | sleep={adaptive_sleep_seconds:.2f}s | "
            f"retry_left={retry_remaining} | pending_checkpoint={pending_since_write} | "
            f"run_statuses: {status_summary} | total_est: {total_status_summary}"
        )

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_start in range(0, len(rows), max_workers):
                batch = rows[batch_start:batch_start + max_workers]
                futures = {
                    executor.submit(
                        _fetch_one_cryptoslate_article,
                        row,
                        cfg=cfg,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        save_article_body=save_article_body,
                        sleep_seconds=adaptive_sleep_seconds,
                        timeout_seconds=article_timeout_seconds,
                    ): row
                    for row in batch
                }
                rate_limited_in_batch = 0

                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        meta, body = future.result()
                    except Exception as exc:
                        fetched_at = pd.Timestamp.utcnow().tz_localize(None)
                        meta = _cryptoslate_article_error_row(row, fetched_at=fetched_at, error=exc)
                        body = None
                        print(f"[WARN] CryptoSlate article skipped: {row.get('url')} | {exc}")

                    status = str(meta.get("fetch_status") or "")
                    status_counts[status] = status_counts.get(status, 0) + 1
                    meta_url = meta.get("url")
                    if meta_url:
                        latest_status_by_url[str(meta_url)] = status
                    if status == "rate_limited":
                        rate_limited_in_batch += 1
                        consecutive_rate_limited += 1
                    elif status in CRYPTOSLATE_PROGRESS_HIDDEN_STATUSES:
                        consecutive_rate_limited = 0
                        print(
                            "CryptoSlate terminal article status: "
                            f"{status} | {meta.get('url')}"
                        )
                    elif status not in {"ok", "out_of_range"}:
                        consecutive_rate_limited = 0
                        print(
                            "[WARN] CryptoSlate non-ok article status: "
                            f"{status} | {meta.get('url')}"
                        )
                    else:
                        consecutive_rate_limited = 0

                    records.append(meta)
                    if body is not None:
                        body_records.append(body)
                    pending_since_write += 1
                    completed_count += 1

                    if completed_count % progress_log_every == 0:
                        log_progress("interval")
                    if pending_since_write >= max(1, int(progress_save_every)):
                        flush_parts()

                threshold = max(1, int(rate_limit_threshold))
                if 0 < rate_limited_in_batch < threshold:
                    print(
                        "[WARN] CryptoSlate rate limit seen below threshold: "
                        f"batch_rate_limited={rate_limited_in_batch}, "
                        f"threshold={threshold}, sleep={adaptive_sleep_seconds:.2f}s"
                    )
                if rate_limited_in_batch >= threshold or consecutive_rate_limited >= threshold:
                    old_sleep = adaptive_sleep_seconds
                    adaptive_sleep_seconds = min(
                        max(0.0, float(max_sleep_seconds)),
                        max(
                            adaptive_sleep_seconds * float(rate_limit_sleep_multiplier),
                            adaptive_sleep_seconds + max(0.5, base_sleep_seconds),
                        ),
                    )
                    clean_batches = 0
                    consecutive_rate_limited = 0
                    print(
                        "[WARN] CryptoSlate rate limit pressure detected: "
                        f"batch_rate_limited={rate_limited_in_batch}, "
                        f"sleep {old_sleep:.2f}s -> {adaptive_sleep_seconds:.2f}s"
                    )
                    log_progress("rate_limit")
                    _polite_sleep(adaptive_sleep_seconds)
                elif rate_limited_in_batch == 0:
                    clean_batches += 1
                    if clean_batches >= 10 and adaptive_sleep_seconds > base_sleep_seconds:
                        old_sleep = adaptive_sleep_seconds
                        adaptive_sleep_seconds = max(base_sleep_seconds, adaptive_sleep_seconds * 0.8)
                        clean_batches = 0
                        print(
                            "CryptoSlate rate limit pressure cooled: "
                            f"sleep {old_sleep:.2f}s -> {adaptive_sleep_seconds:.2f}s"
                        )
                        log_progress("cooldown")
    finally:
        flush_parts()
        log_progress("final")

    out = _load_coindesk_table_with_parts(
        out_path,
        detail_parts_dir,
        CRYPTOSLATE_ARTICLE_COLUMNS,
        _dedupe_cryptoslate_article_details,
    )
    if save_article_body:
        _load_coindesk_table_with_parts(
            body_out_path,
            body_parts_dir,
            CRYPTOSLATE_ARTICLE_BODY_COLUMNS,
            _dedupe_cryptoslate_article_bodies,
        )
    print(f"saved CryptoSlate article details checkpoint parts -> {detail_parts_dir}")
    return out


def compact_cryptoslate_article_outputs(
    *,
    details_path: str | Path | None = None,
    body_path: str | Path | None = None,
    delete_parts: bool = False,
) -> pd.DataFrame:
    details = Path(details_path) if details_path else (
        config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.CRYPTOSLATE_ARTICLE_DETAIL_OUTPUT_NAME
    )
    detail_parts = _coindesk_parts_dir(details)
    detail_main = pd.read_parquet(details) if details.exists() else pd.DataFrame(columns=CRYPTOSLATE_ARTICLE_COLUMNS)
    detail_part_files = sorted(detail_parts.glob("*.parquet")) if detail_parts.exists() else []
    detail_parts_df = (
        pd.concat([pd.read_parquet(p) for p in detail_part_files], ignore_index=True)
        if detail_part_files else pd.DataFrame(columns=CRYPTOSLATE_ARTICLE_COLUMNS)
    )
    _print_compact_part_summary(
        label="CryptoSlate article details",
        main_df=detail_main,
        parts_df=detail_parts_df,
        part_files=detail_part_files,
    )
    detail_frames = [detail_main] + ([detail_parts_df] if not detail_parts_df.empty else [])
    detail_out = _dedupe_cryptoslate_article_details(
        pd.concat(detail_frames, ignore_index=True)
    )
    _write_parquet_atomic(details, detail_out)
    if delete_parts:
        _delete_parquet_files(detail_part_files)

    if body_path is not None:
        body = Path(body_path)
    else:
        body = config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.CRYPTOSLATE_ARTICLE_BODY_OUTPUT_NAME
    body_parts = _coindesk_parts_dir(body)
    body_main = pd.read_parquet(body) if body.exists() else pd.DataFrame(columns=CRYPTOSLATE_ARTICLE_BODY_COLUMNS)
    body_part_files = sorted(body_parts.glob("*.parquet")) if body_parts.exists() else []
    body_parts_df = (
        pd.concat([pd.read_parquet(p) for p in body_part_files], ignore_index=True)
        if body_part_files else pd.DataFrame(columns=CRYPTOSLATE_ARTICLE_BODY_COLUMNS)
    )
    _print_compact_part_summary(
        label="CryptoSlate article bodies",
        main_df=body_main,
        parts_df=body_parts_df,
        part_files=body_part_files,
    )
    body_frames = [body_main] + ([body_parts_df] if not body_parts_df.empty else [])
    body_out = _dedupe_cryptoslate_article_bodies(
        pd.concat(body_frames, ignore_index=True)
    )
    _write_parquet_atomic(body, body_out)
    if delete_parts:
        _delete_parquet_files(body_part_files)

    print(f"compacted CryptoSlate article details: {len(detail_out)} rows -> {details}")
    print(f"compacted CryptoSlate article bodies: {len(body_out)} rows -> {body}")

    return detail_out


def _print_compact_part_summary(
    *,
    label: str,
    main_df: pd.DataFrame,
    parts_df: pd.DataFrame,
    part_files: list[Path],
) -> None:
    main_urls = set(main_df["url"].dropna()) if main_df is not None and "url" in main_df.columns else set()
    part_urls = set(parts_df["url"].dropna()) if parts_df is not None and "url" in parts_df.columns else set()
    new_urls = part_urls - main_urls
    overlap_urls = part_urls & main_urls
    print(
        f"compact source summary - {label}: "
        f"part_files={len(part_files)} | part_rows={len(parts_df)} | "
        f"part_unique_urls={len(part_urls)} | new_urls={len(new_urls)} | "
        f"overlap_urls={len(overlap_urls)} | existing_rows={len(main_df)}"
    )


def _completed_cryptoslate_archive_pages(df: pd.DataFrame) -> set[int]:
    if df is None or df.empty or "archive_page" not in df.columns:
        return set()
    pages = pd.to_numeric(df["archive_page"], errors="coerce").dropna()
    return {int(page) for page in pages}


def _completed_cryptoslate_article_urls(
    details: pd.DataFrame,
    bodies: pd.DataFrame,
    *,
    save_article_body: bool,
) -> set[str]:
    if details is None or details.empty or "fetch_status" not in details.columns:
        return set()
    terminal_details = details[
        details["fetch_status"].isin(CRYPTOSLATE_TERMINAL_DETAIL_STATUSES)
    ][["url", "fetch_status"]].dropna(subset=["url"])
    if terminal_details.empty:
        return set()

    no_body_needed = set(
        terminal_details.loc[
            terminal_details["fetch_status"].isin({"http_404", "out_of_range"}),
            "url",
        ]
    )
    ok_detail_urls = set(terminal_details.loc[terminal_details["fetch_status"].eq("ok"), "url"])
    if not save_article_body:
        return no_body_needed | ok_detail_urls
    if bodies is None or bodies.empty or "fetch_status" not in bodies.columns:
        return no_body_needed
    ok_body_urls = set(bodies.loc[bodies["fetch_status"].eq("ok"), "url"].dropna())
    return no_body_needed | (ok_detail_urls & ok_body_urls)


def _write_cryptoslate_archive_index(
    path: Path,
    existing: pd.DataFrame,
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    parts = [existing] + [frame for frame in frames if frame is not None and not frame.empty]
    out = _dedupe_cryptoslate_archive(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame(columns=CRYPTOSLATE_ARCHIVE_COLUMNS)
    _write_parquet_atomic(path, out)
    return out


def _dedupe_cryptoslate_archive(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=CRYPTOSLATE_ARCHIVE_COLUMNS)
    out = df.copy()
    for col in CRYPTOSLATE_ARCHIVE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["published_date"] = pd.to_datetime(out["published_date"], errors="coerce")
    out["archive_page"] = pd.to_numeric(out["archive_page"], errors="coerce").astype("Int64")
    out["fetched_at"] = pd.to_datetime(out["fetched_at"], errors="coerce")
    out = out.drop_duplicates(subset=["url"], keep="last")
    return out[CRYPTOSLATE_ARCHIVE_COLUMNS].sort_values(
        ["archive_page", "published_date", "url"],
        na_position="last",
    ).reset_index(drop=True)


def _dedupe_cryptoslate_article_details(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=CRYPTOSLATE_ARTICLE_COLUMNS)
    out = df.copy()
    for col in CRYPTOSLATE_ARTICLE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["published_date"] = pd.to_datetime(out["published_date"], errors="coerce")
    out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce")
    out["fetched_at"] = pd.to_datetime(out["fetched_at"], errors="coerce")
    out = out.drop_duplicates(subset=["url"], keep="last")
    return out[CRYPTOSLATE_ARTICLE_COLUMNS].sort_values(
        ["published_at", "published_date", "url"],
        na_position="last",
    ).reset_index(drop=True)


def _dedupe_cryptoslate_article_bodies(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=CRYPTOSLATE_ARTICLE_BODY_COLUMNS)
    out = df.copy()
    for col in CRYPTOSLATE_ARTICLE_BODY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["fetched_at"] = pd.to_datetime(out["fetched_at"], errors="coerce")
    out["body_char_count"] = pd.to_numeric(out["body_char_count"], errors="coerce").fillna(0).astype(int)
    out["body_word_count"] = pd.to_numeric(out["body_word_count"], errors="coerce").fillna(0).astype(int)
    out = out.drop_duplicates(subset=["url"], keep="last")
    return out[CRYPTOSLATE_ARTICLE_BODY_COLUMNS].sort_values(
        ["fetched_at", "url"],
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
