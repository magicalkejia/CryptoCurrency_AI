"""Fetch CryptoSlate public news archive metadata and optional body text.

This is the non-login counterpart to scripts/fetch_coindesk_archive.py.
It uses normal HTTP requests, so it should be faster and less fragile than the
CoinDesk logged-in/browser flow. Edit constants here and run with taskipy.
"""
from __future__ import annotations

import pandas as pd

import config
from etl.sentiment_updater import (
    RSSConfig,
    compact_cryptoslate_article_outputs,
    fetch_cryptoslate_archive_index,
    fetch_cryptoslate_article_details,
)


# Recommended first pass: recent enough to avoid source/time artifacts, but not
# so broad that body-text NLP becomes expensive before we validate usefulness.
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

# Keep this script friendly to run from taskipy / IDE. Edit constants instead
# of passing a long command line.
FETCH_ARCHIVE_INDEX = True
FETCH_ARCHIVE_INDEX_IF_MISSING = True
FETCH_ARTICLE_DETAILS = True
SAVE_ARTICLE_BODY = True
COMPACT_AT_END = True
COMPACT_ONLY = False

# Conservative guard for the first large run. If CryptoSlate exposes list-page
# dates, the date-stop guard below should stop earlier; if not, this prevents
# accidentally walking the whole site before we validate coverage.
MAX_ARCHIVE_PAGES = 1500
STOP_AFTER_OLDER_PAGES = 1
STOP_AFTER_EMPTY_PAGES = 1
INDEX_SLEEP_SECONDS = 0.5

# First network test cap. Set to None after confirming pagination / body
# extraction on a few hundred rows.
ARTICLE_DETAIL_LIMIT = None
ARTICLE_WORKERS = 4
ARTICLE_SLEEP_SECONDS = 0.3
ARTICLE_TIMEOUT_SECONDS = 60
RATE_LIMIT_THRESHOLD = 1
RATE_LIMIT_SLEEP_MULTIPLIER = 2.0
MAX_ARTICLE_SLEEP_SECONDS = 8.0
PROGRESS_SAVE_EVERY = 30


def main() -> None:
    config.init_directories()
    cfg = RSSConfig()

    if COMPACT_ONLY:
        compact_cryptoslate_article_outputs(delete_parts=True)
        return

    archive = None
    archive_path = config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.CRYPTOSLATE_ARCHIVE_OUTPUT_NAME

    if FETCH_ARCHIVE_INDEX or (FETCH_ARCHIVE_INDEX_IF_MISSING and not archive_path.exists()):
        archive = fetch_cryptoslate_archive_index(
            start_date=START_DATE,
            end_date=END_DATE,
            max_pages=MAX_ARCHIVE_PAGES,
            append=True,
            stop_after_older_pages=STOP_AFTER_OLDER_PAGES,
            stop_after_empty_pages=STOP_AFTER_EMPTY_PAGES,
            sleep_seconds=INDEX_SLEEP_SECONDS,
            cfg=cfg,
        )

    if FETCH_ARTICLE_DETAILS:
        fetch_cryptoslate_article_details(
            archive_index=archive,
            start_date=START_DATE,
            end_date=END_DATE,
            limit=ARTICLE_DETAIL_LIMIT,
            append=True,
            workers=ARTICLE_WORKERS,
            progress_save_every=PROGRESS_SAVE_EVERY,
            save_article_body=SAVE_ARTICLE_BODY,
            sleep_seconds=ARTICLE_SLEEP_SECONDS,
            article_timeout_seconds=ARTICLE_TIMEOUT_SECONDS,
            rate_limit_threshold=RATE_LIMIT_THRESHOLD,
            rate_limit_sleep_multiplier=RATE_LIMIT_SLEEP_MULTIPLIER,
            max_sleep_seconds=MAX_ARTICLE_SLEEP_SECONDS,
            cfg=cfg,
        )

    if COMPACT_AT_END:
        compact_cryptoslate_article_outputs(delete_parts=True)

    print(f"CryptoSlate archive task finished at {pd.Timestamp.utcnow()}")


if __name__ == "__main__":
    main()
