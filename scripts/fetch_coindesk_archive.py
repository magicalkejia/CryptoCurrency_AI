"""Fetch CoinDesk archive URLs, article metadata and optional body text.

Workflow:
    1. Fetch public archive index pages: url, title, published_date, source.
    2. Optionally visit article pages with a logged-in Playwright browser
       profile and extract exact published_at / section / author metadata.
    3. Optionally store article body text in a separate local parquet table.

Body text is intentionally kept separate from metadata so it can be deleted
after downstream FinBERT/CryptoBERT feature extraction.
"""
from __future__ import annotations

import pandas as pd

import config
from etl.sentiment_updater import (
    RSSConfig,
    fetch_coindesk_archive_index,
    fetch_coindesk_archive_index_with_browser,
    fetch_coindesk_article_details,
)


START_YEAR = 2021
END_YEAR = pd.Timestamp.utcnow().year

# Keep this script friendly to run from taskipy / IDE. Edit these constants
# instead of passing a long command line.
# Rebuild once after pagination support was added. Set back to False after the
# full index has been refreshed locally.
FETCH_ARCHIVE_INDEX = False
FETCH_ARCHIVE_INDEX_IF_MISSING = True
FETCH_ARTICLE_DETAILS = True
SAVE_ARTICLE_BODY = True
FETCH_ARCHIVE_INDEX_WITH_BROWSER = True

# None means fetch all remaining archive URLs from 2021-01 through now.
ARTICLE_DETAIL_LIMIT = None
# Number of article records per checkpoint part. Keep this large enough to
# avoid thousands of tiny parquet files; reruns still skip ok records from
# existing main/part parquet files.
PROGRESS_SAVE_EVERY = 10
MAX_CONSECUTIVE_NON_OK = 20

# Preferred logged-in mode: connect to a normal Chrome you launched yourself
# with --remote-debugging-port=9222. This avoids Google OAuth blocking
# Playwright's automated browser during login.
USE_EXISTING_CHROME_CDP = True
CHROME_CDP_URL = "http://127.0.0.1:9222"

# Fallback logged-in browser mode. First run opens a visible browser and pauses
# so you can log in to CoinDesk. Google OAuth may reject this browser; use CDP
# mode above when logging in with Google.
USE_LOGGED_IN_BROWSER = False
LOGIN_FIRST = True
BROWSER_HEADLESS = False

# If Chrome is not installed, try "msedge" or set this to None after running:
#     python -m playwright install chromium
BROWSER_CHANNEL = "chrome"

# Be polite and reduce account / site risk. If CoinDesk starts redirecting or
# showing rate-limit behavior, keep this conservative.
SLEEP_SECONDS = 1.5

# Performance knobs. CoinDesk pages often keep analytics/ad requests alive, so
# waiting for networkidle can cost a fixed 8s per article. Metadata/body are
# usually available after domcontentloaded plus a short settle wait.
PAGE_SETTLE_SECONDS = 1.0
WAIT_FOR_NETWORK_IDLE = False
NETWORK_IDLE_TIMEOUT_MS = 1500
BLOCK_HEAVY_RESOURCES = True
BROWSER_PAGE_WORKERS = 1


def main() -> None:
    config.init_directories()
    cfg = RSSConfig()
    archive = None
    archive_path = config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.COINDESK_ARCHIVE_OUTPUT_NAME

    if FETCH_ARCHIVE_INDEX or (FETCH_ARCHIVE_INDEX_IF_MISSING and not archive_path.exists()):
        if FETCH_ARCHIVE_INDEX_WITH_BROWSER and USE_EXISTING_CHROME_CDP:
            archive = fetch_coindesk_archive_index_with_browser(
                start_year=START_YEAR,
                end_year=END_YEAR,
                append=True,
                cdp_url=CHROME_CDP_URL,
                page_settle_seconds=PAGE_SETTLE_SECONDS,
                wait_for_network_idle=WAIT_FOR_NETWORK_IDLE,
                network_idle_timeout_ms=NETWORK_IDLE_TIMEOUT_MS,
                block_heavy_resources=BLOCK_HEAVY_RESOURCES,
            )
        else:
            archive = fetch_coindesk_archive_index(
                start_year=START_YEAR,
                end_year=END_YEAR,
                append=True,
                cfg=cfg,
            )

    if FETCH_ARTICLE_DETAILS:
        fetch_coindesk_article_details(
            archive_index=archive,
            limit=ARTICLE_DETAIL_LIMIT,
            append=True,
            use_existing_browser_cdp=USE_EXISTING_CHROME_CDP,
            cdp_url=CHROME_CDP_URL,
            use_browser=USE_LOGGED_IN_BROWSER,
            browser_headless=BROWSER_HEADLESS,
            browser_channel=BROWSER_CHANNEL,
            login_first=LOGIN_FIRST,
            sleep_seconds=SLEEP_SECONDS,
            page_settle_seconds=PAGE_SETTLE_SECONDS,
            wait_for_network_idle=WAIT_FOR_NETWORK_IDLE,
            network_idle_timeout_ms=NETWORK_IDLE_TIMEOUT_MS,
            block_heavy_resources=BLOCK_HEAVY_RESOURCES,
            browser_page_workers=BROWSER_PAGE_WORKERS,
            progress_save_every=PROGRESS_SAVE_EVERY,
            max_consecutive_non_ok=MAX_CONSECUTIVE_NON_OK,
            save_article_body=SAVE_ARTICLE_BODY,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
