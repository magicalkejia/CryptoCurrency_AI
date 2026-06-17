"""Compact CoinDesk article checkpoint parts into main parquet files."""
from __future__ import annotations

import config
from etl.sentiment_updater import compact_coindesk_article_outputs


def main() -> None:
    config.init_directories()
    details_path = config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.COINDESK_ARTICLE_DETAIL_OUTPUT_NAME
    body_path = config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.COINDESK_ARTICLE_BODY_OUTPUT_NAME
    out = compact_coindesk_article_outputs(
        details_path=details_path,
        body_path=body_path,
        delete_parts=True,
    )
    print(f"compacted CoinDesk article details: {len(out)} rows -> {details_path}")
    print(f"compacted CoinDesk article bodies -> {body_path}")


if __name__ == "__main__":
    main()
