"""Compact CryptoSlate article checkpoint parts into main parquet files."""
from __future__ import annotations

import config
from etl.sentiment_updater import compact_cryptoslate_article_outputs


def main() -> None:
    config.init_directories()
    details_path = config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.CRYPTOSLATE_ARTICLE_DETAIL_OUTPUT_NAME
    body_path = config.PathConfig.RAW_SENTIMENT / config.SentimentConfig.CRYPTOSLATE_ARTICLE_BODY_OUTPUT_NAME
    compact_cryptoslate_article_outputs(
        details_path=details_path,
        body_path=body_path,
        delete_parts=True,
    )


if __name__ == "__main__":
    main()
