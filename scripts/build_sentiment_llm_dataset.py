"""Build a merged news dataset for LLM sentiment / asset labeling.

Inputs are raw article details + body parquet files from CoinDesk and
CryptoSlate. The output intentionally drops crawler bookkeeping columns such as
fetch timestamps, body counts and hashes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

import config


OUTPUT_NAME = "merged_news_for_llm.parquet"
WRITE_JSONL = False
JSONL_OUTPUT_NAME = "merged_news_for_llm.jsonl"

OUTPUT_COLUMNS = [
    "source",
    "url",
    "title",
    "published_at",
    "published_date",
    "section",
    "author",
    "description",
    "asset_type",
    "sentiment_label",
    "article_text",
]

BASE_SYMBOL_ALIASES = {
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


def _label_symbols_from_config() -> list[str]:
    symbols = getattr(config.TargetConfig, "DIVERSIFIED_10_COINS", None)
    if not symbols:
        symbols = getattr(config.TargetConfig, "COINS", [])
    out = []
    for item in symbols:
        base = str(item).split("/", 1)[0].upper()
        if base and base not in out:
            out.append(base)
    return out


LABEL_SYMBOLS = _label_symbols_from_config()
SYMBOL_ALIASES = {
    symbol: BASE_SYMBOL_ALIASES.get(symbol, [symbol])
    for symbol in LABEL_SYMBOLS
}


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input parquet: {path}")
    return pd.read_parquet(path)


def _normalize_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _normalize_body_text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_asset_tags(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        raw = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = re.split(r"[,;\s]+", text)
    out = []
    for item in raw:
        symbol = str(item).strip().upper()
        if symbol in SYMBOL_ALIASES and symbol not in out:
            out.append(symbol)
    return out


def _infer_asset_type(row: pd.Series) -> str | None:
    found = _parse_asset_tags(row.get("asset_tags"))
    # Keep this weak pre-label cheap and explainable. Full body text is left
    # for the downstream LLM/CryptoBERT labeling pass.
    searchable = " ".join(
        str(row.get(col) or "")
        for col in ["title", "description", "section"]
    )
    for symbol, aliases in SYMBOL_ALIASES.items():
        for alias in aliases:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", searchable, flags=re.IGNORECASE):
                if symbol not in found:
                    found.append(symbol)
                break
    return json.dumps(found, ensure_ascii=True) if found else None


def _load_source(
    *,
    source: str,
    details_path: Path,
    bodies_path: Path,
) -> pd.DataFrame:
    details = _read_parquet(details_path).copy()
    bodies = _read_parquet(bodies_path).copy()

    details = details.loc[details["fetch_status"].eq("ok")].copy()
    bodies = bodies.loc[bodies["fetch_status"].eq("ok")].copy()

    details = details.drop_duplicates(subset=["url"], keep="last")
    bodies = bodies.drop_duplicates(subset=["url"], keep="last")
    merged = details.merge(
        bodies[["url", "body_text"]],
        on="url",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.rename(columns={"body_text": "article_text"})
    merged["source"] = source

    if "asset_tags" not in merged.columns:
        merged["asset_tags"] = pd.NA
    if "sentiment_label" not in merged.columns:
        merged["sentiment_label"] = pd.NA

    merged["published_at"] = pd.to_datetime(merged["published_at"], errors="coerce")
    merged["published_date"] = pd.to_datetime(merged["published_date"], errors="coerce")

    for col in ["title", "section", "author", "description", "sentiment_label"]:
        if col not in merged.columns:
            merged[col] = pd.NA
        merged[col] = merged[col].map(_normalize_text)
    merged["article_text"] = merged["article_text"].map(_normalize_body_text)

    merged["asset_type"] = merged.apply(_infer_asset_type, axis=1)
    return merged[OUTPUT_COLUMNS]


def build_dataset() -> pd.DataFrame:
    config.init_directories()
    raw = config.PathConfig.RAW_SENTIMENT
    out_path = config.PathConfig.PROCESSED_SENTIMENT / OUTPUT_NAME
    print(f"asset_type label symbols: {LABEL_SYMBOLS}")

    frames = [
        _load_source(
            source="coindesk",
            details_path=raw / config.SentimentConfig.COINDESK_ARTICLE_DETAIL_OUTPUT_NAME,
            bodies_path=raw / config.SentimentConfig.COINDESK_ARTICLE_BODY_OUTPUT_NAME,
        ),
        _load_source(
            source="cryptoslate",
            details_path=raw / config.SentimentConfig.CRYPTOSLATE_ARTICLE_DETAIL_OUTPUT_NAME,
            bodies_path=raw / config.SentimentConfig.CRYPTOSLATE_ARTICLE_BODY_OUTPUT_NAME,
        ),
    ]
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["url"], keep="last")
    out = out.sort_values(["published_at", "published_date", "source", "url"], na_position="last")
    out = out[OUTPUT_COLUMNS].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    out.to_parquet(tmp_path, engine="pyarrow", compression="zstd", index=False)
    tmp_path.replace(out_path)

    if WRITE_JSONL:
        jsonl_path = out_path.with_name(JSONL_OUTPUT_NAME)
        out.to_json(jsonl_path, orient="records", lines=True, force_ascii=False, date_format="iso")

    print(f"saved merged sentiment LLM dataset: {len(out)} rows -> {out_path}")
    print("rows by source:")
    print(out["source"].value_counts(dropna=False).to_string())
    print("non-null asset_type:", int(out["asset_type"].notna().sum()))
    print("non-null sentiment_label:", int(out["sentiment_label"].notna().sum()))
    return out


def main() -> None:
    build_dataset()


if __name__ == "__main__":
    main()
