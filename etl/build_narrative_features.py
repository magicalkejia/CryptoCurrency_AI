"""
etl/build_narrative_features.py
===============================
Stage 2 of the narrative modality: turn per-title CryptoBERT scores into a PIT-safe,
PER-SYMBOL narrative factor on a regular 4h grid, written to
data_storage/factors/sentiment/narrative_features.parquet.

Design (all constants are structural and frozen, not tuned):
  * Per-symbol attribution comes from the `coin` column (mapped to our tickers). A
    market-wide factor would be common-mode and demeaned out of the neutral book, so we
    build a cross-sectional (per-symbol) factor instead.
  * News are binned to 4h RIGHT-edges: a bin labelled T contains news with timestamp in
    [T-4h, T), i.e. strictly before T. Combined with the 1-minute decision offset, the
    asof-merge in narrative_loader is point-in-time safe with no future leakage.
  * Three causal features per (symbol, 4h):
      narr_sent_ewm          EWMA (half-life 3d, adjust=False) of per-bin mean sentiment
                             (empty bins contribute 0 -> the factor decays toward neutral
                             when there is no recent news; naturally handles sparse/absent
                             coverage such as TRX=0 or thin BNB).
      narr_news_count_7d_log log1p of the trailing-7d news count (attention; no
                             normalization -> no look-ahead).
      narr_sent_mom          short minus long sentiment EWMA (1d vs 7d) -> sentiment momentum.

The function `build_narrative_features` is pure pandas and is unit-tested offline; the CLI
wrapper handles I/O (parquet on the user's machine, csv supported for offline testing).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Coin Type (lower-cased) -> base ticker. Ripple and XRP both map to XRP. TRX has no rows
# in the current dataset, so it simply never appears -> downstream fills it with 0 (neutral).
COIN_TO_TICKER = {
    # full names (legacy CryptoDataSet_v1.xlsx 'Coin Type')
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL",
    "ripple": "XRP", "xrp": "XRP", "dogecoin": "DOGE", "cardano": "ADA",
    "litecoin": "LTC", "chainlink": "LINK", "binance coin": "BNB", "tron": "TRX",
    # base tickers (merged_news_for_llm 'asset_type' = ["BTC","ETH", ...])
    "btc": "BTC", "eth": "ETH", "sol": "SOL", "bnb": "BNB", "doge": "DOGE",
    "ltc": "LTC", "link": "LINK", "trx": "TRX", "ada": "ADA",
}
BAR = "4h"
FEATURE_COLS = ["narr_sent_ewm", "narr_news_count_7d_log", "narr_sent_mom"]


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(path) if p.suffix.lower() == ".csv" else pd.read_parquet(path)


def _write_any(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if Path(path).suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def build_narrative_features(scored: pd.DataFrame,
                             halflife_days: float = 3.0,
                             count_window_days: int = 7,
                             mom_short_days: float = 1.0,
                             mom_long_days: float = 7.0,
                             bar: str = BAR) -> pd.DataFrame:
    """scored: DataFrame with at least [datetime_utc, coin, sentiment].
    Returns long DataFrame [symbol, ts, narr_sent_ewm, narr_news_count_7d_log, narr_sent_mom]
    on a regular 4h grid per symbol (symbol = base ticker, e.g. 'BTC')."""
    df = scored.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime_utc", "sentiment", "coin"])
    df["ticker"] = df["coin"].astype(str).str.strip().str.lower().map(COIN_TO_TICKER)
    df = df.dropna(subset=["ticker"])                       # drop non-universe coins
    if df.empty:
        return pd.DataFrame(columns=["symbol", "ts"] + FEATURE_COLS)

    bar_td = pd.Timedelta(bar)
    bars_per_day = pd.Timedelta("1D") / bar_td              # 6 for 4h
    hl = float(halflife_days * bars_per_day)                # half-life in bars
    cnt_win = int(round(count_window_days * bars_per_day))  # 42 bars for 7d
    hl_short = float(mom_short_days * bars_per_day)
    hl_long = float(mom_long_days * bars_per_day)

    # RIGHT-edge 4h bin: news in [T-4h, T) -> label T (strictly-before-T information).
    df["ts"] = df["datetime_utc"].dt.floor(bar) + bar_td

    out = []
    for tkr, g in df.groupby("ticker"):
        binned = g.groupby("ts").agg(sent_mean=("sentiment", "mean"),
                                     n=("sentiment", "size"))
        full = pd.date_range(binned.index.min(), binned.index.max(), freq=bar, tz="UTC")
        binned = binned.reindex(full)
        sent_in = binned["sent_mean"].fillna(0.0)           # empty bins -> 0 (decay to neutral)
        cnt_in = binned["n"].fillna(0.0)
        feat = pd.DataFrame(index=full)
        feat["narr_sent_ewm"] = sent_in.ewm(halflife=hl, adjust=False).mean()
        feat["narr_news_count_7d_log"] = np.log1p(cnt_in.rolling(cnt_win, min_periods=1).sum())
        feat["narr_sent_mom"] = (sent_in.ewm(halflife=hl_short, adjust=False).mean()
                                 - sent_in.ewm(halflife=hl_long, adjust=False).mean())
        feat = feat.reset_index().rename(columns={"index": "ts"})
        feat["symbol"] = tkr
        out.append(feat)

    res = pd.concat(out, ignore_index=True)
    return res[["symbol", "ts"] + FEATURE_COLS].sort_values(["symbol", "ts"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Build PIT-safe per-symbol narrative features.")
    ap.add_argument("--scored", default="data_storage/factors/sentiment/news_scored.parquet")
    ap.add_argument("--out", default="data_storage/factors/sentiment/narrative_features.parquet")
    ap.add_argument("--halflife_days", type=float, default=3.0)
    ap.add_argument("--count_window_days", type=int, default=7)
    args = ap.parse_args()

    scored = _read_any(args.scored)
    feats = build_narrative_features(scored, halflife_days=args.halflife_days,
                                     count_window_days=args.count_window_days)
    _write_any(feats, args.out)
    print(f"wrote {len(feats)} rows for {feats['symbol'].nunique()} symbols -> {args.out}")
    print("coverage per symbol:")
    print(feats.groupby("symbol")["ts"].agg(["count", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
