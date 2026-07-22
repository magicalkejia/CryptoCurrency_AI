import os
import time
from datetime import datetime
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

import config


# =====================================================================
# 工具函数
# =====================================================================
def symbol_key(symbol: str) -> str:
    """BTC/USDT -> BTCUSDT"""
    return symbol.replace("/", "")


def _raw_funding_path(symbol: str) -> Path:
    return config.PathConfig.RAW_FUNDING / f"{symbol_key(symbol)}.parquet"


def _legacy_raw_funding_path(symbol: str) -> Path:
    return config.PathConfig.RAW / f"{symbol_key(symbol)}_funding.parquet"


def _raw_oi_path(symbol: str) -> Path:
    return config.PathConfig.RAW_OI / f"{symbol_key(symbol)}.parquet"


def _legacy_raw_oi_path(symbol: str) -> Path:
    return config.PathConfig.RAW / f"{symbol_key(symbol)}_oi.parquet"


def _raw_spot_path(symbol: str, timeframe: str = "1m") -> Path:
    return config.PathConfig.RAW_SPOT / f"{symbol_key(symbol)}_{timeframe}.parquet"


def merge_incremental_data(
    df_old: pd.DataFrame | None,
    df_new: pd.DataFrame,
    key_col: str | list[str],
) -> pd.DataFrame:
    """
    合并旧数据和增量数据。

    设计原则：
    - 支持单字段主键，也支持复合主键，例如 ["symbol", "timestamp"]；
    - concat + drop_duplicates；
    - 保留最新数据，即 keep="last"；
    - 如果存在 timestamp 字段，最终按 symbol + timestamp 或 timestamp 排序。
    """
    if isinstance(key_col, str):
        key_cols = [key_col]
    else:
        key_cols = list(key_col)

    if df_new is None or df_new.empty:
        return df_old.copy() if df_old is not None else pd.DataFrame()

    df_new = df_new.copy()

    if df_old is None or df_old.empty:
        df_final = df_new
    else:
        df_old = df_old.copy()
        df_final = pd.concat([df_old, df_new], ignore_index=True)

    missing = [c for c in key_cols if c not in df_final.columns]
    if missing:
        raise ValueError(f"merge_incremental_data missing key columns: {missing}")

    df_final = df_final.drop_duplicates(subset=key_cols, keep="last")

    sort_cols = []
    if "symbol" in df_final.columns:
        sort_cols.append("symbol")
    if "period" in df_final.columns:
        sort_cols.append("period")
    if "timestamp" in df_final.columns:
        sort_cols.append("timestamp")
    if not sort_cols:
        sort_cols = key_cols

    return df_final.sort_values(sort_cols).reset_index(drop=True)


def get_exchange():
    """初始化交易所，使用 SpiderConfig 配置。"""
    return ccxt.binanceusdm({
        "enableRateLimit": True,
        "timeout": config.SpiderConfig.TIMEOUT,
        "proxies": config.SpiderConfig.PROXY,
    })


def get_spot_exchange():
    """Initialize Binance spot exchange with the same proxy/timeout settings."""
    return ccxt.binance({
        "enableRateLimit": True,
        "timeout": config.SpiderConfig.TIMEOUT,
        "proxies": config.SpiderConfig.PROXY,
    })


def get_last_timestamp(file_path):
    """读取 Parquet 最后一行的时间戳，用于增量更新。"""
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_parquet(file_path, columns=["timestamp"])
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df["timestamp"].max()
    except Exception as e:
        print(f"⚠️ 读取旧文件时间戳失败: {e}")
        return None


# =====================================================================
# Crypto 行情数据采集
# =====================================================================
def fetch_data(symbol):
    """采集单个币种 1m K 线数据，写入 data_storage/raw/{SYMBOL}_1m.parquet。"""
    exchange = get_exchange()
    symbol_clean = symbol_key(symbol)
    timeframe = config.TargetConfig.TIMEFRAMES["base"]
    file_path = config.PathConfig.RAW / f"{symbol_clean}_{timeframe}.parquet"

    last_ts = get_last_timestamp(file_path)

    if last_ts is not None:
        print(f"🔄 [增量] {symbol} 上次更新至: {last_ts}")
        since = int(pd.Timestamp(last_ts).timestamp() * 1000) + 1
    else:
        print(f"🆕 [全量] {symbol} 从头下载 ({config.SpiderConfig.START_TIME})...")
        since = exchange.parse8601(config.SpiderConfig.START_TIME)

    new_data = []
    now = exchange.milliseconds()
    print(f"开始抓取 {symbol}...")
    retry_count = 0
    max_retry = 5

    while True:
        try:
            klines = exchange.fapiPublicGetKlines({
                "symbol": symbol_clean,
                "interval": timeframe,
                "limit": 1500,
                "startTime": since,
            })

            if not klines:
                break

            batch = [
                [
                    int(k[0]),
                    float(k[1]),
                    float(k[2]),
                    float(k[3]),
                    float(k[4]),
                    float(k[5]),
                    float(k[9]),
                ]
                for k in klines
            ]
            new_data.extend(batch)

            last_ts_ms = batch[-1][0]
            since = last_ts_ms + 1

            if len(new_data) % 15000 < 1500:
                curr_date = datetime.fromtimestamp(last_ts_ms / 1000)
                print(f"   ⏳ 进度: {curr_date} | 本次新增: {len(new_data)} 行")

            retry_count = 0
            if last_ts_ms >= now - 120000:
                break
            time.sleep(0.1)

        except Exception as e:
            retry_count += 1
            print(f"❌ 网络错误 {retry_count}/{max_retry}: {e}")
            if retry_count >= max_retry:
                return False
            time.sleep(5)
            continue

    if not new_data:
        print(f"✅ {symbol} 已是最新，无需更新")
        return False

    df_new = pd.DataFrame(
        new_data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "taker_buy_vol",
        ],
    )
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")

    if not df_new["timestamp"].is_monotonic_increasing:
        df_new = df_new.sort_values("timestamp").reset_index(drop=True)

    if os.path.exists(file_path):
        df_old = pd.read_parquet(file_path)
        df_old["timestamp"] = pd.to_datetime(df_old["timestamp"])
        df_final = merge_incremental_data(df_old, df_new, key_col="timestamp")
    else:
        df_final = df_new.reset_index(drop=True)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(
        file_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    if not file_path.exists():
        raise RuntimeError(f"写入失败，文件未生成: {file_path.resolve()}")

    print(f"💾 {symbol} 写入完成，共包含 {len(df_final)} 行数据: {file_path}")
    return True


# =====================================================================
# Crypto 衍生品数据采集：funding rate / open interest
# =====================================================================
def fetch_spot_data(symbol: str, timeframe: str | None = None) -> bool:
    """
    Fetch Binance spot K-lines and save a per-symbol raw parquet file.

    RAW output:
        data_storage/raw/spot/{SYMBOL}_{timeframe}.parquet

    Columns:
        timestamp, symbol, open, high, low, close, volume, taker_buy_vol,
        source, created_at
    """
    exchange = get_spot_exchange()
    symbol_clean = symbol_key(symbol)
    timeframe = timeframe or config.TargetConfig.TIMEFRAMES["base"]
    file_path = _raw_spot_path(symbol, timeframe=timeframe)

    last_ts = get_last_timestamp(file_path)
    if last_ts is not None:
        since = int(pd.Timestamp(last_ts).timestamp() * 1000) + 1
        print(f"[incremental] {symbol} spot {timeframe} from {last_ts}")
    else:
        since = exchange.parse8601(config.SpiderConfig.START_TIME)
        print(f"[full] {symbol} spot {timeframe} from {config.SpiderConfig.START_TIME}")

    rows = []
    retry = 0
    max_retry = 5
    now = exchange.milliseconds()
    created_at = pd.Timestamp.now(tz="UTC").tz_localize(None)

    while True:
        try:
            klines = exchange.publicGetKlines({
                "symbol": symbol_clean,
                "interval": timeframe,
                "limit": 1000,
                "startTime": since,
            })

            if not klines:
                break

            batch = [
                [
                    int(k[0]),
                    symbol,
                    float(k[1]),
                    float(k[2]),
                    float(k[3]),
                    float(k[4]),
                    float(k[5]),
                    float(k[9]),
                    "binance_spot",
                    created_at,
                ]
                for k in klines
            ]
            rows.extend(batch)

            last_ts_ms = batch[-1][0]
            since = last_ts_ms + 1
            retry = 0

            if len(rows) % 10000 < len(batch):
                curr_date = datetime.fromtimestamp(last_ts_ms / 1000)
                print(f"   spot progress: {symbol} {curr_date} | new rows={len(rows)}")

            if last_ts_ms >= now - 120000 or len(klines) < 1000:
                break
            time.sleep(0.1)

        except Exception as e:
            retry += 1
            print(f"spot network error {retry}/{max_retry}: {symbol} | {e}")
            if retry >= max_retry:
                return False
            time.sleep(5)

    if not rows:
        print(f"{symbol} spot {timeframe} is already up to date")
        return False

    raw_columns = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "taker_buy_vol",
        "source",
        "created_at",
    ]
    df_new = pd.DataFrame(rows, columns=raw_columns)
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")
    df_new["created_at"] = pd.to_datetime(df_new["created_at"])

    df_old = None
    if file_path.exists():
        df_old = pd.read_parquet(file_path)
        for col in raw_columns:
            if col not in df_old.columns:
                df_old[col] = np.nan
        df_old = df_old[raw_columns].copy()
        df_old["timestamp"] = pd.to_datetime(df_old["timestamp"], errors="coerce")
        df_old["created_at"] = pd.to_datetime(df_old["created_at"], errors="coerce")

    df_final = merge_incremental_data(
        df_old=df_old,
        df_new=df_new,
        key_col=["symbol", "timestamp"],
    )
    df_final = df_final[raw_columns].sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(file_path, engine="pyarrow", compression="zstd", index=False)
    print(f"saved {symbol} spot RAW rows={len(df_final)} -> {file_path}")
    return True


def fetch_funding_rate(symbol):
    """
    抓取 Binance USDM 资金费率历史，保存 RAW 原始事件表。

    RAW 输出：
        data_storage/raw/derivatives/funding/{SYMBOL}.parquet

    columns:
        timestamp, symbol, funding_rate, source, created_at

    注意：
    - timestamp = 实际 fundingTime；
    - funding 是事件数据，不假设固定 8h；
    - mark_price 当前在你的环境中为 missing，因此不写入 RAW；
    - interval / z-score / change 等派生字段由 data_processor.py 生成。
    """
    exchange = get_exchange()
    symbol_clean = symbol_key(symbol)
    file_path = _raw_funding_path(symbol)
    legacy_path = _legacy_raw_funding_path(symbol)

    limit = 1000
    raw_columns = [
        "timestamp",
        "symbol",
        "funding_rate",
        "source",
        "created_at",
    ]

    # 兼容迁移：新路径优先；若不存在，则使用旧路径的最后 timestamp 继续增量。
    last_ts = get_last_timestamp(file_path)
    if last_ts is None and legacy_path.exists():
        last_ts = get_last_timestamp(legacy_path)

    if last_ts is not None:
        since = int(pd.Timestamp(last_ts).timestamp() * 1000) + 1
        print(f"🔄 [增量] {symbol} funding 上次至: {last_ts}")
    else:
        since = exchange.parse8601(config.SpiderConfig.START_TIME)
        print(f"🆕 [全量] {symbol} funding 从 {config.SpiderConfig.START_TIME}")

    rows = []
    retry = 0
    max_retry = 5
    created_at = pd.Timestamp.now(tz="UTC").tz_localize(None)

    while True:
        try:
            data = exchange.fapiPublicGetFundingRate({
                "symbol": symbol_clean,
                "startTime": since,
                "limit": limit,
            })

            if not data:
                break

            batch = []
            for d in data:
                batch.append([
                    int(d["fundingTime"]),
                    symbol,
                    float(d["fundingRate"]),
                    "binance_usdm",
                    created_at,
                ])

            rows.extend(batch)

            last = batch[-1][0]
            since = last + 1
            retry = 0

            if len(data) < limit:
                break

            time.sleep(0.1)

        except Exception as e:
            retry += 1
            print(f"funding 网络错误 {retry}/{max_retry}: {e}")
            if retry >= max_retry:
                return False
            time.sleep(5)

    if not rows:
        print(f"{symbol} funding 已是最新")
        return False

    df_new = pd.DataFrame(rows, columns=raw_columns)
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")
    df_new["created_at"] = pd.to_datetime(df_new["created_at"])

    df_old = None
    old_candidates = [p for p in [file_path, legacy_path] if p.exists()]
    if old_candidates:
        dfs = []
        for p in old_candidates:
            old = pd.read_parquet(p)
            if old.empty:
                continue

            if "timestamp" in old.columns:
                old["timestamp"] = pd.to_datetime(old["timestamp"], errors="coerce")
            if "symbol" not in old.columns:
                old["symbol"] = symbol
            else:
                old["symbol"] = old["symbol"].fillna(symbol)
            if "source" not in old.columns:
                old["source"] = "binance_usdm"
            else:
                old["source"] = old["source"].fillna("binance_usdm")
            if "created_at" not in old.columns:
                old["created_at"] = pd.NaT
            else:
                old["created_at"] = pd.to_datetime(old["created_at"], errors="coerce")

            for col in raw_columns:
                if col not in old.columns:
                    old[col] = np.nan
            dfs.append(old[raw_columns].copy())

        if dfs:
            df_old = pd.concat(dfs, ignore_index=True)

    df_final = merge_incremental_data(
        df_old=df_old,
        df_new=df_new,
        key_col=["symbol", "timestamp"],
    )
    df_final = df_final[raw_columns].sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(
        file_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    print(f"💾 {symbol} funding RAW 写入 {len(df_final)} 行: {file_path}")
    return True


def fetch_open_interest(symbol, period="1h"):
    """
    抓取 Binance USDM 持仓量历史，保存 RAW 原始事件表。

    RAW 输出：
        data_storage/raw/derivatives/oi/{SYMBOL}.parquet

    columns:
        timestamp, symbol, open_interest, open_interest_value, period, source, created_at

    注意：Binance 该接口通常仅提供约最近 30 天历史。
    """
    exchange = get_exchange()
    symbol_clean = symbol_key(symbol)
    file_path = _raw_oi_path(symbol)
    legacy_path = _legacy_raw_oi_path(symbol)

    raw_columns = [
        "timestamp",
        "symbol",
        "open_interest",
        "open_interest_value",
        "period",
        "source",
        "created_at",
    ]

    last_ts = get_last_timestamp(file_path)
    if last_ts is None and legacy_path.exists():
        last_ts = get_last_timestamp(legacy_path)

    earliest = exchange.milliseconds() - 30 * 24 * 3600 * 1000
    since = int(pd.Timestamp(last_ts).timestamp() * 1000) + 1 if last_ts is not None else earliest
    since = max(since, earliest)

    rows = []
    retry = 0
    max_retry = 5
    now = exchange.milliseconds()
    created_at = pd.Timestamp.now(tz="UTC").tz_localize(None)

    while True:
        try:
            data = exchange.fapiDataGetOpenInterestHist({
                "symbol": symbol_clean,
                "period": period,
                "limit": 500,
                "startTime": since,
            })

            if not data:
                break

            for d in data:
                rows.append([
                    int(d["timestamp"]),
                    symbol,
                    float(d["sumOpenInterest"]),
                    float(d["sumOpenInterestValue"]),
                    period,
                    "binance_usdm",
                    created_at,
                ])

            last = int(data[-1]["timestamp"])
            since = last + 1
            retry = 0

            if last >= now - 3600 * 1000 or len(data) < 500:
                break
            time.sleep(0.1)

        except Exception as e:
            retry += 1
            print(f"❌ OI 网络错误 {retry}/{max_retry}: {e}")
            if retry >= max_retry:
                return False
            time.sleep(5)

    if not rows:
        print(f"✅ {symbol} OI 已是最新")
        return False

    df_new = pd.DataFrame(rows, columns=raw_columns)
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")
    df_new["created_at"] = pd.to_datetime(df_new["created_at"])

    df_old = None
    old_candidates = [p for p in [file_path, legacy_path] if p.exists()]
    if old_candidates:
        dfs = []
        for p in old_candidates:
            old = pd.read_parquet(p)
            if old.empty:
                continue

            if "timestamp" in old.columns:
                old["timestamp"] = pd.to_datetime(old["timestamp"], errors="coerce")
            if "symbol" not in old.columns:
                old["symbol"] = symbol
            if "period" not in old.columns:
                old["period"] = period
            if "source" not in old.columns:
                old["source"] = "binance_usdm"
            if "created_at" not in old.columns:
                old["created_at"] = pd.NaT
            else:
                old["created_at"] = pd.to_datetime(old["created_at"], errors="coerce")

            for col in raw_columns:
                if col not in old.columns:
                    old[col] = np.nan
            dfs.append(old[raw_columns].copy())

        if dfs:
            df_old = pd.concat(dfs, ignore_index=True)

    df_final = merge_incremental_data(
        df_old=df_old,
        df_new=df_new,
        key_col=["symbol", "period", "timestamp"],
    )
    df_final = df_final[raw_columns].sort_values(["symbol", "period", "timestamp"]).reset_index(drop=True)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(
        file_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    print(f"💾 {symbol} OI RAW 写入 {len(df_final)} 行: {file_path}")
    return True


if __name__ == "__main__":
    fetch_funding_rate("BTC/USDT")
