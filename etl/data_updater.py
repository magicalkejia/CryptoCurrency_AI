import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import config  
import akshare as ak
from tqdm import tqdm
import baostock as bs
import numpy as np
import traceback
##工具函数
def merge_incremental_data(
    df_old: pd.DataFrame | None,
    df_new: pd.DataFrame,
    key_col: str,
) -> pd.DataFrame:
    """
    合并旧数据和增量数据。
    设计原则：
    - 默认相信旧数据和新数据都是按 key_col 升序；
    - 先 concat + drop_duplicates；
    - 只有发现非单调时才排序；
    - 保留最新数据，即 keep='last'。
    """
    if df_new is None or df_new.empty:
        return df_old if df_old is not None else pd.DataFrame()

    if not df_new[key_col].is_monotonic_increasing:
        df_new = df_new.sort_values(key_col).reset_index(drop=True)

    if df_old is None or df_old.empty:
        return df_new.reset_index(drop=True)

    if not df_old[key_col].is_monotonic_increasing:
        df_old = df_old.sort_values(key_col).reset_index(drop=True)

    df_final = pd.concat([df_old, df_new], ignore_index=True)
    df_final = df_final.drop_duplicates(subset=[key_col], keep="last")

    if not df_final[key_col].is_monotonic_increasing:
        df_final = df_final.sort_values(key_col)

    return df_final.reset_index(drop=True)
# =====================================================================
# Crypto 数据采集 
# =====================================================================
def get_exchange():
    """初始化交易所，使用 SpiderConfig 配置"""
    return ccxt.binanceusdm({
        'enableRateLimit': True,
        'timeout': config.SpiderConfig.TIMEOUT,
        'proxies': config.SpiderConfig.PROXY, 
    })

def get_last_timestamp(file_path):
    """读取 Parquet 最后一行的时间戳，用于增量更新"""
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_parquet(file_path, columns=['timestamp'])
        if df.empty: return None
        return df['timestamp'].iloc[-1]
    except Exception as e:
        print(f"⚠️ 读取旧文件时间戳失败: {e}")
        return None

def fetch_data(symbol):
    """采集单个币种数据"""
    exchange = get_exchange()
    symbol_clean = symbol.replace('/', '')
    timeframe = config.TargetConfig.TIMEFRAMES['base']
    file_path = config.PathConfig.RAW / f"{symbol_clean}_{timeframe}.parquet"
    
    last_ts = get_last_timestamp(file_path)
    
    if last_ts:
        print(f"🔄 [增量] {symbol} 上次更新至: {last_ts}")
        since = int(last_ts.timestamp() * 1000) + 1
    else:
        print(f"🆕 [全量] {symbol} 从头下载 ({config.SpiderConfig.START_TIME})...")
        since = exchange.parse8601(config.SpiderConfig.START_TIME)

    new_data = []
    now = exchange.milliseconds()
    print(f" 开始抓取 {symbol}...")
    retry_count = 0
    max_retry = 5
    while True:

        try:
            klines = exchange.fapiPublicGetKlines({
                'symbol': symbol_clean,
                'interval': timeframe,
                'limit': 1500,
                'startTime': since
            })
            
            if not klines: break
            
            batch = [
                [
                    int(k[0]), float(k[1]), float(k[2]), float(k[3]), 
                    float(k[4]), float(k[5]), float(k[9])
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
            if last_ts_ms >= now - 120000: break
            time.sleep(0.1) 

        except Exception as e:
            retry_count += 1
            print(f"❌ 网络错误 {retry_count}/{max_retry}: {e}")
            if retry_count >= max_retry:
                return False
            time.sleep(5)
            continue

    if new_data:
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

        # API 正常情况下返回有序，但这里做轻量防御
        if not df_new["timestamp"].is_monotonic_increasing:
            df_new = df_new.sort_values("timestamp").reset_index(drop=True)

        if os.path.exists(file_path):
            df_old = pd.read_parquet(file_path)

            # 如果旧文件本身乱序，先修一次
            if not df_old["timestamp"].is_monotonic_increasing:
                print(f"⚠️ 旧文件时间戳乱序，执行一次排序修复: {file_path}")
                df_old = df_old.sort_values("timestamp").reset_index(drop=True)

            df_final = pd.concat([df_old, df_new], ignore_index=True)
            df_final = df_final.drop_duplicates(subset=["timestamp"], keep="last")

            # 正常增量不需要排序；只有发现异常才排序
            if not df_final["timestamp"].is_monotonic_increasing:
                print(f"⚠️ 合并后时间戳乱序，执行排序修复: {symbol}")
                df_final = df_final.sort_values("timestamp").reset_index(drop=True)
            else:
                df_final = df_final.reset_index(drop=True)

        else:
            df_final = df_new.reset_index(drop=True)

        # 轻量防御：不依赖调用方一定先 init_directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df_final.to_parquet(
            file_path,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )

        print(f"💾 {symbol} 写入完成，共包含 {len(df_final)} 行数据")
        return True

    else:
        print(f"✅ {symbol} 已是最新，无需更新")
        return False
    
# =====================================================================
# Crypto 衍生品数据采集 (funding rate / open interest)  —— v6 新增
# 与 fetch_data 风格一致：增量更新、轻量防御、写 RAW/PROCESSED
# =====================================================================
def fetch_funding_rate(symbol):
    """
    抓取 Binance USDM 资金费率历史 (fapiPublicGetFundingRate)。
    funding 每 8h 结算一次。输出 RAW/{symbol}_funding.parquet:
        columns = [timestamp, funding_rate]   (timestamp = fundingTime)
    返回 True/False 表示是否有更新。
    """
    exchange = get_exchange()
    symbol_clean = symbol.replace('/', '')
    file_path = config.PathConfig.RAW / f"{symbol_clean}_funding.parquet"

    last_ts = get_last_timestamp(file_path)
    if last_ts is not None:
        since = int(last_ts.timestamp() * 1000) + 1
        print(f"🔄 [增量] {symbol} funding 上次至: {last_ts}")
    else:
        since = exchange.parse8601(config.SpiderConfig.START_TIME)
        print(f"🆕 [全量] {symbol} funding 从 {config.SpiderConfig.START_TIME}")

    now = exchange.milliseconds()
    rows, retry, max_retry = [], 0, 5
    while True:
        try:
            data = exchange.fapiPublicGetFundingRate({
                'symbol': symbol_clean, 'startTime': since, 'limit': 1000,
            })
            if not data:
                break
            for d in data:
                rows.append([int(d['fundingTime']), float(d['fundingRate'])])
            last = int(data[-1]['fundingTime'])
            since = last + 1
            retry = 0
            if last >= now - 8 * 3600 * 1000 or len(data) < 1000:
                break
            time.sleep(0.1)
        except Exception as e:
            retry += 1
            print(f"❌ funding 网络错误 {retry}/{max_retry}: {e}")
            if retry >= max_retry:
                return False
            time.sleep(5)

    if not rows:
        print(f"✅ {symbol} funding 已是最新")
        return False
    df_new = pd.DataFrame(rows, columns=['timestamp', 'funding_rate'])
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
    df_old = pd.read_parquet(file_path) if os.path.exists(file_path) else None
    df_final = merge_incremental_data(df_old, df_new, key_col='timestamp')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(file_path, engine='pyarrow', compression='zstd', index=False)
    print(f"💾 {symbol} funding 写入 {len(df_final)} 行")
    return True


def fetch_open_interest(symbol, period='1h'):
    """
    抓取 Binance USDM 持仓量历史 (fapiDataGetOpenInterestHist)。
    注意：Binance 该接口仅提供约最近 30 天历史 —— 多年回测的 OI 会缺失早期段，
    属已知数据源限制(见 INTEGRATION_NOTES 数据缺口)。
    输出 RAW/{symbol}_oi.parquet: [timestamp, open_interest, open_interest_value]
    """
    exchange = get_exchange()
    symbol_clean = symbol.replace('/', '')
    file_path = config.PathConfig.RAW / f"{symbol_clean}_oi.parquet"

    last_ts = get_last_timestamp(file_path)
    # 该接口最多回溯 30 天
    earliest = exchange.milliseconds() - 30 * 24 * 3600 * 1000
    since = int(last_ts.timestamp() * 1000) + 1 if last_ts is not None else earliest
    since = max(since, earliest)

    rows, retry, max_retry = [], 0, 5
    now = exchange.milliseconds()
    while True:
        try:
            data = exchange.fapiDataGetOpenInterestHist({
                'symbol': symbol_clean, 'period': period, 'limit': 500, 'startTime': since,
            })
            if not data:
                break
            for d in data:
                rows.append([int(d['timestamp']), float(d['sumOpenInterest']),
                             float(d['sumOpenInterestValue'])])
            last = int(data[-1]['timestamp'])
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
    df_new = pd.DataFrame(rows, columns=['timestamp', 'open_interest', 'open_interest_value'])
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
    df_old = pd.read_parquet(file_path) if os.path.exists(file_path) else None
    df_final = merge_incremental_data(df_old, df_new, key_col='timestamp')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(file_path, engine='pyarrow', compression='zstd', index=False)
    print(f"💾 {symbol} OI 写入 {len(df_final)} 行")
    return True



if __name__ == "__main__":
    fetch_data("BTC/USDT")
