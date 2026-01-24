import ccxt
import pandas as pd
import os
import time
from datetime import datetime
import config  

def get_exchange():
    """初始化交易所，使用 SpiderConfig 配置"""
    return ccxt.binanceusdm({
        'enableRateLimit': True,
        # 使用配置类中的超时设置
        'timeout': config.SpiderConfig.TIMEOUT,
        # 直接使用配置类中的代理字典
        'proxies': config.SpiderConfig.PROXY, 
    })

def get_last_timestamp(file_path):
    """读取 Parquet 最后一行的时间戳，用于增量更新"""
    # 兼容 pathlib.Path 对象，将其转为字符串判断是否存在
    if not os.path.exists(file_path):
        return None
    try:
        # 只读取 timestamp 列，极大提高速度
        df = pd.read_parquet(file_path, columns=['timestamp'])
        if df.empty: return None
        return df['timestamp'].iloc[-1]
    except Exception as e:
        print(f"⚠️ 读取旧文件时间戳失败: {e}")
        return None

def fetch_data(symbol):
    """
    采集单个币种数据 (全量/增量自动切换)
    """
    exchange = get_exchange()
    symbol_clean = symbol.replace('/', '')
    
    # --- 1. 路径与配置映射 ---
    # 从 TargetConfig 获取基础时间粒度 (例如 '1m')
    timeframe = config.TargetConfig.TIMEFRAMES['base']
    
    # 从 PathConfig 获取 Raw 目录路径
    #以此构建文件路径: ./data_storage/raw/BTCUSDT_1m.parquet
    file_path = config.PathConfig.RAW / f"{symbol_clean}_{timeframe}.parquet"
    
    # --- 2. 确定下载起点 ---
    last_ts = get_last_timestamp(file_path)
    
    if last_ts:
        print(f"🔄 [增量] {symbol} 上次更新至: {last_ts}")
        # 接着上一条继续下
        since = int(last_ts.timestamp() * 1000) + 1
    else:
        # 从 SpiderConfig 获取起始时间字符串
        print(f"🆕 [全量] {symbol} 从头下载 ({config.SpiderConfig.START_TIME})...")
        since = exchange.parse8601(config.SpiderConfig.START_TIME)

    new_data = []
    now = exchange.milliseconds()
    
    # --- 3. 循环下载逻辑  ---
    print(f" 开始抓取 {symbol}...")
    
    while True:
        try:
            klines = exchange.fapiPublicGetKlines({
                'symbol': symbol_clean,
                'interval': timeframe,
                'limit': 1500,
                'startTime': since
            })
            
            if not klines:
                break
            

            batch = [
                [
                    int(k[0]), float(k[1]), float(k[2]), float(k[3]), 
                    float(k[4]), float(k[5]), float(k[9])
                ]
                for k in klines
            ]
            new_data.extend(batch)
            
            # 更新游标
            last_ts_ms = batch[-1][0]
            since = last_ts_ms + 1
            
            # 进度提示
            if len(new_data) % 15000 < 1500:
                curr_date = datetime.fromtimestamp(last_ts_ms / 1000)
                print(f"   ⏳ 进度: {curr_date} | 本次新增: {len(new_data)} 行")
            
            # 追赶检测 (距离现在小于2分钟)
            if last_ts_ms >= now - 120000:
                break
                
            time.sleep(0.1) 

        except Exception as e:
            print(f"   ❌ 网络错误: {e}")
            time.sleep(5)
            # 报错后不中断，继续重试
            continue

    # --- 4. 保存逻辑 (Concat + Rewrite) ---
    if new_data:
        df_new = pd.DataFrame(new_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_vol'])
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
        
        if os.path.exists(file_path):
            # 读取旧数据
            df_old = pd.read_parquet(file_path)
            # 合并
            df_final = pd.concat([df_old, df_new], ignore_index=True)
            # 去重 (保留最后一条，防止边界重叠)
            df_final = df_final.drop_duplicates(subset=['timestamp'], keep='last')
        else:
            df_final = df_new
            
        # 保存 (使用 zstd 压缩)
        df_final.to_parquet(file_path, engine='pyarrow', compression='zstd', index=False)
        print(f"💾 {symbol} 写入完成，共包含 {len(df_final)} 行数据")
        return True # 返回 True 告诉主程序：有新数据，需要清洗
    else:
        print(f"✅ {symbol} 已是最新，无需更新")
        return False