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
    """采集单个币种数据 (全量/增量自动切换)"""
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
    
    while True:
        retry_count = 0
        max_retry = 5
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
        df_new = pd.DataFrame(new_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_vol'])
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
        
        if os.path.exists(file_path):
            df_old = pd.read_parquet(file_path)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
            df_final = df_final.drop_duplicates(subset=['timestamp'], keep='last')
        else:
            df_final = df_new
            
        df_final.to_parquet(file_path, engine='pyarrow', compression='zstd', index=False)
        print(f"💾 {symbol} 写入完成，共包含 {len(df_final)} 行数据")
        return True 
    else:
        print(f"✅ {symbol} 已是最新，无需更新")
        return False
    
# =====================================================================
# 模块一：A股数据 - 辅助与基表
# =====================================================================
def fetch_akshare_with_retry(fetch_func, max_retries=3, delay=2, **kwargs):
    """AKShare 接口重试保护，防止新浪/东财突然限流断开"""
    for i in range(max_retries):
        try:
            return fetch_func(**kwargs)
        except Exception:
            time.sleep(delay)
    return pd.DataFrame()

def update_instrument_master():
    print("\n========================================")
    print(" 🧊 开始更新静态基表")
    print("========================================")
    
    master_path = config.PathConfig.DATA_ROOT / 'meta' / 'instrument_master.parquet'
    os.makedirs(master_path.parent, exist_ok=True)
    
    bs.login()
    rs = bs.query_stock_basic(code="") 
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()
    
    if not data_list:
        print("❌ 获取基础资料失败！")
        return pd.DataFrame()
        
    df_raw = pd.DataFrame(data_list, columns=rs.fields)
    
    def parse_asset_type(t):
        if t == '1': return 'stock'
        elif t == '2': return 'index'
        elif t == '4': return 'cb' 
        elif t == '5': return 'etf' 
        return 'other' 
        
    df_raw['AssetType'] = df_raw['type'].apply(parse_asset_type)
    df_raw['Status'] = df_raw['status'].apply(lambda x: 'active' if x == '1' else 'delisted')
    df_raw['BsCode'] = df_raw['code']
    df_raw['StockCode'] = df_raw['code'].apply(lambda x: x.split('.')[1] if '.' in x else x)
    
    def make_suffix_and_sina(bs_code):
        if '.' not in bs_code: return bs_code, bs_code
        exchange, num = bs_code.split('.')
        return f"{num}.{exchange.upper()}", f"{exchange.lower()}{num}"
        
    codes = df_raw['BsCode'].apply(make_suffix_and_sina)
    df_raw['SuffixStockNum'] = [x[0] for x in codes]
    df_raw['SinaCode']       = [x[1] for x in codes]
    
    df_new = df_raw.rename(columns={'code_name': 'StockAbbreviation', 'ipoDate': 'ListingDate', 'outDate': 'DelistingDate'})
    df_new['ListingDate'] = pd.to_datetime(df_new['ListingDate'], errors='coerce')
    df_new['DelistingDate'] = pd.to_datetime(df_new['DelistingDate'], errors='coerce')
    
    cols = ['SuffixStockNum', 'StockCode', 'BsCode', 'SinaCode', 'StockAbbreviation', 'AssetType', 'Status', 'ListingDate', 'DelistingDate']
    df_new = df_new[cols].drop_duplicates(subset=['SuffixStockNum'], keep='first')
    
    if os.path.exists(master_path):
        df_master = pd.read_parquet(master_path).set_index('SuffixStockNum')
        df_new = df_new.set_index('SuffixStockNum')
        df_master.update(df_new) 
        new_indices = df_new.index.difference(df_master.index)
        if not new_indices.empty:
            df_master = pd.concat([df_master, df_new.loc[new_indices]]) 
        df_final = df_master.reset_index()
    else:
        df_final = df_new

    df_final.to_parquet(master_path, engine='pyarrow', compression='zstd')
    print(f"✅ 基表 更新完毕 系统总标的数: {len(df_final)}")
    return df_final

def fetch_realtime_snapshot(save=False):
    """实时截面由于精度要求不高（仅作最新状态获取），可保留东财接口"""
    today_str = datetime.now().strftime('%Y%m%d')
    print(f"\n========================================")
    print(f" 📊 生成每日全资产截面快照 ({today_str})")
    print("========================================")
    
    cross_section_path = config.PathConfig.DATA_ROOT / 'cross_section' / f'meta_{today_str}.parquet'
    master_path = config.PathConfig.DATA_ROOT / 'meta' / 'instrument_master.parquet'
    
    if not os.path.exists(master_path): return
    df_master = pd.read_parquet(master_path)
    df_universe = df_master[df_master['AssetType'].isin(['stock', 'etf', 'index'])].copy()
    
    df_stock = fetch_akshare_with_retry(ak.stock_zh_a_spot_em)
    df_etf = fetch_akshare_with_retry(ak.fund_etf_spot_em)
    df_index = fetch_akshare_with_retry(ak.stock_zh_index_spot_em)
    
    spot_list = []
    for df, name in zip([df_stock, df_etf, df_index], ['Stock', 'ETF', 'Index']):
        if not df.empty:
            try:
                temp_df = df[['代码', '最新价', '最高', '最低', '成交量', '成交额']].copy()
                temp_df = temp_df.rename(columns={'代码': 'StockCode', '最新价': 'Close', '最高': 'High', '最低': 'Low', '成交量': 'Volume', '成交额': 'Amount'})
                spot_list.append(temp_df)
            except KeyError: pass
                
    if not spot_list: return
        
    df_spot_all = pd.concat(spot_list, ignore_index=True).drop_duplicates(subset=['StockCode'], keep='first')
    df_daily = pd.merge(df_universe, df_spot_all, on='StockCode', how='left')
    df_daily['ST'] = df_daily['StockAbbreviation'].str.contains('ST|退', na=False).astype(int)
    
    for col in ['Close', 'High', 'Low', 'Volume', 'Amount']:
        df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')
        
    if save:
        df_daily.to_parquet(config.PathConfig.CROSS_SECTION / "latest.parquet")
    return df_daily

# =====================================================================
# 模块二：A股历史数据 - 新浪与 Baostock 兼用
# =====================================================================
def fetch_etf_history_sina(suffix_code, start_date, end_date):
    """
     ETF ：使用新浪 ETF 接口 (fund_etf_hist_sina)
    该接口只能全量拉取且无复权参数，我们在本地进行日期截断和复权字段对齐。
    """
    # 格式转换：'510300.SH' -> 'sh510300'
    code, exchange = suffix_code.split('.')
    sina_symbol = f"{exchange.lower()}{code}" 
    
    # 调用新浪 ETF 专属接口 (无 adjust 参数)
    df_normal = fetch_akshare_with_retry(
        ak.fund_etf_hist_sina, symbol=sina_symbol
    )

    if df_normal.empty: 
        return pd.DataFrame()

    # 1. 格式化时间并进行本地切片 (该接口默认返回上市至今全量数据)
    df_normal['date'] = pd.to_datetime(df_normal['date']).dt.strftime('%Y-%m-%d')
    df_normal = df_normal[(df_normal['date'] >= start_date) & (df_normal['date'] <= end_date)].copy()
    
    if df_normal.empty:
        return pd.DataFrame()

    # 2. 结构对齐映射
    df_combined = pd.DataFrame()
    df_combined['date'] = df_normal['date']
    
    # 原始价格
    df_combined['open'] = df_normal['open']
    df_combined['high'] = df_normal['high']
    df_combined['low'] = df_normal['low']
    df_combined['close'] = df_normal['close']
    
    #  ETF 除权影响极小，为保证下游宽表结构统一，将原始价复制给 hfq 字段
    df_combined['open_hfq'] = df_normal['open']
    df_combined['high_hfq'] = df_normal['high']
    df_combined['low_hfq'] = df_normal['low']
    df_combined['close_hfq'] = df_normal['close']
    
    # 交易量
    df_combined['volume'] = df_normal['volume']
    
    # 新浪该接口通常不提供成交额(amount)和换手率(turn)，用 NaN 填充保证 Schema 不崩
    df_combined['amount'] = np.nan 
    df_combined['turn'] = np.nan
    df_combined['pctChg'] = np.nan
    
    return df_combined

def fetch_symbol_history(bs_code, suffix_code, start_date, end_date, asset_type):
    """
    数据路由器：根据 asset_type 决定底层使用 Baostock 还是 新浪(AKShare)
    """
    # ====== 路由分支 1: ETF 走新浪通道 ======
    if asset_type == 'etf':
        df_final = fetch_etf_history_sina(suffix_code, start_date, end_date)
        if not df_final.empty:
            for col in [c for c in df_final.columns if c != 'date']:
                df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        return df_final

    # ====== 路由分支 2: 股票与指数走 Baostock 通道 ======
    if asset_type == 'stock':
        fields_unadj = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST"
    elif asset_type == 'index':
        fields_unadj = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
    else:
        return pd.DataFrame() 

    rs_unadj = bs.query_history_k_data_plus(bs_code, fields_unadj, start_date=start_date, end_date=end_date, frequency="d", adjustflag="3")
    if rs_unadj.error_code != '0': return pd.DataFrame()
    
    data_unadj = []
    while rs_unadj.next(): data_unadj.append(rs_unadj.get_row_data())
    if not data_unadj: return pd.DataFrame()
    
    df_unadj = pd.DataFrame(data_unadj, columns=rs_unadj.fields)

    # 股票处理复权
    if asset_type == 'stock':
        fields_adj = "date,open,high,low,close,preclose"
        rs_adj = bs.query_history_k_data_plus(bs_code, fields_adj, start_date=start_date, end_date=end_date, frequency="d", adjustflag="2")
        data_adj = []
        while rs_adj.next(): data_adj.append(rs_adj.get_row_data())
        df_adj = pd.DataFrame(data_adj, columns=rs_adj.fields)
        
        rename_dict = {col: f"{col}_hfq" for col in df_adj.columns if col != 'date'}
        df_adj = df_adj.rename(columns=rename_dict)
        df_final = pd.merge(df_unadj, df_adj, on='date', how='left')
    else:
        # 指数填充复权列
        df_final = df_unadj.copy()
        for col in ['open', 'high', 'low', 'close', 'preclose']:
            df_final[f'{col}_hfq'] = df_final[col]

    for col in [c for c in df_final.columns if c not in ['date', 'code']]:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    if asset_type == 'stock' and 'turn' in df_final.columns:
        df_final['circulating_market_cap'] = np.where(
            df_final['turn'] > 0,
            (df_final['volume'] * 100 / df_final['turn']) * df_final['close'],
            np.nan 
        )
    return df_final

#最好不要修改默认的开始时间，提前开始时间并不能让更新变快,目前baostock数据源最早支持1990-12-19
def update_all_history_data(global_start_date = "1990-12-19"):
    print("\n========================================")
    print(" 历史数据更新中")
    print("========================================")
    
    master_path = config.PathConfig.DATA_ROOT / 'meta' / 'instrument_master.parquet'
    if not os.path.exists(master_path): return
        
    df_master = pd.read_parquet(master_path)
    allowed_types = ['stock', 'etf', 'index']
    df_filtered = df_master[df_master['AssetType'].isin(allowed_types)]
    target_list = df_filtered[['SuffixStockNum', 'BsCode', 'AssetType', 'Status', 'ListingDate']].dropna(subset=['SuffixStockNum']).to_dict('records')
    
    save_dir = config.PathConfig.DATA_ROOT / 'history_k'
    os.makedirs(save_dir, exist_ok=True)
    
    bs.login()
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    failed_symbols_exception = [] 
    failed_symbols_empty = []
    
    for item in tqdm(target_list, desc="拉取进度"):
        suffix_code = item['SuffixStockNum']
        file_path = save_dir / f"{suffix_code}.parquet"
        
        try:
            last_date = None
            if os.path.exists(file_path):
                try:
                    last_date = pd.to_datetime(pd.read_parquet(file_path, columns=['date'])['date'].iloc[-1])
                except: pass
                
            if last_date:
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                if start_date > today_str: continue 
            else:
                start_date = global_start_date
                
            df_new = fetch_symbol_history(item['BsCode'], suffix_code, start_date, today_str, item['AssetType'])
            
            # 如果返回空表，检查区别 真的污染还是别的问题
            if df_new.empty: 
                # 规则 1：退市标的过滤 (status == 0 表示目前不上市)
                if item.get('Status') == 'delisted':
                    continue
                    
                # 规则 2：增量更新碰壁 = 处于停牌期
                if last_date is not None:
                    # 本地已经存有它的历史数据，说明 API 对该标的有效。
                    # 今天做增量查询却为空，说明它处于停牌期，无需记录。
                    continue
                    
                # 规则 3：首次全量拉取 (起于 1990) 依然为空，排查是否未上市
                list_date = item.get('ListingDate')
                if pd.notnull(list_date):
                    # 如果上市日期就是今天，甚至在未来（有些基表会提前录入即将发行的股票）
                    # 此时拉不到历史 K 线是绝对正常的。
                    if pd.to_datetime(list_date) >= pd.to_datetime(today_str):
                        continue
                        
                # 规则 4：排除以上所有情况的“真凶”
                # 老股票、没退市、本地无历史数据、从 1990 年拉到现在依然全空
                # 可能是数据源的盲区或代码映射错误，排查
                failed_symbols_empty.append(f"{suffix_code} (上市时间: {list_date}，全史无数据)")
                continue
                
            df_new['code'] = suffix_code 
            
            if last_date and os.path.exists(file_path):
                df_old = pd.read_parquet(file_path)
                df_final = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=['date'], keep='last')
            else:
                df_final = df_new
                
            df_final.to_parquet(file_path, engine='pyarrow', compression='zstd')
            
        except Exception as e:
            print(f"\n❌ [异常报错] {suffix_code} 拉取失败: {str(e)}")
            failed_symbols_exception.append(f"{suffix_code} ({str(e)})")
            continue 

    bs.logout()
    print("\n✅ 历史数据更新完毕")
    
    # 生成日志报告
    if failed_symbols_exception or failed_symbols_empty:
        log_dir = config.PathConfig.LOG
        os.makedirs(log_dir, exist_ok=True)
        log_file = log_dir / f"failed_history_update_{today_str}.txt"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 历史数据拉取失败记录 ({today_str}) ===\n\n")
            
            if failed_symbols_exception:
                f.write(f" [抛出异常组] 共 {len(failed_symbols_exception)} 个 (通常由于网络断开或解析错误):\n")
                for symbol_info in failed_symbols_exception:
                    f.write(f"  - {symbol_info}\n")
                f.write("\n")
                
            if failed_symbols_empty:
                f.write(f"[返回空值组] 共 {len(failed_symbols_empty)} 个 (通常由于新上市、长期停牌或接口限流):\n")
                for symbol in failed_symbols_empty:
                    f.write(f"  - {symbol}\n")
                    
        print(f"📄 详细名单已保存至日志: {log_file}")

if __name__ == "__main__":
    # update_instrument_master()
    # update_daily_cross_section()
    
    # 局部测试新浪通道是否生效：拉取沪深300 ETF 历史数据
    # print("\n--- 测试 ETF 新浪通道 ---")
    # if not df.empty:
    #     print(df.head())
    #     print(f"✅ 成功拉取 {len(df)} 条新浪源数据。")
    
    #不要轻易运行，更新一次需要45分钟到1小时左右
    update_all_history_data()