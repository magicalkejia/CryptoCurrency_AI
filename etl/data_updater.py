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
    
# A股数据
# =====================================================================
# 辅助函数，akahare等爬虫接口重试做封装
# =====================================================================
def fetch_akshare_with_retry(fetch_func, max_retries=3, delay=2, **kwargs):
    for i in range(max_retries):
        try:
            return fetch_func(**kwargs)
        except Exception:
            time.sleep(delay)
    return pd.DataFrame()

# =====================================================================
# 1. 基表更新 (Baostock 数据源)
# =====================================================================
def update_instrument_master():
    print("\n========================================")
    print(" 🧊 开始更新静态基表")
    print("========================================")
    
    master_path = config.PathConfig.DATA_ROOT / 'meta' / 'instrument_master.parquet'
    os.makedirs(master_path.parent, exist_ok=True)
    
    bs.login()
    print(" 正在拉取全市场基础资料...")
    rs = bs.query_stock_basic(code="") 
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()
    
    if not data_list:
        print("❌ 获取基础资料失败！")
        return pd.DataFrame()
        
    df_raw = pd.DataFrame(data_list, columns=rs.fields)
    
    # 解析类型与状态
    def parse_asset_type(t):
        if t == '1': return 'stock'
        elif t == '2': return 'index'
        elif t == '4': return 'cb'       # 可转债 (Convertible Bond)
        elif t == '5': return 'etf'      # ETF基金
        return 'other'                   # 3:其它
        
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
    
    df_new = df_raw.rename(columns={
        'code_name': 'StockAbbreviation',
        'ipoDate': 'ListingDate',
        'outDate': 'DelistingDate'
    })
    
    df_new['ListingDate'] = pd.to_datetime(df_new['ListingDate'], errors='coerce')
    df_new['DelistingDate'] = pd.to_datetime(df_new['DelistingDate'], errors='coerce')
    
    cols = ['SuffixStockNum', 'StockCode', 'BsCode', 'SinaCode', 'StockAbbreviation', 'AssetType', 'Status', 'ListingDate', 'DelistingDate']
    df_new = df_new[cols].drop_duplicates(subset=['SuffixStockNum'], keep='first')
    
    # --- 安全 Upsert 逻辑 ---
    if os.path.exists(master_path):
        df_master = pd.read_parquet(master_path).set_index('SuffixStockNum')
        df_new = df_new.set_index('SuffixStockNum')
        
        df_master.update(df_new) # 覆盖更新已存在的（如名字变更、退市）
        
        new_indices = df_new.index.difference(df_master.index)
        if not new_indices.empty:
            df_master = pd.concat([df_master, df_new.loc[new_indices]]) # 追加新增的
            
        df_final = df_master.reset_index()
        print(f"🔄 基表 Upsert 完成: 新增了 {len(new_indices)} 个标的。")
    else:
        df_final = df_new
        print("🆕 本地无历史基表，执行首次全量初始化。")

    df_final.to_parquet(master_path, engine='pyarrow', compression='zstd')
    print(f"✅ 基表 更新完毕 系统总标的数: {len(df_final)}")
    return df_final


# =====================================================================
# 2. 每日截面生成 (以后这个可以拓展 横向获取更多维度的数据)
# =====================================================================
def update_daily_cross_section():
    today_str = datetime.now().strftime('%Y%m%d')
    print(f"\n========================================")
    print(f" 📊 生成每日全资产截面快照 ({today_str})")
    print("========================================")
    
    cross_section_path = config.PathConfig.DATA_ROOT / 'cross_section' / f'meta_{today_str}.parquet'
    master_path = config.PathConfig.DATA_ROOT / 'meta' / 'instrument_master.parquet'
    
    if not os.path.exists(master_path):
        print("❌ 缺少基表，请先执行 update_instrument_master()")
        return
        
    df_master = pd.read_parquet(master_path)
    

    df_universe = df_master[df_master['AssetType'].isin(['stock', 'etf', 'index'])].copy()
    
    # 2. 分布式拉取三大资产的实时行情 (采用高并发稳定的东财 _em 接口（新浪实时行情方面不如东财）)
    print(" 拉取 股票 现货数据...")
    df_stock = fetch_akshare_with_retry(ak.stock_zh_a_spot_em)
    
    print(" 拉取 ETF 现货数据...")
    df_etf = fetch_akshare_with_retry(ak.fund_etf_spot_em)
    
    print(" 拉取 指数 现货数据...")
    df_index = fetch_akshare_with_retry(ak.stock_zh_index_spot_em)
    
    # 3. 统一规范清洗与垂直拼接 (Union All)
    spot_list = []
    for df, name in zip([df_stock, df_etf, df_index], ['Stock', 'ETF', 'Index']):
        if not df.empty:
            try:
                temp_df = df[['代码', '最新价', '最高', '最低', '成交量', '成交额']].copy()
                temp_df = temp_df.rename(columns={
                    '代码': 'StockCode', 
                    '最新价': 'Close',
                    '最高': 'High',
                    '最低': 'Low',
                    '成交量': 'Volume', 
                    '成交额': 'Amount'
                })
                spot_list.append(temp_df)
            except KeyError as e:
                print(f"⚠️ {name} 接口字段发生变化，缺少列: {e}")
                
    if not spot_list:
        print("❌ 所有现货源拉取失败，截面生成中止！")
        return
        
    # 将三大资产的动态价格融合成一张大表
    df_spot_all = pd.concat(spot_list, ignore_index=True)
    # 极小概率防御：不同市场可能存在六位数字代码碰撞，以防万一做个去重
    df_spot_all = df_spot_all.drop_duplicates(subset=['StockCode'], keep='first')
    
    
    # 
    # 合并行情数据
    df_daily = pd.merge(df_universe, df_spot_all, on='StockCode', how='left')
    
    # 5. ST 状态判断
    # 指数和 ETF 的简称里通常没有 ST，所以会自动被判定为 0
    df_daily['ST'] = df_daily['StockAbbreviation'].str.contains('ST|退', na=False).astype(int)
    
    # 安全类型转换，防止含有 NaN 的列变成 object 类型导致 VectorBT 读取失败
    numeric_cols = ['Close', 'High', 'Low', 'Volume', 'Amount']
    for col in numeric_cols:
        df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')
        
    os.makedirs(cross_section_path.parent, exist_ok=True)
    df_daily.to_parquet(cross_section_path, engine='pyarrow', compression='zstd')
    
    print(f"💾 截面数据已保存至: {cross_section_path} (共 {len(df_daily)} 条)")


# =====================================================================
# 3. 历史 K 线更新
# =====================================================================
def fetch_symbol_history(bs_code, start_date, end_date, asset_type):
    """
    带类型适配的单标的历史数据拉取内核
    """
    # 1. 严格适配不同资产类型的字段
    if asset_type == 'stock':
        fields_unadj = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST"
    elif asset_type == 'etf':
        fields_unadj = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg"
    elif asset_type == 'index':
        # 指数没有复权，也没有估值
        fields_unadj = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
    else:
        return pd.DataFrame() # 兜底拒绝不支持的类型

    # 2. 拉取主数据 (不复权)
    rs_unadj = bs.query_history_k_data_plus(bs_code, fields_unadj, start_date=start_date, end_date=end_date, frequency="d", adjustflag="3")
    if rs_unadj.error_code != '0': return pd.DataFrame()
    
    data_unadj = []
    while rs_unadj.next(): data_unadj.append(rs_unadj.get_row_data())
    if not data_unadj: return pd.DataFrame()
    
    df_unadj = pd.DataFrame(data_unadj, columns=rs_unadj.fields)

    # 3. 处理复权数据 (指数跳过此步骤)
    if asset_type in ['stock', 'etf']:
        fields_adj = "date,open,high,low,close,preclose"
        rs_adj = bs.query_history_k_data_plus(bs_code, fields_adj, start_date=start_date, end_date=end_date, frequency="d", adjustflag="2")
        data_adj = []
        while rs_adj.next(): data_adj.append(rs_adj.get_row_data())
        df_adj = pd.DataFrame(data_adj, columns=rs_adj.fields)
        
        rename_dict = {col: f"{col}_hfq" for col in df_adj.columns if col != 'date'}
        df_adj = df_adj.rename(columns=rename_dict)
        df_final = pd.merge(df_unadj, df_adj, on='date', how='left')
    else:
        # 指数：为了保持 DataFrame 结构统一（让策略无缝调用 xxx_hfq），直接把不复权价格复制一份作为后复权
        df_final = df_unadj.copy()
        for col in ['open', 'high', 'low', 'close', 'preclose']:
            df_final[f'{col}_hfq'] = df_final[col]

    # 4. 类型转换与物化特征 (在数据更新时计算好存储)
    for col in [c for c in df_final.columns if c not in ['date', 'code']]:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    if asset_type == 'stock' and 'turn' in df_final.columns:
        import numpy as np
        df_final['circulating_market_cap'] = np.where(
            df_final['turn'] > 0,
            (df_final['volume'] * 100 / df_final['turn']) * df_final['close'],
            np.nan 
        )
    return df_final

def update_all_history_data(global_start_date = "1990-12-19"):
    """驱动所有标的进行历史数据更新 默认从最早时间1990-12-19开始"""
    print("\n========================================")
    print(" 📚 开始历史数据更新 (Baostock)")
    print("========================================")
    
    master_path = config.PathConfig.DATA_ROOT / 'meta' / 'instrument_master.parquet'
    if not os.path.exists(master_path): return
        
    df_master = pd.read_parquet(master_path)
    
    #  只更新我们关注的三大核心资产，直接抛弃可转债和其他废弃物
    allowed_types = ['stock', 'etf', 'index']
    df_filtered = df_master[df_master['AssetType'].isin(allowed_types)]
    
    target_list = df_filtered[['SuffixStockNum', 'BsCode', 'AssetType']].dropna().to_dict('records')
    
    save_dir = config.PathConfig.DATA_ROOT / 'history_k'
    os.makedirs(save_dir, exist_ok=True)
    
    bs.login()
    today_str = datetime.now().strftime('%Y-%m-%d')
    # global_start_date = "1990-12-19" 
    
    failed_symbols = [] # 记录失败名单
    
    for item in tqdm(target_list, desc="拉取进度"):
        suffix_code = item['SuffixStockNum']
        file_path = save_dir / f"{suffix_code}.parquet"
        
        try:
            # --- 断点续传逻辑 ---
            last_date = None
            if os.path.exists(file_path):
                try:
                    # 只读取最后一行的时间戳
                    last_date = pd.to_datetime(pd.read_parquet(file_path, columns=['date'])['date'].iloc[-1])
                except: pass
                
            if last_date:
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                if start_date > today_str: continue # 已经是最新，直接跳过
            else:
                start_date = global_start_date
                
            # --- 拉取 ---
            df_new = fetch_symbol_history(item['BsCode'], start_date, today_str, item['AssetType'])
            if df_new.empty: continue
                
            df_new['code'] = suffix_code 
            
            # --- 合并 ---
            if last_date and os.path.exists(file_path):
                df_old = pd.read_parquet(file_path)
                df_final = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=['date'], keep='last')
            else:
                df_final = df_new
                
            df_final.to_parquet(file_path, engine='pyarrow', compression='zstd')
            
        except Exception as e:
            # 断网、解析错误、写入冲突时，记录并继续
            print(f"\n❌ [异常] {suffix_code} 拉取失败: {str(e)}")
            failed_symbols.append(suffix_code)
            continue 

    bs.logout()
    print("\n✅ 历史数据更新完毕！")
    
    if failed_symbols:
        print(f"⚠️ 共有 {len(failed_symbols)} 个标的更新失败，下次运行时系统将自动重试。")
        
        # 获取配置中的日志目录并确保它存在
        log_dir = config.PathConfig.LOG
        os.makedirs(log_dir, exist_ok=True)
        
        # 按照当天的日期生成独立的日志文件，防止覆盖旧账
        log_file = log_dir / f"failed_history_update_{today_str}.txt"
        
        # 使用 utf-8 编码安全写入
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Baostock 历史数据拉取失败记录 ({today_str}) ===\n")
            f.write(f"总计失败数: {len(failed_symbols)}\n")
            f.write("-" * 40 + "\n")
            for symbol in failed_symbols:
                f.write(f"{symbol}\n")
                
        print(f"📄 详细失败名单已保存至日志: {log_file}")


if __name__ == "__main__":

    # update_instrument_master()
    # update_daily_cross_section()
    # bs.login()

    # df = fetch_symbol_history("sh.600000", '2024-07-01', '2024-12-31', "stock") 
    # print(df)
    # bs.logout()

    update_all_history_data() # 初次运行可能需要几个小时，建议单独执行
    pass