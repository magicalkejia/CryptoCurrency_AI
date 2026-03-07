import duckdb
import pandas as pd
import config
from pathlib import Path

class DataLoader:
    def __init__(self):
        """
        初始化数据加载器，连接到 DuckDB 内存实例，并配置所有数据源的路径
        """
        # DuckDB 内存实例 
        self.conn = duckdb.connect(database=':memory:')
        
        # 加密货币路径
        self.crypto_dir = Path(config.PathConfig.PROCESSED)
        
        # A 股路径配置
        self.a_share_meta_dir = Path(config.PathConfig.DATA_ROOT) / 'meta'
        self.a_share_cross_dir = Path(config.PathConfig.DATA_ROOT) / 'cross_section'
        self.a_share_history_dir = Path(config.PathConfig.DATA_ROOT) / 'history_k'

    # =====================================================================
    # 模块一：A 股专属数据接口 (A-Share Data Access)
    # =====================================================================
    def get_instrument_master(self, asset_type=None, status='active', include_st=False):
        """获取 A 股基表 """
        master_path = self.a_share_meta_dir / 'instrument_master.parquet'
        if not master_path.exists():
            print("⚠️ A股基表不存在")
            return pd.DataFrame()

        query = f"SELECT * FROM read_parquet('{master_path}') WHERE 1=1"
        if status != 'all':
            query += f" AND Status = '{status}'"
        if asset_type:
            query += f" AND AssetType = '{asset_type}'"
        if not include_st:
            query += " AND StockAbbreviation NOT LIKE '%ST%' AND StockAbbreviation NOT LIKE '%退%'"
            
        return self.conn.execute(query).df()

    def get_cross_section(self, date_str, asset_type='stock'):
        """获取 A 股某一日的截面快照"""
        file_path = self.a_share_cross_dir / f'meta_{date_str}.parquet'
        if not file_path.exists():
            return pd.DataFrame()
            
        query = f"SELECT * FROM read_parquet('{file_path}') WHERE AssetType = '{asset_type}'"
        return self.conn.execute(query).df()

    def get_a_share_matrix(self, suffix_codes=None, start_date=None, end_date=None, use_hfq=True):
        """
        通过 DuckDB 读取文件，并透视转换成字典宽表矩阵
        
        参数:
        - suffix_codes: 指定标的列表，不传则读取所有
        - use_hfq: 默认使用后复权(hfq)的 OHLC 数据喂给回测引擎
        """
        
        # 1. 构造 DuckDB 全局批量查询
        query = f"SELECT * FROM read_parquet('{self.a_share_history_dir}/*.parquet') WHERE 1=1"
        
        if suffix_codes:
            codes_tuple = tuple(suffix_codes) if len(suffix_codes) > 1 else f"('{suffix_codes[0]}')"
            query += f" AND code IN {codes_tuple}"
            
        if start_date: query += f" AND date >= '{start_date}'"
        if end_date: query += f" AND date <= '{end_date}'"

        # 2. 一次性将所有满足条件的数据拉入内存
        try:
            flat_df = self.conn.execute(query).df()
        except Exception as e:
            print(f"❌ A股数据加载失败: {e}")
            return {}

        if flat_df.empty:
            return {}

        # 3. 处理时间索引
        flat_df['date'] = pd.to_datetime(flat_df['date'])
        
        # 4. 透视转换 (Pivot)
        # 把长表 (百万行) 瞬间变成宽表 (Index=Date, Columns=Symbols)
        aligned_data = {}
        
        # 定义你要喂给回测引擎的核心价格列
        price_cols = ['open_hfq', 'high_hfq', 'low_hfq', 'close_hfq'] if use_hfq else ['open', 'high', 'low', 'close']
        
        # 定义【黑名单】：除了时间、代码、以及不需要的另一套价格列外，其他的全当做因子列提取
        exclude_cols = ['date', 'code', 'open', 'high', 'low', 'close', 'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq', 'adjustflag', 'tradestatus']
        
        # 动态找出底层 flat_df 中实际存在的所有其他有效特征
        other_cols = [c for c in flat_df.columns if c not in exclude_cols]
        
        # 遍历所有需要提取的列
        for col in price_cols + other_cols:
            if col in flat_df.columns:
                # 把复权后缀洗掉，方便 VectorBT 统一调用
                clean_col_name = col.replace('_hfq', '') 
                
                # 透视成宽表
                pivot_df = flat_df.pivot(index='date', columns='code', values=col)
                aligned_data[clean_col_name] = pivot_df
                
        print(f"查询数据列: {list(aligned_data.keys())}")
        return aligned_data


    # =====================================================================
    # 模块二：加密货币数据查询接口 
    # =====================================================================
    def get_crypto_kline_data(self, symbol, timeframe='1m', start_date=None, end_date=None, columns=None):
        """获取单个加密货币的 K 线数据"""
        symbol_clean = symbol.replace('/', '')
        file_name = f"{symbol_clean}_{timeframe}.parquet"
        file_path = self.crypto_dir / file_name

        if not file_path.exists(): return None

        cols_sql = ",".join(columns) if columns else "*"
        query = f"SELECT {cols_sql} FROM read_parquet('{file_path}')"
        
        conditions = []
        if start_date: conditions.append(f"timestamp >= '{start_date}'")
        if end_date: conditions.append(f"timestamp <= '{end_date}'")
        
        if conditions: query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp ASC"

        try:
            df = self.conn.execute(query).df()
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"❌ Crypto 数据加载失败 {symbol}: {e}")
            return None

    def get_all_crypto_symbols(self, timeframe='1m'):
        """扫描本地拥有哪些加密货币资产"""
        files = list(self.crypto_dir.glob(f"*_{timeframe}.parquet"))
        symbols = []
        for f in files:
            clean_name = f.name.replace(f'_{timeframe}.parquet', '')
            if clean_name.endswith('USDT'):
                symbols.append(f"{clean_name[:-4]}/USDT")
            else:
                symbols.append(clean_name)
        return symbols

    def get_crypto_matrix(self, symbols=None, timeframe='1m', start_date=None, end_date=None):
        """
        将多个币种的序列对齐并拼接为 VectorBT 适用的宽表字典
        """
        if symbols is None:
            symbols = self.get_all_crypto_symbols(timeframe)

        datasets = {} 
        print(f"读取加密货币数据 (周期: {timeframe})...")
        
        for s in symbols:
            df = self.get_crypto_kline_data(s, timeframe, start_date, end_date)
            if df is not None and not df.empty:
                for col_name in df.columns:
                    if col_name not in datasets: datasets[col_name] = []
                    datasets[col_name].append(df[col_name].rename(s))

        aligned_data = {}
        for col_name, series_list in datasets.items():
            aligned_data[col_name] = pd.concat(series_list, axis=1) if series_list else pd.DataFrame()
                
        return aligned_data

# --- 测试代码 ---
if __name__ == "__main__":
    loader = DataLoader()
    
    # ---------------- 测试加密货币 ----------------
    # print("\n--- 1. 测试 Crypto 读取 ---")
    # crypto_symbols = loader.get_all_crypto_symbols('1h')
    # if crypto_symbols:
    #     crypto_data = loader.get_crypto_matrix(start_date='2023-01-01')
    #     print(f"Crypto 特征: {list(crypto_data.keys())}")
        
    # ---------------- 测试 A 股 ----------------
    print("\n--- 2. 测试 A 股读取 ---")
    # 获取正常的、非 ST 的大盘股名单
    df_universe = loader.get_instrument_master(asset_type= "etf", include_st=False)
    # print(df_universe)
    
    if not df_universe.empty:
        # 取前 5 只股票测试
        test_codes = df_universe['SuffixStockNum'].head(5).tolist()
        
        # 获取矩阵喂给 VectorBT
        a_share_data = loader.get_a_share_matrix(suffix_codes=test_codes, start_date='2020-01-01', use_hfq=True)
        
        if a_share_data:
            print(f"A股特征: {list(a_share_data.keys())}")
            print("\n收盘价矩阵 (Close):")
            print(a_share_data)