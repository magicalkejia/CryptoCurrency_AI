import duckdb
import pandas as pd
import config
from pathlib import Path
import os
import numpy as np

class DataLoader:
    def __init__(self):
        """
        初始化数据加载器，连接到 DuckDB 内存实例，并配置所有数据源的路径
        """
        self.conn = duckdb.connect(database=':memory:')
        
        self.crypto_dir = Path(config.PathConfig.PROCESSED)
        self.a_share_meta_dir = Path(config.PathConfig.DATA_ROOT) / 'meta'
        self.a_share_cross_dir = Path(config.PathConfig.DATA_ROOT) / 'cross_section'
        self.a_share_history_dir = Path(config.PathConfig.DATA_ROOT) / 'history_k'

    # =====================================================================
    # 模块一：A 股数据接口
    # =====================================================================
    def get_instrument_master(self, asset_type=None, status='active', include_st=False):
        """获取 A 股基表"""
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
        透视宽表矩阵 
        """
        query = f"SELECT * FROM read_parquet('{self.a_share_history_dir}/*.parquet') WHERE 1=1"
        
        if suffix_codes:
            codes_tuple = tuple(suffix_codes) if len(suffix_codes) > 1 else f"('{suffix_codes[0]}')"
            query += f" AND code IN {codes_tuple}"
            
        if start_date: query += f" AND date >= '{start_date}'"
        if end_date: query += f" AND date <= '{end_date}'"

        try:
            flat_df = self.conn.execute(query).df()
        except Exception as e:
            print(f"❌ A股数据加载失败: {e}")
            return {}

        if flat_df.empty:
            return {}

        flat_df['date'] = pd.to_datetime(flat_df['date'])
        
        aligned_data = {}
        price_cols = ['open_hfq', 'high_hfq', 'low_hfq', 'close_hfq'] if use_hfq else ['open', 'high', 'low', 'close']
        exclude_cols = ['date', 'code', 'open', 'high', 'low', 'close', 'open_hfq', 'high_hfq', 'low_hfq', 'close_hfq', 'adjustflag', 'tradestatus']
        other_cols = [c for c in flat_df.columns if c not in exclude_cols]
        
        for col in price_cols + other_cols:
            if col in flat_df.columns:
                clean_col_name = col.replace('_hfq', '') 
                pivot_df = flat_df.pivot(index='date', columns='code', values=col)
                aligned_data[clean_col_name] = pivot_df
                
        print(f"✅ 成功加载 {len(aligned_data)} 个特征矩阵，维度: {pivot_df.shape}")
        return aligned_data

    def get_factor_matrix(self, factor_name: str) -> pd.DataFrame:
        """加载独立因子矩阵宽表"""
        factor_path = config.PathConfig.DATA_ROOT / 'factors' / f'{factor_name}.parquet'
        if not os.path.exists(factor_path):
            print(f"❌ 因子文件不存在: {factor_path}")
            return pd.DataFrame()
        return pd.read_parquet(factor_path)

    # =====================================================================
    # 模块二：加密货币数据查询接口 
    # =====================================================================
    def get_crypto_kline_data(self, symbol, timeframe='1m', start_date=None, end_date=None, columns=None):
        symbol_clean = symbol.replace('/', '')
        file_path = self.crypto_dir / f"{symbol_clean}_{timeframe}.parquet"

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
                
        print(f"✅ 成功加载 {len(aligned_data)} 个特征矩阵。")
        return aligned_data

# --- 测试代码 ---
if __name__ == "__main__":
    loader = DataLoader()
    print("\n--- 测试 A 股读取 ---")
    # df_universe = loader.get_instrument_master(asset_type= "etf", include_st=False)
    
    # if not df_universe.empty:
    #     test_codes = df_universe['SuffixStockNum'].head(5).tolist()
    #     a_share_data = loader.get_a_share_matrix(suffix_codes=test_codes, start_date='2020-01-01', use_hfq=True)
    #     if a_share_data:
    #         print("\n收盘价矩阵 (Close) 头部概览:")
    #         print(a_share_data['close'].tail())

    a_share_data = loader.get_a_share_matrix(suffix_codes=["510300.SH"], start_date='1990-12-19', use_hfq=True)
    print(a_share_data)