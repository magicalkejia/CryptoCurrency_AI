import duckdb
import pandas as pd
import config
from pathlib import Path

class DataLoader:
    def __init__(self):
        """
        初始化数据加载器，连接到 DuckDB 内存实例
        """
        self.base_dir = Path(config.PathConfig.PROCESSED)
        # 创建一个内存数据库连接，用于查询 Parquet
        self.conn = duckdb.connect(database=':memory:')

    def get_kline_data(self, symbol, timeframe='1m', start_date=None, end_date=None, columns=None):
        """
        获取单币种 K 线数据
        参数 columns:如果不传(None)，默认查询所有列 (*)
        """
        # 1. 构造文件路径
        symbol_clean = symbol.replace('/', '')
        file_name = f"{symbol_clean}_{timeframe}.parquet"
        file_path = self.base_dir / file_name

        if not file_path.exists():
            return None

        # 2. 构造 SQL 查询
        # 如果 columns 是 None，就查 * (所有列)
        cols_sql = ",".join(columns) if columns else "*"
        
        query = f"SELECT {cols_sql} FROM read_parquet('{file_path}')"
        
        # 3. 过滤条件
        conditions = []
        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp ASC"

        try:
            df = self.conn.execute(query).df()
            if df.empty:
                return df

            # 4. 设置索引
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df

        except Exception as e:
            print(f"❌ 数据加载失败 {symbol}: {e}")
            return None

    def get_all_symbols(self, timeframe='1m'):
        """扫描本地有哪些币种"""
        files = list(self.base_dir.glob(f"*_{timeframe}.parquet"))
        symbols = []
        for f in files:
            clean_name = f.name.replace(f'_{timeframe}.parquet', '')
            if clean_name.endswith('USDT'):
                symbol = f"{clean_name[:-4]}/USDT"
                symbols.append(symbol)
            else:
                symbols.append(clean_name)
        return symbols

    def get_all_kline(self, symbols=None, timeframe='1m', start_date=None, end_date=None):
        """
        动态全量加载函数
        不指定的话会把本地有的币种K线数据全部查询整理成方便回测的字典矩阵格式
        """
        if symbols is None:
            symbols = self.get_all_symbols(timeframe)
            # print(f"🔍 自动扫描到 {len(symbols)} 个币种 ({timeframe})")

        # 1. 使用字典动态存储所有列的数据
        # 结构: {'close': [Series_BTC, Series_ETH], 'vol': [...], 'funding': [...]}
        datasets = {} 
        
        print(f"读取数据 (周期: {timeframe})...")
        
        count = 0
        for s in symbols:
            # 默认 columns=None，即获取所有列
            df = self.get_kline_data(s, timeframe, start_date, end_date)
            
            if df is not None and not df.empty:
                count += 1
                
                # 遍历该 DataFrame 的所有列
                for col_name in df.columns:
                    # 如果这个列名第一次出现，在 datasets 里初始化一个空列表
                    if col_name not in datasets:
                        datasets[col_name] = []
                    
                    # 将该列改名为币种名，并加入列表
                    datasets[col_name].append(df[col_name].rename(s))
        
        # print(f"✅ 成功加载 {count}/{len(symbols)} 个币种")

        # 2. 合并对齐
        aligned_data = {}
        for col_name, series_list in datasets.items():
            if series_list:
                # 拼接成宽表 (Index=Time, Columns=Symbols)
                aligned_data[col_name] = pd.concat(series_list, axis=1)
            else:
                aligned_data[col_name] = pd.DataFrame()
                
        return aligned_data

# --- 测试代码 ---
if __name__ == "__main__":
    loader = DataLoader()
    
    all_symbols = loader.get_all_symbols('1h')[:3] # 只测前3个
    
    if all_symbols:
        data = loader.get_all_kline(all_symbols, '1h', start_date='2021-01-01')
        
        print("\n识别到的所有数据字段:")
        print(list(data.keys())) 
        # 输出示例: ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'trades']
        
        print(data['high'].tail(5))