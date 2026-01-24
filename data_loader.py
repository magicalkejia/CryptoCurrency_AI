import duckdb
import pandas as pd
import config


class DataLoader:
    def __init__(self):
        """
        初始化数据加载器，连接到 DuckDB 内存实例
        """
        self.base_dir = config.PathConfig.PROCESSED
        # 创建一个内存数据库连接，用于快速查询 Parquet
        self.conn = duckdb.connect(database=':memory:')

    def get_data(self, symbol, timeframe='1m', start_date=None, end_date=None, columns=None):
        """
        核心接口：获取 K 线数据
        
        参数:
            symbol (str): 币种，如 'BTC/USDT'
            timeframe (str): 周期，如 '1m', '1h', '4h'
            start_date (str): '2023-01-01' (可选)
            end_date (str): '2023-01-31' (可选)
            columns (list): 需要加载的列，默认全部 (可选)
        
        返回:
            pd.DataFrame: 设置好 DateTimeIndex 的数据
        """
        # 1. 构造文件路径
        symbol_clean = symbol.replace('/', '')
        file_name = f"{symbol_clean}_{timeframe}.parquet"
        file_path = self.base_dir / file_name

        if not file_path.exists():
            print(f"❌ 数据文件不存在: {file_path}")
            return None

        # 2. 构造 SQL 查询
        # 如果用户没指定列，就查所有 (*)
        cols_sql = ",".join(columns) if columns else "*"
        

        query = f"SELECT {cols_sql} FROM read_parquet('{file_path}')"
        
        # 3. 添加过滤条件 (WHERE 子句)
        conditions = []
        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # 4. 排序 (确保时序正确)
        query += " ORDER BY timestamp ASC"

        try:
            # 5. 执行查询并转为 Pandas
            # print(f"Executing: {query}") # 调试时可以打开
            df = self.conn.execute(query).df()
            
            if df.empty:
                print(f"⚠️ 查询结果为空: {symbol} {timeframe}")
                return df

            # 6. 后处理：设置时间索引 (Quants 习惯)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return None

    def get_all_symbols(self):
        """
        辅助功能：查看本地都有哪些币种的数据
        """
        files = list(self.base_dir.glob("*_1m.parquet"))
        symbols = [f.name.replace('_1m.parquet', '') for f in files]
        # 尝试还原 /USDT 格式 (假设都是 USDT 结尾)
        return [f"{s[:-4]}/USDT" for s in symbols]

# --- 单例测试 ---
if __name__ == "__main__":
    loader = DataLoader()
    
    # 测试加载
    # print("🔍 正在查询 BTC 2023年1月 的 1小时数据...")
    # df = loader.get_data(
    #     symbol='BTC/USDT', 
    #     timeframe='1h', 
    #     start_date='2024-01-01', 
    #     end_date='2024-01-31'
    # )
    
    # if df is not None:
    #     print(df.head())
    #     print(f"📊 数据形状: {df.shape}")
    symbol_list = loader.get_all_symbols()
    print(symbol_list)