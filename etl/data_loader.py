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
    # 模块二：加密货币数据查询接口 
    # =====================================================================
    def get_crypto_kline_data(
        self,
        symbol,
        timeframe="1m",
        start_date=None,
        end_date=None,
        columns=None,
    ):
        symbol_clean = symbol.replace("/", "")
        file_path = self.crypto_dir / f"{symbol_clean}_{timeframe}.parquet"

        if not file_path.exists():
            print(f"⚠️ Crypto 文件不存在: {file_path}")
            return None

        allowed_cols = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "taker_buy_vol",
            "net_taker_vol",
        }

        # 默认全字段；如果指定 columns，强制带 timestamp，便于后面 set_index
        if columns is None:
            selected_cols = list(allowed_cols)
        else:
            if isinstance(columns, str):
                columns = [columns]

            invalid_cols = set(columns) - allowed_cols
            if invalid_cols:
                raise ValueError(f"Invalid crypto columns: {invalid_cols}")

            selected_cols = ["timestamp"] + [c for c in columns if c != "timestamp"]

        # DuckDB 标识符安全引用
        cols_sql = ", ".join([f'"{c}"' for c in selected_cols])

        query = f"""
            SELECT {cols_sql}
            FROM read_parquet(?)
            WHERE 1=1
        """

        params = [file_path.as_posix()]

        if start_date is not None:
            query += " AND timestamp >= ?"
            params.append(pd.to_datetime(start_date))

        if end_date is not None:
            query += " AND timestamp <= ?"
            params.append(pd.to_datetime(end_date))

        query += " ORDER BY timestamp ASC"

        try:
            df = self.conn.execute(query, params).df()

            if df.empty:
                return df

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

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

    def get_crypto_matrix(
        self,
        symbols=None,
        timeframe="1m",
        start_date=None,
        end_date=None,
        columns=None,
    ):
        if symbols is None:
            symbols = self.get_all_crypto_symbols(timeframe)

        datasets = {}
        print(f"读取加密货币数据 (周期: {timeframe})...")

        for s in symbols:
            df = self.get_crypto_kline_data(
                symbol=s,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                columns=columns,
            )

            if df is not None and not df.empty:
                for col_name in df.columns:
                    if col_name not in datasets:
                        datasets[col_name] = []
                    datasets[col_name].append(df[col_name].rename(s))

        aligned_data = {}
        for col_name, series_list in datasets.items():
            aligned_data[col_name] = pd.concat(series_list, axis=1) if series_list else pd.DataFrame()

        print(f"✅ 成功加载 {len(aligned_data)} 个特征矩阵。")
        return aligned_data

# --- 测试代码 ---
if __name__ == "__main__":
    loader = DataLoader()
