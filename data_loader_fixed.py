import duckdb
import pandas as pd
import config
from pathlib import Path
import numpy as np


class DataLoader:
    def __init__(self):
        """
        初始化数据加载器，连接到 DuckDB 内存实例，并配置所有数据源路径。

        路径兼容两种情况：
        1. config.PathConfig 已显式定义 RAW_FUNDING / PROCESSED_DERIVATIVES 等目录；
        2. config.PathConfig 只有 RAW / PROCESSED 基础目录。
        """
        self.conn = duckdb.connect(database=":memory:")

        self.raw_dir = Path(config.PathConfig.RAW)
        self.processed_dir = Path(config.PathConfig.PROCESSED)
        self.crypto_dir = self.processed_dir

        self.raw_derivatives_dir = Path(
            getattr(config.PathConfig, "RAW_DERIVATIVES", self.raw_dir / "derivatives")
        )
        self.raw_funding_dir = Path(
            getattr(config.PathConfig, "RAW_FUNDING", self.raw_derivatives_dir / "funding")
        )
        self.raw_oi_dir = Path(
            getattr(config.PathConfig, "RAW_OI", self.raw_derivatives_dir / "oi")
        )

        self.processed_derivatives_dir = Path(
            getattr(config.PathConfig, "PROCESSED_DERIVATIVES", self.processed_dir / "derivatives")
        )

        # 历史股票路径，暂时保留兼容，不参与 crypto 主线。
        self.a_share_meta_dir = Path(config.PathConfig.DATA_ROOT) / "meta"
        self.a_share_cross_dir = Path(config.PathConfig.DATA_ROOT) / "cross_section"
        self.a_share_history_dir = Path(config.PathConfig.DATA_ROOT) / "history_k"

    # =====================================================================
    # Crypto K线查询接口
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

        if columns is None:
            selected_cols = list(allowed_cols)
        else:
            if isinstance(columns, str):
                columns = [columns]

            invalid_cols = set(columns) - allowed_cols
            if invalid_cols:
                raise ValueError(f"Invalid crypto columns: {invalid_cols}")

            selected_cols = ["timestamp"] + [c for c in columns if c != "timestamp"]

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
            return df.set_index("timestamp")
        except Exception as e:
            print(f"❌ Crypto 数据加载失败 {symbol}: {e}")
            return None

    def get_all_crypto_symbols(self, timeframe="1m"):
        files = list(self.crypto_dir.glob(f"*_{timeframe}.parquet"))
        symbols = []
        for f in files:
            clean_name = f.name.replace(f"_{timeframe}.parquet", "")
            if clean_name.endswith("USDT"):
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
                    datasets.setdefault(col_name, []).append(df[col_name].rename(s))

        aligned_data = {
            col_name: pd.concat(series_list, axis=1)
            for col_name, series_list in datasets.items()
            if series_list
        }

        print(f"✅ 成功加载 {len(aligned_data)} 个特征矩阵。")
        return aligned_data

    # =====================================================================
    # Funding rate 查询接口
    # =====================================================================
    def get_funding_rate_data(
        self,
        symbol=None,
        symbols=None,
        start_date=None,
        end_date=None,
        columns=None,
        processed: bool = True,
        as_index: bool = True,
    ):
        """
        查询资金费率数据。

        默认读取 processed long table：
            data_storage/processed/derivatives/funding.parquet

        返回：
        - 不传 symbol / symbols：返回完整 funding long table；
        - 传 symbol="BTC/USDT"：默认返回该 symbol，index=timestamp；
        - 传 symbols=[...]：返回多币种 long table；
        - processed=False：读取 raw per-symbol 文件并 concat。

        参数：
        - symbol: 单个币种，例如 "BTC/USDT"；
        - symbols: 多币种列表。若 symbol 和 symbols 同时传入，优先使用 symbol；
        - columns: 除 timestamp/symbol 外需要保留的字段；
        - processed: True 读取 processed long table；False 读取 raw per-symbol 文件；
        - as_index: 单币种查询时是否设置 timestamp 为 index。
        """
        if symbol is not None:
            symbols = [symbol]
        elif symbols is not None:
            symbols = list(symbols)

        if processed:
            df = self._load_processed_funding_long()
            if df is None:
                return None

            if symbols is not None:
                df = df[df["symbol"].isin(symbols)].copy()
        else:
            # raw 是 per-symbol 文件；不传 symbol/symbols 时默认读取 TargetConfig.COINS。
            if symbols is None:
                symbols = list(config.TargetConfig.COINS)

            frames = []
            for s in symbols:
                raw = self.get_raw_funding_rate_data(
                    symbol=s,
                    start_date=start_date,
                    end_date=end_date,
                    as_index=False,
                )
                if raw is not None and not raw.empty:
                    frames.append(raw)

            if not frames:
                return None
            df = pd.concat(frames, ignore_index=True)

        if df.empty:
            return df

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        if start_date is not None:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]

        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]
            base_cols = ["timestamp", "symbol"]
            selected = base_cols + [c for c in columns if c not in base_cols]
            missing = [c for c in selected if c not in df.columns]
            if missing:
                raise ValueError(f"Invalid funding columns: {missing}")
            df = df[selected]

        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        # 只有明确查单币种时才默认设置 index。
        if as_index and symbols is not None and len(symbols) == 1:
            return df.set_index("timestamp")

        return df

    def _load_processed_funding_long(self):
        file_path = self.processed_derivatives_dir / "funding.parquet"

        if not file_path.exists():
            print(f"⚠️ Processed funding long table 不存在: {file_path}")
            return None

        try:
            df = pd.read_parquet(file_path)
            if df.empty:
                return df

            required_cols = {"timestamp", "symbol", "funding_rate"}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Processed funding table missing columns: {missing}")

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp", "symbol"])
            return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        except Exception as e:
            print(f"❌ Funding long table 加载失败: {e}")
            return None

    def get_raw_funding_rate_data(
        self,
        symbol,
        start_date=None,
        end_date=None,
        as_index: bool = True,
    ):
        """
        查询 raw per-symbol funding 文件。

        新路径：
            data_storage/raw/derivatives/funding/{SYMBOL}.parquet

        兼容旧路径：
            data_storage/raw/{SYMBOL}_funding.parquet
        """
        if symbol is None:
            raise ValueError("get_raw_funding_rate_data(symbol=...) 必须指定 symbol，因为 raw funding 是 per-symbol 文件。")

        symbol_clean = symbol.replace("/", "")
        new_path = self.raw_funding_dir / f"{symbol_clean}.parquet"
        legacy_path = self.raw_dir / f"{symbol_clean}_funding.parquet"

        file_path = new_path if new_path.exists() else legacy_path
        if not file_path.exists():
            print(f"⚠️ Funding 文件不存在: {new_path} or {legacy_path}")
            return None

        try:
            df = pd.read_parquet(file_path)
            if df.empty:
                return df

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            if "symbol" not in df.columns:
                df["symbol"] = symbol
            else:
                df["symbol"] = df["symbol"].fillna(symbol)

            if start_date is not None:
                df = df[df["timestamp"] >= pd.to_datetime(start_date)]
            if end_date is not None:
                df = df[df["timestamp"] <= pd.to_datetime(end_date)]

            df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
            if as_index:
                return df.set_index("timestamp")
            return df
        except Exception as e:
            print(f"❌ Raw funding 加载失败 {symbol}: {e}")
            return None

    def get_funding_matrix(
        self,
        symbols=None,
        field: str = "funding_rate",
        start_date=None,
        end_date=None,
    ):
        """
        将 processed funding long table 转成宽表矩阵。

        输出：
            index = timestamp
            columns = symbol
            values = field
        """
        df = self.get_funding_rate_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            processed=True,
            as_index=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if field not in df.columns:
            raise ValueError(f"Invalid funding field: {field}")

        return (
            df.pivot_table(
                index="timestamp",
                columns="symbol",
                values=field,
                aggfunc="last",
            )
            .sort_index()
        )

    def get_latest_funding_rates(
        self,
        symbols=None,
        field: str = "funding_rate",
    ):
        """返回每个 symbol 最新一条 funding 记录。"""
        df = self.get_funding_rate_data(symbols=symbols, processed=True, as_index=False)
        if df is None or df.empty:
            return pd.DataFrame()

        if field not in df.columns:
            raise ValueError(f"Invalid funding field: {field}")

        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        idx = df.groupby("symbol")["timestamp"].idxmax()
        latest = df.loc[idx].sort_values("symbol").reset_index(drop=True)
        return latest[["symbol", "timestamp", field]]


if __name__ == "__main__":
    loader = DataLoader()
    print(loader.get_funding_rate_data(symbol=None).head())
