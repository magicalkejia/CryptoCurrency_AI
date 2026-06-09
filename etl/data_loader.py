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

    #region
    # 模块一：A 股数据接口
    #endregion
    def get_instrument_master(
        self,
        asset_type=None,
        status="all",
        include_st=True,
        keyword=None,
        only_with_history=False,
        sort_by=None,
    ):
        """
        获取 A股 / ETF / 指数基表。

        设计原则：
        - 默认 status="all"，避免因为 active/delisted 状态口径导致查不到
        - 默认 only_with_history=False，先查基表，再决定是否要求本地历史数据
        - asset_type 支持单值、列表、别名、all
        - keyword 支持代码 / 简称 / suffix 模糊搜索
        """
        master_path = self.a_share_meta_dir / "instrument_master.parquet"

        if not master_path.exists():
            print("⚠️ A股基表不存在，请先运行 update_instrument_master()")
            return pd.DataFrame()

        df = pd.read_parquet(master_path).copy()

        # 基础字段标准化，避免大小写/空格导致筛不到
        for col in ["AssetType", "Status", "StockCode", "SuffixStockNum", "StockAbbreviation"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        if "AssetType" in df.columns:
            df["AssetType"] = df["AssetType"].str.lower()

        if "Status" in df.columns:
            df["Status"] = df["Status"].str.lower()

        # 标记本地是否已有 history_k 文件
        history_files = {p.stem for p in self.a_share_history_dir.glob("*.parquet")}
        if "SuffixStockNum" in df.columns:
            df["HasHistory"] = df["SuffixStockNum"].isin(history_files)
        else:
            df["HasHistory"] = False

        # 资产类型别名
        asset_alias = {
            None: None,
            "all": None,
            "market": ["stock", "etf", "index"],
            "tradable": ["stock", "etf"],
            "stocks": ["stock"],
            "stock": ["stock"],
            "etfs": ["etf"],
            "etf": ["etf"],
            "fund": ["etf"],
            "index": ["index"],
            "idx": ["index"],
        }

        if isinstance(asset_type, str):
            asset_type = asset_alias.get(asset_type.lower(), [asset_type.lower()])
        elif isinstance(asset_type, (list, tuple, set)):
            asset_type = [str(x).lower() for x in asset_type]

        if asset_type is not None and "AssetType" in df.columns:
            df = df[df["AssetType"].isin(asset_type)].copy()

        # 状态别名
        status_alias = {
            None: "all",
            "all": "all",
            "active": ["active"],
            "listed": ["active"],
            "list": ["active"],
            "上市": ["active"],
            "delisted": ["delisted"],
            "delist": ["delisted"],
            "退市": ["delisted"],
        }

        if isinstance(status, str):
            status_key = status.lower()
            status = status_alias.get(status_key, [status_key])
        elif isinstance(status, (list, tuple, set)):
            status = [str(x).lower() for x in status]

        if status != "all" and "Status" in df.columns:
            df = df[df["Status"].isin(status)].copy()

        # ST / 退过滤
        if not include_st and "StockAbbreviation" in df.columns:
            df = df[
                ~df["StockAbbreviation"].astype(str).str.contains("ST|退", case=False, na=False)
            ].copy()

        # 是否只看本地已有历史数据
        if only_with_history:
            df = df[df["HasHistory"]].copy()

        # 关键词搜索
        if keyword:
            keyword = str(keyword).strip()
            mask = pd.Series(False, index=df.index)

            search_cols = [
                "SuffixStockNum",
                "StockCode",
                "SinaCode",
                "StockAbbreviation",
                "AssetType",
                "Status",
            ]

            for col in search_cols:
                if col in df.columns:
                    mask = mask | df[col].astype(str).str.contains(keyword, case=False, na=False)

            df = df[mask].copy()

        # 列顺序
        preferred_cols = [
            "SuffixStockNum",
            "StockCode",
            "SinaCode",
            "StockAbbreviation",
            "AssetType",
            "Status",
            "ListingDate",
            "DelistingDate",
            "HasHistory",
        ]

        preferred_cols = [c for c in preferred_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in preferred_cols]
        df = df[preferred_cols + other_cols]

        # 排序
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by)
        else:
            sort_cols = [c for c in ["AssetType", "StockCode", "SuffixStockNum"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols)

        return df.reset_index(drop=True)
        
    def list_etf(self, keyword=None, only_with_history=True):
        return self.get_instrument_master(
            asset_type="etf",
            status="active",
            include_st=True,
            keyword=keyword,
            only_with_history=only_with_history,
        )


    def list_stock(self, keyword=None, include_st=True, only_with_history=True):
        return self.get_instrument_master(
            asset_type="stock",
            status="active",
            include_st=include_st,
            keyword=keyword,
            only_with_history=only_with_history,
        )


    def list_index(self, keyword=None, only_with_history=True):
        return self.get_instrument_master(
            asset_type="index",
            status="active",
            include_st=True,
            keyword=keyword,
            only_with_history=only_with_history,
        )


    def list_market(self, keyword=None, status="active", only_with_history=True):
        return self.get_instrument_master(
            asset_type="market",
            status=status,
            include_st=True,
            keyword=keyword,
            only_with_history=only_with_history,
        )

    def get_cross_section(self, date_str, asset_type='stock'):
        """获取 A 股某一日的截面快照"""
        file_path = self.a_share_cross_dir / f'meta_{date_str}.parquet'
        if not file_path.exists():
            return pd.DataFrame()
            
        query = f"SELECT * FROM read_parquet('{file_path}') WHERE AssetType = '{asset_type}'"
        return self.conn.execute(query).df()

    def get_a_share_matrix(
        self,
        suffix_codes=None,
        start_date=None,
        end_date=None,
        use_hfq=True,
        columns=None,
        allow_full_market=False,
    ):
        """
        读取 A股/ETF/指数历史行情，并透视成宽表矩阵。

        1. 如果传入 suffix_codes，只读取对应 parquet 文件，不再 glob 全市场；
        2. 使用 union_by_name=True 兼容不同 parquet schema；
        3. 使用 pivot_table 处理重复 date-code；
        4. 如果 hfq 字段不存在，自动 fallback 到原始 OHLC；
        5. 不再因为个别文件缺字段导致整体失败。
        """

        if suffix_codes is None and not allow_full_market:
            raise ValueError("请传入 suffix_codes；如确需全市场读取，设置 allow_full_market=True")
        def _quote_sql_string(s: str) -> str:
            return "'" + s.replace("'", "''") + "'"

        def _build_file_list_sql(files):
            return "[" + ", ".join(_quote_sql_string(f) for f in files) + "]"

        # =========================
        # 1. 确定读取哪些文件
        # =========================
        if suffix_codes:
            files = []
            missing_codes = []

            for code in suffix_codes:
                file_path = self.a_share_history_dir / f"{code}.parquet"
                if file_path.exists():
                    files.append(file_path.as_posix())
                else:
                    missing_codes.append(code)

            if missing_codes:
                print(f"⚠️ 以下标的文件不存在，已跳过: {missing_codes}")

            if not files:
                print("❌ 没有可读取的 A股历史行情文件")
                return {}

            source_sql = _build_file_list_sql(files)
            query = f"SELECT * FROM read_parquet({source_sql}, union_by_name=True) WHERE 1=1"

        else:
            glob_path = (self.a_share_history_dir / "*.parquet").as_posix()
            query = f"SELECT * FROM read_parquet({_quote_sql_string(glob_path)}, union_by_name=True) WHERE 1=1"

        # =========================
        # 2. 日期过滤
        # =========================
        if start_date:
            query += f" AND date >= '{start_date}'"
        if end_date:
            query += f" AND date <= '{end_date}'"

        # 如果是 glob 全市场读取，才需要 SQL 层 code 过滤；
        # 如果 suffix_codes 已经转成具体文件列表，则不再需要 code IN。
        if suffix_codes and not files:
            return {}

        try:
            flat_df = self.conn.execute(query).df()
        except Exception as e:
            print(f"❌ A股数据加载失败: {e}")
            return {}

        if flat_df.empty:
            print("⚠️ A股历史行情查询结果为空")
            return {}

        # =========================
        # 3. 基础清洗
        # =========================
        if "date" not in flat_df.columns or "code" not in flat_df.columns:
            print("❌ 数据缺少 date 或 code 字段")
            return {}

        flat_df["date"] = pd.to_datetime(flat_df["date"])
        flat_df = flat_df.sort_values(["date", "code"])

        # 防止不同文件中的 code 为空
        flat_df = flat_df.dropna(subset=["date", "code"])

        # =========================
        # 4. 选择价格字段
        # =========================
        hfq_cols = ["open_hfq", "high_hfq", "low_hfq", "close_hfq"]
        raw_cols = ["open", "high", "low", "close"]

        if use_hfq and all(c in flat_df.columns for c in hfq_cols):
            price_cols = hfq_cols
        else:
            if use_hfq:
                print("⚠️ 部分 hfq 字段不存在，自动使用原始 OHLC 字段")
            price_cols = [c for c in raw_cols if c in flat_df.columns]

        exclude_cols = [
            "date", "code",
            "open", "high", "low", "close",
            "open_hfq", "high_hfq", "low_hfq", "close_hfq",
            "adjustflag",
        ]

        other_cols = [c for c in flat_df.columns if c not in exclude_cols]
        if columns is not None:
            wanted = set(columns)
            selected_cols = [c for c in price_cols + other_cols if c.replace("_hfq", "") in wanted or c in wanted]
        else:
            selected_cols = price_cols + other_cols
        # =========================
        # 5. 透视成矩阵
        # =========================
        aligned_data = {}

        for col in selected_cols:
            # if col not in flat_df.columns:
            #     continue

            if col in ["open_hfq", "high_hfq", "low_hfq", "close_hfq"]:
                clean_col_name = col.replace("_hfq", "")
            else:
                clean_col_name = col

            try:
                pivot_df = flat_df.pivot_table(
                    index="date",
                    columns="code",
                    values=col,
                    aggfunc="last",
                ).sort_index()

                aligned_data[clean_col_name] = pivot_df

            except Exception as e:
                print(f"⚠️ 字段 {col} 透视失败，已跳过: {e}")

        if not aligned_data:
            print("❌ 没有成功生成任何特征矩阵")
            return {}

        first_key = next(iter(aligned_data))
        print(f"✅ 成功加载 {len(aligned_data)} 个特征矩阵，维度: {aligned_data[first_key].shape}")

        return aligned_data
    def get_factor_matrix(self, factor_name: str) -> pd.DataFrame:
        """加载独立因子矩阵宽表"""
        factor_path = config.PathConfig.DATA_ROOT / 'factors' / f'{factor_name}.parquet'
        if not os.path.exists(factor_path):
            print(f"❌ 因子文件不存在: {factor_path}")
            return pd.DataFrame()
        return pd.read_parquet(factor_path)

    def export_local_instrument_catalog(self, output_path=None):
        """
        导出本地标的目录，方便用 Excel / WPS / VSCode Data Viewer 查看。
        """
        catalog = self.list_local_instruments(
            asset_type=None,
            status="all",
            include_st=True,
            only_with_history=False,
        )

        if output_path is None:
            output_path = self.a_share_meta_dir / "local_instrument_catalog.csv"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        catalog.to_csv(output_path, index=False, encoding="utf-8-sig")

        parquet_path = output_path.with_suffix(".parquet")
        catalog.to_parquet(parquet_path, index=False)

        print(f"✅ 已导出本地标的目录: {output_path}")
        print(f"✅ 已导出 Parquet 版本: {parquet_path}")

        return catalog
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