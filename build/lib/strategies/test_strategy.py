
import vectorbt as vbt
from etl.data_loader import DataLoader

def run_taker_flow_strategy():

    # ----------------------------------------------------
    # 1. 加载数据
    # ----------------------------------------------------
    loader = DataLoader()
    
    # 设定回测时间和周期
    start_date = '2021-01-01'
    timeframe = '1h'
    
    # 获取本地所有币种进行测试
    symbols = loader.get_all_symbols(timeframe)
    print(f"测试币种: {symbols}")

    # 获取这些币种处理好的数据
    data = loader.get_all_kline(symbols, timeframe=timeframe, start_date=start_date)
    

    C = data['close']
    H = data['high']
    L = data['low']
    # 假设你的 Parquet 里有这一列 (主动买入量 - 主动卖出量)
    # 如果没有，可以用 taker_buy_vol - (volume - taker_buy_vol) 计算
    NetVol = data['net_taker_vol']

    # ----------------------------------------------------
    # 3. 核心因子计算
    # ----------------------------------------------------
    
    # A. 趋势过滤器: 60小时均线
    ma_trend = vbt.MA.run(C, 60)
    
    # B. 资金流异常检测 (Z-Score)
    # 逻辑：当前的主动买入净量，是否超过了过去 24 小时平均水平的 2 倍标准差？
    # 这代表"异常的巨量买入"
    net_vol_mean = NetVol.rolling(window=24).mean()
    net_vol_std = NetVol.rolling(window=24).std()
    # 避免除以0
    vol_z_score = (NetVol - net_vol_mean) / (net_vol_std + 1e-9)

    # C. 波动率计算 (ATR) - 用于动态止损
    # vbt.ATR 需要 H, L, C 三个矩阵
    atr = vbt.ATR.run(high=H, low=L, close=C, window=14)

    # ----------------------------------------------------
    # 4. 生成交易信号
    # ----------------------------------------------------
    
    # 买入条件 (Entry):
    # 1. 趋势向上 (价格 > MA60)
    # 2. 资金流爆发 (Z-Score > 2.0)
    entries = (C.values > ma_trend.ma.values) & (vol_z_score.values > 0.5)
    
    # 卖出条件 (Exit):
    # 1. 趋势反转 (价格跌破 MA60)
    # 注意：我们下面还会配置硬止损，所以这里是"趋势性离场"
    exits = C.values < ma_trend.ma.values

    # ----------------------------------------------------
    # 5. 运行回测
    # ----------------------------------------------------
    pf = vbt.Portfolio.from_signals(
        close=C,
        entries=entries,
        exits=exits,
        freq=timeframe,
        init_cash=10000,
        fees=0.0004,       # 万4手续费
        sl_stop=0.05,      # 5% 固定止损 (防止黑天鹅)
        tp_stop=0.15,      # 15% 止盈 (吃大波段)
        # 也可以利用 ATR 进行动态止损，VectorBT 支持 sl_stop 传矩阵，这里先用简单的
    )

    # ----------------------------------------------------
    # 6. 分析结果
    # ----------------------------------------------------
    print("\n" + "="*50)
    print(f"📊 资金流策略回测结果 ({start_date} 至今)")
    print("="*50)
    
    # 打印每个币种的收益率
    print("各币种收益率:")
    print(pf.total_return())
    
    print(f"\n平均夏普比率: {pf.sharpe_ratio().mean():.2f}")
    print(f"平均最大回撤: {pf.max_drawdown().mean():.2f}")

    # 挑选表现最好的一个币种画图
    best_symbol = pf.total_return().idxmax()
    print(f"\n📈 展示表现最好的币种: {best_symbol}")
    
    # 如果在 Jupyter 中，加上 .show()
    # pf[best_symbol].plot().show() 
    
    return pf

if __name__ == "__main__":
    run_taker_flow_strategy()