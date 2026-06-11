# crypto 集成说明 / Bug 报告 / 范围声明

本文件说明:在现有项目(`CryptoCurrency_AI-main`)基础上,按 v6 方案 + 第六轮审计修订新增的 `crypto/` 包,如何集成、修了哪些 bug、实现了哪些、还差哪些、怎么跑。

---

## 1. 对现有代码的改动

只改了两处,且都是"明显 bug / 纯配置增项":

1. **`etl/crypto_pipeline.py`(明显 bug,已修)**
   原来 `df_result = pd.DataFrame(results); print(...); return df_result` 三行**缩进在 `for symbol` 循环体内**,导致 `run_crypto_pipeline()` 处理完**第一个币种(BTC)就 return** 了,ETH/SOL 永远不会被抓取/清洗/重采样。已把这三行 de-indent 到循环外。改动处有注释标明。

2. **`pyproject.toml`(纯配置增项)**
   注册了 `crypto` 及其子包到 `[tool.setuptools].packages`;在依赖里加了 `scikit-learn / scipy / lightgbm / pytest`。没有改你们既有依赖版本。

**没有改动**:`backtest/`、`etl/data_loader.py`、`data_processor.py`、`data_updater.py`、`config.py`、`main.py` 的任何逻辑。`crypto` 通过 `adapters.py` 单向调用你们的 `DataLoader` 和 `backtest.engine`。

---

## 2. 其它发现的问题(未改,仅指出)

- **`strategies/test_strategy.py`**:`import vectorbt`(不在 `pyproject` 依赖里),且调用了**不存在的方法** `loader.get_all_symbols` / `loader.get_all_kline`(实际是 `get_all_crypto_symbols` / `get_crypto_matrix`)。这是个练习脚本,我没动它。若要用,需加 vectorbt 依赖并改方法名。

- **数据字段缺口(与 v6 不兼容,重要)**:`data_updater.fetch_data` 只从 Binance USDM 抓 `open/high/low/close/volume/taker_buy_vol`,**没有 funding rate / open interest / basis / liquidation**。因此 v6 的"衍生品因子"和"funding 一等成本"目前**无数据可用**。我把 funding 成本/PnL 的接口写好了(`exec_price.funding_return`,符号正确,审计 #12),但回测里只有在你补充 funding 历史(Binance `fapiPublic GetFundingRate`)后才会真正生效;无 funding 时自动降级为 0,不报错。

- **回测引擎频率**:`backtest.engine` 是 close-to-close、`annual_days=252`(股票口径)。引擎本身是通用按行计算的,我**没改**;在 v6 适配里调用时传 `annual_days=2190`(4h crypto)。注意它是"按 bar close 成交"的向量化口径,与 v6 triple-barrier 用的"下一根 open 成交"是两套口径——前者用于组合级权重回测,后者用于事件级标签,二者目的不同,已在范围说明里区分。

- **`config.TargetConfig.COINS`** 目前是 BTC/ETH/SOL(无 BNB)。v6 用四标的,按需在 config 里加 `'BNB/USDT'` 即可。

---

## 3. 第六轮审计必修项的落地情况

| 审计必修项 | 落地 | 位置 / 测试 |
|---|---|---|
| 空头收益用 `1 - exit/entry`(非 `entry/exit-1`) | ✅ | `triple_barrier.py`;测试 `T1b_16` |
| 同 bar 双触碰判 ambiguous(不偏置方向) | ✅ | `triple_barrier.py`(`intrabar_dual_touch="ambiguous"`);`T1b_02` |
| 拆分 entry / exit 成交价,滑点不重复扣 | ✅ | `exec_price.py`(`get_entry_price` + `apply_slippage`);约定:价格不含滑点,成本项扣一次 |
| 净收益 `raw + funding_return - costs`(funding 符号不反转) | ✅ | `exec_price.net_return` / `funding_return`;`T1b_17` |
| 多空净收益分列 `net_exit_return_long/short` | ✅ | `triple_barrier.LABEL_COLUMNS`;`build_meta_label`;`T1b_10/11/12/16` |
| 强烈建议:OOF train/test indices hash | ✅ | `purged_kfold.FoldSplit.train_indices_hash` |
| bars 加 `ts_close` + `availability_ts ≥ ts_close` | ✅ | `adapters.to_bars_schema` |
| pooled uniqueness 资产均衡公式 | ✅ | `uniqueness.average_uniqueness(scope="pooled")` |
| 深度代理最小样本数 | ✅ | `exec_price.depth_proxy`(`min_depth_samples`) |
| 短大幅波动 / dual touch / 不重复扣滑点 等测试 | ✅ | `tests/test_crypto.py` |
| MVP 双头去循环依赖(meta_label 来自 OOF) | ✅ | `meta_label.build_meta_label(source="oof")`;`T1b_13` |
| Holdout-A 冻结全部参数 + config/env hash | ✅ | `schemas.FrozenConfig.config_hash` / `environment_hash` |

其余 P2/P3(GTC→GTD、TWAP 漂移、kill switch 恢复、N=5 等)属于实盘执行,见第 5 节范围。

---

## 4. 已实现并通过测试的部分(阶段 1a/1b 核心)

`crypto/` 包结构:

```
crypto/
  schemas.py           # FrozenConfig + 5 个配置 dataclass + config/env hash
  pit.py               # audit_lookahead, make_supervised_dataset(防标签泄露/PIT)
  exec_price.py        # 成交价(entry/exit 分离)+ 滑点 + funding(符号正确)+ 净收益
  adapters.py          # 桥接现有 DataLoader -> bars schema(ts_close/availability_ts)
  labels/
    triple_barrier.py  # 三屏障(ATR、半开区间、dual-touch ambiguous、多空净收益)
    meta_label.py      # 多空元标签,强制 OOF 来源
  features/uniqueness.py  # 平均唯一性(半开 O(N+T)、within/pooled 资产均衡)
  cv/purged_kfold.py      # purged+embargo+多资产 time-block,FoldSplit 审计
  benchmark/tsmom.py      # 波动率平价 TSMOM(协方差 port_vol、eps 防御、no-trade band)
  models/base_lgb.py      # LightGBM 优先 / sklearn 兜底
  models/calibrate.py     # Platt/isotonic + compute_ece
  eval/significance.py    # Newey-West IC t / block bootstrap / Deflated Sharpe
  pipeline_1b.py          # 两阶段 meta-labeling walk-forward OOF + 校准
tests/test_crypto.py  # 18 个单元测试,对应 v6 附录 A(全部通过)
demo.py                  # 合成数据端到端演示(含现有回测引擎对比 TSMOM)
```

**测试结果**:`python -m tests.run_all` → 18 passed, 0 failed(覆盖 T1a_10、T1b_01~20 的关键项 + 滑点量纲 + ECE)。

**端到端**:`python demo.py` 在合成数据上跑通 标签→唯一性→数据集→1b 两阶段 OOF→校准→生成权重→现有回测引擎 vs TSMOM 基准。诊断会打印 `tb_label_distribution`(演示中偏空,正是非对称屏障 tp=2/sl=1 的体现,pipeline 按审计 #5 如实报告)。

---

> 实盘部分务必注意:永续合约有真实亏损(含爆仓)风险,本代码不构成投资建议;任何实盘前请先 testnet 充分验证、用可承受的小资金灰度。

---

## 6. 如何运行

```bash
# 1) 安装(在虚拟环境里)
pip install -e .          # 会带上 scikit-learn/scipy/lightgbm/pytest

# 2) 跑单元测试(任选其一)
pytest tests/ -q
python -m tests.run_all     # 不依赖 pytest 的运行器


# 4) 接真实数据(在已跑过 ETL、PROCESSED 下有 *_1h.parquet 之后)
#    用 crypto.adapters.load_crypto_bars(DataLoader(), "BTC/USDT", "1h", ...) 取 bars,
#    再走 triple_barrier -> uniqueness -> make_supervised_dataset -> run_phase1b。
```

---

```bash
python run_on_real_data.py --symbols BTC/USDT ETH/USDT SOL/USDT BNB/USDT
```

## 跑一下
```bash
python agent_demo.py          # 7-Agent 状态机端到端(合成数据)，打印结构化决策 + 审计日志 + Risk 否决演示
python -m tests.run_all    # 38 passed
```

## 与之前确定性 pipeline 的关系
`pipeline_1b.py`（确定性、walk-forward OOF）用于**研究/回测/评估**（严谨、可复现）；Agent 编排层用于**决策时的推理流**（可解释、可审计、可降级、可接实盘）。二者共用同一批 Skills/模型，互不冲突——这正是 v6 "研究闭环 + 决策编排" 的设计。

---

## 3. PatchTST A/B/C/D 正式对比实验 

## 跑一下
```bash
python -m tests.run_all    # 46 passed
```

## 还有如下开发工作，可佳重点关注
1. **Dune**：(a) 你的 Dune **API key**；(b) 你在 dune.com 建好的 **query_id**（或要我把 `DUNE_SQL_TEMPLATES` 扩成 BTC/SOL/BNB 各链的可复算指标 SQL，你建查询后回填 id）；(c) **每个标的对应哪条链**做链上因子？BTC 本身链上活跃度对价格信号弱，常见做法是用 **ETH 链上 + 稳定币流** 作为整体 risk-on/off 代理——你倾向哪种映射？SOL→Solana、BNB→BNB Chain 的 Dune 表覆盖也要确认。
2. **链上日频 vs 4h 决策**：确认接受链上作为**慢因子**（asof 前向填充到 4h）即可，对吧？
3. **实盘行情**：用哪个交易所 **testnet**？需要 `ccxt.pro`（付费）还是用免费 websocket 自己聚合?（`CCXTProFeed` 现按 ccxt.pro 写，若你们用免费源我再加一个 `websocket-client` 版本。）
