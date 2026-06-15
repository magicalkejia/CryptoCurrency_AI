## 📂 系统架构 (Project Structure)

```text
TRADINGSYSTEM/
├── backtest/
│   ├── engine.py              # Vectorized backtest engine
│   ├── metrics.py             # Performance and risk metrics
│   ├── records.py             # Trade and position record generation
│   ├── report.py              # QuantStats report and plots
│   └── quick.py               # One-call backtest wrapper
│
├── crypto/
│   ├── agents/                # Agent implementations
│   ├── benchmark/             # Benchmark strategies
│   ├── cv/                    # Purged / embargoed CV utilities
│   ├── eval/                  # Evaluation helpers
│   ├── experiments/           # Ablation and incremental experiment modules
│   ├── features/              # Market, derivative, on-chain, narrative features
│   ├── governance/            # PBO, holdout, registry, freeze / pre-registration logic
│   ├── labels/                # Triple-barrier labels and related label logic
│   ├── live/                  # Paper broker / simulated live execution support
│   ├── models/                # Model bundle, PatchTST fallback, classifiers
│   ├── orchestration/         # TradingGraph and agent decision orchestration
│   ├── skills/                # Registered tools / skills used by agents
│   ├── __init__.py
│   ├── adapters.py            # Bar schema and decision-time adapters
│   ├── exec_price.py          # Execution price helpers
│   ├── INTEGRATION_NOTES.md   # Integration notes
│   ├── pipeline_1b.py         # Phase-1b model pipeline
│   ├── pit.py                 # Point-in-time dataset construction and audit
│   └── schemas.py             # Frozen config, hashes, and shared schema definitions
│
├── etl/
│   ├── data_updater.py        # Market data, funding rate, open interest collection
│   ├── data_processor.py      # Cleaning and resampling
│   ├── data_loader.py         # Processed crypto data loader
│   └── dune_loader.py         # Dune on-chain data loader
│
├── data_storage/
│   ├── raw/                   # Raw market / funding / OI / on-chain data
│   ├── processed/             # Cleaned and resampled data
│   ├── factors/               # Materialized feature matrices
│   ├── signals/               # Model / agent signal outputs
│   ├── backtest_results/      # Backtest reports and records
│   └── experiments/           # Experiment outputs
│
├── examples/
│   ├── demo.py                # Synthetic end-to-end model demo
│   ├── agent_demo.py          # Synthetic agent orchestration demo
│   ├── incremental_demo.py    # Synthetic incremental ablation demo
│   └── experiment_demo.py     # Synthetic governance / holdout demo
│
├── scripts/
│   ├── run_agent.py           # Command-line agent decision entry point
│   └── run_on_real_data.py    # Real-data Phase-1b experiment runner
│
├── config.py                  # Global path and runtime configuration
├── main.py                    # Main crypto data pipeline entry
├── pyproject.toml             # Project dependencies and task runner
├── .gitignore
└── README.md
```

## 如何使用：

首先构造一个虚拟环境防止污染（本地回测可以不用，但是服务器为了保证运行稳定强制要求）

```
python -m venv .venv
```

引入了pyproject.toml 做python 项目管理，这样跨层级引用文件就不容易出乱七八糟的问题，我受够了os 写绝对路径

需要看到终端控制台提示绿色字符知道进入虚拟环境，如果没有

```
.\.venv\Scripts\activate
```

然后项目根目录终端输入

```
pip install -e .   
```

就可以用了，如果还有config.py这样希望全局引用的文件可以去修改pyproject.toml

用于云服务器时，不需要可编辑，省略e

```
pip install .
```

终端输入

```
task sysrun  
```

等效于 “pip install -e .” 其他的常用长命令也可以这样写入pyproject 里面

scripts里面的参考脚本演示了如何提交标准规范的交易决策信号记录给回测功能并接收结果

里面标注viewer的脚本用来方便查看parquet内的数据（二进制文件没有直接可用的图形化工具查看）

如果使用了虚拟环境：

```
pip install ipykernel
```

```
python -m ipykernel install --user --name=quant_system --display-name="Quant System (.venv)"
```

以解决jupter兼容性问题

如果你本地没有历史数据，运行时，执行main.py 会初始化目录并更新数据

```
python main.py
```

### 日常维护：

有新依赖时，上面的pip install -e .也行

```
task sysrun 
```

要手动更新数据时：

如果增加新的币种： config.py里面 coin 列表添加对应的合约（记得核实币安接口是否有）

随后输入：

```
task dataupdate
```

如果只需要纯K线数据（OHLC） 

```
task marketonly_dataupdate
```

如果要更新链上数据：

```
task onchain_dataupdate
```

如果想要构建特征集：

1. 可以直接运行feature_builder.py 配置在文件最上方
2. 终端输入如下命令：

```
task feature
```
