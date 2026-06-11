# Crypto AI Strategy Research System

## 1. Project Overview

This project is a  **cryptocurrency AI strategy research system** . It is designed for data collection, feature construction, model experimentation, agent-based decision generation, and backtesting evaluation.

The system does **not** perform real-money automated trading. Its current scope is:

1. Collect and process cryptocurrency market data.
2. Collect supplementary data such as funding rate, open interest, and on-chain indicators.
3. Build point-in-time features and labels for machine learning research.
4. Generate strategy signals through statistical models and AI/Agent modules.
5. Evaluate strategies through a reusable vectorized backtesting engine.
6. Produce auditable decision records, performance metrics, and experiment outputs.

The project is currently focused on the graduation project topic:  **AI-enhanced cryptocurrency strategy research** . Stock / A-share modules are not part of the current thesis submission scope.

---

## 2. System Boundary

The system is divided into four major layers:

```text
Data Layer
  ├── Market data
  ├── Funding rate
  ├── Open interest
  └── On-chain data

Research Layer
  ├── Feature engineering
  ├── Triple-barrier labels
  ├── Point-in-time dataset construction
  ├── Cross-validation
  ├── Model training
  └── Experiment governance

Agent Layer
  ├── Data Agent
  ├── Signal Research Agent
  ├── Narrative Agent
  ├── Fusion Agent
  ├── Risk Agent
  ├── Execution Agent
  └── Review Agent

Evaluation Layer
  ├── Vectorized backtest
  ├── Performance metrics
  ├── Trade records
  ├── Position records
  └── HTML / chart reports
```

The system intentionally separates:

```text
ETL / data processing
    Responsible for collecting, cleaning, resampling, and loading data.

Crypto research modules
    Responsible for features, labels, model training, agent orchestration, and governance.

Backtest engine
    Responsible for converting target weights into simulated portfolio performance.

Demo / scripts
    Responsible for running synthetic demos, real-data tests, and agent experiments.
```

---

## 3. Current Project Structure

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

Note: if the repository still keeps demo files in the root directory, they should be moved into `examples/` or `scripts/` according to the structure above before final submission.

---

## 4. Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.\.venv\Scripts\activate
```

Install the project in editable mode:

```bash
pip install -e .
```

If using Jupyter Notebook / VS Code Notebook:

```bash
pip install ipykernel
python -m ipykernel install --user --name=quant_system --display-name="Quant System (.venv)"
```

---

## 5. Environment Variables

Create a `.env` file in the project root if needed.

Example:

```env
ENABLE_PROXY=false
PROXY_URL=http://127.0.0.1:7897
DUNE_API_KEY=your_dune_api_key_here
```

Explanation:

```text
ENABLE_PROXY
    Whether to use proxy for external data requests.

PROXY_URL
    Local proxy address, used mainly for Binance / CCXT access if required.

DUNE_API_KEY
    API key for Dune Analytics on-chain data fetching.
```

The `.env` file should never be committed.

---

## 6. Data Pipeline

### 6.1 Market Data

The main crypto market data pipeline is:

```text
fetch raw 1m data
    -> clean raw data
    -> generate processed 1m data
    -> resample to 1h / 4h / 1d
    -> save to data_storage/processed/
```

Run:

```bash
python main.py --mode crypto
```

or:

```bash
task crypto
```

The default target symbols are configured in `config.py`:

```python
class TargetConfig:
    COINS = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
    ]

    TIMEFRAMES = {
        "base": "1m",
        "resample": ["1h", "4h", "1d"],
    }
```

Processed files are saved as:

```text
data_storage/processed/BTCUSDT_1m.parquet
data_storage/processed/BTCUSDT_1h.parquet
data_storage/processed/BTCUSDT_4h.parquet
data_storage/processed/BTCUSDT_1d.parquet
```

---

### 6.2 Funding Rate and Open Interest

`etl/data_updater.py` also supports optional derivative data collection:

```text
funding rate
open interest
```

The crypto pipeline supports an optional derivative switch:

```python
from etl.crypto_pipeline import run_crypto_pipeline

run_crypto_pipeline(fetch_derivatives=True)
```

Funding rate is saved as:

```text
data_storage/raw/BTCUSDT_funding.parquet
```

Open interest is saved as:

```text
data_storage/raw/BTCUSDT_oi.parquet
```

Known limitation:

```text
Binance open interest history has limited lookback. It is suitable for recent analysis,
but not sufficient for long historical backtests without another data source.
```

---

### 6.3 On-chain Data

On-chain data is loaded through `etl/dune_loader.py`.

The current Dune workflow is:

```text
Dune SQL query
    -> Dune API execution or cached result fetch
    -> normalize result into a DataFrame
    -> save as data_storage/raw/onchain_{name}.parquet
```

Example:

```python
from etl.dune_loader import fetch_dune_onchain
import config
import os

df = fetch_dune_onchain(
    query_id=YOUR_DUNE_QUERY_ID,
    api_key=os.getenv("DUNE_API_KEY"),
    name="eth_core_daily",
    raw_dir=config.PathConfig.RAW,
    use_cached=True,
)
```

Current recommended minimum on-chain metrics:

```text
active_address
tx_count
transfer_volume
gas_used
```

On-chain data is generally daily-frequency and should be treated as a low-frequency contextual factor rather than a high-frequency trading signal.

---

## 7. Data Loading

Use `DataLoader` to read processed crypto data.

```python
from etl.data_loader import DataLoader

loader = DataLoader()

df = loader.get_crypto_kline_data(
    symbol="BTC/USDT",
    timeframe="1h",
    start_date="2022-01-01",
    columns=["open", "high", "low", "close", "volume"],
)

display(df.tail())
```

To load multiple symbols into aligned feature matrices:

```python
matrix = loader.get_crypto_matrix(
    symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    timeframe="1h",
    start_date="2022-01-01",
    columns=["close", "volume"],
)

close = matrix["close"]
volume = matrix["volume"]
```

The returned matrix format is:

```text
index   = timestamp
columns = symbols
values  = selected field values
```

Example:

```text
matrix["close"]
    BTC/USDT    ETH/USDT    SOL/USDT
ts
...
```

---

## 8. Feature, Label, and PIT Dataset Pipeline

The crypto research modules support:

```text
bar schema normalization
decision-time grid construction
triple-barrier label generation
uniqueness weighting
point-in-time dataset construction
look-ahead audit
PatchTST / fallback temporal features
Phase-1b model pipeline
```

A typical real-data flow is:

```text
processed 1h market data
    -> 4h decision grid
    -> triple-barrier labels
    -> market features
    -> funding features
    -> optional on-chain features
    -> PatchTST / temporal model features
    -> point-in-time supervised dataset
    -> Phase-1b model pipeline
```

Run the real-data experiment entry:

```bash
python scripts/run_on_real_data.py --symbols BTC/USDT ETH/USDT SOL/USDT
```

If the script is still in the root directory, run:

```bash
python run_on_real_data.py --symbols BTC/USDT ETH/USDT SOL/USDT
```

The real-data runner should be used to validate:

```text
1. Whether processed market data exists
2. Whether decision_time is correctly constructed
3. Whether labels are generated
4. Whether point-in-time audit passes
5. Whether feature columns are available
6. Whether Phase-1b diagnostics can be produced
```

---

## 9. Agent Orchestration

The Agent system is implemented through `crypto/orchestration/graph.py`.

The decision graph is:

```text
START
  -> data
  -> quality_gate
        ├── if data_quality_score < threshold:
        │       -> no_trade
        │       -> review
        │       -> END
        │
        └── else:
                -> signal
                -> narrative
                -> fusion
                -> risk
                       ├── if not approved:
                       │       -> review
                       │       -> END
                       │
                       └── if approved:
                               -> execution
                               -> review
                               -> END
```

The graph supports two execution paths:

```text
LangGraph path
    Used if langgraph is installed.

Built-in state machine path
    Used as default fallback, so the system can run without LangGraph.
```

The core programmatic call is:

```python
state = graph.run_decision(
    symbol="BTC/USDT",
    decision_time=decision_time,
    broker=broker,
    ref_price=ref_price,
    cb_level=0,
)
```

The structured decision output is generated by:

```python
decision_to_json(state, fcfg)
```

The output includes:

```text
decision_time
symbol
regime
base_model_pred
combined_alpha
primary_direction
meta_trade_prob_calibrated
confidence
action
target_position
vol_target_scalar
stop_loss
take_profit
risk_level
circuit_breaker_level
data_quality_score
narrative_stub
execution_status
reason
audit_id
```

---

## 10. Agent CLI

Run the agent entry point:

```bash
python scripts/run_agent.py --symbol BTC/USDT
```

If the script is still in the root directory:

```bash
python run_agent.py --symbol BTC/USDT
```

Run the latest 5 decisions:

```bash
python scripts/run_agent.py --symbol BTC/USDT --n 5
```

Simulate circuit-breaker level 3:

```bash
python scripts/run_agent.py --symbol BTC/USDT --cb 3
```

Current limitation:

```text
The current agent demo path can run on synthetic data. For formal project submission,
the agent should consume real-data features and labels generated by the data pipeline.
```

---

## 11. Integration Contract Between Data System and Agent System

The system should integrate through files, not through direct coupling of internal modules.

### 11.1 Data System Outputs

The data and research pipeline should produce:

```text
data_storage/processed/features/crypto_features.parquet
data_storage/processed/labels/crypto_labels.parquet
```

Recommended feature schema:

```text
symbol
decision_time
ts_open
max_feature_availability_ts
ret_1
ret_6
ret_24
vol_24
mom_z
funding_rate
funding_rate_z
funding_rate_chg
onchain_active_z
onchain_flow_z
narrative_sentiment
narrative_event_risk
patchtst_*
```

Recommended label schema:

```text
symbol
decision_time
entry_time
exit_time
tb_label
primary_direction
raw_exit_return_long
raw_exit_return_short
uniqueness_weight
label_config_hash
cost_model_hash
```

### 11.2 Agent System Output

The Agent system should output:

```text
data_storage/signals/agent_decisions.parquet
```

Recommended schema:

```text
decision_time
symbol
agent_name
action
target_position
signal_score
confidence
risk_approved
reason
audit_log
created_at
```

The backtest system only requires:

```text
decision_time
symbol
target_position
```

Other fields are used for explainability, governance, and audit.

---

## 12. Backtesting

The backtest engine expects:

```text
close: DataFrame
    index = timestamp
    columns = symbol

target_weight: DataFrame
    index = timestamp
    columns = symbol
```

Example:

```python
import pandas as pd
from backtest.quick import quick_backtest

agent_df = pd.read_parquet("data_storage/signals/agent_decisions.parquet")

target_weight = agent_df.pivot_table(
    index="decision_time",
    columns="symbol",
    values="target_position",
    aggfunc="last",
).fillna(0)

result = quick_backtest(
    close=close_df,
    target_weight=target_weight,
    strategy_name="agent_strategy",
    fee_rate=0.0004,
    slippage_rate=0.0003,
    execution_lag=1,
    annual_days=2190,
)
```

The backtest output is saved under:

```text
data_storage/backtest_results/{experiment_id}/
```

Typical outputs:

```text
returns.parquet
equity_curve.parquet
weights.parquet
target_weight.parquet
turnover.parquet
cost.parquet
trades.parquet
trades.csv
positions.parquet
positions.csv
metrics.json
config.json
quantstats_report.html
nav.png
drawdown.png
```

---

## 13. Synthetic Demos

Synthetic demos are useful for checking whether the research machinery runs end-to-end, but they are not real trading results.

Recommended location:

```text
examples/
```

Current synthetic demos include:

```text
demo.py
    End-to-end synthetic model pipeline.

agent_demo.py
    Synthetic Agent + Skills orchestration demo.

incremental_demo.py
    Synthetic incremental ablation study.

experiment_demo.py
    Synthetic governance, PBO, freeze, and pre-registration demo.
```

Run examples:

```bash
python examples/agent_demo.py
```

If the files are still in the root directory:

```bash
python agent_demo.py
```

Important:

```text
Synthetic demo results must not be reported as empirical trading performance.
They only prove that the pipeline, interfaces, and experiment machinery can run.
```

---

## 14. Recommended Workflow for Thesis Submission

### Step 1: Update crypto market data

```bash
python main.py --mode crypto
```

### Step 2: Validate processed data

```python
from etl.data_loader import DataLoader

loader = DataLoader()
df = loader.get_crypto_kline_data("BTC/USDT", timeframe="1h")
display(df.tail())
```

### Step 3: Run real-data research pipeline

```bash
python scripts/run_on_real_data.py --symbols BTC/USDT ETH/USDT SOL/USDT
```

### Step 4: Run Agent decision generation

```bash
python scripts/run_agent.py --symbol BTC/USDT --n 5
```

### Step 5: Convert Agent decisions into target weights

```python
agent_df = pd.read_parquet("data_storage/signals/agent_decisions.parquet")

target_weight = agent_df.pivot_table(
    index="decision_time",
    columns="symbol",
    values="target_position",
    aggfunc="last",
).fillna(0)
```

### Step 6: Run backtest

```python
from backtest.quick import quick_backtest

result = quick_backtest(
    close=close_df,
    target_weight=target_weight,
    strategy_name="agent_strategy",
    fee_rate=0.0004,
    slippage_rate=0.0003,
    execution_lag=1,
    annual_days=2190,
)
```

### Step 7: Review experiment outputs

```text
data_storage/backtest_results/
```

---

## 15. Development Tasks

Install package:

```bash
task sysrun
```

Run crypto data pipeline:

```bash
task crypto
```

If task aliases are not available:

```bash
python main.py --mode crypto
```

Recommended task aliases in `pyproject.toml`:

```toml
[tool.taskipy.tasks]
sysrun = "pip install -e ."
crypto = "python main.py --mode crypto"
dataupdate = "python main.py --mode crypto"
realdata = "python scripts/run_on_real_data.py"
agent = "python scripts/run_agent.py"
```

Avoid keeping obsolete stock-related tasks in the thesis submission branch.

---

## 16. Current Limitations

1. The system is a research and backtesting system, not a production trading system.
2. The Agent execution module uses paper execution / simulated broker behavior only.
3. Synthetic demos are not empirical results.
4. Dune on-chain data is generally low-frequency and may have availability lag.
5. Binance open interest data may have limited historical lookback.
6. Real conclusions require locked holdout evaluation and no retuning after holdout inspection.
7. The current Agent integration should use file-based contracts before deeper module coupling.
8. The final thesis submission should clearly distinguish:
   * market data pipeline,
   * feature / label construction,
   * model and agent decision generation,
   * backtest and evaluation results.

---

## 17. Project Status

Current completed components:

```text
Crypto market data collection
Crypto data cleaning and resampling
Funding rate and open interest collection
Dune on-chain data loader
Crypto data loader
Vectorized backtest engine
Performance metrics and reports
Synthetic model pipeline demos
Synthetic Agent orchestration demos
Real-data Phase-1b runner
```

Components still requiring integration:

```text
Persisting real-data features / labels from run_on_real_data.py
Agent reading real-data features instead of synthetic features
Saving agent decisions to data_storage/signals/agent_decisions.parquet
Backtesting agent decisions on real market data
Final README / report alignment with thesis scope
```

---

## 18. One-line Summary

This project is a **cryptocurrency AI strategy research system** that combines market data, derivative data, on-chain indicators, point-in-time machine learning, agent-based decision orchestration, and vectorized backtesting to evaluate AI-enhanced crypto trading hypotheses in a controlled research environment.
