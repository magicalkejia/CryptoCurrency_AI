## 📂 系统架构 (Project Structure)

```text
、TRADINGSYSTEM/
├── .venv/                  # 虚拟环境
├── data_storage/           # 数据存储
│   ├── cross_section/      # 每日全资产截面快照 (A股/ETF/指数)
│   ├── history_k/          # 历史 K 线数据(Baostock 源)
│   ├── meta/               # 资产基表 
│   ├── factors/            # 因子库
│   └── models/             # 机器学习模型权重归档
├── etl/                    # 数据管道模块 (ETL)
│   ├── data_loader.py      # 读取
│   ├── data_updater.py     # 更新
│   └── data_processor.py   # 数据清洗
├── backtest/               # 回测引擎层 
├── strategies/             # 交易策略（上线）
├── models/                 # 算法与预测模型
├── research/               # 投研草稿(Jupyter Notebooks)、乱七八糟啊的策略研究文件都可以放这里
├── logs/                   # 日志 
├── .env                    # 环境变量与私密 Key 
├── .gitignore              # Git 忽略
├── config.py               # 全局路径配置
├── main.py                 # 系统入口
└── pyproject.toml          # Python 项目配置文件 (依赖管理与 Task Runner)
```

如何使用：

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

strategies 里面是我随便弄的人工策略

用来熟悉新的回测库的

如果使用了虚拟环境：

```
pip install ipykernel
```

```
python -m ipykernel install --user --name=quant_system --display-name="Quant System (.venv)"
```

解决jupter兼容性问题
