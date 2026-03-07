Quant_System/

├── 📂 models/                 # 3. AI 算法层 (DeepSeek/LGBM/Transformer)
│
├── 📂 backtest/               # 4. 回测引擎层
│   ├── __init__.py
│   ├── engine_vector.py      # VectorBT 快速回测逻辑
│   ├── engine_event.py       # 事件驱动回测逻辑 (backtesting.py)
│   └── analysis.py           # 绩效分析/画图
│
├── 📂 data_storage/           # 7. 数据仓库 (git ignore)
│   ├── raw/
│   ├── processed/
│   ├── factors/              # 存储算好的因子 Parquet
│   └── model_weights/
│
├── .gitignore                 # Git 忽略文件
├── .env                       # 环境变量
├── main.py                    # 程序统一入口
└── requirements.txt           # 依赖包

├── config.py      #各种配置

如何使用：

首先构造一个虚拟环境防止污染

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

pipreqs ./ --encoding=utf8 --force    （新装的包可以用这个更新requirements.txt)

strategies 里面是我随便弄的人工策略

用来熟悉新的回测库的
