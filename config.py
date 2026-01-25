import os
from pathlib import Path
from dotenv import load_dotenv
# 加载 .env
load_dotenv()
# 获取项目根目录 
BASE_DIR = Path(__file__).parent.absolute()

class PathConfig:
    """所有跟文件路径有关的配置"""
    DATA_ROOT = BASE_DIR / 'data_storage'
    BACKTEST_ROOT = BASE_DIR / 'backtest'  #回测
    MODELS_ROOT = BASE_DIR / 'models'   #模型
    ETL = BASE_DIR / 'etl'          # 数据处理
    STRATEGY = BASE_DIR / 'strategies'  #策略
    RAW = DATA_ROOT / 'raw'
    PROCESSED = DATA_ROOT / 'processed'
    MODELS = DATA_ROOT / 'models'     

class SpiderConfig:
    """爬虫相关配置"""
    # 1. 读取开关 (默认为 false)
    _enable_proxy = os.getenv('ENABLE_PROXY', 'false').lower() == 'true'
    
    # 2. 读取地址
    _proxy_url = os.getenv('PROXY_URL', 'http://127.0.0.1:7897')
    if _enable_proxy:
        PROXY = {
            'http': _proxy_url,
            'https': _proxy_url
        }
    else:
        PROXY = None  # CCXT 收到 None 会直接直连
    TIMEOUT = 30000
    RATE_LIMIT = True
    START_TIME = '2021-01-01 00:00:00'
    # print(f"🌐 网络模式: {'代理模式 ' if _enable_proxy else '直连模式'}")
    
class TargetConfig:
    """交易标的配置"""
    COINS = [
        'BTC/USDT', 
        'ETH/USDT', 
        'SOL/USDT',
        # ...
    ]
    TIMEFRAMES = {
        'base': '1m',         # 下载用的
        'resample': ['1h', '4h', '1d'] # 合成用的
    }

def init_directories():
    """
    初始化项目目录结构
    应在程序启动时显式调用
    """
    # 获取 PathConfig 中所有不以 '__' 开头的属性（即我们定义的路径）
    paths_to_create = [
        PathConfig.DATA_ROOT,
        PathConfig.BACKTEST_ROOT,
        PathConfig.MODELS_ROOT,

        PathConfig.RAW,
        PathConfig.PROCESSED,
        PathConfig.MODELS,
        PathConfig.ETL
    ]
    
    for p in paths_to_create:
        try:
            # exist_ok=True 表示如果文件夹已存在就不报错
            # parents=True 表示如果父目录不存在也一并创建 (比如先建 data_storage 再建 raw)
            os.makedirs(p, exist_ok=True)
            print(f"✅ 目录检查/创建成功: {p}")
        except Exception as e:
            print(f"❌ 目录创建失败 {p}: {e}")