import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env
load_dotenv()

# 获取项目根目录
BASE_DIR = Path(__file__).parent.absolute()


class PathConfig:
    """所有跟文件路径有关的配置。"""
    DATA_ROOT = BASE_DIR / "data_storage"
    BACKTEST_ROOT = BASE_DIR / "backtest"
    MODELS_ROOT = BASE_DIR / "models"
    ETL = BASE_DIR / "etl"
    STRATEGY = BASE_DIR / "strategies"
    LOG = BASE_DIR / "logs"

    RAW = DATA_ROOT / "raw"
    PROCESSED = DATA_ROOT / "processed"
    MODELS = DATA_ROOT / "models"
    META = DATA_ROOT / "meta"

    CROSS_SECTION = DATA_ROOT / "cross_section"
    FACTORS = DATA_ROOT / "factors"
    SIGNALS = DATA_ROOT / "signals"
    BACKTEST_RESULTS = DATA_ROOT / "backtest_results"
    EXPERIMENTS = DATA_ROOT / "experiments"

    # Raw source-specific directories
    RAW_DERIVATIVES = RAW / "derivatives"
    RAW_FUNDING = RAW_DERIVATIVES / "funding"
    RAW_OI = RAW_DERIVATIVES / "oi"
    RAW_LONG_SHORT_RATIO = RAW_DERIVATIVES / "long_short_ratio"
    RAW_SPOT = RAW / "spot"

    RAW_FLOW = RAW / "flow"
    RAW_SPOT_CVD = RAW_FLOW / "spot_cvd"

    RAW_SENTIMENT = RAW / "sentiment"
    RAW_ONCHAIN = RAW / "onchain"

    # Processed source-specific directories
    PROCESSED_DERIVATIVES = PROCESSED / "derivatives"
    PROCESSED_SPOT = PROCESSED / "spot"
    PROCESSED_FLOW = PROCESSED / "flow"
    PROCESSED_SENTIMENT = PROCESSED / "sentiment"
    PROCESSED_ONCHAIN = PROCESSED / "onchain"


class SpiderConfig:
    """爬虫相关配置。"""
    _enable_proxy = os.getenv("ENABLE_PROXY", "false").lower() == "true"
    _proxy_url = os.getenv("PROXY_URL", "http://127.0.0.1:7897")

    if _enable_proxy:
        PROXY = {
            "http": _proxy_url,
            "https": _proxy_url,
        }
    else:
        PROXY = None

    TIMEOUT = 30000
    RATE_LIMIT = True
    START_TIME = "2021-01-01 00:00:00"


class TargetConfig:
    """交易标的配置。"""
    # Default trading universe for pooled cross-sectional crypto modeling.
    # Keep this list focused on Binance USDT-margined perpetuals with broad
    # history and different narrative / sector drivers.
    DIVERSIFIED_10_COINS = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "XRP/USDT",
        "DOGE/USDT",
        "LTC/USDT",
        "LINK/USDT",
        "TRX/USDT",
        "ADA/USDT",
    ]

    # Research candidates mentioned for lower BTC correlation checks. Keep
    # them out of the default universe until availability/history is verified.
    EXTRA_DIVERSIFICATION_CANDIDATES = [
        "HYPE/USDT",
        "XMR/USDT",
        "ZEC/USDT",
    ]

    # Not a Binance USDM perpetual symbol. Treat as a future macro/cash-regime
    # proxy only if a separate data source is added.
    CASH_REGIME_PROXIES = [
        "USDT.D",
    ]

    # DATA_TEST_COINS = DIVERSIFIED_10_COINS + EXTRA_DIVERSIFICATION_CANDIDATES

    COINS =  DIVERSIFIED_10_COINS + EXTRA_DIVERSIFICATION_CANDIDATES

    TIMEFRAMES = {
        "base": "1m",
        "resample": ["1h", "4h", "1d"],
    }


class OnchainConfig:
    """On-chain / DeFi source configuration."""
    SYMBOL_TO_DEFILLAMA_CHAIN = {
        "BTC/USDT": "Bitcoin",
        "ETH/USDT": "Ethereum",
        "SOL/USDT": "Solana",
        "BNB/USDT": "BSC",
    }
    DEFILLAMA_CHAINS = list(SYMBOL_TO_DEFILLAMA_CHAIN.values())
    ONCHAIN_FACTOR_TABLE = "onchain_features.parquet"


class SentimentConfig:
    """News/social sentiment source configuration."""

    RSS_OUTPUT_NAME = "rss_articles.parquet"
    RSS_USE_PROXY = os.getenv("RSS_USE_PROXY", os.getenv("ENABLE_PROXY", "false")).lower() == "true"
    RSS_PROXY_URL = os.getenv("RSS_PROXY_URL", os.getenv("PROXY_URL", "http://127.0.0.1:7897"))
    RSS_RETRIES = 2
    RSS_BACKOFF_SECONDS = 5.0

    RSS_FEEDS = {
        "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "cointelegraph": "https://cointelegraph.com/rss",
        "decrypt": "https://decrypt.co/feed",
        "cryptoslate": "https://cryptoslate.com/feed/",
    }

    COINDESK_ARCHIVE_BASE_URL = "https://www.coindesk.com/sitemap/archive"
    COINDESK_ARCHIVE_OUTPUT_NAME = "coindesk_archive_index.parquet"
    # DEPRECATED: the two raw CoinDesk files below were merged (with cryptoslate) by a
    # teammate into a single processed file and then deleted. Kept only for reference.
    COINDESK_ARTICLE_DETAIL_OUTPUT_NAME = "coindesk_article_details.parquet"
    COINDESK_ARTICLE_BODY_OUTPUT_NAME = "coindesk_article_bodies.parquet"

    # Merged news corpus (coindesk + cryptoslate), the single input to the LLM event
    # extraction. Lives under PathConfig.PROCESSED_SENTIMENT
    # (= data_storage/processed/sentiment/). Columns: source, url, title,
    # published_at (ms epoch), published_date, section, author, description,
    # asset_type, sentiment_label, article_text.
    MERGED_NEWS_INPUT_NAME = "merged_news_for_llm.parquet"

    # LLM event/narrative extraction (A1; produced by etl.extract_events_llm +
    # etl.build_event_features). Stored under data_storage/factors/sentiment/.
    EVENTS_LLM_OUTPUT_NAME = "events_llm.parquet"
    EVENT_FEATURES_OUTPUT_NAME = "event_features.parquet"


class LLMConfig:
    """DeepSeek (and compatible) LLM configuration for narrative/event extraction.

    Set DEEPSEEK_API_KEY in the project .env. DeepSeek is OpenAI-compatible, so the
    extractor uses the `openai` SDK pointed at DEEPSEEK_BASE_URL.
    """
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")          # <-- put your key in .env
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))   # 0 for reproducibility
    LLM_MAX_BODY_CHARS = int(os.getenv("LLM_MAX_BODY_CHARS", "4000"))
    LLM_CONSISTENCY_RUNS = int(os.getenv("LLM_CONSISTENCY_RUNS", "1"))  # >1 = agreement score


def init_directories():
    """初始化项目目录结构，应在程序启动时显式调用。"""
    paths_to_create = [
        PathConfig.DATA_ROOT,
        PathConfig.BACKTEST_ROOT,
        PathConfig.MODELS_ROOT,
        PathConfig.LOG,

        PathConfig.RAW,
        PathConfig.PROCESSED,
        PathConfig.MODELS,
        PathConfig.META,

        PathConfig.CROSS_SECTION,
        PathConfig.FACTORS,
        PathConfig.SIGNALS,
        PathConfig.BACKTEST_RESULTS,
        PathConfig.EXPERIMENTS,

        PathConfig.RAW_DERIVATIVES,
        PathConfig.RAW_FUNDING,
        PathConfig.RAW_OI,
        PathConfig.RAW_LONG_SHORT_RATIO,
        PathConfig.RAW_SPOT,
        PathConfig.RAW_FLOW,
        PathConfig.RAW_SPOT_CVD,
        PathConfig.RAW_SENTIMENT,
        PathConfig.RAW_ONCHAIN,

        PathConfig.PROCESSED_DERIVATIVES,
        PathConfig.PROCESSED_SPOT,
        PathConfig.PROCESSED_FLOW,
        PathConfig.PROCESSED_SENTIMENT,
        PathConfig.PROCESSED_ONCHAIN,
    ]

    for p in paths_to_create:
        try:
            os.makedirs(p, exist_ok=True)
            # print(f"目录检查/创建成功: {p}")
        except Exception as e:
            print(f"目录创建失败 {p}: {e}")
