"""Check CoinDesk access through urllib proxy and Chrome CDP."""
from __future__ import annotations

import urllib.request

import config


URL = "https://www.coindesk.com/sitemap/archive/2021"
CDP_VERSION_URL = "http://127.0.0.1:9222/json/version"


def check_urllib() -> None:
    proxy_url = config.SentimentConfig.RSS_PROXY_URL
    print(f"RSS_USE_PROXY={config.SentimentConfig.RSS_USE_PROXY}")
    print(f"RSS_PROXY_URL={proxy_url}")
    if config.SentimentConfig.RSS_USE_PROXY:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        )
    else:
        opener = urllib.request.build_opener()
    req = urllib.request.Request(URL, headers={"User-Agent": "TradingSystem/0.1 access check"})
    with opener.open(req, timeout=20) as resp:
        payload = resp.read(200)
    print(f"urllib CoinDesk OK: first bytes={payload[:80]!r}")


def check_cdp() -> None:
    req = urllib.request.Request(CDP_VERSION_URL)
    with urllib.request.urlopen(req, timeout=5) as resp:
        payload = resp.read(200)
    print(f"Chrome CDP OK: first bytes={payload[:80]!r}")


def main() -> None:
    for name, fn in [("urllib", check_urllib), ("chrome_cdp", check_cdp)]:
        try:
            fn()
        except Exception as exc:
            print(f"{name} FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
