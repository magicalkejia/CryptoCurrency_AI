"""
etl/llm_deepseek.py
===================
DeepSeek LLM client for the narrative/event extraction stage (v6 §7.3).

Design (mirrors score_news.py's discipline):
  * The LLM is a TOOL: it returns STRUCTURED fields only (event type, themes,
    entities, sentiment, severity, rumor flag, source credibility) — never a
    trade direction or position size.
  * Determinism for a frozen config: temperature=0, JSON mode, and the extractor
    is run OFFLINE once; results are cached by content hash and persisted to
    parquet. The backtest/decision loop reads the cached factors, so the trading
    path stays fully reproducible even though the LLM itself is not bit-exact.
  * Offline fallback: if the `openai` SDK or DEEPSEEK_API_KEY is unavailable, a
    deterministic keyword heuristic is used and every row is flagged `_offline=True`
    so it is never mistaken for a real LLM result.

API facts (verified): DeepSeek is OpenAI-compatible at base_url https://api.deepseek.com;
current model id `deepseek-v4-flash` (low-latency, JSON output supported via
response_format={"type":"json_object"}; the prompt must also mention JSON).

Where to put your key: set DEEPSEEK_API_KEY in the project .env (see config.LLMConfig).
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import List, Optional

# Load .env so DEEPSEEK_API_KEY is available even when this module is used via
# `python -m etl.extract_events_llm` (which does not import config). Safe no-op if
# python-dotenv is missing or there is no .env.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- structured schema (allowed values) ----------------------------------- #
EVENT_TYPES = ["listing", "delisting", "hack", "regulation", "partnership",
               "unlock", "funding", "etf", "macro", "rumor", "none"]
THEMES = ["etf", "regulation", "defi", "l2", "ai", "meme", "none"]
RISK_EVENT_TYPES = {"hack", "delisting", "regulation"}     # used by event-risk feature

# base tickers the project trades (entity extraction is restricted to these)
UNIVERSE_TICKERS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "LTC", "LINK", "TRX", "ADA"]

# name/keyword -> ticker (offline entity heuristic; the live LLM does this far better)
_NAME_TO_TICKER = {
    "bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "ether": "ETH", "eth": "ETH",
    "solana": "SOL", "sol": "SOL", "binance coin": "BNB", "bnb": "BNB",
    "ripple": "XRP", "xrp": "XRP", "dogecoin": "DOGE", "doge": "DOGE",
    "litecoin": "LTC", "ltc": "LTC", "chainlink": "LINK", "link": "LINK",
    "tron": "TRX", "trx": "TRX", "cardano": "ADA", "ada": "ADA",
}
_EVENT_KEYWORDS = {
    "hack": ["hack", "exploit", "stolen", "breach", "drained"],
    "delisting": ["delist", "remove", "suspend trading"],
    "listing": ["list", "lists", "listing", "debut"],
    "regulation": ["sec", "regulat", "lawsuit", "ban", "court", "subpoena", "settle"],
    "etf": ["etf", "exchange-traded fund", "spot etf"],
    "partnership": ["partner", "integration", "collaborat"],
    "unlock": ["token unlock", "vesting", "unlock"],
    "funding": ["raise", "funding round", "series a", "series b", "invest"],
    "macro": ["fed", "inflation", "cpi", "rate", "macro"],
}
_THEME_KEYWORDS = {
    "etf": ["etf"], "regulation": ["sec", "regulat", "lawsuit", "ban"],
    "defi": ["defi", "yield", "tvl", "liquidity pool"], "l2": ["layer 2", "l2", "rollup"],
    "ai": [" ai ", "artificial intelligence", "agent"], "meme": ["meme", "dogecoin", "shib"],
}

SYSTEM_PROMPT = (
    "You are a precise financial information extractor for crypto news. "
    "Read the article and return ONLY a single JSON object with the schema described. "
    "You classify and extract facts; you NEVER give trading advice, price targets, "
    "directions, or position sizes. List a coin under `entities` ONLY when it is a "
    "central subject of the article, never for incidental or passing mentions. "
    "If unsure, use conservative/neutral values."
)


def build_user_prompt(title: str, body: str, max_body_chars: int = 4000) -> str:
    body = (body or "")[:max_body_chars]
    tickers = ", ".join(UNIVERSE_TICKERS)
    return (
        f"Article title: {title}\n\n"
        f"Article body (truncated):\n{body}\n\n"
        "Return ONLY this JSON object (no prose, no markdown):\n"
        "{\n"
        f'  "event_type": one of {EVENT_TYPES},\n'
        f'  "entities": ONLY the coins from [{tickers}] that are a PRIMARY SUBJECT of the '
        'article (the story is centrally about them). EXCLUDE coins merely mentioned in '
        'passing, used as a unit/medium of exchange, or named only for price/comparison '
        'context. Example: an article about an Ethereum (Parity) hack that notes the hacker '
        'cashed out into Bitcoin -> entities=["ETH"], NOT ["ETH","BTC"]. If no coin from the '
        'list is a primary subject, return []. Do NOT guess,\n'
        f'  "themes": subset of {THEMES} (use ["none"] if no theme applies),\n'
        '  "sentiment": number in [-1,1] (market sentiment toward the entities),\n'
        '  "severity": number in [0,1] (how market-moving),\n'
        '  "is_rumor": true/false (unconfirmed/speculative),\n'
        '  "source_credibility": number in [0,1],\n'
        '  "horizon": one of ["intraday","days","weeks"],\n'
        '  "confidence": number in [0,1] (your confidence in this extraction)\n'
        "}"
    )


def _coerce(d: dict) -> dict:
    """Validate/clamp an extraction dict to the schema; fill safe defaults."""
    def num(x, lo, hi, default):
        try:
            return float(min(hi, max(lo, float(x))))
        except Exception:
            return default
    et = str(d.get("event_type", "none")).lower()
    if et not in EVENT_TYPES:
        et = "none"
    ents = [str(e).upper() for e in (d.get("entities") or []) if str(e).upper() in UNIVERSE_TICKERS]
    th = [str(t).lower() for t in (d.get("themes") or []) if str(t).lower() in THEMES] or ["none"]
    return {
        "event_type": et,
        "entities": sorted(set(ents)),
        "themes": sorted(set(th)),
        "sentiment": num(d.get("sentiment"), -1, 1, 0.0),
        "severity": num(d.get("severity"), 0, 1, 0.0),
        "is_rumor": bool(d.get("is_rumor", False)),
        "source_credibility": num(d.get("source_credibility"), 0, 1, 0.5),
        "horizon": str(d.get("horizon", "days")) if str(d.get("horizon", "days")) in
                   ("intraday", "days", "weeks") else "days",
        "confidence": num(d.get("confidence"), 0, 1, 0.5),
    }


def offline_extract(title: str, body: str) -> dict:
    """Deterministic keyword heuristic used when the LLM API is unavailable.
    Clearly flagged `_offline=True` so it is never confused with a real LLM result."""
    text = f"{title or ''} {body or ''}".lower()
    ents = sorted({t for name, t in _NAME_TO_TICKER.items() if re.search(rf"\b{name}\b", text)})
    event_type, severity = "none", 0.1
    for et, kws in _EVENT_KEYWORDS.items():
        if any(k in text for k in kws):
            event_type = et
            severity = 0.7 if et in RISK_EVENT_TYPES or et == "etf" else 0.4
            break
    themes = sorted({th for th, kws in _THEME_KEYWORDS.items() if any(k in text for k in kws)}) or ["none"]
    pos = sum(text.count(w) for w in ("surge", "rally", "approve", "bullish", "gain", "soar"))
    neg = sum(text.count(w) for w in ("hack", "ban", "lawsuit", "crash", "plunge", "bearish", "delist"))
    sentiment = max(-1.0, min(1.0, 0.15 * (pos - neg)))
    is_rumor = any(k in text for k in ("rumor", "reportedly", "sources say", "speculation", "alleged"))
    out = _coerce({"event_type": event_type, "entities": ents, "themes": themes,
                   "sentiment": sentiment, "severity": severity, "is_rumor": is_rumor,
                   "source_credibility": 0.6, "horizon": "days", "confidence": 0.3})
    out["_offline"] = True
    return out


class DeepSeekExtractor:
    """Thin OpenAI-compatible DeepSeek client. Falls back to offline_extract if the
    SDK/key is missing OR if offline=True is forced."""

    def __init__(self, model: str = "deepseek-v4-flash",
                 base_url: str = "https://api.deepseek.com",
                 api_key: Optional[str] = None, temperature: float = 0.0,
                 max_retries: int = 4, max_body_chars: int = 4000,
                 offline: bool = False, consistency_runs: int = 1):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_body_chars = max_body_chars
        self.consistency_runs = max(1, int(consistency_runs))
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.offline = offline
        self._client = None
        if not self.offline and self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=base_url)
            except Exception as e:
                print(f"[llm_deepseek] OpenAI SDK unavailable ({e!r}); using offline heuristic.")
                self.offline = True
        else:
            self.offline = True
            if not api_key and not os.getenv("DEEPSEEK_API_KEY"):
                print("[llm_deepseek] DEEPSEEK_API_KEY not set; using offline heuristic. "
                      "Set it in .env to use the real model.")

    def _call_once(self, title: str, body: str) -> dict:
        resp = self._client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": build_user_prompt(title, body, self.max_body_chars)}],
        )
        raw = resp.choices[0].message.content
        # robust parse: strip stray markdown fences if any slipped through
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        return _coerce(json.loads(raw))

    def extract(self, title: str, body: str) -> dict:
        if self.offline or self._client is None:
            return offline_extract(title, body)
        outs, last_err = [], None
        for _ in range(self.consistency_runs):
            for attempt in range(self.max_retries):
                try:
                    outs.append(self._call_once(title, body))
                    break
                except Exception as e:               # 429/5xx/JSON errors -> backoff
                    last_err = e
                    time.sleep(min(2 ** attempt, 8))
        if not outs:
            print(f"[llm_deepseek] all retries failed ({last_err!r}); offline fallback for this row.")
            return offline_extract(title, body)
        merged = _merge_consistency(outs)
        merged["_offline"] = False
        return merged


def _merge_consistency(outs: List[dict]) -> dict:
    """Majority/mean merge over repeated calls + an agreement score in [0,1]."""
    import statistics as st
    et = st.mode([o["event_type"] for o in outs])
    agree = sum(o["event_type"] == et for o in outs) / len(outs)
    ents = sorted({e for o in outs for e in o["entities"]})
    themes = sorted({t for o in outs for t in o["themes"]})
    mean = lambda k: float(sum(o[k] for o in outs) / len(outs))
    return {
        "event_type": et, "entities": ents, "themes": themes,
        "sentiment": mean("sentiment"), "severity": mean("severity"),
        "is_rumor": sum(o["is_rumor"] for o in outs) > len(outs) / 2,
        "source_credibility": mean("source_credibility"),
        "horizon": st.mode([o["horizon"] for o in outs]),
        "confidence": mean("confidence"), "consistency": float(agree),
    }
