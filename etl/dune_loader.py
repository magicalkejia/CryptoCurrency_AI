"""
etl.dune_loader
===============
On-chain data via Dune Analytics (chosen for: free tier + low dev effort).

Workflow (Dune API v1):
  1. You author a Dune SQL query that aggregates ONLY recomputable, non-revised
     metrics from decoded on-chain tables (v6 §4.2.1) — e.g. active addresses,
     tx count, transfer volume, gas, new addresses — at daily granularity, with
     a `day` (timestamp) column.
  2. fetch_dune_onchain(query_id, api_key) executes it (or fetches cached latest
     results), normalizes to a tidy frame, and saves RAW/onchain_{name}.parquet.
  3. crypto.features.onchain.onchain_factors() turns it into PIT factors.

Honest constraints:
  * Dune metrics are typically DAILY -> a slow factor relative to 4h decisions
    (fine; on-chain is a low-frequency signal by design).
  * availability_lag must cover block finality + Dune refresh latency; default 1d.
  * only pass recomputable metrics into the CORE experiment; the onchain factor
    builder already filters non-recomputable names.

Uses urllib (zero extra deps). Needs network + a Dune API key at run time.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

DUNE_BASE = "https://api.dune.com/api/v1"

# Example SQL templates (Ethereum mainnet, daily). Author/adjust on dune.com,
# then pass the resulting query_id to fetch_dune_onchain. These compute only
# recomputable metrics (v6 §4.2.1).
DUNE_SQL_TEMPLATES = {
    "eth_core_daily": """
        -- recomputable daily on-chain metrics (Ethereum)
        SELECT
          date_trunc('day', block_time) AS day,
          count(DISTINCT "from")        AS active_address,
          count(*)                      AS tx_count,
          sum(value) / 1e18             AS transfer_volume,
          sum(gas_used)                 AS gas_used
        FROM ethereum.transactions
        WHERE block_time >= now() - interval '730' day
        GROUP BY 1
        ORDER BY 1
    """,
}


def _http_get(url: str, api_key: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"X-Dune-API-Key": api_key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _http_post(url: str, api_key: str, body: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"X-Dune-API-Key": api_key,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run_query(query_id: int, api_key: str, params: Optional[dict] = None,
              poll_s: float = 3.0, max_wait_s: float = 300.0) -> list:
    """Execute a Dune query and return rows (list of dicts)."""
    body = {"query_parameters": params} if params else {}
    ex = _http_post(f"{DUNE_BASE}/query/{query_id}/execute", api_key, body)
    eid = ex["execution_id"]
    waited = 0.0
    while waited < max_wait_s:
        st = _http_get(f"{DUNE_BASE}/execution/{eid}/status", api_key)
        state = st.get("state")
        if state == "QUERY_STATE_COMPLETED":
            res = _http_get(f"{DUNE_BASE}/execution/{eid}/results", api_key)
            return res["result"]["rows"]
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            raise RuntimeError(f"Dune execution {state}")
        time.sleep(poll_s)
        waited += poll_s
    raise TimeoutError("Dune query timed out")


def get_latest_results(query_id: int, api_key: str) -> list:
    """Fetch the cached latest results (no fresh execution -> cheaper/faster)."""
    res = _http_get(f"{DUNE_BASE}/query/{query_id}/results", api_key)
    return res["result"]["rows"]


def rows_to_frame(rows: list, time_col: str = "day") -> pd.DataFrame:
    """Normalize Dune rows -> DataFrame indexed by timestamp (the `day` column)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_dune_onchain(query_id: int, api_key: str, name: str,
                       raw_dir, use_cached: bool = True,
                       params: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """Fetch + persist on-chain metrics. Returns the frame (indexed by day)."""
    rows = get_latest_results(query_id, api_key) if use_cached \
        else run_query(query_id, api_key, params)
    df = rows_to_frame(rows)
    if df.empty:
        return None
    out = Path(raw_dir) / f"onchain_{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_parquet(out, engine="pyarrow", compression="zstd", index=False)
    print(f"💾 Dune on-chain '{name}' saved {len(df)} rows -> {out}")
    return df


def load_onchain(raw_dir, name: str) -> Optional[pd.DataFrame]:
    """Load a previously fetched on-chain parquet -> frame indexed by day."""
    p = Path(raw_dir) / f"onchain_{name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df.set_index(pd.to_datetime(df["day"])).drop(columns=["day"]).sort_index()
