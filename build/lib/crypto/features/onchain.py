"""
crypto.features.onchain
==========================
Tabularize on-chain data into point-in-time factors (v6 §5, discipline §4.2.1).

Input is a generic on-chain DataFrame indexed by timestamp with metric columns
(e.g. active_address, exchange_netflow, stablecoin_netflow, ...).  The CALLER is
responsible for passing only metrics that are (a) recomputable from raw blocks
and (b) non-revised, plus a per-metric availability_lag.  This module:

  * applies availability_lag (shifts timestamp forward) -> PIT,
  * builds rolling z-scores and growth rates (right-aligned, never full-sample),
  * asof-aligns to decision_time,
  * tags every factor with max availability_ts so make_supervised_dataset can
    enforce PIT.

If no on-chain data is supplied, returns an empty (all-NaN) factor frame so the
pipeline degrades gracefully (current project has no on-chain source yet).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# only metrics considered safe for the CORE experiment (v6 §4.2.1)
CORE_RECOMPUTABLE = {"active_address", "tx_count", "transfer_volume", "gas_used", "new_addresses"}


def onchain_factors(
    onchain: Optional[pd.DataFrame],
    decision_times: pd.DatetimeIndex,
    availability_lag: Dict[str, pd.Timedelta] | None = None,
    z_window: int = 30,
    metrics: Optional[List[str]] = None,
    core_only: bool = True,
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by decision_time with columns:
      <metric>_z, <metric>_growth  for each metric, plus max_onchain_availability_ts.
    """
    base_cols = ["max_onchain_availability_ts"]
    if onchain is None or len(onchain) == 0:
        return pd.DataFrame(index=decision_times, columns=base_cols, dtype="datetime64[ns]")

    df = onchain.sort_index().copy()
    metrics = metrics or [c for c in df.columns]
    if core_only:
        metrics = [m for m in metrics if m in CORE_RECOMPUTABLE]
    if not metrics:
        return pd.DataFrame(index=decision_times, columns=base_cols, dtype="datetime64[ns]")

    availability_lag = availability_lag or {}
    out_frames = []
    avail_ts = pd.Series(pd.NaT, index=df.index)
    for m in metrics:
        lag = availability_lag.get(m, pd.Timedelta(hours=1))
        s = df[m].copy()
        s.index = s.index + lag                       # PIT availability
        z = (s - s.rolling(z_window, min_periods=5).mean()) / (s.rolling(z_window, min_periods=5).std() + 1e-12)
        growth = s.pct_change()
        fm = pd.DataFrame({f"{m}_z": z, f"{m}_growth": growth})
        out_frames.append(fm)

    feat = pd.concat(out_frames, axis=1).sort_index()
    feat["max_onchain_availability_ts"] = feat.index
    aligned = feat.reindex(feat.index.union(decision_times)).sort_index().ffill().reindex(decision_times)
    aligned.index = decision_times
    return aligned
