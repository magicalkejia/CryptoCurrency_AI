"""
crypto.labels.meta_label
============================
Stage-2 meta-label (v6 §7.5) with audit fixes:
  #1/#3  uses separate net_exit_return_long / net_exit_return_short.
  #5     meta_label must be built from OOF primary_direction, never in-sample.

  primary_direction = flat  -> sample dropped
  primary_direction = long  -> 1 if net_exit_return_long  > 0 else 0
  primary_direction = short -> 1 if net_exit_return_short > 0 else 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_meta_label(
    primary_direction_oof: pd.Series,
    net_exit_return_long: pd.Series,
    net_exit_return_short: pd.Series,
    source: str = "oof",
) -> pd.DataFrame:
    """
    Returns DataFrame[meta_label, selected_net_exit_return] for non-flat rows.

    `source` must be "oof" to certify the direction came from out-of-fold
    predictions (audit #5).  Passing source="in_sample" raises, so the circular
    dependency is structurally blocked (test T1b_13).
    """
    if source != "oof":
        raise ValueError(
            "primary_direction must come from OOF predictions (source='oof'); "
            "building meta_label from in-sample direction is forbidden (audit #5)."
        )

    pdir = primary_direction_oof.astype(str)
    long_mask = pdir == "long"
    short_mask = pdir == "short"
    keep = long_mask | short_mask

    selected = pd.Series(np.nan, index=pdir.index, dtype=float)
    selected[long_mask] = net_exit_return_long[long_mask]
    selected[short_mask] = net_exit_return_short[short_mask]

    meta = (selected > 0).astype(int)
    out = pd.DataFrame({"meta_label": meta, "selected_net_exit_return": selected})
    return out[keep]
