"""
crypto.governance.holdout
============================
Holdout-A discipline (v6 §6.4): dev / locked-holdout split, a max_train_time
guard (training never reads holdout data), and frozen_config write/verify with
tamper detection.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from crypto.schemas import FrozenConfig, environment_hash


def dev_holdout_split(decision_time: pd.Series, holdout_start) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dev_idx, holdout_idx). dev = strictly before holdout_start."""
    dt = pd.to_datetime(pd.Series(decision_time).reset_index(drop=True))
    holdout_start = pd.Timestamp(holdout_start)
    pos = np.arange(len(dt))
    dev = pos[(dt < holdout_start).to_numpy()]
    hold = pos[(dt >= holdout_start).to_numpy()]
    return dev, hold


class MaxTrainTimeGuard:
    """
    Wrap any training-data access. During Holdout-A, set max_train_time =
    holdout_start; any attempt to read rows with decision_time >= max_train_time
    raises (v6 §6.4 hard isolation).
    """
    def __init__(self, max_train_time):
        self.max_train_time = pd.Timestamp(max_train_time)

    def check(self, decision_time: pd.Series):
        dt = pd.to_datetime(pd.Series(decision_time))
        if (dt >= self.max_train_time).any():
            raise ValueError(
                f"training data accessed at/after max_train_time={self.max_train_time} "
                f"(holdout leakage)")
        return True


def freeze_config(fcfg: FrozenConfig, path) -> str:
    """Write frozen_config.json (+ environment) and return its config_hash."""
    payload = {"config": fcfg.to_dict(),
               "config_hash": fcfg.config_hash(),
               "environment_hash": environment_hash()}
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2))
    return payload["config_hash"]


def load_frozen(path) -> dict:
    """Load and VERIFY a frozen config; raises if the file was tampered with."""
    payload = json.loads(Path(path).read_text())
    # recompute hash from the stored config and compare
    fc = FrozenConfig()  # default skeleton to access hashing of a rebuilt dict
    stored = payload["config"]
    import hashlib
    recomputed = hashlib.sha256(
        json.dumps(stored, sort_keys=True, default=str).encode()).hexdigest()[:12]
    if recomputed != payload["config_hash"]:
        raise ValueError("frozen_config tamper detected: config_hash mismatch")
    return payload
