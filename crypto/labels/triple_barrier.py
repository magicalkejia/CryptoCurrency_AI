"""
crypto.labels.triple_barrier
===============================
Triple-barrier labelling per v6 §6.1, with the round-6 audit fixes:

  #1/#4  separate net_exit_return_long / net_exit_return_short (funding is
         sign-dependent, cannot reuse -long for short).
  short  raw_exit_return_short = 1 - exit/entry   (NOT entry/exit - 1) -> correct
         linear-perp notional return, no overstatement on big moves.
  #2     dual-touch in the same 1h bar -> tb_label = 0, reason = dual_touch_ambiguous
         (configurable to stop_loss_first), avoids direction-model bias.
  vert   vertical barrier measured from entry_time (not decision_time).
  entry  label_entry_price == backtest entry_ref_price (shared exec_price fn).

Time semantics (locked):
  decision_time -> entry_time(=t0) -> exit_time(=t1); holding interval [t0, t1).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from crypto.exec_price import get_entry_price, funding_return, net_return


def _atr(bars: pd.DataFrame, window: int) -> pd.Series:
    """Wilder-style ATR on the given OHLC frame (right-aligned, no future)."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


LABEL_COLUMNS = [
    "symbol", "decision_time", "entry_time", "exit_time",
    "label_entry_price", "exit_price",
    "raw_exit_return_long", "raw_exit_return_short",
    "net_exit_return_long", "net_exit_return_short",
    "tb_label", "tb_exit_reason",
    "barrier_width_pct", "atr20_at_decision",
    "label_config_hash", "cost_model_hash", "uniqueness_weight",
]


def compute_triple_barrier(
    bars_1h: pd.DataFrame,
    decision_times: pd.DatetimeIndex,
    symbol: str,
    label_cfg,
    cost_cfg,
    bars_1m: Optional[pd.DataFrame] = None,
    funding: Optional[pd.Series] = None,
    label_config_hash: str = "",
    cost_model_hash: str = "",
) -> pd.DataFrame:
    """
    bars_1h : DataFrame indexed by ts_open with columns open/high/low/close.
    decision_times : decision timestamps (= 4h bar close + offset).
    Returns a DataFrame with LABEL_COLUMNS.
    """
    if label_cfg.entry_rule == "next_1m_open" and bars_1m is None:
        raise ValueError("bars_1m required for next_1m_open")

    bars_1h = bars_1h.sort_index()
    atr = _atr(bars_1h, label_cfg.atr_window)
    close = bars_1h["close"]

    # scan-array (1h) used for the path between entry and vertical barrier
    scan = bars_1h if label_cfg.entry_rule == "next_1h_open" else bars_1m.sort_index()
    scan_idx = scan.index
    scan_high = scan["high"].to_numpy()
    scan_low = scan["low"].to_numpy()
    scan_close = scan["close"].to_numpy()

    rows = []
    vertical = pd.Timedelta(days=label_cfg.vertical_days)
    for dt in decision_times:
        # ATR & close known at decision_time (last 1h bar <= dt)
        prior = atr.index[atr.index <= dt]
        if len(prior) == 0 or not np.isfinite(atr.loc[prior[-1]]):
            continue
        atr_at = float(atr.loc[prior[-1]])
        close_at = float(close.loc[prior[-1]])
        if close_at <= 0:
            continue
        barrier_width_pct = atr_at / close_at

        entry_time, entry_price = get_entry_price(bars_1h, dt, label_cfg.entry_rule, bars_1m)
        if entry_time is None or entry_price is None or entry_price <= 0:
            continue

        upper = entry_price * (1 + label_cfg.tp_mult * barrier_width_pct)
        lower = entry_price * (1 - label_cfg.sl_mult * barrier_width_pct)
        vexit = entry_time + vertical

        # forward scan over [entry_time, vexit]
        lo = scan_idx.searchsorted(entry_time, side="left")
        hi = scan_idx.searchsorted(vexit, side="right")
        tb_label, reason, exit_time, exit_price = 0, "vertical_neutral", None, None
        for j in range(lo, hi):
            hit_up = scan_high[j] >= upper
            hit_dn = scan_low[j] <= lower
            if hit_up and hit_dn:
                # ambiguous order within the bar (audit #2)
                if label_cfg.intrabar_dual_touch == "stop_loss_first":
                    tb_label, reason, exit_price = -1, "dual_touch_stop", lower
                else:
                    tb_label, reason, exit_price = 0, "dual_touch_ambiguous", scan_close[j]
                exit_time = scan_idx[j]
                break
            if hit_up:
                tb_label, reason, exit_time, exit_price = 1, "upper_touch", scan_idx[j], upper
                break
            if hit_dn:
                tb_label, reason, exit_time, exit_price = -1, "lower_touch", scan_idx[j], lower
                break
        if exit_time is None:
            # vertical expiry
            k = hi - 1
            if k < lo:
                continue
            exit_time = scan_idx[k]
            exit_price = float(scan_close[k])

        # raw returns (audit: short = 1 - exit/entry, NOT entry/exit - 1)
        raw_long = exit_price / entry_price - 1.0
        raw_short = 1.0 - exit_price / entry_price

        # resolve neutral tb_label on vertical expiry
        if reason == "vertical_neutral":
            neutral_thr = label_cfg.neutral_threshold_frac * barrier_width_pct
            if abs(raw_long) < neutral_thr:
                tb_label = 0
            else:
                tb_label = int(np.sign(raw_long))
                reason = "vertical_up" if tb_label > 0 else "vertical_down"

        f_long = funding_return("long", entry_time, exit_time, entry_price, funding)
        f_short = funding_return("short", entry_time, exit_time, entry_price, funding)
        net_long = net_return(raw_long, "long", cost_cfg, funding_ret=f_long)
        net_short = net_return(raw_short, "short", cost_cfg, funding_ret=f_short)

        rows.append({
            "symbol": symbol, "decision_time": dt,
            "entry_time": entry_time, "exit_time": exit_time,
            "label_entry_price": entry_price, "exit_price": exit_price,
            "raw_exit_return_long": raw_long, "raw_exit_return_short": raw_short,
            "net_exit_return_long": net_long, "net_exit_return_short": net_short,
            "tb_label": int(tb_label), "tb_exit_reason": reason,
            "barrier_width_pct": barrier_width_pct, "atr20_at_decision": atr_at,
            "label_config_hash": label_config_hash, "cost_model_hash": cost_model_hash,
            "uniqueness_weight": np.nan,
        })

    return pd.DataFrame(rows, columns=LABEL_COLUMNS)
