"""
crypto.live.oms  &  crypto.live.exchange
==============================================
Order-management skeleton (v6 §14.4/§14.5) with audit fixes:
  * default = limit GTD (or GTC + local timed cancel) — NOT "guaranteed fill"
    (detail #16 / simp #3).  Partial fills update internal position to the
    ACTUAL filled qty (detail #15).
  * large order (> 0.1 * avg_depth_proxy) -> TWAP slicing with max_price_drift
    stop (detail #17 / simp #2.3).
  * idempotent client_order_id; reconciliation against exchange (detail #8/#14.5).

A PaperBroker (in-memory fill simulator) makes this runnable offline; a CCXT
wrapper (`CCXTExchange`) provides the same interface for live use (needs ccxt).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class OrderStatus(str, Enum):
    NOT_SUBMITTED = "not_submitted"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    EXPIRED_CANCELLED = "expired_cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    symbol: str
    side: str                 # "buy" | "sell"
    qty: float
    order_type: str = "limit_gtd"   # limit_gtd | market | twap
    limit_price: Optional[float] = None
    tif_seconds: float = 4 * 3600   # GTD validity = current 4h window
    client_order_id: str = field(default_factory=lambda: f"v6-{uuid.uuid4().hex[:12]}")
    status: OrderStatus = OrderStatus.NOT_SUBMITTED
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0


class PaperBroker:
    """Deterministic fill simulator for dry-run / paper trading."""

    def __init__(self, max_slippage_bps: float = 10.0, seed: int = 0):
        self.max_slippage_bps = max_slippage_bps
        self.rng = np.random.default_rng(seed)
        self.positions: Dict[str, float] = {}
        self.seen_client_ids: set = set()

    def submit(self, order: Order, ref_price: float, available_liquidity: float) -> Order:
        if order.client_order_id in self.seen_client_ids:   # idempotency
            return order
        self.seen_client_ids.add(order.client_order_id)
        order.status = OrderStatus.SUBMITTED

        want = order.qty
        # liquidity-limited partial fill (detail #15)
        fillable = min(want, max(available_liquidity, 0.0))
        if fillable <= 0:
            order.status = OrderStatus.EXPIRED_CANCELLED
            return order

        slip = self.max_slippage_bps / 1e4
        px = ref_price * (1 + slip) if order.side == "buy" else ref_price * (1 - slip)

        # limit GTD: only fills if marketable vs limit
        if order.order_type == "limit_gtd" and order.limit_price is not None:
            if order.side == "buy" and px > order.limit_price:
                order.status = OrderStatus.EXPIRED_CANCELLED
                return order
            if order.side == "sell" and px < order.limit_price:
                order.status = OrderStatus.EXPIRED_CANCELLED
                return order

        order.filled_qty = fillable
        order.avg_fill_price = px
        order.status = OrderStatus.FILLED if fillable >= want - 1e-12 else OrderStatus.PARTIAL_FILLED
        signed = fillable if order.side == "buy" else -fillable
        self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + signed
        return order

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)


def is_large_order(order_notional: float, avg_depth_proxy: float, frac: float = 0.1) -> bool:
    """detail #2.3: TWAP threshold."""
    if avg_depth_proxy is None or avg_depth_proxy <= 0:
        return True
    return order_notional > frac * avg_depth_proxy


def twap_slice(total_qty: float, n_slices: int = 5) -> List[float]:
    base = total_qty / n_slices
    return [base] * n_slices


def execute_twap(broker, symbol, side, total_qty, n_slices, signal_ref_price,
                 price_feed, max_price_drift_bps: float = 50.0,
                 available_liquidity_per_slice: float = 1e9) -> List[Order]:
    """
    TWAP with price-drift stop (detail #17): abort remaining slices if price
    deviates from signal_ref_price beyond max_price_drift_bps.
    price_feed: callable() -> current ref price.
    """
    orders, slices = [], twap_slice(total_qty, n_slices)
    drift = max_price_drift_bps / 1e4
    for q in slices:
        px = price_feed()
        if abs(px / signal_ref_price - 1.0) > drift:
            break  # stop remaining slices, re-evaluate signal
        o = Order(symbol=symbol, side=side, qty=q, order_type="market")
        orders.append(broker.submit(o, px, available_liquidity_per_slice))
    return orders


def reconcile(internal_positions: Dict[str, float], exchange_positions: Dict[str, float],
              tol: float = 1e-6) -> Dict[str, bool]:
    """detail #8/#14.5: compare internal vs exchange; flag drift."""
    out = {}
    for sym in set(internal_positions) | set(exchange_positions):
        a = internal_positions.get(sym, 0.0)
        b = exchange_positions.get(sym, 0.0)
        out[sym] = abs(a - b) <= tol + 1e-6 * max(abs(a), abs(b))
    return out


# --------------------------------------------------------------------------- #
# CCXT wrapper (live; needs ccxt + keys). Interface mirrors PaperBroker.
# --------------------------------------------------------------------------- #
class CCXTExchange:
    """Thin live wrapper. Not runnable offline; provided as the live interface."""

    def __init__(self, exchange_id: str = "binanceusdm", proxies=None, timeout=30000):
        try:
            import ccxt
        except Exception as e:
            raise RuntimeError("ccxt required for live trading") from e
        klass = getattr(ccxt, exchange_id)
        self.ex = klass({"enableRateLimit": True, "timeout": timeout, "proxies": proxies})
        self.seen_client_ids: set = set()

    def submit(self, order: Order, ref_price: float = None, available_liquidity: float = None) -> Order:
        if order.client_order_id in self.seen_client_ids:
            return order
        self.seen_client_ids.add(order.client_order_id)
        params = {"newClientOrderId": order.client_order_id}
        otype = "market" if order.order_type == "market" else "limit"
        res = self.ex.create_order(order.symbol, otype, order.side, order.qty,
                                   order.limit_price, params)
        order.status = OrderStatus.SUBMITTED
        order.filled_qty = float(res.get("filled", 0.0) or 0.0)
        order.avg_fill_price = float(res.get("average", 0.0) or 0.0)
        return order

    def fetch_positions(self) -> Dict[str, float]:
        pos = self.ex.fetch_positions()
        return {p["symbol"]: float(p.get("contracts", 0) or 0) *
                (1 if p.get("side") == "long" else -1) for p in pos}
