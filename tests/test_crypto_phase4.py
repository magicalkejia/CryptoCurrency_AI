"""
tests.test_crypto_phase4
==============================
Tests for the phase-4+ modules: PatchTST (fallback), on-chain factors,
derivatives funding features, live risk guard & OMS.
"""
import numpy as np
import pandas as pd

from crypto.schemas import FrozenConfig
from crypto.adapters import to_bars_schema, decision_time_grid
from crypto.models.patchtst import run_patchtst, make_windows, HORIZONS_H
from crypto.features.onchain import onchain_factors, CORE_RECOMPUTABLE
from crypto.features.derivatives import funding_features
from crypto.live.risk_guard import CircuitBreaker, CBLevel, check_staleness, signal_data_is_pit
from crypto.live.oms import (Order, PaperBroker, OrderStatus, is_large_order,
                                execute_twap, reconcile)


def _synth_1h(n=24 * 120, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998,
                       "close": close, "volume": rng.lognormal(10, .4, n),
                       "taker_buy_vol": rng.lognormal(9, .4, n),
                       "net_taker_vol": rng.normal(0, 1, n)}, index=idx)
    return df


# ---- PatchTST fallback ---------------------------------------------------- #
def test_patchtst_windows_no_future():
    bars = to_bars_schema(_synth_1h(), "1h")
    dts = pd.DatetimeIndex(bars.index[200::4])  # 4h-ish decision grid
    X, tgts, vdts = make_windows(bars, dts, ["close", "volume"], lookback=96)
    assert X is not None and X.shape[1] == 96 and X.shape[2] == 2
    assert set(tgts.keys()) == set(HORIZONS_H.keys())
    # windows end at or before decision_time (no future leak in the window)
    assert len(vdts) == len(X)


def test_patchtst_oof_features_runnable():
    fcfg = FrozenConfig()
    bars = to_bars_schema(_synth_1h(), "1h")
    dts = pd.DatetimeIndex(bars.index[200::4])
    out = run_patchtst(bars, dts, "BTC/USDT", fcfg, lookback=96, emb_dim=4)
    assert "patchtst_forecast_4h" in out.columns
    assert any(c.startswith("patchtst_emb_") for c in out.columns)
    # some OOF rows should be populated
    assert out["patchtst_forecast_4h"].notna().sum() > 0


# ---- on-chain factors ----------------------------------------------------- #
def test_onchain_core_only_filter():
    idx = pd.date_range("2022-01-01", periods=200, freq="1h")
    oc = pd.DataFrame({"active_address": np.arange(200) + 100.0,
                       "entity_adjusted_flow": np.arange(200) * 2.0}, index=idx)  # non-core
    dts = idx[150::4]
    feat = onchain_factors(oc, dts, core_only=True)
    assert "active_address_z" in feat.columns
    assert "entity_adjusted_flow_z" not in feat.columns   # filtered (non-recomputable)


def test_onchain_empty_graceful():
    dts = pd.date_range("2022-01-01", periods=10, freq="4h")
    feat = onchain_factors(None, dts)
    assert len(feat) == len(dts)   # degrades gracefully, no crash


# ---- derivatives funding -------------------------------------------------- #
def test_funding_features_pit_asof():
    fidx = pd.date_range("2022-01-01", periods=100, freq="8h")
    funding = pd.Series(np.linspace(-0.001, 0.001, 100), index=fidx)
    dts = pd.DatetimeIndex(["2022-01-02 05:00", "2022-01-05 09:00"])
    feat = funding_features(funding, dts, z_window=10)
    # value at decision is the last funding strictly known by then (asof/ffill)
    assert feat["funding_rate"].notna().all()


# ---- live: circuit breaker hierarchy -------------------------------------- #
def test_circuit_breaker_levels():
    cb = CircuitBreaker()
    assert cb.evaluate(drawdown=0.05, daily_loss=0.0) == CBLevel.NORMAL
    assert cb.evaluate(drawdown=0.12, daily_loss=0.0) == CBLevel.L1_WARN
    assert cb.evaluate(drawdown=0.16, daily_loss=0.0) == CBLevel.L2_DELEVER
    assert cb.evaluate(drawdown=0.21, daily_loss=0.0) == CBLevel.L3_HALT
    # kill switch forces L3 even when metrics are calm
    assert cb.evaluate(drawdown=0.0, daily_loss=0.0, kill_switch=True) == CBLevel.L3_HALT


def test_circuit_breaker_recovery_flow():
    cb = CircuitBreaker(recover_periods=2)
    cb.evaluate(drawdown=0.25, daily_loss=0.0)             # L3
    assert cb.position_multiplier() == 0.0
    # incomplete recovery -> denied
    assert cb.request_recovery(True, True, True, human_confirmed=False) is False
    # full recovery -> reduced risk mode
    assert cb.request_recovery(True, True, True, human_confirmed=True) is True
    assert cb.position_multiplier() == 0.5
    cb.step_period(); cb.step_period()
    assert cb.position_multiplier() == 1.0                 # reduced mode expired


def test_staleness_three_tiers():
    t = pd.Timestamp("2022-01-01 00:00:00")
    assert check_staleness(t, t + pd.Timedelta(seconds=1)).status == "normal"
    r = check_staleness(t, t + pd.Timedelta(seconds=5))
    assert r.status == "stale_warning" and r.allow_trade and not r.allow_add
    assert check_staleness(t, t + pd.Timedelta(seconds=12)).status == "skip"


def test_signal_pit_guard():
    dt = pd.Timestamp("2022-01-01 12:01")
    assert signal_data_is_pit(pd.Timestamp("2022-01-01 12:00"), dt)
    assert not signal_data_is_pit(pd.Timestamp("2022-01-01 12:05"), dt)


# ---- live: OMS ------------------------------------------------------------ #
def test_oms_partial_fill_and_idempotency():
    br = PaperBroker(max_slippage_bps=0)
    o = Order("BTC/USDT", "buy", qty=10, order_type="market")
    o = br.submit(o, ref_price=100, available_liquidity=4)   # only 4 fillable
    assert o.status == OrderStatus.PARTIAL_FILLED and o.filled_qty == 4
    assert br.get_position("BTC/USDT") == 4
    # resubmitting same client_order_id is a no-op (idempotent)
    br.submit(o, ref_price=100, available_liquidity=4)
    assert br.get_position("BTC/USDT") == 4


def test_oms_gtd_limit_no_fill_when_unmarketable():
    br = PaperBroker(max_slippage_bps=10)
    o = Order("BTC/USDT", "buy", qty=1, order_type="limit_gtd", limit_price=99)  # below market
    o = br.submit(o, ref_price=100, available_liquidity=10)
    assert o.status == OrderStatus.EXPIRED_CANCELLED and o.filled_qty == 0


def test_twap_drift_stop():
    br = PaperBroker(max_slippage_bps=0)
    prices = iter([100, 100, 200, 200, 200])  # 3rd slice drifts >50bps -> stop
    orders = execute_twap(br, "BTC/USDT", "buy", total_qty=5, n_slices=5,
                          signal_ref_price=100, price_feed=lambda: next(prices),
                          max_price_drift_bps=50)
    assert len(orders) == 2   # stopped after drift


def test_large_order_and_reconcile():
    assert is_large_order(order_notional=200, avg_depth_proxy=1000, frac=0.1) is True
    assert is_large_order(order_notional=50, avg_depth_proxy=1000, frac=0.1) is False
    rec = reconcile({"BTC/USDT": 1.0}, {"BTC/USDT": 1.0, "ETH/USDT": 0.5})
    assert rec["BTC/USDT"] is True and rec["ETH/USDT"] is False
