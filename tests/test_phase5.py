"""
tests.test_phase5
====================
Tests for: Dune row parsing, live market feed + snapshot consistency,
PatchTST A/B/C/D ablation, Holdout-A freeze/guard, PBO (CSCV), experiment registry.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from crypto.schemas import FrozenConfig
from etl.dune_loader import rows_to_frame
from crypto.live.market_feed import ReplayFeed, snapshot_consistency_check
from crypto.experiments.patchtst_ablation import run_ablation
from crypto.governance.holdout import (dev_holdout_split, MaxTrainTimeGuard,
                                          freeze_config, load_frozen)
from crypto.governance.pbo import cscv_pbo
from crypto.governance.registry import pre_register, assert_preregistered, is_preregistered


# ---- Dune parsing --------------------------------------------------------- #
def test_dune_rows_to_frame():
    rows = [{"day": "2023-01-01", "active_address": "100", "tx_count": "5"},
            {"day": "2023-01-02", "active_address": "120", "tx_count": "7"}]
    df = rows_to_frame(rows)
    assert list(df.columns) == ["active_address", "tx_count"]
    assert df["active_address"].dtype.kind in "if" and df.index[0] == pd.Timestamp("2023-01-01")


# ---- market feed + staleness --------------------------------------------- #
def test_replay_feed_snapshot_consistency():
    idx = pd.date_range("2022-01-01", periods=10, freq="1h")
    bars = {"BTC/USDT": pd.DataFrame({"close": np.arange(10) + 100.0}, index=idx),
            "ETH/USDT": pd.DataFrame({"close": np.arange(10) + 50.0}, index=idx)}
    feed = ReplayFeed(bars)
    feed.set_now(pd.Timestamp("2022-01-01 05:00:00"))
    rt = pd.Timestamp("2022-01-01 05:00:01")  # 1s after last bar -> normal
    res = snapshot_consistency_check(feed, ["BTC/USDT", "ETH/USDT"], rt)
    assert set(res["tradeable"]) == {"BTC/USDT", "ETH/USDT"}
    # stale read 12s later -> skip
    res2 = snapshot_consistency_check(feed, ["BTC/USDT"], pd.Timestamp("2022-01-01 05:00:12"))
    assert res2["per_symbol"]["BTC/USDT"].status == "skip"


# ---- PatchTST ablation ---------------------------------------------------- #
def _ablation_dataset(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2022-01-01", periods=n, freq="4h")
    signal = rng.normal(0, 1, n)
    tab = signal + rng.normal(0, 0.5, n)                 # informative tabular feature
    fwd = 0.01 * np.tanh(signal) + rng.normal(0, 0.01, n)
    tb = np.sign(fwd); tb[np.abs(fwd) < 0.005] = 0
    df = pd.DataFrame({
        "symbol": "BTC/USDT", "decision_time": dt, "entry_time": dt,
        "exit_time": dt + pd.Timedelta(hours=8),
        "tb_label": tb.astype(int), "uniqueness_weight": 1.0,
        "raw_exit_return_long": fwd,
        "tab_feat": tab,
        "patchtst_forecast_4h": 0.5 * signal + rng.normal(0, 0.5, n),
        "patchtst_forecast_24h": 0.5 * signal + rng.normal(0, 0.5, n),
    })
    for j in range(4):
        df[f"patchtst_emb_{j}"] = signal * rng.normal(0, 1) + rng.normal(0, 1, n)
    return df


def test_ablation_runs_four_configs():
    fcfg = FrozenConfig()
    ds = _ablation_dataset()
    res = run_ablation(ds, ["tab_feat"], fcfg, max_label_horizon_bars=2)
    assert set(res.index) == {"A_tabular", "B_patchtst_forecast",
                              "C_emb_plus_tabular", "D_full_fusion"}
    assert "IC" in res.columns and "deflated_sharpe" in res.columns
    assert res["n_oof"].min() > 0


# ---- Holdout-A freeze / guard -------------------------------------------- #
def test_dev_holdout_split_and_guard():
    dt = pd.Series(pd.date_range("2022-01-01", periods=100, freq="D"))
    dev, hold = dev_holdout_split(dt, "2022-03-01")
    assert len(dev) + len(hold) == 100
    assert dt.iloc[dev].max() < pd.Timestamp("2022-03-01") <= dt.iloc[hold].min()
    g = MaxTrainTimeGuard("2022-03-01")
    g.check(dt.iloc[dev])  # ok
    try:
        g.check(dt.iloc[hold])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_freeze_config_tamper_detection():
    fcfg = FrozenConfig()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "frozen.json"
        h = freeze_config(fcfg, p)
        payload = load_frozen(p)        # ok
        assert payload["config_hash"] == h
        # tamper: change a config value but keep old hash
        raw = json.loads(p.read_text())
        raw["config"]["risk"]["p_threshold"] = 0.99
        p.write_text(json.dumps(raw))
        try:
            load_frozen(p)
            raised = False
        except ValueError:
            raised = True
        assert raised


# ---- PBO ------------------------------------------------------------------ #
def test_pbo_dominant_signal_low():
    rng = np.random.default_rng(1)
    T, N = 240, 6
    R = rng.normal(0, 0.01, (T, N))
    R[:, 0] += 0.004           # config 0 has a genuine, persistent edge
    out = cscv_pbo(R, n_blocks=8)
    assert 0.0 <= out["pbo"] <= 1.0
    assert out["pbo"] < 0.5    # a truly-best config -> low overfitting prob


def test_pbo_pure_noise_high():
    rng = np.random.default_rng(2)
    R = rng.normal(0, 0.01, (240, 8))   # no real edge anywhere
    out = cscv_pbo(R, n_blocks=8)
    assert out["pbo"] >= 0.4            # noise selection -> high overfitting prob


# ---- experiment registry -------------------------------------------------- #
def test_registry_preregistration_gate():
    cfg = {"label": {"tp": 2}, "model": {"seed": 42}}
    other = {"label": {"tp": 3}, "model": {"seed": 42}}
    with tempfile.TemporaryDirectory() as d:
        reg = Path(d) / "registry.json"
        try:
            assert_preregistered(cfg, reg)   # not registered yet -> raises
            raised = False
        except PermissionError:
            raised = True
        assert raised
        pre_register(cfg, reg, label="confirmatory_run_1")
        assert is_preregistered(cfg, reg)
        assert assert_preregistered(cfg, reg) is True
        assert not is_preregistered(other, reg)   # different config not registered


# ---- incremental study (Step0->Step6 ladder) ------------------------------ #
def test_incremental_study_runs_all_steps():
    from crypto.experiments.incremental_study import run_incremental_study
    rng = np.random.default_rng(0)
    n = 360
    dt = pd.date_range("2022-01-01", periods=n, freq="4h")
    sig = rng.normal(0, 1, n)
    fwd = 0.01 * np.tanh(sig) + rng.normal(0, 0.01, n)
    tb = np.sign(fwd); tb[np.abs(fwd) < 0.005] = 0
    close = pd.Series(100 * np.exp(np.cumsum(0.01 * sig)), index=dt)
    df = pd.DataFrame({
        "symbol": "BTC/USDT", "decision_time": dt, "entry_time": dt,
        "exit_time": dt + pd.Timedelta(hours=8), "tb_label": tb.astype(int),
        "uniqueness_weight": 1.0, "raw_exit_return_long": fwd,
        "net_exit_return_long": fwd, "net_exit_return_short": -fwd,
        "mkt": sig + rng.normal(0, .5, n),
        "onchain_z": 0.3 * sig + rng.normal(0, 1, n),
        "narr": 0.2 * sig + rng.normal(0, 1, n),
        "patchtst_forecast_4h": 0.4 * sig + rng.normal(0, .6, n),
    })
    close_panel = pd.DataFrame({"BTC/USDT": close})
    modality = {"market": ["mkt"], "onchain": ["onchain_z"],
                "narrative": ["narr"], "patchtst": ["patchtst_forecast_4h"]}
    fcfg = FrozenConfig()
    table = run_incremental_study(df, close_panel, modality, fcfg, bars_per_year=2190, max_pos=0.2)
    assert list(table.index) == ["Step0_baseline_tsmom", "Step1_market", "Step2_+onchain",
                                 "Step3_+narrative", "Step4_+patchtst", "Step5_fusion",
                                 "Step6_meta_gate"]
    assert "incr_NW_t" in table.columns and "research_question" in table.columns
    # every non-baseline step has a research question mapped
    assert table.loc["Step1_market", "research_question"] != ""
