"""
tests.test_crypto
=======================
Unit tests mapped to v6 Appendix A (T1a_*, T1b_*).  Runnable with pytest, or
with `python tests/run_all.py` (no pytest dependency).
"""
import numpy as np
import pandas as pd

from crypto.schemas import LabelConfig, CostConfig, CVConfig, ModelConfig, FrozenConfig
from crypto.labels.triple_barrier import compute_triple_barrier
from crypto.labels.meta_label import build_meta_label
from crypto.features.uniqueness import average_uniqueness
from crypto.cv.purged_kfold import purged_embargoed_splits
from crypto.benchmark.tsmom import vol_parity_tsmom_weights
from crypto.exec_price import get_entry_price, funding_return, net_return, slippage_bps
from crypto.pit import make_supervised_dataset, audit_lookahead
from crypto.models.calibrate import compute_ece


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _bars_from_path(prices, start="2021-01-01", freq="1h"):
    idx = pd.date_range(start, periods=len(prices), freq=freq)
    p = np.asarray(prices, float)
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p}, index=idx)


# --------------------------------------------------------------------------- #
# T1b_01 triple-barrier upper touch
# --------------------------------------------------------------------------- #
def test_T1b_01_upper_touch():
    # warmup oscillates (ATR>0 -> real barrier band), entry flat, then spike up
    warm = [101 if i % 2 == 0 else 99 for i in range(26)]   # ATR ~2 -> width ~2%
    prices = warm + [100, 100] + [100] * 40
    bars = _bars_from_path(prices)
    bars.loc[bars.index[27], "high"] = 130                  # clear spike above upper
    lc = LabelConfig(entry_rule="next_1h_open", tp_mult=2.0, sl_mult=1.0, atr_window=20)
    cc = CostConfig()
    dts = pd.DatetimeIndex([bars.index[25]])                # entry = bar 26 (open 100)
    out = compute_triple_barrier(bars, dts, "BTC/USDT", lc, cc)
    assert len(out) == 1
    assert out.iloc[0]["tb_label"] == 1
    assert out.iloc[0]["tb_exit_reason"] == "upper_touch"


# T1b_02 / T1b_21 dual touch ambiguous
def test_T1b_02_dual_touch_ambiguous():
    prices = [100.0] * 30 + [100.0] * 40
    bars = _bars_from_path(prices)
    # at bar 27, high and low straddle both barriers
    j = 27
    bars.loc[bars.index[j], "high"] = 200.0
    bars.loc[bars.index[j], "low"] = 1.0
    lc = LabelConfig(entry_rule="next_1h_open", intrabar_dual_touch="ambiguous", atr_window=20)
    out = compute_triple_barrier(bars, pd.DatetimeIndex([bars.index[25]]), "BTC/USDT", lc, CostConfig())
    assert out.iloc[0]["tb_exit_reason"] == "dual_touch_ambiguous"
    assert out.iloc[0]["tb_label"] == 0


# T1b_03 vertical neutral expiry
def test_T1b_03_vertical_neutral():
    # warmup oscillates (ATR>0), then perfectly flat -> price stays inside band
    # for the whole 1-day vertical window and expires neutral.
    warm = [101 if i % 2 == 0 else 99 for i in range(26)]
    prices = warm + [100.0] * 60
    bars = _bars_from_path(prices)
    lc = LabelConfig(vertical_days=1, neutral_threshold_frac=0.5, atr_window=20)  # 1 day = 24 bars
    out = compute_triple_barrier(bars, pd.DatetimeIndex([bars.index[25]]), "BTC/USDT", lc, CostConfig())
    assert out.iloc[0]["tb_label"] == 0
    assert "vertical" in out.iloc[0]["tb_exit_reason"]


# T1b_04 label entry price == backtest entry price (same fn)
def test_T1b_04_label_entry_price_consistency():
    bars = _bars_from_path([100 + i for i in range(50)])
    lc = LabelConfig(entry_rule="next_1h_open", atr_window=20)
    dt = bars.index[25]
    et, ep = get_entry_price(bars, dt, "next_1h_open")
    out = compute_triple_barrier(bars, pd.DatetimeIndex([dt]), "BTC/USDT", lc, CostConfig())
    assert abs(out.iloc[0]["label_entry_price"] - ep) < 1e-9
    assert out.iloc[0]["entry_time"] == et


# T1b_05 uniqueness no overlap -> 1
def test_T1b_05_uniqueness_no_overlap():
    t0 = pd.to_datetime(["2021-01-01 00:00", "2021-01-02 00:00", "2021-01-03 00:00"])
    t1 = pd.to_datetime(["2021-01-01 06:00", "2021-01-02 06:00", "2021-01-03 06:00"])
    w = average_uniqueness(pd.Series(t0), pd.Series(t1), normalize=False)
    assert np.allclose(w.to_numpy(), 1.0)


# T1b_06 uniqueness known overlap matches hand calc
def test_T1b_06_uniqueness_overlap():
    # two identical fully-overlapping intervals -> each uniqueness 0.5
    t0 = pd.to_datetime(["2021-01-01 00:00", "2021-01-01 00:00"])
    t1 = pd.to_datetime(["2021-01-01 04:00", "2021-01-01 04:00"])
    w = average_uniqueness(pd.Series(t0), pd.Series(t1), normalize=False)
    assert np.allclose(w.to_numpy(), 0.5)


# T1b_07 / T1b_18 half-open endpoint -> no overlap
def test_T1b_07_half_open_endpoint():
    t0 = pd.to_datetime(["2021-01-01 00:00", "2021-01-01 04:00"])
    t1 = pd.to_datetime(["2021-01-01 04:00", "2021-01-01 08:00"])  # a.t1 == b.t0
    w = average_uniqueness(pd.Series(t0), pd.Series(t1), normalize=False)
    assert np.allclose(w.to_numpy(), 1.0)  # touching but not overlapping


# T1b_08 purged double-sided overlap removed
def test_T1b_08_purged_overlap():
    dt = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"])
    t0 = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"])
    t1 = pd.to_datetime(["2021-01-03", "2021-01-04", "2021-01-05", "2021-01-06"])  # long, overlapping
    sym = pd.Series(["BTC"] * 4)
    folds = purged_embargoed_splits(pd.Series(dt), pd.Series(t0), pd.Series(t1),
                                    n_splits=2, embargo_delta=pd.Timedelta(0), symbol=sym)
    # for any fold, train labels must not overlap test labels
    for f in folds:
        for tr in f.train_idx:
            for te in f.test_idx:
                assert not (t0[tr] < t1[te] and t1[tr] > t0[te])


# T1b_09 multi-asset time-block: same period other symbols not in train
def test_T1b_09_multi_asset_time_block():
    times = pd.date_range("2021-01-01", periods=6, freq="D")
    dt = pd.Series(list(times) * 2)
    t0 = dt.copy()
    t1 = dt + pd.Timedelta(hours=1)
    sym = pd.Series(["BTC"] * 6 + ["ETH"] * 6)
    folds = purged_embargoed_splits(dt, t0, t1, n_splits=3, embargo_delta=pd.Timedelta(0), symbol=sym)
    for f in folds:
        test_times = set(dt.iloc[f.test_idx])
        train_times = set(dt.iloc[f.train_idx])
        assert test_times.isdisjoint(train_times)  # no same-block time in train


# T1b_10/11/12/16 meta-label mapping (long/short via separate net columns)
def test_T1b_10_11_12_meta_label():
    pdir = pd.Series(["long", "short", "flat"])
    net_long = pd.Series([0.01, -0.01, 0.0])
    net_short = pd.Series([-0.01, 0.009, 0.0])
    out = build_meta_label(pdir, net_long, net_short, source="oof")
    assert len(out) == 2                      # flat dropped (T1b_12)
    assert out.loc[0, "meta_label"] == 1      # long, net_long>0 (T1b_10)
    assert out.loc[1, "meta_label"] == 1      # short, net_short>0 (T1b_11/16)


# T1b_13 MVP double-head circular-dependency block
def test_T1b_13_no_in_sample_meta_label():
    try:
        build_meta_label(pd.Series(["long"]), pd.Series([0.01]), pd.Series([-0.01]),
                         source="in_sample")
        raised = False
    except ValueError:
        raised = True
    assert raised


# T1b_16 short large-move formula (1 - exit/entry, NOT entry/exit - 1)
def test_T1b_16_short_large_move():
    # entry=100, exit=50 -> short raw should be +50%, not +100%
    prices = [100.0] * 25 + [100.0] + [50.0] + [50.0] * 40  # drops to 50 -> hits lower
    bars = _bars_from_path(prices)
    bars.loc[bars.index[26], "low"] = 50.0
    lc = LabelConfig(entry_rule="next_1h_open", tp_mult=2.0, sl_mult=1.0, atr_window=20)
    out = compute_triple_barrier(bars, pd.DatetimeIndex([bars.index[25]]), "BTC/USDT", lc, CostConfig())
    row = out.iloc[0]
    # exit at lower barrier; verify short raw uses 1 - exit/entry
    expected_short = 1.0 - row["exit_price"] / row["label_entry_price"]
    assert abs(row["raw_exit_return_short"] - expected_short) < 1e-9
    # and explicitly: with entry 100 exit<=100, entry/exit-1 would be larger
    assert row["raw_exit_return_short"] <= 1.0


# T1b_17 funding flips meta-label sign for short
def test_T1b_17_funding_flip():
    cc = CostConfig(fee_bps=0, spread_proxy_bps=0, base_slippage_bps=0, rounding_bps=0)
    # short tiny profit raw=+0.001
    raw_short = 0.001
    entry, exit_t = pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02")
    # positive funding -> short RECEIVES -> helps; negative funding -> short PAYS -> hurts
    f_idx = pd.date_range("2021-01-01 08:00", "2021-01-01 16:00", freq="8h")
    neg_funding = pd.Series([-0.01, -0.01], index=f_idx)  # negative -> short pays
    fr = funding_return("short", entry, exit_t, 100.0, neg_funding)
    net = net_return(raw_short, "short", cc, funding_ret=fr)
    assert fr < 0          # short pays under negative funding
    assert net < 0         # flipped from +0.001 to negative


# T1a_10 tsmom all-zero direction -> no div by zero
def test_T1a_10_tsmom_zero_direction():
    idx = pd.date_range("2021-01-01", periods=120, freq="4h")
    flat = pd.DataFrame({"BTC": 100.0, "ETH": 100.0}, index=idx)  # zero momentum
    w = vol_parity_tsmom_weights(flat, lookback_mom=10, vol_window=10, cov_window=10)
    assert np.isfinite(w.to_numpy()).all()
    assert (w.abs().sum().sum()) == 0.0


# T1b_19 feature availability > decision_time -> raises
def test_T1b_19_feature_availability():
    feats = pd.DataFrame({
        "symbol": ["BTC"], "decision_time": [pd.Timestamp("2021-01-01 12:01")],
        "f1": [1.0], "max_feature_availability_ts": [pd.Timestamp("2021-01-01 13:00")],  # future!
    })
    labels = pd.DataFrame({"symbol": ["BTC"], "decision_time": [pd.Timestamp("2021-01-01 12:01")],
                           "tb_label": [1]})
    try:
        make_supervised_dataset(feats, labels, require_pit=True)
        raised = False
    except ValueError:
        raised = True
    assert raised


# T1b_20 label columns not allowed in feature matrix
def test_T1b_20_label_not_in_features():
    feats = pd.DataFrame({
        "symbol": ["BTC"], "decision_time": [pd.Timestamp("2021-01-01 12:01")],
        "net_exit_return_long": [0.1],  # a label column masquerading as a feature
        "max_feature_availability_ts": [pd.Timestamp("2021-01-01 12:00")],
    })
    labels = pd.DataFrame({"symbol": ["BTC"], "decision_time": [pd.Timestamp("2021-01-01 12:01")],
                           "tb_label": [1]})
    try:
        make_supervised_dataset(feats, labels, require_pit=True)
        raised = False
    except ValueError:
        raised = True
    assert raised


# slippage units sanity (audit #11): depth_ratio=0.2 -> liq term 2bps at k_liq=10
def test_slippage_units():
    cc = CostConfig(base_slippage_bps=3, k_vol_bps=2, k_liq_bps=10)
    s = slippage_bps(realized_vol_short_bps=100, vol_benchmark_bps=100,
                     order_notional=20, avg_depth_proxy=100, cost_cfg=cc)
    assert abs(s - (3 + 0 + 10 * 0.2)) < 1e-9  # = 5 bps


# ECE sanity
def test_ece_perfect_calibration():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    p = y.astype(float)  # perfectly calibrated (prob == outcome)
    assert compute_ece(y, p, n_bins=5) < 1e-9


def test_multiclass_learner_binary_labels():
    # 2-class data (the real-data case that crashed LightGBM with multiclass obj)
    from crypto.models.base_lgb import MultiClassLearner
    from crypto.schemas import ModelConfig
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 4))
    y = np.where(X[:, 0] > 0, 1, -1)            # only {-1, 1}, no neutral
    m = MultiClassLearner(ModelConfig()).fit(X, y)
    p_down, p_neu, p_up = m.predict_proba_df(X)
    assert np.allclose(p_neu, 0.0)              # missing neutral class -> zeros
    assert np.allclose(p_down + p_up, 1.0, atol=1e-6)


def test_multiclass_learner_single_class_fold():
    # degenerate fold: only one class -> constant model, no crash
    from crypto.models.base_lgb import MultiClassLearner
    from crypto.schemas import ModelConfig
    X = np.random.default_rng(1).normal(0, 1, (50, 4))
    y = np.full(50, -1)
    m = MultiClassLearner(ModelConfig()).fit(X, y)
    p_down, p_neu, p_up = m.predict_proba_df(X)
    assert np.allclose(p_down, 1.0) and np.allclose(p_up, 0.0)
