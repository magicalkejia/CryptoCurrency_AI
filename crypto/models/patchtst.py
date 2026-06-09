"""
crypto.models.patchtst
=========================
Multi-horizon temporal representation (v6 §7.4, audit #23):

  * input  : past LOOKBACK 1h bars, multivariate, right-aligned (no future).
  * output : forecast_{4h,12h,24h,3d} + embedding[D]  -> used as FEATURES in the
             Stage-1 fusion model (NOT trained on triple-barrier labels).
  * leakage: produced via walk-forward / purged OOF; horizon purging by keying
             label interval as [decision_time, decision_time + max_horizon).

Two backends, identical interface:
  * PatchTSTTorch  : real PatchTST (patch embedding + transformer) — needs torch.
  * TemporalFallback: StandardScaler + TruncatedSVD embedding + per-horizon Ridge
                      forecast — pure sklearn, always available.
`run_patchtst()` auto-selects torch if importable, else the fallback.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from crypto.cv.purged_kfold import purged_embargoed_splits, default_embargo_delta

HORIZONS_H = {"4h": 4, "12h": 12, "24h": 24, "3d": 72}   # in 1h bars

try:
    import torch  # noqa
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
def make_windows(bars_1h: pd.DataFrame, decision_times: pd.DatetimeIndex,
                 channels: List[str], lookback: int = 96):
    """
    Returns (X, target_dict, valid_decision_times).
    X: [n, lookback, C] standardized per-window; targets: future returns by horizon.
    Right-aligned: window ends at the last 1h bar <= decision_time.
    """
    bars = bars_1h.sort_index()
    close = bars["close"]
    idx = bars.index
    arr = bars[channels].to_numpy(dtype=float)

    Xs, tgts, dts = [], {h: [] for h in HORIZONS_H}, []
    max_h = max(HORIZONS_H.values())
    for dt in decision_times:
        loc = idx.searchsorted(dt, side="right") - 1   # last bar <= dt
        if loc < lookback or loc + max_h >= len(idx):
            continue
        w = arr[loc - lookback + 1: loc + 1]            # [lookback, C], ends at dt
        mu, sd = w.mean(0), w.std(0) + 1e-9
        Xs.append((w - mu) / sd)
        c0 = close.iloc[loc]
        for h, hh in HORIZONS_H.items():
            tgts[h].append(close.iloc[loc + hh] / c0 - 1.0)
        dts.append(dt)
    if not Xs:
        return None, None, pd.DatetimeIndex([])
    X = np.stack(Xs)
    tgts = {h: np.array(v) for h, v in tgts.items()}
    return X, tgts, pd.DatetimeIndex(dts)


# --------------------------------------------------------------------------- #
# sklearn fallback
# --------------------------------------------------------------------------- #
class TemporalFallback:
    def __init__(self, emb_dim: int = 8, seed: int = 42):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.linear_model import Ridge
        self.emb_dim = emb_dim
        self.svd = TruncatedSVD(n_components=emb_dim, random_state=seed)
        self.ridges = {h: Ridge(alpha=1.0) for h in HORIZONS_H}
        self._fit = False

    def fit(self, X, tgts):
        flat = X.reshape(len(X), -1)
        self.svd.fit(flat)
        emb = self.svd.transform(flat)
        for h in HORIZONS_H:
            self.ridges[h].fit(emb, tgts[h])
        self._fit = True
        return self

    def predict(self, X):
        flat = X.reshape(len(X), -1)
        emb = self.svd.transform(flat)
        fc = {f"patchtst_forecast_{h}": self.ridges[h].predict(emb) for h in HORIZONS_H}
        return fc, emb


# --------------------------------------------------------------------------- #
# torch PatchTST (used when torch present)
# --------------------------------------------------------------------------- #
if _HAS_TORCH:
    class _PatchTST(nn.Module):
        def __init__(self, n_ch, lookback, patch=16, stride=8, d_model=64,
                     n_heads=4, n_layers=2, emb_dim=8, n_horizons=4):
            super().__init__()
            self.patch, self.stride = patch, stride
            n_patches = (lookback - patch) // stride + 1
            self.proj = nn.Linear(patch, d_model)
            self.pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
            enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2,
                                             batch_first=True, dropout=0.1)
            self.encoder = nn.TransformerEncoder(enc, n_layers)
            self.n_ch = n_ch
            self.flat_dim = n_ch * n_patches * d_model
            self.emb_head = nn.Linear(self.flat_dim, emb_dim)
            self.fc_head = nn.Linear(emb_dim, n_horizons)

        def _patchify(self, x):  # x: [B, L, C]
            x = x.permute(0, 2, 1)                       # [B, C, L]
            patches = x.unfold(-1, self.patch, self.stride)  # [B, C, n_patches, patch]
            return patches

        def forward(self, x):
            B = x.shape[0]
            p = self._patchify(x)                        # [B,C,P,patch]
            C, P = p.shape[1], p.shape[2]
            p = p.reshape(B * C, P, self.patch)
            h = self.proj(p) + self.pos
            h = self.encoder(h)                          # [B*C, P, d_model]
            h = h.reshape(B, C * P * h.shape[-1])
            emb = self.emb_head(h)
            return self.fc_head(emb), emb

    class PatchTSTTorch:
        def __init__(self, lookback, n_ch, emb_dim=8, epochs=30, lr=1e-3, seed=42):
            torch.manual_seed(seed)
            self.lookback, self.n_ch, self.emb_dim = lookback, n_ch, emb_dim
            self.epochs, self.lr = epochs, lr
            self.model = None

        def fit(self, X, tgts):
            self.model = _PatchTST(self.n_ch, self.lookback, emb_dim=self.emb_dim,
                                   n_horizons=len(HORIZONS_H))
            Y = np.stack([tgts[h] for h in HORIZONS_H], axis=1)
            xb = torch.tensor(X, dtype=torch.float32)
            yb = torch.tensor(Y, dtype=torch.float32)
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            lossf = nn.MSELoss()
            self.model.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                pred, _ = self.model(xb)
                loss = lossf(pred, yb)
                loss.backward()
                opt.step()
            return self

        def predict(self, X):
            self.model.eval()
            with torch.no_grad():
                pred, emb = self.model(torch.tensor(X, dtype=torch.float32))
            pred, emb = pred.numpy(), emb.numpy()
            fc = {f"patchtst_forecast_{h}": pred[:, i] for i, h in enumerate(HORIZONS_H)}
            return fc, emb


def run_patchtst(
    bars_1h: pd.DataFrame,
    decision_times: pd.DatetimeIndex,
    symbol: str,
    fcfg,
    channels: Optional[List[str]] = None,
    lookback: int = 96,
    emb_dim: int = 8,
    prefer: str = "auto",
) -> pd.DataFrame:
    """
    Walk-forward OOF PatchTST features for one symbol.
    Returns DataFrame[symbol, decision_time, patchtst_forecast_*, patchtst_emb_*].
    """
    channels = channels or [c for c in ["close", "volume", "taker_buy_vol", "net_taker_vol"]
                            if c in bars_1h.columns]
    # right-aligned channel transforms (returns) to stabilize
    feat = bars_1h.copy()
    feat["close"] = feat["close"].pct_change().fillna(0)
    if "volume" in feat:
        feat["volume"] = np.log1p(feat["volume"]).diff().fillna(0)
    X, tgts, dts = make_windows(feat, decision_times, channels, lookback)
    if X is None:
        return pd.DataFrame(columns=["symbol", "decision_time"])

    max_h = max(HORIZONS_H.values())
    t0 = pd.Series(dts)
    t1 = pd.Series(dts) + pd.Timedelta(hours=max_h)     # horizon purging
    splits = purged_embargoed_splits(pd.Series(dts), t0, t1,
                                     n_splits=fcfg.cv.n_splits,
                                     embargo_delta=default_embargo_delta(fcfg.cv),
                                     symbol=pd.Series([symbol] * len(dts)))

    n = len(dts)
    fc_oof = {f"patchtst_forecast_{h}": np.full(n, np.nan) for h in HORIZONS_H}
    emb_oof = np.full((n, emb_dim), np.nan)

    use_torch = _HAS_TORCH and prefer in ("auto", "torch")
    for fold in splits:
        tr, te = fold.train_idx, fold.test_idx
        if len(tr) < 50 or len(te) == 0:
            continue
        if use_torch:
            m = PatchTSTTorch(lookback, X.shape[2], emb_dim=emb_dim,
                              seed=fcfg.model.random_seed)
        else:
            m = TemporalFallback(emb_dim=emb_dim, seed=fcfg.model.random_seed)
        m.fit(X[tr], {h: tgts[h][tr] for h in HORIZONS_H})
        fc, emb = m.predict(X[te])
        for h in HORIZONS_H:
            fc_oof[f"patchtst_forecast_{h}"][te] = fc[f"patchtst_forecast_{h}"]
        emb_oof[te] = emb

    out = pd.DataFrame({"symbol": symbol, "decision_time": dts})
    for k, v in fc_oof.items():
        out[k] = v
    for j in range(emb_dim):
        out[f"patchtst_emb_{j}"] = emb_oof[:, j]
    return out
