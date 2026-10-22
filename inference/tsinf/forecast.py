"""과제 3. 60분 예측 — 데이터셋 만들기, 모델(GRU/LSTM/TCN), 학습·평가.

입력 (공간 하나, 시점 t 기준)
- 과거 L칸(기본 24칸 = 2시간): CO₂·온도·습도(공간별 학습 구간 평균·표준편차로 정규화) + 시간대·요일 + 지금 일정 인원
- 정적: 어느 공간인지(원-핫)
- 미래에 미리 아는 정보: 앞으로 60분(12칸)의 일정 인원 (수업 시간표·회의실 예약)
출력: 5분 간격 12칸 뒤까지 CO₂·온도의 '지금 값 대비 변화량' (정규화) → 지금 값에 더해 원래 단위로 복원
  (변화량을 맞히게 하면 '현재값 유지'가 기본값이 되어 학습이 안정적이고, 기준선보다 나빠지기 어렵다)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
try:                                    # 학습·평가에만 필요. 서비스(보드)는 ONNX Runtime 만 쓰므로 torch 없이도 import 되게
    import torch
    from torch import nn
    _Module = nn.Module
except ImportError:                     # pragma: no cover
    torch = None
    nn = None
    _Module = object

H = 12                      # 예측 칸 수 (5분 × 12 = 60분)
FEAT = ("co2", "temperature", "humidity")
TARGETS = ("co2", "temperature")


@dataclass
class DatasetSpec:
    spaces: List[str]
    lookback: int = 24
    use_schedule: bool = True
    stats: Optional[pd.DataFrame] = None           # index (space, metric), cols mean/std
    capacity: Dict[str, float] = field(default_factory=dict)
    space_onehot: bool = True                      # False: 공간 구분 없이(처음 보는 실제 공간에도 쓰는 범용 모델)


def schedule_grid(schedule: Optional[pd.DataFrame], index: pd.DatetimeIndex, spaces: Sequence[str],
                  capacity: Dict[str, float]) -> pd.DataFrame:
    """5분 격자의 '계획 인원 / 정원' (0~1). 일정이 없는 공간은 0."""
    g = pd.DataFrame(0.0, index=index, columns=list(spaces))
    if schedule is None or schedule.empty:
        return g
    for r in schedule.itertuples():
        if r.space not in g.columns:
            continue
        s = pd.Timestamp(r.start).tz_convert(index.tz)
        e = pd.Timestamp(r.end).tz_convert(index.tz)
        mask = (index >= s.floor("5min")) & (index < e)
        g.loc[mask, r.space] = np.maximum(g.loc[mask, r.space], r.planned_people / capacity.get(r.space, 30))
    return g


def time_features(index: pd.DatetimeIndex) -> np.ndarray:
    h = index.hour + index.minute / 60.0
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * index.dayofweek / 7), np.cos(2 * np.pi * index.dayofweek / 7),
                     (index.dayofweek >= 5).astype(float)], axis=1).astype(np.float32)


def build_samples(grid: pd.DataFrame, sched: pd.DataFrame, spec: DatasetSpec, times: pd.DatetimeIndex,
                  exclude: Optional[Dict[str, List[tuple]]] = None):
    """times 에 속한 기준 시점 t 마다 샘플을 만든다. 결측이 있는 창은 건너뛴다.

    반환: dict(x_seq, x_static, x_future, y, base, space, time)
    """
    L = spec.lookback
    tf = time_features(grid.index)
    pos = pd.Series(np.arange(len(grid)), index=grid.index)
    out = {k: [] for k in ("x_seq", "x_static", "x_future", "y", "base", "space", "time")}
    for si, sp in enumerate(spec.spaces):
        raw = np.stack([grid[(sp, m)].to_numpy(np.float32) for m in FEAT], axis=1)       # [T,3]
        mu = np.array([spec.stats.loc[(sp, m), "mean"] for m in FEAT], np.float32)
        sd = np.array([spec.stats.loc[(sp, m), "std"] for m in FEAT], np.float32)
        z = (raw - mu) / sd
        sc = sched[sp].to_numpy(np.float32) if spec.use_schedule else np.zeros(len(grid), np.float32)
        onehot = np.eye(len(spec.spaces), dtype=np.float32)[si] if spec.space_onehot else np.ones(1, np.float32)
        bad = np.isnan(raw).any(axis=1)
        bad_cum = np.concatenate([[0], np.cumsum(bad)])
        excl = np.zeros(len(grid), bool)
        for (s, e) in (exclude or {}).get(sp, []):
            excl |= (grid.index >= s - pd.Timedelta(minutes=5 * (L + 1))) & (grid.index <= e + pd.Timedelta(minutes=5 * H))
        for t in times:
            i = pos.get(t)
            if i is None or i < L - 1 or i + H >= len(grid) or excl[i]:
                continue
            if bad_cum[i + H + 1] - bad_cum[i - L + 1] > 0:
                continue
            seq = np.concatenate([z[i - L + 1:i + 1], tf[i - L + 1:i + 1], sc[i - L + 1:i + 1, None]], axis=1)
            fut = sc[i + 1:i + H + 1]
            y = np.stack([(raw[i + 1:i + H + 1, FEAT.index(m)] - raw[i, FEAT.index(m)]) / sd[FEAT.index(m)]
                          for m in TARGETS], axis=1)                                      # [H,2]
            out["x_seq"].append(seq); out["x_static"].append(onehot); out["x_future"].append(fut)
            out["y"].append(y); out["base"].append(raw[i, [FEAT.index(m) for m in TARGETS]])
            out["space"].append(sp); out["time"].append(t)
    if not out["y"]:
        raise ValueError("샘플이 없습니다 (결측이 너무 많거나 기간이 짧음)")
    res = {k: np.stack(v) if k not in ("space", "time") else np.array(v) for k, v in out.items()}
    res["sd"] = np.array([[spec.stats.loc[(sp, m), "std"] for m in TARGETS] for sp in res["space"]], np.float32)
    return res


# ---------------------------------------------------------------- 모델
class Head(_Module):
    def __init__(self, enc_dim, n_static, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(enc_dim + n_static + H, hidden), nn.ReLU(),
                                 nn.Linear(hidden, H * len(TARGETS)))

    def forward(self, h, x_static, x_future):
        return self.mlp(torch.cat([h, x_static, x_future], dim=1)).view(-1, H, len(TARGETS))


class RNNForecaster(_Module):
    def __init__(self, n_in, n_static, kind="gru", hidden=64, layers=2):
        super().__init__()
        rnn = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn(n_in, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.head = Head(hidden, n_static)

    def forward(self, x_seq, x_static, x_future):
        out, _ = self.rnn(x_seq)
        return self.head(out[:, -1], x_static, x_future)


class TCNForecaster(_Module):
    def __init__(self, n_in, n_static, hidden=64, levels=4, k=3):
        super().__init__()
        layers, ch = [], n_in
        for i in range(levels):
            d = 2 ** i
            layers += [nn.ConstantPad1d(((k - 1) * d, 0), 0.0), nn.Conv1d(ch, hidden, k, dilation=d), nn.ReLU()]
            ch = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = Head(hidden, n_static)

    def forward(self, x_seq, x_static, x_future):
        h = self.tcn(x_seq.transpose(1, 2))[:, :, -1]
        return self.head(h, x_static, x_future)


def make_model(kind: str, n_in: int, n_static: int) -> nn.Module:
    return TCNForecaster(n_in, n_static) if kind == "tcn" else RNNForecaster(n_in, n_static, kind)


def _tensors(d, idx=None):
    sel = slice(None) if idx is None else idx
    return (torch.from_numpy(d["x_seq"][sel]), torch.from_numpy(d["x_static"][sel]),
            torch.from_numpy(d["x_future"][sel]))


def predict(model, d, batch=2048) -> np.ndarray:
    """원래 단위 예측 [N, H, 2]"""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(d["y"]), batch):
            xs, xst, xf = _tensors(d, slice(i, i + batch))
            outs.append(model(xs, xst, xf).numpy())
    dz = np.concatenate(outs)
    return d["base"][:, None, :] + dz * d["sd"][:, None, :]


def mae_by_h(pred, d) -> np.ndarray:
    true = d["base"][:, None, :] + d["y"] * d["sd"][:, None, :]
    return np.abs(pred - true).mean(axis=0)                     # [H,2]


def train(kind, tr, va, epochs=40, lr=1e-3, batch=256, patience=6, seed=0, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model(kind, tr["x_seq"].shape[2], tr["x_static"].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xs, xst, xf = _tensors(tr)
    y = torch.from_numpy(tr["y"])
    best, best_state, bad = np.inf, None, 0
    n = len(y)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            loss = nn.functional.smooth_l1_loss(model(xs[b], xst[b], xf[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        v = mae_by_h(predict(model, va), va)[:, 0].mean()       # 검증: CO₂ 평균 MAE (ppm)
        if verbose:
            print(f"  [{kind}] epoch {ep + 1:2d} val CO2 MAE {v:.2f}")
        if v < best - 1e-3:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


def baseline_preds(grid: pd.DataFrame, d) -> Dict[str, np.ndarray]:
    """같은 샘플(같은 시점)에서 기준선 3종 예측 [N,H,2]"""
    pos = pd.Series(np.arange(len(grid)), index=grid.index)
    out = {"현재값 유지": np.repeat(d["base"][:, None, :], H, axis=1)}
    m1 = np.zeros((len(d["y"]), H, 2), np.float32)
    yd = np.full((len(d["y"]), H, 2), np.nan, np.float32)
    for n, (sp, t) in enumerate(zip(d["space"], d["time"])):
        i = pos[t]
        for j, m in enumerate(TARGETS):
            s = grid[(sp, m)].to_numpy()
            m1[n, :, j] = np.nanmean(s[max(0, i - 11):i + 1])
            if i + H - 288 >= 0:
                yd[n, :, j] = s[i + 1 - 288:i + H + 1 - 288]
    out["최근 1시간 평균"] = m1
    out["어제 같은 시각"] = yd
    return out
