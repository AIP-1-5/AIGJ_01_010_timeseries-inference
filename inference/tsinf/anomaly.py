"""과제 4. 이상 탐지 — 규칙 기반 + 모델 기반(오토인코더) + 예측 오차 기반 + 영점 틀어짐 전용.

모든 탐지기는 (공간, 시각) 별 '경보 여부'와 '원인 항목'을 낸다. 서비스(과제 8)에서도 그대로 쓴다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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

FEAT = ("co2", "temperature", "humidity")


# ------------------------------------------------------------------ 규칙
STUCK_MIN = {"co2": 60, "temperature": 120, "humidity": 120}


def rule_alarms(raw: pd.DataFrame, spike_thr=None, stuck_min=None) -> pd.DataFrame:
    """1분 원본으로 규칙 경보. raw: time 인덱스, 열 = 항목 (한 공간).

    - 튄 값: 직전 10분 중앙값과 차이가 임계값 이상 **이고 바로 다음 값이 제자리로 돌아옴** (CO₂ 300ppm, 온도 3℃, 습도 15%)
      (쉬는 시간 창문 개방처럼 진짜로 빠르게 바뀌는 변화는 다음 값도 그대로라 경보가 아님, 대신 1분 늦게 판단)
    - 값 고정: 같은 값이 CO₂ 60분 · 온도/습도 120분 이상 (실데이터 확인: 냉방 중 0.1℃ 해상도 온도는 1시간쯤 그대로일 수 있음)
    반환: time × [rule_spike_<m>, rule_stuck_<m>] bool
    """
    spike_thr = spike_thr or {"co2": 300.0, "temperature": 3.0, "humidity": 15.0}
    out = pd.DataFrame(index=raw.index)
    for m in raw.columns:
        x = raw[m]
        med = x.shift(1).rolling(10, min_periods=3).median()
        thr = spike_thr.get(m, np.inf)
        out[f"rule_spike_{m}"] = ((x - med).abs() >= thr) & ((x.shift(-1) - med).abs() < thr / 2)
        run_id = (x != x.shift()).cumsum()
        run_len = x.groupby(run_id).cumcount() + 1
        lim = (stuck_min or STUCK_MIN).get(m, 60) if isinstance(stuck_min or STUCK_MIN, dict) else stuck_min
        out[f"rule_stuck_{m}"] = (run_len >= lim) & x.notna()
    return out.fillna(False)


def drift_alarm(grid_space: pd.DataFrame, occupied_plan: pd.Series, c_out: float = 420.0,
                min_empty_h: float = 1.5, rise_ppm: float = 25.0) -> pd.Series:
    """영점 틀어짐 전용 (심화): 비어 있는 방의 CO₂ 는 환기로 외기(≈420ppm) 쪽으로 내려가야 한다.
    밤(20~07시)에 일정상 1.5시간 이상 비어 있는데 CO₂ 가 1시간 동안 25ppm 이상 '올라가면' 센서가 틀어지는 것으로 본다."""
    co2 = grid_space["co2"]
    hour = co2.index.hour
    night = (hour >= 20) | (hour < 7)          # 일정 밖 사용(라운지·예약 없는 회의)을 피하려고 밤 시간만
    empty = (occupied_plan.rolling(int(min_empty_h * 12), min_periods=1).max() == 0) & night
    rising = (co2 - co2.shift(12)) >= rise_ppm
    above = co2 > c_out + 80
    return (empty & rising & above).fillna(False)


# ------------------------------------------------------------------ 시간대 기대값(프로필)
def profile_fit(gs: pd.DataFrame, train_mask) -> dict:
    """학습 구간에서 (요일, 하루 중 5분 칸)별 중앙값과 퍼짐(MAD)을 구한다."""
    key = _profile_key(gs.index)
    tr = gs[train_mask]
    med = tr.groupby(key[train_mask]).median()
    mad = (tr - med.reindex(key[train_mask]).values).abs().groupby(key[train_mask]).median()
    floor = {"co2": 60.0, "temperature": 0.3, "humidity": 2.0}    # 학습 주가 3주뿐이라 퍼짐을 너무 작게 잡지 않도록
    for c in mad.columns:
        mad[c] = mad[c].clip(lower=floor.get(c, 0.1))
    return {"med": med, "mad": mad}


def _profile_key(index: pd.DatetimeIndex) -> np.ndarray:
    # 요일마다 수업 시간표가 달라(예: 강의실2 는 월·수·금 오전, 화·목 오후) 요일별로 따로 본다
    return (index.dayofweek * 1000 + index.hour * 12 + index.minute // 5).to_numpy()


def profile_score(gs: pd.DataFrame, prof: dict) -> pd.DataFrame:
    """(실제 − 시간대 기대값) / 퍼짐. 양수 = 평소보다 높음"""
    key = _profile_key(gs.index)
    med = prof["med"].reindex(key).set_axis(gs.index)
    mad = prof["mad"].reindex(key).set_axis(gs.index)
    return (gs[med.columns] - med) / (1.4826 * mad)


# ------------------------------------------------------------------ 오토인코더
class AE(_Module):
    def __init__(self, n_in, n_cond, bottleneck=6):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in + n_cond, 64), nn.ReLU(), nn.Linear(64, bottleneck))
        self.dec = nn.Sequential(nn.Linear(bottleneck + n_cond, 64), nn.ReLU(), nn.Linear(64, n_in))

    def forward(self, x, c):
        return self.dec(torch.cat([self.enc(torch.cat([x, c], 1)), c], 1))


def ae_windows(z: np.ndarray, cond: np.ndarray, W: int):
    """z [T,3] 정규화 값, cond [T,k] 시간·일정 → 끝 시점 t 마다 [W*3] 창 (결측 창은 제외)"""
    T = len(z)
    idx = [t for t in range(W - 1, T) if not np.isnan(z[t - W + 1:t + 1]).any()]
    X = np.stack([z[t - W + 1:t + 1].reshape(-1) for t in idx]).astype(np.float32) if idx else np.zeros((0, W * 3), np.float32)
    C = cond[idx].astype(np.float32) if idx else np.zeros((0, cond.shape[1]), np.float32)
    return X, C, np.array(idx)


def train_ae(X, C, epochs=30, seed=0):
    torch.manual_seed(seed)
    m = AE(X.shape[1], C.shape[1])
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xt, Ct = torch.from_numpy(X), torch.from_numpy(C)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            b = perm[i:i + 512]
            loss = ((m(Xt[b], Ct[b]) - Xt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def ae_scores(m, X, C, W) -> Tuple[np.ndarray, np.ndarray]:
    """창 전체 재구성 오차(점수)와 항목별 오차(원인 지목용)"""
    with torch.no_grad():
        r = m(torch.from_numpy(X), torch.from_numpy(C)).numpy()
    e = (r - X) ** 2
    per_metric = e.reshape(len(X), W, 3).mean(axis=1)
    return e.mean(axis=1), per_metric


# ------------------------------------------------------------------ 이벤트 평가
def to_events(alarm: pd.Series, merge_gap: str = "30min", min_len: int = 1) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """연속 경보를 하나의 '경보 이벤트'로 묶는다 (같은 고장에 매분 울리지 않도록)."""
    t = alarm[alarm].index
    if len(t) == 0:
        return []
    ev, s, e = [], t[0], t[0]
    gap = pd.Timedelta(merge_gap)
    for x in t[1:]:
        if x - e <= gap:
            e = x
        else:
            ev.append((s, e)); s = e = x
    ev.append((s, e))
    return ev


def evaluate(events: Dict[str, List[Tuple]], faults: pd.DataFrame, t0, t1, margin="30min") -> dict:
    """events: {space: [(start,end),...]}. 고장 하나라도 겹치는 경보가 있으면 '탐지'.
    고장과 겹치지 않는 경보 이벤트 = 헛경보."""
    mg = pd.Timedelta(margin)
    det, delays, per_type = [], [], {}
    used = {sp: [False] * len(ev) for sp, ev in events.items()}
    for f in faults.itertuples():
        hit, first = False, None
        for k, (s, e) in enumerate(events.get(f.space, [])):
            if s <= f.end + mg and e >= f.start:
                used[f.space][k] = True
                hit = True
                first = s if first is None else min(first, s)
        det.append(hit)
        if hit:
            delays.append(max(0.0, (first - f.start).total_seconds() / 60))
        pt = per_type.setdefault(f.type, [0, 0])
        pt[0] += hit; pt[1] += 1
    days = (t1 - t0 + pd.Timedelta(minutes=5)).total_seconds() / 86400    # 마지막 칸(23:55)까지 포함 → 7일
    fa = sum(1 for sp, u in used.items() for x in u if not x)
    return {"detected": int(sum(det)), "total": len(det), "rate": float(np.mean(det)) if det else np.nan,
            "delay_median_min": float(np.median(delays)) if delays else np.nan,
            "false_alarms": fa, "fa_per_day": fa / days if days else np.nan, "per_type": per_type}
