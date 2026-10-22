"""전처리 (과제 2 정제 규칙). 학습(과제 3~5)과 상시 서비스(과제 8)가 **똑같이** 이 모듈을 쓴다.

정제 규칙
1. 중복 제거: 같은 기기·항목·값이 5초 안에 다시 받아진 행은 재전송(게이트웨이 2대)으로 보고 버린다.
   (수집기 2026-10-21 버전부터는 fCnt 로 저장 단계에서 거르지만, 그 전 데이터용)
2. 시각 보정: 센서 시각이 받은 시각보다 30초 넘게 미래면(게이트웨이 시계 약 5분 빠름) 받은 시각을 쓴다.
3. 범위 밖 값 제거: CO₂ 300~5000ppm, 온도 -10~50℃, 습도 0~100% 밖은 물리적으로 불가능한 값(0ppm 등)으로 본다.
4. 분석 간격 5분: 기기별로 5분 평균 → 같은 공간의 기기들을 평균해 공간 대표값.
   (센서 주기가 1·2·10분으로 제각각이라 1분 칸은 비는 곳이 많고, 예측도 5분 간격이므로)
5. 결측: 15분(3칸) 이하 짧은 결측은 선형 보간, 그보다 긴 결측은 비워 둔다(학습 창에서 제외).
6. 시계 미설정: 보드 재부팅 직후 NTP 동기화 전(약 1분) 받은 행은 1970년 시각 → 센서 시각으로 대체
7. 위치 보정: 태그는 '실습실'인데 설치 위치(point)가 '복도'인 센서는 복도 공기(18℃·83%)를 재므로
   실습실 평균에서 빼고 별도 공간 '복도'로 둔다.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

METRICS = ("co2", "temperature", "humidity")
VALID_RANGE = {"co2": (300.0, 5000.0), "temperature": (-10.0, 50.0), "humidity": (0.0, 100.0)}
FREQ = "5min"
SHORT_GAP_STEPS = 3          # 15분
DUP_WINDOW_SEC = 5.0
MAX_FUTURE_SEC = 30.0
TZ = "Asia/Seoul"
POINT_TO_SPACE = {"복도": "복도"}   # 설치 위치가 이 이름이면 공간을 바꿔 기록


@dataclass
class CleanStats:
    raw: int
    duplicates: int
    clock_fixed: int
    out_of_range: Dict[str, int]
    kept: int


def load_readings(db_path: str, since: Optional[float] = None, spaces=None) -> pd.DataFrame:
    q = ("SELECT space, metric, device, point, value, ts, sensor_ts, recv_ts FROM readings "
         "WHERE metric IN ('co2','temperature','humidity')")
    params: list = []
    if since is not None:
        q += " AND ts >= ?"
        params.append(since)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        df = pd.read_sql(q, con, params=params)
    finally:
        con.close()
    if spaces:
        df = df[df.space.isin(spaces)]
    return df


def clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, CleanStats]:
    raw = len(df)
    df = df.sort_values(["device", "metric", "recv_ts"]).copy()
    for point, space in POINT_TO_SPACE.items():
        df.loc[df["point"] == point, "space"] = space

    # 1) 재전송 중복
    prev_val = df.groupby(["device", "metric"])["value"].shift()
    prev_recv = df.groupby(["device", "metric"])["recv_ts"].shift()
    dup = (df["value"] == prev_val) & ((df["recv_ts"] - prev_recv) < DUP_WINDOW_SEC)
    n_dup = int(dup.sum())
    df = df[~dup]

    # 2) 시계 보정
    future = df["sensor_ts"].notna() & ((df["sensor_ts"] - df["recv_ts"]) > MAX_FUTURE_SEC)
    df.loc[future, "ts"] = df.loc[future, "recv_ts"]
    n_clock = int(future.sum())

    # 3) 범위 밖 값
    oor: Dict[str, int] = {}
    keep = pd.Series(True, index=df.index)
    for m, (lo, hi) in VALID_RANGE.items():
        bad = (df["metric"] == m) & ((df["value"] < lo) | (df["value"] > hi))
        oor[m] = int(bad.sum())
        keep &= ~bad
    df = df[keep]

    # 재부팅 직후 보드 시계가 1970년으로 찍힌 행은 받은 시각을 쓸 수 없다 → 센서 시각으로 대체, 그것도 없으면 버림
    bad_clock = df["ts"] < 1_577_836_800
    df.loc[bad_clock & df["sensor_ts"].notna(), "ts"] = df.loc[bad_clock & df["sensor_ts"].notna(), "sensor_ts"]
    df = df[df["ts"] >= 1_577_836_800]
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(TZ)
    return df, CleanStats(raw=raw, duplicates=n_dup, clock_fixed=n_clock, out_of_range=oor, kept=len(df))


def to_grid(df: pd.DataFrame, interpolate: bool = True) -> pd.DataFrame:
    """공간 × 항목 5분 격자. 열은 MultiIndex (space, metric).

    기기마다 보고 주기가 달라(1·2·10분) 5분 칸마다 들어오는 기기 조합이 바뀌면, 기기별 치우침 때문에
    공간 평균이 톱니처럼 오르내린다. 그래서 **기기별로 먼저 짧은 결측을 메운 뒤** 공간 평균을 낸다.
    """
    dev = (df.set_index("time")
             .groupby(["space", "device", "metric"])["value"]
             .resample(FREQ).mean()
             .unstack(["space", "device", "metric"])
             .sort_index())
    full = pd.date_range(dev.index.min().floor(FREQ), dev.index.max().floor(FREQ), freq=FREQ, tz=TZ)
    dev = dev.reindex(full)
    if interpolate:
        # 10분 주기 센서도 메워지도록 기기 단위는 15분(3칸)까지
        dev = dev.interpolate(method="time", limit=SHORT_GAP_STEPS, limit_area="inside")
    sp = dev.T.groupby(level=["space", "metric"]).mean().T
    if interpolate:
        sp = sp.interpolate(method="time", limit=SHORT_GAP_STEPS, limit_area="inside")
    sp.index.name = "time"
    return sp.sort_index(axis=1)


def split_time(index: pd.DatetimeIndex, train=0.7, val=0.15) -> Dict[str, pd.DatetimeIndex]:
    """시간순으로 자른다 (무작위로 섞으면 미래 정보가 학습에 새어 들어감)."""
    n = len(index)
    a, b = int(n * train), int(n * (train + val))
    return {"train": index[:a], "val": index[a:b], "test": index[b:]}


def norm_stats(grid: pd.DataFrame, train_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """정규화 평균·표준편차는 **학습 기간 데이터로만** 계산한다."""
    tr = grid.loc[train_idx]
    return pd.DataFrame({"mean": tr.mean(), "std": tr.std().replace(0, 1.0)})


def add_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    h = index.hour + index.minute / 60.0
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
        "dow_sin": np.sin(2 * np.pi * index.dayofweek / 7), "dow_cos": np.cos(2 * np.pi * index.dayofweek / 7),
        "weekend": (index.dayofweek >= 5).astype(float),
    }, index=index)
