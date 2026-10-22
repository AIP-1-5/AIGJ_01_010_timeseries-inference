"""합성 데이터 생성기 (제공 키트가 없을 때 README 설명대로 직접 만든 것).

    .venv/bin/python -m tsinf.synth --out data/synth --weeks 5 --seed 7

만드는 것 (data/synth/)
- sensors.db     : 수집기와 같은 스키마(readings)의 1분 센서 데이터 → tsinf.prep 로 그대로 읽힘
- schedule.csv   : 일정(수업 시간표·회의실 예약). 미래까지 미리 알 수 있는 '계획' 정보
- occupancy.csv  : 실제 재실 인원(정답). 일정과 다를 수 있음(결석·노쇼·연장·예약 없는 사용)
- observed.csv   : '직접 관찰' 기록이 있는 시간대(과제 5 정답 검증용) = 무작위 6일의 실제 사용 여부
- faults.csv     : 마지막 1주에 심은 고장 5종의 정답 (종류·공간·항목·시작·끝)
- truth_state.csv: 냉방·환기 실제 상태(분석용)

물리 모델 (1분 단위)
- CO₂: dC/dt = 사람 수 × 호흡 발생량 / 부피 − 환기율 × (C − 외기)
  호흡 0.0052 L/s/인(≈ 0.31 L/min), 외기 약 420ppm, 환기율(ACH) 운영 시간 2~3회/h, 야간 0.3회/h, 쉬는 시간 창문 개방
- 온도: 냉방 가동 시 설정 온도로 수렴(시정수 15분) + 사람 열, 꺼지면 건물 온도(일교차)로 서서히 수렴
- 습도: 일교차 + 사람 수에 따라 약간 상승
- 센서: 측정 잡음, 반올림, 게이트웨이 끊김(결측), 0ppm 같은 깨진 값
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

TZ = "Asia/Seoul"
C_OUT = 420.0
GEN_M3_PER_MIN = 0.0052 * 60 / 1000          # 1인 1분 CO₂ 발생량 (m³)


@dataclass
class Space:
    name: str
    volume: float      # m³
    capacity: int
    ach_on: float      # 운영 시간 환기 횟수/h
    setpoint: float    # 냉방 설정 온도


SPACES = [
    Space("강의실1", 210, 30, 2.5, 24.0),
    Space("강의실2", 160, 22, 2.5, 24.0),
    Space("회의실", 60, 8, 1.5, 24.0),
    Space("라운지", 140, 15, 2.0, 25.0),
]


def _minutes(start: pd.Timestamp, weeks: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=weeks * 7 * 1440, freq="1min", tz=TZ)


def make_schedule(days: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for d in days:
        if d.dayofweek >= 5:
            continue
        # 강의실1: 오전 3교시(9~12) + 오후 5교시(13~18), 50분 수업 + 10분 쉬는 시간
        # 강의실2: 요일에 따라 오전 또는 오후만
        for sp, blocks in (("강의실1", [(9, 12), (13, 18)]),
                           ("강의실2", [(9, 12)] if d.dayofweek in (0, 2, 4) else [(13, 17)])):
            cap = next(s.capacity for s in SPACES if s.name == sp)
            planned = int(round(cap * rng.uniform(0.75, 0.95)))
            for h0, h1 in blocks:
                for h in range(h0, h1):
                    rows.append((sp, d + pd.Timedelta(hours=h), d + pd.Timedelta(hours=h, minutes=50), "수업", planned))
        # 회의실 예약: 하루 0~4건, 30~120분, 2~8명
        t = d + pd.Timedelta(hours=9)
        for _ in range(rng.integers(0, 5)):
            t = t + pd.Timedelta(minutes=int(rng.integers(0, 120)))
            dur = int(rng.choice([30, 60, 60, 90, 120]))
            end = t + pd.Timedelta(minutes=dur)
            if end > d + pd.Timedelta(hours=19):
                break
            rows.append(("회의실", t, end, "예약", int(rng.integers(2, 9))))
            t = end + pd.Timedelta(minutes=10)
    return pd.DataFrame(rows, columns=["space", "start", "end", "kind", "planned_people"])


def make_occupancy(idx: pd.DatetimeIndex, sched: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    occ = pd.DataFrame(0.0, index=idx, columns=[s.name for s in SPACES])
    pos = {t: i for i, t in enumerate(idx)}
    n = len(idx)

    def add(space, t0, t1, people):
        i0 = pos.get(t0.floor("1min"), None)
        i1 = pos.get(t1.floor("1min"), None)
        if i0 is None:
            return
        i1 = n if i1 is None else i1
        occ.iloc[i0:i1, occ.columns.get_loc(space)] = np.maximum(occ.iloc[i0:i1, occ.columns.get_loc(space)], people)

    for r in sched.itertuples():
        if r.kind == "수업":
            people = max(0, int(round(r.planned_people * rng.normal(0.9, 0.08))))     # 결석
            late = int(rng.integers(0, 6))
            ext = int(rng.integers(0, 8)) if rng.random() < 0.2 else 0                 # 연장
            add(r.space, r.start + pd.Timedelta(minutes=late), r.end + pd.Timedelta(minutes=ext), people)
            # 쉬는 시간에도 일부는 남아 있음
            add(r.space, r.end + pd.Timedelta(minutes=ext), r.end + pd.Timedelta(minutes=10),
                int(people * rng.uniform(0.2, 0.5)))
        else:
            if rng.random() < 0.15:                                                    # 노쇼
                continue
            ext = int(rng.integers(5, 30)) if rng.random() < 0.15 else 0
            add(r.space, r.start + pd.Timedelta(minutes=int(rng.integers(0, 6))),
                r.end + pd.Timedelta(minutes=ext), max(1, int(r.planned_people * rng.uniform(0.6, 1.0))))

    for d in pd.date_range(idx[0].normalize(), idx[-1].normalize(), freq="D"):
        if d.dayofweek >= 5:
            continue
        # 회의실 예약 없는 사용 (하루 0~1건)
        if rng.random() < 0.35:
            t0 = d + pd.Timedelta(hours=int(rng.integers(10, 17)), minutes=int(rng.integers(0, 60)))
            add("회의실", t0, t0 + pd.Timedelta(minutes=int(rng.integers(20, 60))), int(rng.integers(2, 5)))
        # 라운지: 점심·쉬는 시간 위주
        for h, m, dur, lo, hi in ((12, 0, 60, 6, 15), (9, 50, 10, 1, 6), (10, 50, 10, 1, 6), (14, 50, 10, 1, 6),
                                  (15, 50, 10, 1, 6), (16, 50, 10, 1, 6)):
            if rng.random() < 0.9:
                t0 = d + pd.Timedelta(hours=h, minutes=m + int(rng.integers(-3, 4)))
                add("라운지", t0, t0 + pd.Timedelta(minutes=dur + int(rng.integers(-5, 10))), int(rng.integers(lo, hi)))
    return occ


def simulate(idx, occ, rng, faults_hvac=None, faults_vent=None):
    """faults_hvac/vent: {space: [(i0, i1), ...]} 냉방 정지 / 환기 정지 구간 (분 인덱스)."""
    n = len(idx)
    minute_of_day = idx.hour * 60 + idx.minute
    weekday = idx.dayofweek < 5
    op = weekday & (minute_of_day >= 8 * 60 + 30) & (minute_of_day < 18 * 60 + 30)      # 냉난방·환기 운영
    # 건물 기저 온도: 일교차 (새벽 19℃, 오후 22℃) + 날마다 조금씩 다름
    day_off = np.repeat(rng.normal(0, 0.6, n // 1440 + 1), 1440)[:n]
    t_base = 20.5 + 1.5 * np.sin(2 * np.pi * (minute_of_day - 9 * 60) / 1440) + day_off
    c_out = C_OUT + 15 * np.sin(2 * np.pi * (minute_of_day - 6 * 60) / 1440) + rng.normal(0, 3, n)

    out = {}
    states = {}
    for s in SPACES:
        N = occ[s.name].to_numpy()
        hvac = op.copy()
        vent_ach = np.where(op, s.ach_on, 0.3)
        # 쉬는 시간(매시 50~60분) 강의실은 가끔 창문 개방
        if s.name.startswith("강의실"):
            brk = weekday & (idx.minute >= 50) & (idx.hour >= 9) & (idx.hour < 18)
            day_open = np.repeat(rng.random(n // 1440 + 1) < 0.5, 1440)[:n]
            vent_ach = np.where(brk & day_open, 6.0, vent_ach)
        for (i0, i1) in (faults_hvac or {}).get(s.name, []):
            hvac[i0:i1] = False
        vent_scale = np.ones(n)
        for (i0, i1) in (faults_vent or {}).get(s.name, []):
            vent_scale[i0:i1] = 1 / 1.8      # 환기 정지 → 사람이 있을 때 상승 폭 약 1.8배
        ach = vent_ach * vent_scale

        C = np.empty(n); T = np.empty(n); H = np.empty(n)
        c, t = 450.0, t_base[0]
        for i in range(n):
            gen = N[i] * GEN_M3_PER_MIN / s.volume * 1e6
            c = c + gen - ach[i] / 60 * (c - c_out[i])
            heat = 0.012 * N[i] / s.capacity * 30 / s.volume * 10     # 사람 열 (℃/min)
            if hvac[i]:
                t = t + (s.setpoint + 0.4 * N[i] / s.capacity - t) / 15 + heat * 0.2
            elif op[i]:
                # 운영 시간인데 냉방이 꺼짐(고장): 낮 열부하(일사·기기)로 실외 쪽(기저+6℃)으로 끌려감
                t = t + (t_base[i] + 6.0 - t) / 120 + heat
            else:
                t = t + (t_base[i] - t) / 240 + heat
            C[i], T[i] = c, t
        H[:] = 55 - 2.0 * (T - 22) + 4 * N / s.capacity + rng.normal(0, 0.3, n).cumsum() * 0.01
        out[s.name] = {"co2": C, "temperature": T, "humidity": np.clip(H, 20, 90)}
        states[s.name] = {"hvac_on": hvac.astype(int), "ach": ach}
    return out, states


def inject_sensor_faults(idx, vals, rng, last_week_start):
    """튄 값·값 고정·영점 틀어짐 (센서 고장). 냉방·환기 정지는 simulate 에서 물리적으로 반영."""
    n = len(idx)
    i_lw = int(np.searchsorted(idx, last_week_start))
    faults = []

    def pick(dur, hours=None):
        while True:
            i0 = int(rng.integers(i_lw, n - dur - 1))
            if hours is None or (idx[i0].dayofweek < 5 and hours[0] <= idx[i0].hour < hours[1]):
                return i0

    for _ in range(6):          # 튄 값: 한 칸만 크게
        sp = SPACES[int(rng.integers(len(SPACES)))].name
        m = "co2" if rng.random() < 0.6 else "temperature"
        i0 = pick(1)
        if m == "co2":
            up = rng.random() < 0.6 or vals[sp][m][i0] < 900
            vals[sp][m][i0] += rng.uniform(500, 1200) if up else -rng.uniform(400, vals[sp][m][i0] - 350)
        else:
            vals[sp][m][i0] += rng.uniform(6, 12)
        faults.append(("튄 값", sp, m, idx[i0], idx[i0] + pd.Timedelta(minutes=1)))
    stuck = []
    for _ in range(3):          # 값 고정: 1~4시간 같은 값 (측정 잡음을 넣은 '뒤'에 적용해야 진짜로 멈춤 → main 에서)
        sp = SPACES[int(rng.integers(len(SPACES)))].name
        m = str(rng.choice(["co2", "temperature", "humidity"]))
        dur = int(rng.integers(60, 240))
        i0 = pick(dur, (8, 17))
        stuck.append((sp, m, i0, i0 + dur))
        faults.append(("값 고정", sp, m, idx[i0], idx[i0 + dur]))
    for _ in range(2):          # 영점 틀어짐: 수 시간에 걸쳐 수백 ppm 상승
        sp = SPACES[int(rng.integers(len(SPACES)))].name
        dur = int(rng.integers(6 * 60, 12 * 60))
        i0 = pick(dur)
        vals[sp]["co2"][i0:i0 + dur] += np.linspace(0, rng.uniform(250, 400), dur)
        faults.append(("영점 틀어짐", sp, "co2", idx[i0], idx[i0 + dur]))
    return faults, stuck


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--weeks", type=int, default=5)
    ap.add_argument("--start", default="2026-09-14")    # 월요일
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    idx = _minutes(pd.Timestamp(args.start, tz=TZ), args.weeks)
    days = pd.date_range(idx[0], idx[-1].normalize(), freq="D")
    sched = make_schedule(days, rng)
    occ = make_occupancy(idx, sched, rng)
    last_week = idx[-1].normalize() - pd.Timedelta(days=6)
    i_lw = int(np.searchsorted(idx, last_week))
    pos = lambda t: int(np.searchsorted(idx, t))

    # 설비 고장(냉방 정지·환기 정지): 마지막 1주, 사용 중인 시간대에 심는다
    faults = []
    hv, ve = {}, {}
    lw_days = [d for d in pd.date_range(last_week, idx[-1].normalize(), freq="D") if d.dayofweek < 5]
    for k, (kind, store, hours) in enumerate((("냉방 정지", hv, (13, 17)), ("냉방 정지", hv, (9, 12)),
                                               ("환기 정지", ve, (9, 12)), ("환기 정지", ve, (13, 18)))):
        sp = ["강의실1", "강의실2", "강의실1", "회의실"][k]
        d = lw_days[(k * 2 + 1) % len(lw_days)]
        t0 = d + pd.Timedelta(hours=hours[0])
        t1 = d + pd.Timedelta(hours=hours[1])
        store.setdefault(sp, []).append((pos(t0), pos(t1)))
        faults.append((kind, sp, "temperature" if kind == "냉방 정지" else "co2", t0, t1))

    vals, states = simulate(idx, occ, rng, hv, ve)
    sf, stuck = inject_sensor_faults(idx, vals, rng, last_week)
    faults += sf

    # 센서 측정: 잡음 + 반올림 + 결측 + 깨진 값
    rows = []
    n = len(idx)
    gw_down = np.zeros(n, bool)
    for _ in range(args.weeks * 2):                       # 게이트웨이 끊김 (모든 공간 동시)
        i0 = int(rng.integers(0, n - 200)); gw_down[i0:i0 + int(rng.integers(5, 120))] = True
    epoch = (idx.view("int64") // 10**9).astype(float)
    for s in SPACES:
        drop = gw_down | (rng.random(n) < 0.01)
        for m, noise, rnd in (("co2", 8, 0), ("temperature", 0.1, 1), ("humidity", 0.5, 1)):
            v = vals[s.name][m] + rng.normal(0, noise, n)
            v = np.round(v, rnd)
            for (ssp, sm, i0, i1) in stuck:
                if ssp == s.name and sm == m:
                    v[i0:i1] = v[i0]
            broken = rng.random(n) < 0.0005
            if m == "co2":
                v[broken] = 0
            elif m == "temperature":
                v[broken] = -40.0
            keep = ~drop
            rows.append(pd.DataFrame({"space": s.name, "metric": m, "device": f"synth-{s.name}", "point": "",
                                      "value": v[keep], "ts": epoch[keep]}))
    df = pd.concat(rows, ignore_index=True)
    df["sensor_ts"] = df["ts"]; df["recv_ts"] = df["ts"] + 0.3; df["topic"] = "synthetic"; df["rule"] = "synth"

    db = os.path.join(args.out, "sensors.db")
    if os.path.exists(db):
        os.remove(db)
    con = sqlite3.connect(db)
    df.to_sql("readings", con, index=False)
    con.execute("CREATE INDEX idx ON readings(space, metric, ts)")
    con.commit(); con.close()

    sched.to_csv(os.path.join(args.out, "schedule.csv"), index=False)
    occ.to_csv(os.path.join(args.out, "occupancy.csv"), index_label="time")
    obs_days = rng.choice([d for d in days if d.dayofweek < 5 and d < last_week], size=6, replace=False)
    obs = occ[np.isin(occ.index.normalize(), pd.DatetimeIndex(obs_days))]
    obs = obs[(obs.index.hour >= 8) & (obs.index.hour < 19)]
    (obs > 0).astype(int).to_csv(os.path.join(args.out, "observed.csv"), index_label="time")
    pd.DataFrame(faults, columns=["type", "space", "metric", "start", "end"]).sort_values("start").to_csv(
        os.path.join(args.out, "faults.csv"), index=False)
    st = pd.DataFrame({f"{sp}_{k}": v for sp, d in states.items() for k, v in d.items()}, index=idx)
    st.iloc[::5].to_csv(os.path.join(args.out, "truth_state.csv"), index_label="time")

    print(f"기간 {idx[0]:%Y-%m-%d} ~ {idx[-1]:%Y-%m-%d} ({args.weeks}주), 센서 행 {len(df):,}, 일정 {len(sched)}건, 고장 {len(faults)}건")
    print(pd.DataFrame(faults, columns=["type", "space", "metric", "start", "end"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
