"""실데이터 + 실제 실습실 시간표로 일정 정보의 가치를 잰다 (과제 2·3·5 보강).

    .venv/bin/python task_schedule_real.py --db data/influx_56d.db

시간표(수연 확인, 2026-10-22): 평일 09:00~11:30 수업, 11:30~13:00 점심, 13:00~18:00 수업.
수업 시작일·휴일은 데이터로 확인해 정한다: 10/1 부터 수업, 10/5(개천절 대체휴일)·10/9(한글날) 휴일.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from tsinf import forecast as fc
from tsinf import prep

SPACE = "실습실"
START = pd.Timestamp("2026-10-01", tz=prep.TZ)          # 이 기수 수업 시작 (9/28~30 은 CO₂ 변화 없음)
HOLIDAYS = {pd.Timestamp("2026-10-05").date(), pd.Timestamp("2026-10-09").date()}
BLOCKS = [("09:00", "11:30"), ("13:00", "18:00")]
HZ = {"5분": 0, "15분": 2, "30분": 5, "60분": 11}
SEEDS = (0, 1, 2)


def class_days(index: pd.DatetimeIndex):
    days = sorted({d for d in index.date if d >= START.date()})
    return [d for d in days if pd.Timestamp(d).dayofweek < 5 and d not in HOLIDAYS]


def schedule_rows(index: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for d in class_days(index):
        for s, e in BLOCKS:
            rows.append({"space": SPACE, "start": pd.Timestamp(f"{d} {s}", tz=prep.TZ),
                         "end": pd.Timestamp(f"{d} {e}", tz=prep.TZ), "planned_people": 1.0})
    return pd.DataFrame(rows)


def md(d, fmt="{:.1f}"):
    d = d.copy()
    for c in d.columns:
        d[c] = d[c].map(lambda v: "-" if pd.isna(v) else (fmt.format(v) if isinstance(v, (float, np.floating)) else v))
    return d.to_markdown()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/influx_56d.db")
    ap.add_argument("--own-obs", default="data/own_observed.csv")
    ap.add_argument("--out", default="reports/real56/task_schedule_real_report.md")
    args = ap.parse_args(argv)
    torch.set_num_threads(4)

    df, _ = prep.clean(prep.load_readings(args.db))
    grid = prep.to_grid(df)
    grid = grid[grid.index >= START - pd.Timedelta(hours=4)]
    sch = schedule_rows(grid.index)
    sched = fc.schedule_grid(sch, grid.index, [SPACE], {SPACE: 1.0})
    splits = prep.split_time(grid.index[grid.index >= START])
    stats = prep.norm_stats(grid, splits["train"])
    lines = ["# 실제 시간표로 잰 일정 정보의 가치 (실습실, 실데이터)", "",
             f"> 측정 조건: 교육장 InfluxDB 실데이터 {START:%Y-%m-%d} ~ {grid.index[-1]:%Y-%m-%d %H:%M}, 공간 {SPACE}, 5분 칸 · "
             f"시간순 70/15/15 (학습 ~{splits['train'][-1]:%m-%d}, 검증 ~{splits['val'][-1]:%m-%d}, 시험 {splits['test'][0]:%m-%d} ~ {splits['test'][-1]:%m-%d}) · "
             f"시간표: 평일 09:00~11:30 · 13:00~18:00 (수업일 {len(class_days(grid.index))}일, 휴일 10/5·10/9 제외) · GRU 씨앗 {len(SEEDS)}개 평균", ""]

    # ---------------------------------------------------- 과제 2: 일정 → CO₂ 반응 시간
    co2 = grid[(SPACE, "co2")]
    days = class_days(grid.index)
    react = []
    for label, hm, sign in (("오전 수업 시작 09:00", "09:00", +1), ("점심 시작 11:30", "11:30", -1),
                            ("오후 수업 시작 13:00", "13:00", +1), ("수업 끝 18:00", "18:00", -1)):
        mins = []
        for d in days:
            t0 = pd.Timestamp(f"{d} {hm}", tz=prep.TZ)
            w = co2[(co2.index >= t0) & (co2.index <= t0 + pd.Timedelta(minutes=90))]
            if len(w) < 10 or np.isnan(w.iloc[0]):
                continue
            hit = w[(w - w.iloc[0]) * sign >= 30]
            mins.append((hit.index[0] - t0).total_seconds() / 60 if len(hit) else np.nan)
        s = pd.Series(mins, dtype=float)
        react.append({"기준 시각": label, "수업일": len(s), "30ppm 변할 때까지(분, 중앙값)": s.median(),
                      "90분 안에 변화 없음(일)": int(s.isna().sum())})
    lines += ["## 1. 일정 변화 뒤 CO₂ 반응 시간 (과제 2)", md(pd.DataFrame(react).set_index("기준 시각")), "",
              "- 시작 시각에서 CO₂ 가 30ppm 오르거나(수업 시작) 내려갈 때(점심·종료)까지 걸린 시간. 5분 칸이라 5분 단위.", ""]

    # 시간대별 평균 (수업일 vs 휴일·주말)
    hour = co2.groupby([np.isin(co2.index.date, days), co2.index.hour]).mean().unstack(0)
    hour.columns = ["수업 없는 날" if not c else "수업일" for c in hour.columns]
    lines += ["### 시간대별 평균 CO₂ (ppm)", md(hour.loc[7:20].T, "{:.0f}"), ""]

    # ---------------------------------------------------- 과제 3: 일정 포함 vs 제외
    res = {}
    preds = {}
    for use in (True, False):
        spec = fc.DatasetSpec([SPACE], 24, use, stats, {SPACE: 1.0})
        data = {k: fc.build_samples(grid, sched, spec, v) for k, v in splits.items()}
        te = data["test"]
        true = te["base"][:, None, :] + te["y"] * te["sd"][:, None, :]
        ps = []
        for seed in SEEDS:
            model, _ = fc.train("gru", data["train"], data["val"], epochs=40, seed=seed, verbose=False)
            ps.append(fc.predict(model, te))
        preds[use] = np.mean(ps, axis=0)
        res[use] = [np.abs(p - true)[:, :, 0].mean(axis=0) for p in ps]
        if use:
            base = fc.baseline_preds(grid, te)
            te_keep, true_keep = te, true
    rows = {}
    for name, p in base.items():
        rows[name] = {h: float(np.nanmean(np.abs(p - true_keep)[:, k, 0])) for h, k in HZ.items()}
    for use, name in ((False, "GRU 일정 제외"), (True, "GRU 일정 포함 (실제 시간표)")):
        arr = np.stack(res[use])
        rows[name] = {h: float(arr[:, k].mean()) for h, k in HZ.items()}
        rows[name]["60분 씨앗 간 편차(±)"] = float(arr[:, 11].std())
    t3 = pd.DataFrame(rows).T
    p60 = t3.loc["현재값 유지", "60분"]
    t3["60분 개선율(%)"] = (1 - t3["60분"] / p60) * 100

    # 일정이 바뀌는 때 vs 그 외
    tt = pd.DatetimeIndex(te_keep["time"])
    hm = tt.hour * 60 + tt.minute
    near = np.zeros(len(tt), bool)
    for c in (9 * 60, 11 * 60 + 30, 13 * 60, 18 * 60):
        near |= (hm >= c - 60) & (hm < c + 30)
    near &= np.isin(tt.date, days)
    seg = {}
    for use, name in ((False, "GRU 일정 제외"), (True, "GRU 일정 포함")):
        e = np.abs(preds[use] - true_keep)[:, 11, 0]
        seg[name] = {"일정 바뀌기 1시간 전~30분 뒤": float(e[near].mean()), "그 외": float(e[~near].mean())}
    seg["현재값 유지"] = {k: float(np.abs(base["현재값 유지"] - true_keep)[:, 11, 0][m].mean())
                     for k, m in (("일정 바뀌기 1시간 전~30분 뒤", near), ("그 외", ~near))}
    lines += ["## 2. 60분 예측: 실제 시간표를 넣으면 얼마나 좋아지나 (과제 3)",
              f"- 시험 시점 {len(te_keep['y']):,}개 (CO₂ MAE ppm)", md(t3), "",
              "### 언제 좋아지나 (60분 CO₂ MAE ppm)", md(pd.DataFrame(seg).T), "",
              f"- 일정 정보로 60분 오차가 {t3.loc['GRU 일정 제외', '60분'] - t3.loc['GRU 일정 포함 (실제 시간표)', '60분']:.1f}ppm 바뀌었다 "
              f"(일정 제외 {t3.loc['GRU 일정 제외', '60분']:.1f} → 포함 {t3.loc['GRU 일정 포함 (실제 시간표)', '60분']:.1f}ppm).", ""]

    # ---------------------------------------------------- 과제 5: 시간표 정답 vs 소음 대리 정답
    if os.path.exists(args.own_obs):
        ob = pd.read_csv(args.own_obs, parse_dates=["start", "end"])
        ob = ob[ob.space == SPACE]
        recs = []
        for r in ob.itertuples():
            s, e = pd.Timestamp(r.start).tz_localize(prep.TZ), pd.Timestamp(r.end).tz_localize(prep.TZ)
            for t in pd.date_range(s, e, freq="5min", inclusive="left"):
                d = t.date()
                hmv = t.hour * 60 + t.minute
                in_class = (pd.Timestamp(d).dayofweek < 5 and d not in HOLIDAYS and
                            ((540 <= hmv < 690) or (780 <= hmv < 1080)))
                recs.append({"t": t, "proxy": int(r.occupied), "sched": int(in_class)})
        R = pd.DataFrame(recs).drop_duplicates("t")
        if len(R):
            agree = (R.proxy == R.sched).mean() * 100
            ct = pd.crosstab(R.sched.map({1: "시간표: 수업", 0: "시간표: 없음"}), R.proxy.map({1: "소음: 사용중", 0: "소음: 비어있음"}))
            lines += ["## 3. 시간표로 만든 '사용 여부' 정답은 얼마나 맞나 (과제 5)",
                      f"- 내 수집기 데이터 {R.t.min():%m-%d %H:%M} ~ {R.t.max():%m-%d %H:%M}, 5분 칸 {len(R):,}개. 비교 대상: 실습실 소음 센서(LAeq) 대리 정답",
                      f"- **일치율 {agree:.1f}%**", "", ct.to_markdown(), "",
                      "- 시간표는 '수업이 있어야 하는 시간'이고 소음은 '실제로 사람이 떠드는 시간'이다. 어긋나는 칸은 조용한 실습·쉬는 시간(시간표 수업·소음 없음)이나 "
                      "수업 전후에 남아 있는 사람(시간표 없음·소음 있음)이다.", ""]
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
