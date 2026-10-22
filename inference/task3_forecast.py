"""과제 3. 60분 예측 — 합성 데이터로 학습·평가하고 보고서를 만든다.

    .venv/bin/python task3_forecast.py --data data/synth --models gru lstm tcn

분할(시간순): 1~3주 학습 · 4주 검증 · 5주 시험. 시험 주에는 고장이 심어져 있으므로
'예측' 성능은 고장 구간을 뺀 정상 구간에서 잰다 (고장 탐지는 과제 4).
산출물: reports/task3_report.md, models/forecast_<kind>[_nosched].pt, models/forecast_meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from tsinf import prep
from tsinf import forecast as fc
from tsinf.synth import SPACES

HZ = {"5분": 0, "15분": 2, "30분": 5, "60분": 11}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--models", nargs="+", default=["gru", "lstm", "tcn"])
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True); os.makedirs("models", exist_ok=True)
    torch.set_num_threads(4)

    df, st = prep.clean(prep.load_readings(f"{args.data}/sensors.db"))
    grid = prep.to_grid(df)
    spaces = [s.name for s in SPACES]
    cap = {s.name: float(s.capacity) for s in SPACES}
    sched_df = pd.read_csv(f"{args.data}/schedule.csv", parse_dates=["start", "end"])
    sched = fc.schedule_grid(sched_df, grid.index, spaces, cap)
    faults = pd.read_csv(f"{args.data}/faults.csv", parse_dates=["start", "end"])
    excl = {}
    for r in faults.itertuples():
        excl.setdefault(r.space, []).append((pd.Timestamp(r.start), pd.Timestamp(r.end)))

    t0 = grid.index[0].normalize()
    tr_t = grid.index[grid.index < t0 + pd.Timedelta(days=21)]
    va_t = grid.index[(grid.index >= t0 + pd.Timedelta(days=21)) & (grid.index < t0 + pd.Timedelta(days=28))]
    te_t = grid.index[grid.index >= t0 + pd.Timedelta(days=28)]
    stats = prep.norm_stats(grid, tr_t)

    results, rows, meta = {}, [], {"spaces": spaces, "capacity": cap, "lookback": args.lookback,
                                   "stats": {f"{s}|{m}": [float(stats.loc[(s, m), 'mean']), float(stats.loc[(s, m), 'std'])]
                                             for (s, m) in stats.index}, "models": {}}
    data = {}
    for use_sched in (True, False):
        spec = fc.DatasetSpec(spaces, args.lookback, use_sched, stats, cap)
        data[use_sched] = {name: fc.build_samples(grid, sched, spec, t, excl if name == "test" else None)
                           for name, t in (("train", tr_t), ("val", va_t), ("test", te_t))}
    print({k: len(v["y"]) for k, v in data[True].items()})
    te = data[True]["test"]

    def add_rows(name, pred):
        for sp in spaces + ["전체"]:
            m = np.ones(len(te["y"]), bool) if sp == "전체" else te["space"] == sp
            true = te["base"][m][:, None, :] + te["y"][m] * te["sd"][m][:, None, :]
            err = np.abs(pred[m] - true)
            for hname, h in HZ.items():
                rows.append({"model": name, "space": sp, "horizon": hname,
                             "CO2": float(np.nanmean(err[:, h, 0])), "temp": float(np.nanmean(err[:, h, 1]))})

    for name, pred in fc.baseline_preds(grid, te).items():
        add_rows(name, pred)

    timing = {}
    for kind in args.models:
        for use_sched in ((True, False) if kind == "gru" else (True,)):
            tag = kind.upper() + ("" if use_sched else " (일정 제외)")
            print(f"== {tag}")
            t1 = time.time()
            model, best = fc.train(kind, data[use_sched]["train"], data[use_sched]["val"], epochs=args.epochs)
            timing[tag] = time.time() - t1
            add_rows(tag, fc.predict(model, data[use_sched]["test"]))
            fname = f"models/forecast_{kind}{'' if use_sched else '_nosched'}.pt"
            torch.save(model.state_dict(), fname)
            meta["models"][tag] = {"file": fname, "kind": kind, "use_schedule": use_sched,
                                   "n_in": int(data[use_sched]["train"]["x_seq"].shape[2]), "val_co2_mae": float(best)}
    with open("models/forecast_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    res = pd.DataFrame(rows)
    allv = res[res.space == "전체"]
    co2 = allv.pivot(index="model", columns="horizon", values="CO2")[list(HZ)]
    tmp = allv.pivot(index="model", columns="horizon", values="temp")[list(HZ)]
    p60 = co2.loc["현재값 유지", "60분"]
    co2["60분 개선율(vs 현재값 유지)"] = (1 - co2["60분"] / p60) * 100
    per_space = res[(res.horizon == "60분") & (res.space != "전체")].pivot(index="model", columns="space", values="CO2")

    hardest = None
    best_model = co2.drop(index=["현재값 유지", "최근 1시간 평균", "어제 같은 시각"]).iloc[:, 3].idxmin()
    hardest = per_space.loc[best_model].idxmax()
    gain_sp = (per_space.loc["GRU (일정 제외)"] - per_space.loc["GRU"]).idxmax() if "GRU" in per_space.index else hardest
    sched_gain = co2.loc["GRU (일정 제외)", "60분"] - co2.loc["GRU", "60분"] if "GRU" in co2.index else np.nan

    def md(d, fmt="{:.2f}"):
        d = d.copy()
        for c in d.columns:
            d[c] = d[c].map(lambda v: "-" if pd.isna(v) else fmt.format(v))
        return d.to_markdown()

    lines = [
        "# 과제 3. 60분 예측",
        "",
        f"> 측정 조건: 합성 데이터 5주 (자체 생성기, seed 7) · 학습 1~3주 / 검증 4주 / 시험 5주 (고장 구간 제외) · "
        f"5분 간격 · 과거 {args.lookback * 5}분 입력 · 시험 샘플 {len(te['y']):,}개 · CPU 학습 (MacBook)",
        "",
        "## 1. 입력·출력 설계",
        f"- 과거 {args.lookback * 5 // 60}시간({args.lookback}칸): CO₂·온도·습도(공간별 정규화) + 시간대·요일(sin/cos) + 그 시점 일정 인원",
        "- 공간 원-핫 (어느 공간인지), 앞으로 60분의 **일정 인원(수업 시간표·회의실 예약)** — 미래에도 미리 아는 정보",
        "- 출력: 60분을 **한 번에** 12칸 출력 (한 칸씩 반복 예측하면 오차가 누적됨)",
        "- 출력은 '지금 값 대비 변화량' → 기본값이 '현재값 유지'가 되어 학습이 안정적",
        "",
        "## 2. 모델 비교 — CO₂ MAE (ppm, 시험 주, 전체 공간)",
        md(co2),
        "",
        "## 3. 온도 MAE (℃)",
        md(tmp),
        "",
        "## 4. 공간별 60분 CO₂ MAE (ppm)",
        md(per_space),
        "",
        "## 5. 학습 시간 (초, CPU)",
        md(pd.DataFrame({"학습 시간(s)": timing}), "{:.0f}"),
        "",
        "## 6. 결론",
        f"- 가장 좋은 모델: **{best_model}** — 60분 CO₂ 오차 {co2.loc[best_model, '60분']:.1f}ppm, "
        f"'현재값 유지'({p60:.1f}ppm) 대비 **{co2.loc[best_model, '60분 개선율(vs 현재값 유지)']:.0f}% 감소** "
        f"({'✅ 목표 30% 달성' if co2.loc[best_model, '60분 개선율(vs 현재값 유지)'] >= 30 else '❌ 목표 30% 미달'})",
        f"- **일정 정보의 가치**: 일정을 넣으면 60분 CO₂ 오차가 {sched_gain:.1f}ppm 줄어든다 "
        f"(GRU 일정 제외 {co2.loc['GRU (일정 제외)', '60분']:.1f} → 포함 {co2.loc['GRU', '60분']:.1f}ppm).",
        f"- 공간별로는 **{hardest}** 가 가장 어렵다(60분 {per_space.loc[best_model, hardest]:.1f}ppm). 인원이 가장 많아(정원 30명) "
        "수업 시작·쉬는 시간마다 CO₂ 가 수백 ppm씩 오르내리고, 쉬는 시간 창문 개방 여부도 날마다 다르기 때문이다.",
        f"- 일정 정보 효과가 가장 큰 공간: **{gain_sp}** (60분 오차 {per_space.loc['GRU (일정 제외)', gain_sp]:.1f} → "
        f"{per_space.loc['GRU', gain_sp]:.1f}ppm). 과거 센서값에는 '곧 누가 들어온다'는 정보가 없어서, 수업 시작·회의 예약을 "
        "미리 알면 상승 시점을 맞힐 수 있다. 다만 노쇼·연장·예약 없는 사용처럼 일정과 실제가 어긋나는 경우는 일정을 넣어도 맞히지 못한다.",
        "- 오차가 큰 순간은 수업 시작·쉬는 시간(창문 개방)·회의 시작처럼 재실 인원이 급변하는 시점이다. "
        "일정은 '계획'만 알려 주므로 지각·결석·연장으로 생기는 차이는 여전히 맞히기 어렵다.",
        "- 온도는 냉방이 설정 온도로 묶어 두어 변화가 작아 모든 방법의 오차가 작다. 대신 냉방 정지 같은 고장 때 크게 벗어난다(과제 4).",
        "- 실데이터에는 일정 자료가 없어, 서비스(과제 8)에는 '일정 제외' 모델을 쓰거나 일정 자료를 받아 연결해야 한다.",
    ]
    with open(f"{args.out}/task3_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[10:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
