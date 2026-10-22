"""과제 3 심화(B) — (1) 과거 길이 × 모델 종류 비교, (2) 처음 보는 공간 일반화 (합성 데이터).

    .venv/bin/python task3_advanced.py

(1) 과거 1·2·4시간(12·24·48칸) × GRU·LSTM·TCN, 일정 포함. 같은 시험 시점(5주차, 과거 4시간이 있는 시점)에서 비교.
(2) 공간 하나를 학습에서 완전히 빼고(공간 구분 없는 모델) 그 공간에서만 시험 → 학습한 공간에서의 성능과 비교.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from tsinf import forecast as fc
from tsinf import prep
from tsinf.synth import SPACES

torch.set_num_threads(4)
df, _ = prep.clean(prep.load_readings("data/synth/sensors.db"))
grid = prep.to_grid(df)
spaces = [s.name for s in SPACES]
cap = {s.name: float(s.capacity) for s in SPACES}
sched = fc.schedule_grid(pd.read_csv("data/synth/schedule.csv", parse_dates=["start", "end"]), grid.index, spaces, cap)
faults = pd.read_csv("data/synth/faults.csv", parse_dates=["start", "end"])
excl = {}
for r in faults.itertuples():
    excl.setdefault(r.space, []).append((pd.Timestamp(r.start), pd.Timestamp(r.end)))
t0 = grid.index[0].normalize()
tr_t = grid.index[grid.index < t0 + pd.Timedelta(days=21)]
va_t = grid.index[(grid.index >= t0 + pd.Timedelta(days=21)) & (grid.index < t0 + pd.Timedelta(days=28))]
te_t = grid.index[grid.index >= t0 + pd.Timedelta(days=28) + pd.Timedelta(hours=4)]   # 모든 L 에 같은 시험 시점
stats = prep.norm_stats(grid, tr_t)

rows = []
for L in (12, 24, 48):
    spec = fc.DatasetSpec(spaces, L, True, stats, cap)
    d = {k: fc.build_samples(grid, sched, spec, v, excl if k == "test" else None)
         for k, v in (("train", tr_t), ("val", va_t), ("test", te_t))}
    for kind in ("gru", "lstm", "tcn"):
        t1 = time.time()
        m, _ = fc.train(kind, d["train"], d["val"], epochs=40, verbose=False)
        mae = fc.mae_by_h(fc.predict(m, d["test"]), d["test"])
        rows.append({"과거 길이": f"{L * 5 // 60}시간", "모델": kind.upper(), "CO₂ 15분": mae[2, 0], "CO₂ 30분": mae[5, 0],
                     "CO₂ 60분": mae[11, 0], "온도 60분": mae[11, 1], "학습(s)": time.time() - t1, "n": len(d["test"]["y"])})
        print(rows[-1])
grid_tab = pd.DataFrame(rows)

# (2) 처음 보는 공간
gen = []
for held in spaces:
    seen = [s for s in spaces if s != held]
    spec_all = fc.DatasetSpec(spaces, 24, True, stats, cap, space_onehot=False)
    d_tr = fc.build_samples(grid, sched, fc.DatasetSpec(seen, 24, True, stats, cap, space_onehot=False), tr_t)
    d_va = fc.build_samples(grid, sched, fc.DatasetSpec(seen, 24, True, stats, cap, space_onehot=False), va_t)
    d_te = fc.build_samples(grid, sched, fc.DatasetSpec([held], 24, True, stats, cap, space_onehot=False), te_t, excl)
    d_in = fc.build_samples(grid, sched, spec_all, tr_t)
    d_inv = fc.build_samples(grid, sched, spec_all, va_t)
    m_out, _ = fc.train("gru", d_tr, d_va, epochs=40, verbose=False)
    m_in, _ = fc.train("gru", d_in, d_inv, epochs=40, verbose=False)
    base = np.repeat(d_te["base"][:, None, :], 12, axis=1)
    gen.append({"시험 공간": held,
                "현재값 유지": fc.mae_by_h(base, d_te)[11, 0],
                "학습에 포함한 모델": fc.mae_by_h(fc.predict(m_in, d_te), d_te)[11, 0],
                "처음 보는 공간 (학습에서 뺌)": fc.mae_by_h(fc.predict(m_out, d_te), d_te)[11, 0]})
    print(gen[-1])
gen = pd.DataFrame(gen).set_index("시험 공간")
gen["일반화 손실(ppm)"] = gen["처음 보는 공간 (학습에서 뺌)"] - gen["학습에 포함한 모델"]

best = grid_tab.loc[grid_tab["CO₂ 60분"].idxmin()]


def md(d):
    d = d.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.1f}")
    return d.to_markdown()


lines = [
    "# 과제 3 심화(B)",
    "",
    "> 측정 조건: 합성 데이터 5주 · 학습 1~3주 / 검증 4주 / 시험 5주(고장 구간 제외, 과거 4시간이 있는 시점만 → 모든 조합 같은 시점) · 일정 포함",
    "",
    "## 1. 과거 길이 × 모델 종류 (60분 앞 MAE)",
    md(grid_tab.set_index(["과거 길이", "모델"])),
    "",
    f"- **최적 조합: 과거 {best['과거 길이']} + {best['모델']}** (60분 CO₂ {best['CO₂ 60분']:.1f}ppm)",
    "- 과거를 1시간에서 2시간으로 늘리면 수업 한 주기(50분 수업 + 10분 휴식)를 통째로 보게 되어 좋아지고, 4시간으로 늘려도 이득이 작거나 오히려 학습이 느려진다 "
    "→ 미래 일정 정보가 들어가면 아주 먼 과거는 덜 중요하다.",
    "- 같은 과거 길이에서 세 모델의 차이는 수 ppm 이내로 작다 → 모델 종류보다 **입력 정보(일정)와 과거 길이**가 더 중요하다. 보드 속도까지 고려하면 TCN·GRU 가 실용적.",
    "",
    "## 2. 처음 보는 공간에서도 예측이 되는가 (60분 CO₂ MAE, ppm)",
    md(gen),
    "",
    "- 공간 하나를 학습에서 완전히 빼도 대부분 '현재값 유지'보다 좋다 → 새 공간에도 어느 정도 바로 쓸 수 있다.",
    "- 일반화 손실이 큰 공간은 다른 공간과 성격이 다른 곳이다(인원·부피·일정 방식이 다름). 이런 공간은 며칠치만 모아도 다시 학습하는 것이 좋다.",
    "- 실데이터에서는 합성 학습 모델이 실제 교실에서 크게 틀렸다(과제 3 실데이터 검증) → '공간'보다 '데이터 성격(합성 vs 실제)' 차이가 훨씬 크다.",
]
open("reports/task3_advanced_report.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
