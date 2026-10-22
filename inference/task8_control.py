"""과제 8 심화(B) — 예측 기반 환기 제어 규칙을 지난 1주(합성 5주차)에 적용했을 때 1,000ppm 초과 시간이 얼마나 줄어드는지.

    .venv/bin/python task8_control.py

같은 사람·같은 일정·같은 날씨로 합성 데이터의 물리 모델(CO₂ 질량 수지)을 다시 돌려 비교한다.
- 기존: 운영 시간 기본 환기 (+ 쉬는 시간 창문 개방)
- 반응형 규칙: 지금 CO₂ ≥ 1,000ppm 이면 15분간 환기 강화 (지금 현장 방식 = 넘고 나서 환기)
- 예측형 규칙: 과제 3 GRU 가 앞으로 30분 안에 1,000ppm 을 넘는다고 예측하면 지금 15분간 환기 강화
환기 강화 = 환기 횟수 +4회/h (창문 개방·환기 장치 최대). 비용 지표 = 환기 강화한 시간.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from tsinf import forecast as fc
from tsinf import prep
from tsinf.synth import C_OUT, GEN_M3_PER_MIN, SPACES

BOOST_ACH, BOOST_MIN = 4.0, 15

df, _ = prep.clean(prep.load_readings("data/synth/sensors.db"))
grid = prep.to_grid(df)
spaces = [s.name for s in SPACES]
cap = {s.name: float(s.capacity) for s in SPACES}
sched_df = pd.read_csv("data/synth/schedule.csv", parse_dates=["start", "end"])
sched = fc.schedule_grid(sched_df, grid.index, spaces, cap)
occ = pd.read_csv("data/synth/occupancy.csv", index_col=0)
occ.index = pd.to_datetime(occ.index, utc=True).tz_convert(prep.TZ)
st = pd.read_csv("data/synth/truth_state.csv", index_col=0)
st.index = pd.to_datetime(st.index, utc=True).tz_convert(prep.TZ)
meta = json.load(open("models/forecast_meta.json", encoding="utf-8"))
g = meta["models"]["GRU"]
model = fc.make_model("gru", g["n_in"], len(spaces)); model.load_state_dict(torch.load(g["file"])); model.eval()
t0 = grid.index[0].normalize()
stats = prep.norm_stats(grid, grid.index[grid.index < t0 + pd.Timedelta(days=21)])
week = occ.index[occ.index >= t0 + pd.Timedelta(days=28)]
tf_all = fc.time_features(grid.index)
gpos = pd.Series(np.arange(len(grid)), index=grid.index)


def simulate(space, mode):
    s = next(x for x in SPACES if x.name == space)
    N = occ.loc[week, space].to_numpy()
    ach_base = st[f"{space}_ach"].reindex(week, method="ffill").to_numpy()
    c = float(grid.loc[week[0].floor("5min"), (space, "co2")])
    co2 = np.empty(len(week))
    boost_left, boosted = 0, 0
    hist5 = list(grid.loc[:week[0], (space, "co2")].iloc[-24:].to_numpy())      # 직전 2시간 5분 평균
    acc = []
    mu = stats.loc[space, "mean"].reindex(prep.METRICS).to_numpy(np.float32)
    sd = stats.loc[space, "std"].reindex(prep.METRICS).to_numpy(np.float32)
    si = spaces.index(space)
    for i, t in enumerate(week):
        if t.minute % 5 == 0 and i > 0 and mode != "기존":
            if mode == "반응형 (넘으면 환기)":
                if c >= 1000:
                    boost_left = BOOST_MIN
            else:                                                         # 예측형
                gi = gpos.get(t.floor("5min"))
                if gi is not None and gi >= 24 and gi + 12 < len(grid):
                    other = grid[[(space, "temperature"), (space, "humidity")]].iloc[gi - 23:gi + 1].to_numpy(np.float32)
                    raw = np.concatenate([np.array(hist5[-24:], np.float32)[:, None], other], axis=1)
                    z = (raw - mu) / sd
                    x = np.concatenate([z, tf_all[gi - 23:gi + 1], sched[space].to_numpy(np.float32)[gi - 23:gi + 1, None]], 1)[None]
                    with torch.no_grad():
                        dz = model(torch.from_numpy(x), torch.from_numpy(np.eye(len(spaces), dtype=np.float32)[si][None]),
                                   torch.from_numpy(sched[space].to_numpy(np.float32)[gi + 1:gi + 13][None])).numpy()[0, :, 0]
                    pred30 = raw[-1, 0] + dz[:6] * sd[0]
                    if c >= 1000 or pred30.max() >= 1000:
                        boost_left = BOOST_MIN
        ach = ach_base[i] + (BOOST_ACH if boost_left > 0 else 0.0)
        if boost_left > 0:
            boost_left -= 1
            boosted += 1
        c = c + N[i] * GEN_M3_PER_MIN / s.volume * 1e6 - ach / 60 * (c - C_OUT)
        co2[i] = c
        acc.append(c)
        if t.minute % 5 == 4:
            hist5.append(float(np.mean(acc[-5:])))
    return co2, boosted


rows = []
for sp in spaces:
    for mode in ("기존", "반응형 (넘으면 환기)", "예측형 (30분 앞 예측)"):
        co2, boosted = simulate(sp, mode)
        rows.append({"공간": sp, "방식": mode, "1,000ppm 초과(분/주)": int((co2 >= 1000).sum()),
                     "1,500ppm 초과(분/주)": int((co2 >= 1500).sum()), "최대 CO₂": float(co2.max()), "환기 강화(분/주)": boosted})
        print(rows[-1])
res = pd.DataFrame(rows)
tot = res.groupby("방식")[["1,000ppm 초과(분/주)", "1,500ppm 초과(분/주)", "환기 강화(분/주)"]].sum().reindex(
    ["기존", "반응형 (넘으면 환기)", "예측형 (30분 앞 예측)"])
base_over = tot.loc["기존", "1,000ppm 초과(분/주)"]
tot["초과 시간 감소율(%)"] = (1 - tot["1,000ppm 초과(분/주)"] / base_over) * 100
lines = [
    "# 과제 8 심화(B) — 예측 기반 환기 제어 시뮬레이션",
    "",
    f"> 조건: 합성 5주차(같은 재실·일정·날씨), 물리 모델 재계산, 환기 강화 = +{BOOST_ACH:.0f}회/h × {BOOST_MIN}분, 판단 주기 5분, 예측 = 과제 3 GRU(일정 포함)",
    "",
    "## 4개 공간 합계 (1주)",
    tot.round(1).to_markdown(),
    "",
    "## 공간별",
    res.round(0).set_index(["공간", "방식"]).to_markdown(),
    "",
    "## 해석",
    f"- 지금 현장 방식(넘고 나서 환기)과 비교해, 예측형은 1,000ppm 초과 시간을 "
    f"{tot.loc['반응형 (넘으면 환기)', '1,000ppm 초과(분/주)']:.0f} → {tot.loc['예측형 (30분 앞 예측)', '1,000ppm 초과(분/주)']:.0f}분/주로 줄인다 "
    f"(제어 없음 대비 {tot.loc['예측형 (30분 앞 예측)', '초과 시간 감소율(%)']:.0f}% 감소).",
    "- 대신 환기 강화 시간이 늘 수 있다(헛경보만큼 미리 환기) → 냉난방 에너지 손실과의 균형을 봐야 한다. 30분 앞이 아니라 15분 앞으로 줄이면 환기 시간은 줄고 초과 시간은 조금 늘 것이다.",
    "- 한계: 합성 데이터의 물리 모델 위에서의 추정이다. 실제 창문 개방 효과(외기 온도·바람)와 사람의 반응 지연은 반영하지 않았다.",
]
open("reports/task8_control_report.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
