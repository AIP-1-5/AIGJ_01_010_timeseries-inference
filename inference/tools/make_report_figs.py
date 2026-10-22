"""결과 보고서용 핵심 그래프 3장 (reports/figs/)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsinf import prep

plt.rcParams["font.family"] = ["AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("reports/figs", exist_ok=True)

df, _ = prep.clean(prep.load_readings("data/influx_56d.db"))
g = prep.to_grid(df)
co2 = g[("실습실", "co2")]

# 1. 개념 변화: 주별 실습실 CO₂
wk = co2.resample("W").agg(["mean", "max"])           # 월~일 한 주, 끝나는 일요일로 표기 (task3_real 표와 같음)
wk = wk[wk.index <= co2.index[-1] + pd.Timedelta(days=6)]
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar(["~" + d for d in wk.index.strftime("%m-%d")], wk["mean"], color="#2F6FDE", label="주 평균")
ax.plot(["~" + d for d in wk.index.strftime("%m-%d")], wk["max"], "o-", color="#C2410C", label="주 최대")
ax.axhline(1000, ls="--", c="gray", lw=1); ax.set_ylabel("CO₂ (ppm)")
ax.set_title("실습실 주별 CO₂ — 9월(전 기수)과 10월(우리 기수)의 사용 패턴이 다르다 (개념 변화)")
ax.legend(); fig.tight_layout(); fig.savefig("reports/figs/fig_drift_weekly.png", dpi=130); plt.close(fig)

# 2. 실제 시간표: 수업일 vs 수업 없는 날 시간대별 CO₂
start = pd.Timestamp("2026-10-01", tz=prep.TZ)
c = co2[co2.index >= start]
hol = {pd.Timestamp("2026-10-05").date(), pd.Timestamp("2026-10-09").date()}
is_class = np.array([(t.dayofweek < 5 and t.date() not in hol) for t in c.index])
prof = c.groupby([is_class, c.index.hour + c.index.minute // 30 * 0.5]).mean().unstack(0)
fig, ax = plt.subplots(figsize=(8, 3.4))
for lo, hi, lab in ((9, 11.5, "수업"), (13, 18, "수업")):
    ax.axvspan(lo, hi, color="#F2E8DA", zorder=0)
ax.axvspan(11.5, 13, color="#E8F0FB", zorder=0)
ax.plot(prof.index, prof[True], c="#C2410C", lw=2, label="수업일 (평일)")
ax.plot(prof.index, prof[False], c="#4A5568", lw=2, label="수업 없는 날 (주말·휴일)")
ax.text(10.2, ax.get_ylim()[1] * 0.97, "수업", ha="center"); ax.text(12.25, ax.get_ylim()[1] * 0.97, "점심", ha="center"); ax.text(15.5, ax.get_ylim()[1] * 0.97, "수업", ha="center")
ax.set_xlim(6, 22); ax.set_xlabel("시각"); ax.set_ylabel("CO₂ (ppm)")
ax.set_title("실습실 실제 시간표와 CO₂ (10/1~10/21) — 시간표가 '곧 바뀐다'를 알려 준다")
ax.legend(loc="upper right", fontsize=8); fig.tight_layout(); fig.savefig("reports/figs/fig_schedule_profile.png", dpi=130); plt.close(fig)

# 3. 60분 예측 오차 비교 (보고서 수치)
labels = ["현재값 유지", "GRU 일정 제외", "GRU 실제 시간표"]
vals = [50.0, 39.4, 29.3]
fig, ax = plt.subplots(figsize=(6, 3.2))
b = ax.bar(labels, vals, color=["#9AA5B1", "#8FB8F2", "#2F6FDE"])
ax.axhline(50.0 * 0.7, ls="--", c="#C2410C", lw=1); ax.text(2.45, 50 * 0.7 + 0.8, "기준(30%↓)", color="#C2410C", ha="right", fontsize=8)
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.8, f"{v}", ha="center")
ax.set_ylabel("60분 CO₂ MAE (ppm)"); ax.set_title("실데이터 실습실 60분 예측 (시험 10/18~21)")
fig.tight_layout(); fig.savefig("reports/figs/fig_schedule_forecast.png", dpi=130); plt.close(fig)

# 4. 엣지: 실행 방식별 지연
bj = json.load(open("reports/bench_board_cpu.json"))
rows = [r for r in bj["rows"] if r["batch"] == 1]
d = pd.DataFrame(rows).pivot(index="model", columns="runtime", values="mean_ms")
ax = d.plot.bar(figsize=(6, 3.2), logy=True, rot=0, color=["#2F6FDE", "#9AA5B1"])
ax.set_ylabel("1회 추론 (ms, 로그)"); ax.set_title("보드 B CPU: PyTorch vs ONNX Runtime (배치 1)")
plt.tight_layout(); plt.savefig("reports/figs/fig_edge_latency.png", dpi=130); plt.close()
print(os.listdir("reports/figs"))
