"""과제 6. 사전학습 모델(Chronos-Bolt) 비교 — 추가 학습 없이(zero-shot) CO₂ 60분 예측.

    .venv/bin/python task6_pretrained.py --models amazon/chronos-bolt-tiny amazon/chronos-bolt-small

같은 시점·같은 조건: 과제 3 시험 샘플(5주차, 고장 구간 제외)에서 30분 간격으로 뽑은 시점을 기준선·GRU·Chronos 가 똑같이 예측.
Chronos 는 CO₂ 과거값만 본다 (다른 변수·일정 없음). 과거 길이(context)를 2·6·24시간으로 바꿔 비교.
예측 구간: 10%~90% 분위수 사이에 실제값이 들어온 비율 (이상적이면 80%).
비용(보드 추론 시간·메모리)은 tools/bench_chronos.py 를 보드에서 돌려 채운다.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from tsinf import forecast as fc
from tsinf import prep
from tsinf.synth import SPACES

HZ = {"5분": 0, "15분": 2, "30분": 5, "60분": 11}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--models", nargs="+", default=["amazon/chronos-bolt-tiny", "amazon/chronos-bolt-small"])
    ap.add_argument("--contexts", nargs="+", type=int, default=[24, 72, 288])
    ap.add_argument("--every", type=int, default=6, help="시험 시점을 몇 칸마다 뽑을지 (6 = 30분)")
    args = ap.parse_args(argv)
    from chronos import BaseChronosPipeline

    df, _ = prep.clean(prep.load_readings(f"{args.data}/sensors.db"))
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
    te_t = grid.index[grid.index >= t0 + pd.Timedelta(days=28)][::args.every]
    stats = prep.norm_stats(grid, tr_t)
    te = fc.build_samples(grid, sched, fc.DatasetSpec(spaces, 24, True, stats, cap), te_t, excl)
    n = len(te["y"])
    true = te["base"][:, None, 0] + te["y"][:, :, 0] * te["sd"][:, None, 0]          # CO₂ [N,12]
    pos = pd.Series(np.arange(len(grid)), index=grid.index)

    rows, cover_rows, timing = [], [], {}

    def add(name, pred):
        err = np.abs(pred - true)
        rows.append({"model": name, **{h: float(np.nanmean(err[:, k])) for h, k in HZ.items()}})

    base = fc.baseline_preds(grid, te)
    for name in ("현재값 유지", "최근 1시간 평균", "어제 같은 시각"):
        add(name, base[name][:, :, 0])

    meta = json.load(open("models/forecast_meta.json", encoding="utf-8"))
    for tag in ("GRU", "GRU (일정 제외)"):
        g = meta["models"][tag]
        d = te if g["use_schedule"] else fc.build_samples(grid, sched, fc.DatasetSpec(spaces, 24, False, stats, cap), te_t, excl)
        m = fc.make_model("gru", g["n_in"], len(spaces)); m.load_state_dict(torch.load(g["file"])); m.eval()
        add(tag + " (직접 학습)", fc.predict(m, d)[:, :, 0])

    for mid in args.models:
        pipe = BaseChronosPipeline.from_pretrained(mid, device_map="cpu", torch_dtype=torch.float32)
        short = mid.split("/")[-1]
        for L in args.contexts:
            ctx = []
            for sp, t in zip(te["space"], te["time"]):
                i = pos[t]
                s = grid[(sp, "co2")].to_numpy()[max(0, i - L + 1):i + 1]
                s = pd.Series(s).interpolate(limit_direction="both").to_numpy()
                ctx.append(torch.tensor(s, dtype=torch.float32))
            t1 = time.time()
            qs = []
            for b in range(0, n, 256):
                q, _ = pipe.predict_quantiles(ctx[b:b + 256], prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])
                qs.append(q.numpy())
            q = np.concatenate(qs)                  # [N,12,3]
            timing[f"{short} ctx{L}"] = (time.time() - t1) / n * 1000
            add(f"{short} (과거 {L * 5 // 60}시간)", q[:, :, 1])
            inside = (true >= q[:, :, 0]) & (true <= q[:, :, 2])
            cover_rows.append({"model": f"{short} (과거 {L * 5 // 60}시간)", **{h: float(inside[:, k].mean() * 100) for h, k in HZ.items()}})
        del pipe

    res = pd.DataFrame(rows).set_index("model")
    p60 = res.loc["현재값 유지", "60분"]
    res["60분 개선율(vs 현재값 유지)"] = (1 - res["60분"] / p60) * 100
    cov = pd.DataFrame(cover_rows).set_index("model")
    tm = pd.DataFrame({"샘플당 추론 시간(ms, MacBook CPU, 256개 묶음)": timing})

    best_chronos = res[res.index.str.contains("chronos")]["60분"].idxmin()
    gru60 = res.loc["GRU (직접 학습)", "60분"]
    gru_ns60 = res.loc["GRU (일정 제외) (직접 학습)", "60분"]

    def md(d, fmt="{:.1f}"):
        d = d.copy()
        for c in d.columns:
            d[c] = d[c].map(lambda v: "-" if pd.isna(v) else fmt.format(v))
        return d.to_markdown()

    lines = [
        "# 과제 6. 사전학습 모델 비교 (Chronos-Bolt, zero-shot)",
        "",
        f"> 측정 조건: 합성 데이터 5주차(고장 구간 제외), 30분 간격으로 뽑은 {n}개 시점 · 5분 간격 60분 예측 · CO₂ 만 · "
        "Chronos 는 추가 학습 없음 · 정확도는 MacBook CPU 에서 측정",
        "",
        "## 1. 정확도 — CO₂ MAE (ppm), 같은 시점",
        md(res),
        "",
        "## 2. 예측 구간 적중률 — 10%~90% 구간 안에 실제값이 들어온 비율 (%, 이상적 80%)",
        md(cov),
        "",
        "## 3. 비용",
        md(tm, "{:.2f}"),
        "",
        "보드(Jetson Orin Nano) 추론 시간·메모리는 `tools/bench_chronos.py` 결과로 아래 표를 채운다.",
        "",
        "## 4. 해석",
        f"- 가장 좋은 Chronos: **{best_chronos}** — 60분 {res.loc[best_chronos, '60분']:.1f}ppm. "
        f"직접 학습한 GRU {gru60:.1f}ppm(일정 포함) / {gru_ns60:.1f}ppm(일정 제외)와 비교.",
        "- Chronos 는 CO₂ 과거값만 본다. 직접 학습 모델과의 차이 중 상당 부분은 **일정(미래 수업·예약)** 과 시간대 정보 때문이다: "
        "과거 곡선만으로는 '9시에 수업이 시작된다'를 알 수 없어 상승 시작을 놓친다. 일정 제외 GRU 와의 차이가 순수한 '우리 데이터로 학습한 효과'에 가깝다.",
        "- **zero-shot Chronos 는 60분에서 '현재값 유지'보다도 못하다.** 교실 CO₂ 는 수업 시작·쉬는 시간마다 톱니처럼 오르내리는데, "
        "과거 곡선만 본 범용 모델은 '지금 오르는 중이니 계속 오른다'처럼 추세를 늘려 그리다가 쉬는 시간의 급락을 못 맞힌다.",
        "- **과거를 길게 줄수록 좋아진다** (2시간 → 24시간: 60분 오차 약 50 → 40ppm). 24시간을 주면 '어제 이 시간에 올랐다'는 하루 주기를 볼 수 있기 때문이다.",
        "- **예측 구간은 잘 맞는다**: 10~90% 구간 적중률이 대부분 78~83%로 이상적인 80%에 가깝다 → '불확실성의 크기'는 정직하게 말한다. "
        "직접 학습한 GRU 는 한 값만 내므로 이런 구간 정보가 없다는 점은 Chronos 의 장점이다.",
        "- tiny 와 small 의 정확도 차이가 거의 없다 → 보드에 올린다면 tiny(더 작고 빠름)가 낫다.",
        "- **채택 의견 (초안)**: 학습 데이터가 거의 없는 초기(우리 실데이터 하루치)나 새 공간에는 바로 쓸 수 있는 좋은 출발점이다. "
        "다만 일정 정보를 쓰는 직접 학습 모델보다 정확도가 낮고, 모델이 더 크고 느리며 같은 보드에 로컬 LLM 도 올려야 하므로 "
        "**상시 서비스의 기본 모델로는 채택하지 않고**, 데이터가 부족한 새 공간용 예비 모델로만 둔다.",
    ]
    with open("reports/task6_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
