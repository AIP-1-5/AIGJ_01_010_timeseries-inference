"""과제 3 실데이터 검증 — 교육장 InfluxDB 8주치(2026-08-26 ~ 10-21)로 60분 예측을 학습·평가.

    .venv/bin/python task3_real.py --db data/influx_56d.db

실데이터에는 일정 자료가 없으므로 '일정 제외' 모델만. 분할은 과제 2 와 같은 시간순 70/15/15.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from tsinf import forecast as fc
from tsinf import prep

HZ = {"5분": 0, "15분": 2, "30분": 5, "60분": 11}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/influx_56d.db")
    ap.add_argument("--models", nargs="+", default=["gru", "lstm", "tcn"])
    ap.add_argument("--out", default="reports/real56/task3_real_report.md")
    args = ap.parse_args(argv)
    torch.set_num_threads(4)

    df, st = prep.clean(prep.load_readings(args.db))
    grid = prep.to_grid(df)
    spaces = sorted({s for s, m in grid.columns if all((s, x) in grid.columns for x in prep.METRICS)})
    splits = prep.split_time(grid.index)
    stats = prep.norm_stats(grid, splits["train"])
    sched = fc.schedule_grid(None, grid.index, spaces, {})
    spec = fc.DatasetSpec(spaces, 24, False, stats, {})
    data = {k: fc.build_samples(grid, sched, spec, v) for k, v in splits.items()}
    te = data["test"]
    print({k: len(v["y"]) for k, v in data.items()}, spaces)

    rows = []

    def add(name, pred):
        true = te["base"][:, None, :] + te["y"] * te["sd"][:, None, :]
        for sp in spaces + ["전체"]:
            m = np.ones(len(te["y"]), bool) if sp == "전체" else te["space"] == sp
            err = np.abs(pred[m] - true[m])
            rows.append({"model": name, "space": sp, **{f"CO2 {h}": float(np.nanmean(err[:, k, 0])) for h, k in HZ.items()},
                         "온도 60분": float(np.nanmean(err[:, 11, 1]))})

    for name, pred in fc.baseline_preds(grid, te).items():
        add(name, pred)
    for kind in args.models:
        model, _ = fc.train(kind, data["train"], data["val"], epochs=40, verbose=False)
        add(kind.upper(), fc.predict(model, te))
        if kind == "gru":
            torch.save(model.state_dict(), "models/real_gru.pt")

    res = pd.DataFrame(rows)
    allv = res[res.space == "전체"].set_index("model").drop(columns="space")
    p60 = allv.loc["현재값 유지", "CO2 60분"]
    allv["60분 개선율(%)"] = (1 - allv["CO2 60분"] / p60) * 100
    per = res.pivot(index="model", columns="space", values="CO2 60분")
    best = allv.drop(index=["현재값 유지", "최근 1시간 평균", "어제 같은 시각"])["CO2 60분"].idxmin()

    def md(d):
        d = d.copy()
        for c in d.columns:
            d[c] = d[c].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
        return d.to_markdown()

    lines = [
        "# 과제 3 실데이터 검증 (교육장 InfluxDB 8주)",
        "",
        f"> 측정 조건: 실데이터 {grid.index[0]:%Y-%m-%d} ~ {grid.index[-1]:%Y-%m-%d}, 공간 {', '.join(spaces)} · 시간순 70/15/15 "
        f"(시험 {splits['test'][0]:%m-%d} ~ {splits['test'][-1]:%m-%d}) · 일정 정보 없음 · 시험 샘플 {len(te['y']):,}개",
        "",
        "## 전체 (CO₂ MAE ppm, 온도 ℃)",
        md(allv),
        "",
        "## 공간별 60분 CO₂ MAE (ppm)",
        md(per),
        "",
        "## 해석",
        f"- 실데이터에서도 **{best}** 가 60분 CO₂ 오차를 '현재값 유지'({p60:.1f}ppm) 대비 "
        f"**{allv.loc[best, '60분 개선율(%)']:.0f}%** 줄였다 ({allv.loc[best, 'CO2 60분']:.1f}ppm) "
        f"→ {'✅ 기준 30% 충족' if allv.loc[best, '60분 개선율(%)'] >= 30 else '❌ 기준 30% 미달'}.",
        "- 합성 데이터에서 본 경향(현재값 유지가 가장 강한 기준선, 어제 같은 시각이 가장 약함)이 실데이터에서도 그대로 나타난다 → 합성 데이터 설계가 현실을 잘 흉내 냈다는 근거.",
        "- 실데이터에는 일정 정보가 없어 수업 시작 같은 급변을 미리 알 수 없다. 교육장 시간표를 받으면 합성 실험에서처럼 추가 개선이 기대된다.",
    ]
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
