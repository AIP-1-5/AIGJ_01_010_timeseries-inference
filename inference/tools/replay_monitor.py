"""과제 8 심화: 서비스 '자기 채점'을 지난 데이터로 다시 돌려 본다 (리플레이).

    .venv/bin/python tools/replay_monitor.py --db data/sensors_boardB.db

서비스와 같은 Engine·_self_check 를 5분마다 호출해, 60분 예측이 실제로 맞았는지와
'현재값 유지'보다 나빠져 재학습 경보가 나는 때를 기록한다.
"""
from __future__ import annotations

import argparse
import sqlite3
import tempfile
import time

import pandas as pd
import yaml

from service.run import Service


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sensors_boardB.db")
    ap.add_argument("--out", default="reports/task8_self_monitor_report.md")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open("service/config.yaml"))
    cfg.update(collector_db=args.db, results_db=tempfile.mkdtemp() + "/replay.db")
    cfg["mqtt"]["enabled"] = False
    svc = Service(cfg)
    con = sqlite3.connect(args.db)
    t_min, t_max = con.execute("SELECT min(ts), max(ts) FROM readings WHERE ts > 1577836800").fetchone()
    start = pd.Timestamp(t_min, unit="s", tz="Asia/Seoul").ceil("5min") + pd.Timedelta(hours=2)
    end = pd.Timestamp(t_max, unit="s", tz="Asia/Seoul").floor("5min")
    events, clock = [], 0.0
    for now in pd.date_range(start, end, freq="5min"):
        clock = now.timestamp()
        out = svc.engine.run(args.db, now=now)
        for sp, res in out["spaces"].items():
            mo = svc._self_check(sp, res, clock)
            if mo.get("degraded"):
                events.append({"시각": now.strftime("%m-%d %H:%M"), "공간": sp, "모델 60분 MAE": mo["mae_60min_model"],
                               "현재값 유지": mo["mae_60min_persistence"]})
    rows = svc.store.q("SELECT space, horizon, AVG(model_err), AVG(persist_err), COUNT(*) FROM scores GROUP BY space, horizon")
    df = pd.DataFrame(rows, columns=["공간", "몇 분 뒤", "모델 MAE", "현재값 유지 MAE", "채점 수"])
    df["개선율(%)"] = (1 - df["모델 MAE"] / df["현재값 유지 MAE"]) * 100
    piv = df[df["몇 분 뒤"].isin([15, 30, 60])].pivot(index="공간", columns="몇 분 뒤", values=["모델 MAE", "현재값 유지 MAE", "개선율(%)"]).round(1)
    ev = pd.DataFrame(events)
    first = ev.groupby("공간").first() if len(ev) else ev
    lines = ["# 과제 8 심화: 서비스 자기 채점 (리플레이)", "",
             f"> 측정 조건: 내 수집기 실데이터 {start:%m-%d %H:%M} ~ {end:%m-%d %H:%M}, 서비스와 같은 코드(`Service._self_check`)를 5분마다 호출. "
             "경보 규칙: 최근 6시간 동안 채점된 60분 예측이 36개 이상이고, 모델 오차가 '현재값 유지'의 1.2배를 넘고 차이가 5ppm 이상이면 'model_degraded'. (처음엔 1.2배만 봤더니 조용한 새벽에 4.3 vs 3.5ppm 같은 작은 차이로 25번 울려 조건을 추가)", "",
             "## 채점 결과 (CO₂ MAE ppm)", piv.to_markdown(), "",
             f"## 재학습 경보 (model_degraded): {len(ev)}회 (5분 칸 기준)",
             (first.to_markdown() if len(ev) else "- 없음"), "",
             "- 예측을 결과 DB(`forecasts`)에 저장해 두었다가, 목표 시각의 실제 5분 값이 들어오면 채점해 `scores` 에 남긴다. "
             "서비스를 재시작해도 이어서 채점된다. 조회: `GET /monitor?hours=24`, 각 공간 결과의 `monitor` 항목.", ""]
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
