"""과제 3 보강 — (1) 검증 구간으로 모델·과거 길이 다시 고르기, (2) '값' vs '변화량' 출력 비교 (합성),
(3) 실데이터 온도 60분 진단.

    PYTHONPATH=. .venv/bin/python addenda_task3.py synth   # (1)(2) → reports/addenda/task3_synth.json
    PYTHONPATH=. .venv/bin/python addenda_task3.py real    # (3)    → reports/addenda/task3_real.json

tsinf 는 고치지 않는다. '값' 출력은 데이터 사전의 base 를 공간 평균(mu)으로, y 를 정규화된 절대값으로 바꿔
fc.train/predict 를 그대로 쓴다 (predict = base + out*sd = mu + z*sd 이므로 원래 단위가 그대로 나온다).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from tsinf import forecast as fc
from tsinf import prep

torch.set_num_threads(4)
EPOCHS = 40


def mae60(model, d):
    return fc.mae_by_h(fc.predict(model, d), d)[11]           # [CO2, temp]


def to_value_target(d, stats):
    """변화량 데이터 → 절대값(정규화) 데이터. 같은 샘플·같은 입력."""
    mu = np.array([[stats.loc[(sp, m), "mean"] for m in fc.TARGETS] for sp in d["space"]], np.float32)
    true = d["base"][:, None, :] + d["y"] * d["sd"][:, None, :]
    e = dict(d)
    e["y"] = ((true - mu[:, None, :]) / d["sd"][:, None, :]).astype(np.float32)
    e["base"] = mu
    return e


def synth():
    from tsinf.synth import SPACES
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
    te_full = grid.index[grid.index >= t0 + pd.Timedelta(days=28)]                         # task3_forecast 와 같은 시험
    te_same = grid.index[grid.index >= t0 + pd.Timedelta(days=28) + pd.Timedelta(hours=4)]  # task3_advanced 와 같은 시험
    stats = prep.norm_stats(grid, tr_t)

    rows, out = [], {}
    for L in (24, 12, 48):
        spec = fc.DatasetSpec(spaces, L, True, stats, cap)
        d = {"train": fc.build_samples(grid, sched, spec, tr_t), "val": fc.build_samples(grid, sched, spec, va_t),
             "test_same": fc.build_samples(grid, sched, spec, te_same, excl)}
        if L == 24:
            d["test_full"] = fc.build_samples(grid, sched, spec, te_full, excl)
        for kind in ("gru", "lstm", "tcn"):
            t1 = time.time()
            m, best = fc.train(kind, d["train"], d["val"], epochs=EPOCHS, verbose=False)
            r = {"L": L, "model": kind.upper(), "val_mean_co2": float(best),
                 "val60_co2": float(mae60(m, d["val"])[0]), "val60_temp": float(mae60(m, d["val"])[1]),
                 "test_same60_co2": float(mae60(m, d["test_same"])[0]), "test_same60_temp": float(mae60(m, d["test_same"])[1]),
                 "sec": time.time() - t1, "n_val": len(d["val"]["y"]), "n_test_same": len(d["test_same"]["y"])}
            if L == 24:
                r["test_full60_co2"] = float(mae60(m, d["test_full"])[0])
                r["test_full60_temp"] = float(mae60(m, d["test_full"])[1])
            rows.append(r); print(r, flush=True)
            json.dump({"grid": rows}, open("reports/addenda/task3_synth.json", "w"), ensure_ascii=False, indent=1)
        if L == 24:
            # persistence 기준선 (같은 샘플)
            for k in ("val", "test_full", "test_same"):
                p = np.repeat(d[k]["base"][:, None, :], 12, axis=1)
                out[f"persist_{k}"] = [float(x) for x in fc.mae_by_h(p, d[k])[11]]
            # (2) 값 vs 변화량 — GRU, L=24, seed 0, 같은 분할
            dv = {k: to_value_target(v, stats) for k, v in d.items()}
            t1 = time.time()
            mv, _ = fc.train("gru", dv["train"], dv["val"], epochs=EPOCHS, verbose=False)
            out["value_gru"] = {"val60": [float(x) for x in mae60(mv, dv["val"])],
                                "test_full60": [float(x) for x in mae60(mv, dv["test_full"])],
                                "test_full_by_h_co2": [float(x) for x in fc.mae_by_h(fc.predict(mv, dv["test_full"]), dv["test_full"])[:, 0]],
                                "sec": time.time() - t1}
            gd = [r for r in rows if r["model"] == "GRU" and r["L"] == 24][0]
            print("value", out["value_gru"], flush=True)
            # 변화량 GRU 의 시점별 (다시 학습하지 않고 같은 설정 재학습 = seed 0 동일 → 같은 결과)
            md_, _ = fc.train("gru", d["train"], d["val"], epochs=EPOCHS, verbose=False)
            out["delta_gru"] = {"val60": [float(x) for x in mae60(md_, d["val"])],
                                "test_full60": [float(x) for x in mae60(md_, d["test_full"])],
                                "test_full_by_h_co2": [float(x) for x in fc.mae_by_h(fc.predict(md_, d["test_full"]), d["test_full"])[:, 0]]}
            out["persist_by_h_co2"] = [float(x) for x in fc.mae_by_h(np.repeat(d["test_full"]["base"][:, None, :], 12, axis=1), d["test_full"])[:, 0]]
            print("delta", out["delta_gru"], gd, flush=True)
        json.dump({"grid": rows, **out}, open("reports/addenda/task3_synth.json", "w"), ensure_ascii=False, indent=1)


def real():
    df, _ = prep.clean(prep.load_readings("data/influx_56d.db"))
    grid = prep.to_grid(df)
    spaces = sorted({s for s, m in grid.columns if all((s, x) in grid.columns for x in prep.METRICS)})
    splits = prep.split_time(grid.index)
    stats = prep.norm_stats(grid, splits["train"])
    sched = fc.schedule_grid(None, grid.index, spaces, {})
    spec = fc.DatasetSpec(spaces, 24, False, stats, {})
    d = {k: fc.build_samples(grid, sched, spec, v) for k, v in splits.items()}
    res = {"spaces": spaces, "splits": {k: [str(v[0]), str(v[-1])] for k, v in splits.items()}}
    # 온도 성격: 구간별 표준편차, 60분 실제 변화 크기
    desc = []
    for sp in spaces:
        for k, v in splits.items():
            s = grid.loc[v, (sp, "temperature")]
            ch = (s.shift(-12) - s).abs()
            desc.append({"space": sp, "split": k, "mean": float(s.mean()), "std": float(s.std()),
                         "p05": float(s.quantile(.05)), "p95": float(s.quantile(.95)), "abs_change60_mean": float(ch.mean())})
    res["temp_desc"] = desc
    m, _ = fc.train("gru", d["train"], d["val"], epochs=EPOCHS, verbose=False)
    out = {}
    for k in ("val", "test"):
        pm = fc.predict(m, d[k])
        base = np.repeat(d[k]["base"][:, None, :], 12, axis=1)
        true = d[k]["base"][:, None, :] + d[k]["y"] * d[k]["sd"][:, None, :]
        out[k] = {"pm": pm, "base": base, "true": true, "space": d[k]["space"]}
    # 혼합: pred = base + a*(gru - base), a 는 검증 온도 60분 MAE 로 고름
    alphas = np.round(np.arange(0, 1.01, 0.1), 2)
    blend = []
    for a in alphas:
        r = {"alpha": float(a)}
        for k in ("val", "test"):
            p = out[k]["base"] + a * (out[k]["pm"] - out[k]["base"])
            r[k + "_temp60"] = float(np.abs(p - out[k]["true"])[:, 11, 1].mean())
        blend.append(r)
    res["blend"] = blend
    a_best = min(blend, key=lambda r: r["val_temp60"])["alpha"]
    res["alpha_best_by_val"] = a_best
    per = []
    for k in ("val", "test"):
        for sp in spaces + ["전체"]:
            msk = np.ones(len(out[k]["space"]), bool) if sp == "전체" else out[k]["space"] == sp
            e = lambda p: float(np.abs(p[msk] - out[k]["true"][msk])[:, 11, 1].mean())
            per.append({"split": k, "space": sp, "persist": e(out[k]["base"]), "gru": e(out[k]["pm"]),
                        "bias_gru": float((out[k]["pm"][msk] - out[k]["true"][msk])[:, 11, 1].mean()),
                        "blend": e(out[k]["base"] + a_best * (out[k]["pm"] - out[k]["base"]))})
    res["per_space"] = per
    json.dump(res, open("reports/addenda/task3_real.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def sched_real_temp():
    """task_schedule_real 설정(실습실, 10/1~, 실제 시간표, GRU)에서 온도 60분."""
    import task_schedule_real as tsr
    df, _ = prep.clean(prep.load_readings("data/influx_56d.db"))
    grid = prep.to_grid(df)
    grid = grid[grid.index >= tsr.START - pd.Timedelta(hours=4)]
    sched = fc.schedule_grid(tsr.schedule_rows(grid.index), grid.index, [tsr.SPACE], {tsr.SPACE: 1.0})
    splits = prep.split_time(grid.index[grid.index >= tsr.START])
    stats = prep.norm_stats(grid, splits["train"])
    res = {}
    for use in (True, False):
        spec = fc.DatasetSpec([tsr.SPACE], 24, use, stats, {tsr.SPACE: 1.0})
        d = {k: fc.build_samples(grid, sched, spec, v) for k, v in splits.items()}
        r = []
        for seed in tsr.SEEDS:
            m, _ = fc.train("gru", d["train"], d["val"], epochs=EPOCHS, seed=seed, verbose=False)
            r.append([float(mae60(m, d["val"])[1]), float(mae60(m, d["test"])[1])])
        pv = float(fc.mae_by_h(np.repeat(d["val"]["base"][:, None, :], 12, axis=1), d["val"])[11, 1])
        pt = float(fc.mae_by_h(np.repeat(d["test"]["base"][:, None, :], 12, axis=1), d["test"])[11, 1])
        res["sched" if use else "nosched"] = {"gru_val_test_per_seed": r, "persist_val": pv, "persist_test": pt,
                                             "n_test": len(d["test"]["y"])}
    res["splits"] = {k: [str(v[0]), str(v[-1])] for k, v in splits.items()}
    json.dump(res, open("reports/addenda/task3_sched_real_temp.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"synth": synth, "real": real, "sched": sched_real_temp}[sys.argv[1]]()
