"""과제 6 보강: (1) GRU + Chronos 앙상블, (2) GRU 입력 제거 실험(시간대·온습도·일정 몫).

    PYTHONPATH=. .venv/bin/python addenda_task6.py --part ens      # 합성·내 데이터 앙상블
    PYTHONPATH=. .venv/bin/python addenda_task6.py --part ablate   # 합성 GRU 입력 제거 실험

공정성: 앙상블 가중치는 검증 구간(합성 4주차 / 내 데이터 앞 절반)에서만 고르고, 시험 구간은 보고만 한다.
결과는 reports/addenda/task6_*.json 으로 저장 (보고서는 손으로 작성).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from tsinf import forecast as fc
from tsinf import prep
from tsinf.synth import SPACES

HZ = {"5분": 0, "15분": 2, "30분": 5, "60분": 11}
W_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
OUT = "reports/addenda"


def synth_setup(data="data/synth"):
    df, _ = prep.clean(prep.load_readings(f"{data}/sensors.db"))
    grid = prep.to_grid(df)
    spaces = [s.name for s in SPACES]
    cap = {s.name: float(s.capacity) for s in SPACES}
    sched_df = pd.read_csv(f"{data}/schedule.csv", parse_dates=["start", "end"])
    sched = fc.schedule_grid(sched_df, grid.index, spaces, cap)
    faults = pd.read_csv(f"{data}/faults.csv", parse_dates=["start", "end"])
    excl = {}
    for r in faults.itertuples():
        excl.setdefault(r.space, []).append((pd.Timestamp(r.start), pd.Timestamp(r.end)))
    t0 = grid.index[0].normalize()
    tr_t = grid.index[grid.index < t0 + pd.Timedelta(days=21)]
    va_t = grid.index[(grid.index >= t0 + pd.Timedelta(days=21)) & (grid.index < t0 + pd.Timedelta(days=28))]
    te_t = grid.index[grid.index >= t0 + pd.Timedelta(days=28)]
    stats = prep.norm_stats(grid, tr_t)
    return dict(grid=grid, spaces=spaces, cap=cap, sched=sched, excl=excl, tr_t=tr_t, va_t=va_t, te_t=te_t, stats=stats)


def true_co2(d):
    return d["base"][:, None, 0] + d["y"][:, :, 0] * d["sd"][:, None, 0]


def chronos_median(pipe, grid, d, L=288, idx=None):
    pos = pd.Series(np.arange(len(grid)), index=grid.index)
    idx = np.arange(len(d["y"])) if idx is None else idx
    ctx = []
    for n in idx:
        i = pos[d["time"][n]]
        s = grid[(d["space"][n], "co2")].to_numpy()[max(0, i - L + 1):i + 1]
        ctx.append(torch.tensor(pd.Series(s).interpolate(limit_direction="both").to_numpy(), dtype=torch.float32))
    qs = []
    for b in range(0, len(ctx), 256):
        q, _ = pipe.predict_quantiles(ctx[b:b + 256], prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])
        qs.append(q.numpy())
    return np.concatenate(qs)[:, :, 1]


def mae_row(pred, true):
    err = np.abs(pred - true)
    return {h: float(np.nanmean(err[:, k])) for h, k in HZ.items()}


def pick_w(g, c, true):
    """60분 MAE 가 가장 작은 GRU 가중치 w (예측 = w·GRU + (1-w)·Chronos)"""
    scores = {float(w): float(np.abs(w * g[:, 11] + (1 - w) * c[:, 11] - true[:, 11]).mean()) for w in W_GRID}
    return min(scores, key=scores.get), scores


# ------------------------------------------------------------------ 앙상블
def part_ens():
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu", torch_dtype=torch.float32)
    res = {}

    # ---- 합성: 검증 = 4주차, 시험 = 5주차 (과제 6 과 같은 1108개 시점)
    S = synth_setup()
    spec = fc.DatasetSpec(S["spaces"], 24, True, S["stats"], S["cap"])
    spec_ns = fc.DatasetSpec(S["spaces"], 24, False, S["stats"], S["cap"])
    meta = json.load(open("models/forecast_meta.json", encoding="utf-8"))
    out = {}
    for split, times in (("val", S["va_t"][::6]), ("test", S["te_t"][::6])):
        d = fc.build_samples(S["grid"], S["sched"], spec, times, S["excl"])
        dns = fc.build_samples(S["grid"], S["sched"], spec_ns, times, S["excl"])
        preds = {}
        for tag, dd in (("GRU", d), ("GRU (일정 제외)", dns)):
            g = meta["models"][tag]
            m = fc.make_model("gru", g["n_in"], len(S["spaces"])); m.load_state_dict(torch.load(g["file"])); m.eval()
            preds[tag] = fc.predict(m, dd)[:, :, 0]
        preds["Chronos"] = chronos_median(pipe, S["grid"], d)
        preds["현재값 유지"] = np.repeat(d["base"][:, None, 0], 12, axis=1)
        out[split] = dict(true=true_co2(d), **preds, n=len(d["y"]))
    syn = {"n_val": out["val"]["n"], "n_test": out["test"]["n"], "rows": {}, "w": {}}
    te = out["test"]
    for name in ("현재값 유지", "GRU", "GRU (일정 제외)", "Chronos"):
        syn["rows"][name] = mae_row(te[name], te["true"])
    for gname in ("GRU", "GRU (일정 제외)"):
        syn["rows"][f"평균 ({gname}+Chronos)/2"] = mae_row(0.5 * te[gname] + 0.5 * te["Chronos"], te["true"])
        w, scores = pick_w(out["val"][gname], out["val"]["Chronos"], out["val"]["true"])
        syn["w"][gname] = {"w": w, "val_60": scores[w], "val_60_gru_only": scores[1.0], "val_60_chronos_only": scores[0.0]}
        syn["rows"][f"가중 {w:.2f}·{gname} + {1 - w:.2f}·Chronos (w 는 검증)"] = mae_row(w * te[gname] + (1 - w) * te["Chronos"], te["true"])
    res["synth"] = syn
    print(json.dumps(syn, ensure_ascii=False, indent=1))

    # ---- 내 데이터: task_own.py 와 같은 230개 시점 (15분 간격)
    raw = prep.load_readings("data/sensors_boardB.db")
    df, _ = prep.clean(raw)
    grid = prep.to_grid(df)
    spaces = sorted({s for s, m in grid.columns if all((s, x) in grid.columns for x in prep.METRICS)})
    stats = prep.norm_stats(grid, grid.index[: len(grid) // 2])
    sched = fc.schedule_grid(None, grid.index, spaces, {})
    d = fc.build_samples(grid, sched, fc.DatasetSpec(spaces, 24, False, stats, {}, space_onehot=False), grid.index[24:])
    sel = np.arange(0, len(d["y"]), 3)
    s = ort.InferenceSession("models/service_gru.onnx")
    dz = s.run(None, {"x_seq": d["x_seq"], "x_static": d["x_static"], "x_future": d["x_future"]})[0]
    gru = (d["base"][:, None, 0] + dz[:, :, 0] * d["sd"][:, None, 0])[sel]
    tt = true_co2(d)[sel]
    chr_ = chronos_median(pipe, grid, d, idx=sel)
    hold = np.repeat(d["base"][sel, None, 0], 12, axis=1)
    times = pd.DatetimeIndex(d["time"][sel])
    cut = times.sort_values()[len(times) // 2]
    va, te_m = times < cut, times >= cut
    own = {"n": int(len(sel)), "period": f"{grid.index[0]:%m-%d %H:%M} ~ {grid.index[-1]:%m-%d %H:%M}",
           "cut": f"{cut:%m-%d %H:%M}", "n_val": int(va.sum()), "n_test": int(te_m.sum()), "all": {}, "test_half": {}}
    for name, p in (("현재값 유지", hold), ("실데이터 GRU", gru), ("Chronos", chr_), ("평균 (GRU+Chronos)/2", 0.5 * gru + 0.5 * chr_)):
        own["all"][name] = mae_row(p, tt)
    w, scores = pick_w(gru[va], chr_[va], tt[va])
    own["w"] = {"w": w, "val_60": scores[w], "val_60_gru_only": scores[1.0], "val_60_chronos_only": scores[0.0]}
    for name, p in (("현재값 유지", hold), ("실데이터 GRU", gru), ("Chronos", chr_), ("평균 (GRU+Chronos)/2", 0.5 * gru + 0.5 * chr_),
                    (f"가중 {w:.2f}·GRU + {1 - w:.2f}·Chronos (w 는 앞 절반)", w * gru + (1 - w) * chr_)):
        own["test_half"][name] = mae_row(p[te_m], tt[te_m])
    # 오차 상관: 두 모델이 같은 곳에서 틀리는가
    own["err_corr_60"] = float(np.corrcoef(gru[:, 11] - tt[:, 11], chr_[:, 11] - tt[:, 11])[0, 1])
    syn["err_corr_60"] = float(np.corrcoef(te["GRU"][:, 11] - te["true"][:, 11], te["Chronos"][:, 11] - te["true"][:, 11])[0, 1])
    res["own"] = own
    print(json.dumps(own, ensure_ascii=False, indent=1))
    json.dump(res, open(f"{OUT}/task6_ens.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ 입력 제거 실험
VARIANTS = {   # x_seq 열: 0 CO₂, 1 온도, 2 습도, 3~7 시간대·요일, 8 현재 일정 / x_future: 앞으로 60분 일정
    "(a) 전체 입력": dict(cols=list(range(9)), sched=True),
    "(b) 시간대·요일 제외": dict(cols=[0, 1, 2, 8], sched=True),
    "(c) 온도·습도 제외 (CO₂ 만)": dict(cols=[0, 3, 4, 5, 6, 7, 8], sched=True),
    "(d) 일정 제외": dict(cols=[0, 1, 2, 3, 4, 5, 6, 7], sched=False),
    "(e) CO₂ 과거값만 (Chronos 와 같은 정보)": dict(cols=[0], sched=False),
}


def _subset(d, cols, sched):
    e = dict(d)
    e["x_seq"] = np.ascontiguousarray(d["x_seq"][:, :, cols])
    if not sched:
        e["x_future"] = np.zeros_like(d["x_future"])
    return e


def part_ablate(epochs, seeds):
    torch.set_num_threads(4)
    S = synth_setup()
    spec = fc.DatasetSpec(S["spaces"], 24, True, S["stats"], S["cap"])
    D = {"train": fc.build_samples(S["grid"], S["sched"], spec, S["tr_t"]),
         "val": fc.build_samples(S["grid"], S["sched"], spec, S["va_t"]),
         "test": fc.build_samples(S["grid"], S["sched"], spec, S["te_t"][::6], S["excl"])}
    print({k: len(v["y"]) for k, v in D.items()})
    true = true_co2(D["test"])
    res = {"epochs": epochs, "seeds": list(seeds), "n": {k: len(v["y"]) for k, v in D.items()}, "rows": {}}
    for name, v in VARIANTS.items():
        per_seed = []
        for seed in seeds:
            t1 = time.time()
            tr, va, te = (_subset(D[k], v["cols"], v["sched"]) for k in ("train", "val", "test"))
            m, best = fc.train("gru", tr, va, epochs=epochs, seed=seed, verbose=False)
            r = mae_row(fc.predict(m, te)[:, :, 0], true)
            r["val60"] = float(fc.mae_by_h(fc.predict(m, va), va)[11, 0])
            per_seed.append(r)
            print(name, seed, f"{time.time() - t1:.0f}s", r, flush=True)
        res["rows"][name] = {k: [p[k] for p in per_seed] for k in per_seed[0]}
        json.dump(res, open(f"{OUT}/task6_ablate.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ Chronos-2 입력 내보내기 (실행은 python3.11 별도 venv)
def part_c2export(L=288):
    S = synth_setup()
    spec = fc.DatasetSpec(S["spaces"], 24, True, S["stats"], S["cap"])
    grid, sched = S["grid"], S["sched"]
    pos = pd.Series(np.arange(len(grid)), index=grid.index)
    tf = fc.time_features(grid.index)
    out = {}
    for split, times in (("val", S["va_t"][::6]), ("test", S["te_t"][::6])):
        d = fc.build_samples(grid, sched, spec, times, S["excl"])
        n = len(d["y"])
        arr = {k: np.full((n, L), np.nan, np.float32) for k in ("co2", "temperature", "humidity")}
        sc = np.zeros((n, L + 12), np.float32); tt = np.zeros((n, L + 12, 5), np.float32)
        for j, (sp, t) in enumerate(zip(d["space"], d["time"])):
            i = pos[t]; a = max(0, i - L + 1); k = L - (i + 1 - a)
            for m in arr:
                s = grid[(sp, m)].to_numpy()[a:i + 1]
                arr[m][j, k:] = pd.Series(s).interpolate(limit_direction="both").to_numpy()
            sc[j, k:] = sched[sp].to_numpy()[a:i + 13]
            tt[j, k:] = tf[a:i + 13]
        out.update({f"{split}_{m}": v for m, v in arr.items()})
        out[f"{split}_sched"] = sc; out[f"{split}_time"] = tt; out[f"{split}_true"] = true_co2(d)
    np.savez_compressed(f"{OUT}/task6_c2_inputs.npz", **out)
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["ens", "ablate", "c2export"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"ens": part_ens, "ablate": lambda: part_ablate(a.epochs, a.seeds), "c2export": part_c2export}[a.part]()
