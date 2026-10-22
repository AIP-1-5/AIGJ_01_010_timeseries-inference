"""과제 7 보강 — ONNX Runtime INT8 양자화 (동적 / 정적) 와 정확도 비교 (맥북에서 실행).

    PYTHONPATH=. .venv/bin/python addenda_task7_int8.py

- FP32: models/forecast_{gru,lstm,tcn}.onnx
- 동적 INT8: onnxruntime.quantization.quantize_dynamic (가중치 INT8, 활성값은 실행 중 범위 계산)
- 정적 INT8: quantize_static (QDQ), 보정 데이터 = 검증 주(4주차) 샘플 512개  ← 시험 주는 쓰지 않음
- 정확도: 시험 주(5주차, 고장 구간 제외) 60분 CO₂ MAE, FP32 대비 최대 절대 차이(ppm)
산출물: models/int8/*.onnx, reports/addenda/task7_int8_accuracy.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from tsinf import forecast as fc
from tsinf import prep
from tsinf.synth import SPACES

DATA = "data/synth"
OUT = "models/int8"
os.makedirs(OUT, exist_ok=True)
os.makedirs("reports/addenda", exist_ok=True)
meta = json.load(open("models/forecast_meta.json", encoding="utf-8"))

# ---- task3_forecast.py 와 같은 분할·전처리
df, _ = prep.clean(prep.load_readings(f"{DATA}/sensors.db"))
grid = prep.to_grid(df)
spaces = [s.name for s in SPACES]
cap = {s.name: float(s.capacity) for s in SPACES}
sched = fc.schedule_grid(pd.read_csv(f"{DATA}/schedule.csv", parse_dates=["start", "end"]), grid.index, spaces, cap)
faults = pd.read_csv(f"{DATA}/faults.csv", parse_dates=["start", "end"])
excl = {}
for r in faults.itertuples():
    excl.setdefault(r.space, []).append((pd.Timestamp(r.start), pd.Timestamp(r.end)))
t0 = grid.index[0].normalize()
tr_t = grid.index[grid.index < t0 + pd.Timedelta(days=21)]
va_t = grid.index[(grid.index >= t0 + pd.Timedelta(days=21)) & (grid.index < t0 + pd.Timedelta(days=28))]
te_t = grid.index[grid.index >= t0 + pd.Timedelta(days=28)]
stats = prep.norm_stats(grid, tr_t)
spec = fc.DatasetSpec(spaces, meta["lookback"], True, stats, cap)
va = fc.build_samples(grid, sched, spec, va_t)
te = fc.build_samples(grid, sched, spec, te_t, excl)
print("val", len(va["y"]), "test", len(te["y"]))


def run(path, d):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outs = []
    for i in range(0, len(d["y"]), 1024):
        outs.append(s.run(None, {"x_seq": d["x_seq"][i:i + 1024], "x_static": d["x_static"][i:i + 1024],
                                 "x_future": d["x_future"][i:i + 1024]})[0])
    dz = np.concatenate(outs)
    return d["base"][:, None, :] + dz * d["sd"][:, None, :]          # 원래 단위 [N,12,2]


def mae60(pred, d):
    true = d["base"][:, None, :] + d["y"] * d["sd"][:, None, :]
    return float(np.abs(pred[:, 11, 0] - true[:, 11, 0]).mean())


class Reader(CalibrationDataReader):
    def __init__(self, d, n=512, bs=1):
        idx = np.random.default_rng(0).choice(len(d["y"]), min(n, len(d["y"])), replace=False)
        self.items = [{"x_seq": d["x_seq"][idx[i:i + bs]], "x_static": d["x_static"][idx[i:i + bs]],
                       "x_future": d["x_future"][idx[i:i + bs]]} for i in range(0, len(idx), bs)]
        self.it = iter(self.items)

    def get_next(self):
        return next(self.it, None)


def op_counts(path):
    m = onnx.load(path)
    c = {}
    for n in m.graph.node:
        c[n.op_type] = c.get(n.op_type, 0) + 1
    return c


rng_info = {k: {"val_min": float(va[k].min()), "val_max": float(va[k].max()), "test_min": float(te[k].min()),
                "test_max": float(te[k].max())} for k in ("x_seq", "x_future")}
res = {"input_ranges": rng_info, "ort": ort.__version__, "onnx": onnx.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
       "calib": "검증 주(4주차) 샘플 512개 (MinMax)", "test_samples": int(len(te["y"])), "models": {}}
for tag in ("GRU", "LSTM", "TCN"):
    kind = meta["models"][tag]["kind"]
    fp = f"models/forecast_{kind}.onnx"
    pre = f"{OUT}/forecast_{kind}_pre.onnx"
    quant_pre_process(fp, pre, skip_symbolic_shape=True)
    variants = {"FP32": fp}
    dyn = f"{OUT}/forecast_{kind}_int8_dyn.onnx"
    quantize_dynamic(pre, dyn, weight_type=QuantType.QInt8)
    try:
        ort.InferenceSession(dyn, providers=["CPUExecutionProvider"])
        res.setdefault("dyn_weight_type", {})[tag] = "QInt8"
    except Exception as e:  # ConvInteger 는 CPU EP 에서 uint8 가중치만 지원
        res.setdefault("errors", {})[f"{tag} dynamic QInt8"] = f"{type(e).__name__}: {str(e)[:200]}"
        quantize_dynamic(pre, dyn, weight_type=QuantType.QUInt8)
        res.setdefault("dyn_weight_type", {})[tag] = "QUInt8 (QInt8 실행 불가로 대체)"
    variants["INT8 동적"] = dyn
    st = f"{OUT}/forecast_{kind}_int8_static.onnx"
    try:
        quantize_static(pre, st, Reader(va), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8, per_channel=True)
        variants["INT8 정적"] = st
    except Exception as e:  # noqa: BLE001
        res.setdefault("errors", {})[f"{tag} static"] = f"{type(e).__name__}: {e}"
    p32 = run(fp, te)
    rows = {}
    for name, path in variants.items():
        p = run(path, te)
        rows[name] = {"file": path, "size_kb": os.path.getsize(path) / 1024, "test_mae60_co2": mae60(p, te),
                      "val_mae60_co2": mae60(run(path, va), va),
                      "max_abs_co2_diff_ppm_vs_fp32": float(np.abs(p[..., 0] - p32[..., 0]).max()),
                      "mean_abs_co2_diff_ppm_vs_fp32": float(np.abs(p[..., 0] - p32[..., 0]).mean()),
                      "p99_abs_co2_diff_ppm_vs_fp32": float(np.percentile(np.abs(p[..., 0] - p32[..., 0]), 99)),
                      "frac_samples_diff_gt_10ppm": float((np.abs(p[..., 0] - p32[..., 0]).max(axis=1) > 10).mean()),
                      "worst_sample": {"space": str(te["space"][np.abs(p[..., 0] - p32[..., 0]).max(axis=1).argmax()]),
                                       "time": str(te["time"][np.abs(p[..., 0] - p32[..., 0]).max(axis=1).argmax()]),
                                       "x_future_max": float(te["x_future"][np.abs(p[..., 0] - p32[..., 0]).max(axis=1).argmax()].max())},
                      "ops": op_counts(path)}
        print(tag, name, {k: v for k, v in rows[name].items() if k != "ops"})
    res["models"][tag] = rows
    os.remove(pre)
json.dump(res, open("reports/addenda/task7_int8_accuracy.json", "w"), ensure_ascii=False, indent=1)
print("saved")
