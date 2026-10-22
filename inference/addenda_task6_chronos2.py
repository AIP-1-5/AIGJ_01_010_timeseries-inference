"""과제 6 보강 (심화): 공변량을 받는 사전학습 모델 Chronos-2 (amazon/chronos-2, 120M) zero-shot.

chronos-forecasting 2.x 는 Python 3.10+ 가 필요해 저장소 .venv(3.9)와 분리된 python3.11 venv 에서 돌린다.
    PYTHONPATH=. .venv/bin/python addenda_task6.py --part c2export          # 입력 만들기 (.venv)
    <py3.11 venv>/bin/python addenda_task6_chronos2.py                       # 예측 (chronos-forecasting 2.3.2)

같은 1108개 시험 시점(합성 5주차, 30분 간격). 조합 선택은 검증(4주차 1228개 시점)으로만.
"""
import json
import time

import numpy as np
import torch
from chronos import BaseChronosPipeline

HZ = {"5분": 0, "15분": 2, "30분": 5, "60분": 11}
OUT = "reports/addenda"
torch.set_num_threads(4)

_z = np.load(f"{OUT}/task6_c2_inputs.npz")
z = {k: _z[k] for k in _z.files}       # NpzFile 은 접근할 때마다 배열 전체를 새로 읽으므로 한 번만 읽어 둔다
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu", dtype=torch.float32)

VARIANTS = {
    "Chronos-2 (CO₂ 만)": dict(past=[], fut=[]),
    "Chronos-2 + 온도·습도 (과거)": dict(past=["temperature", "humidity"], fut=[]),
    "Chronos-2 + 일정 (과거+미래 60분)": dict(past=["sched"], fut=["sched"]),
    "Chronos-2 + 시간대·요일 (과거+미래)": dict(past=["h_sin", "h_cos", "dow_sin", "dow_cos"], fut=["h_sin", "h_cos", "dow_sin", "dow_cos"]),
    "Chronos-2 + 전부 (온습도·일정·시간대)": dict(past=["temperature", "humidity", "sched", "h_sin", "h_cos", "dow_sin", "dow_cos"],
                                         fut=["sched", "h_sin", "h_cos", "dow_sin", "dow_cos"]),
}
TCOL = {"h_sin": 0, "h_cos": 1, "dow_sin": 2, "dow_cos": 3}
L = z["test_co2"].shape[1]


def cov(split, name, j, part):
    if name in ("temperature", "humidity"):
        return z[f"{split}_{name}"][j]
    a = z[f"{split}_sched"][j] if name == "sched" else z[f"{split}_time"][j, :, TCOL[name]]
    return a[:L] if part == "past" else a[L:]


res = {}
for split in ("val", "test"):
    n = len(z[f"{split}_true"])
    true = z[f"{split}_true"]
    for vname, v in VARIANTS.items():
        inputs = []
        for j in range(n):
            item = {"target": z[f"{split}_co2"][j]}
            if v["past"]:
                item["past_covariates"] = {c: cov(split, c, j, "past") for c in v["past"]}
            if v["fut"]:
                item["future_covariates"] = {c: cov(split, c, j, "fut") for c in v["fut"]}
            inputs.append(item)
        t1 = time.time()
        q, _ = pipe.predict_quantiles(inputs, prediction_length=12, quantile_levels=[0.1, 0.5, 0.9], batch_size=256)
        ms = (time.time() - t1) / n * 1000
        q = np.stack([x[0].numpy() for x in q])          # [n,12,3]
        err = np.abs(q[:, :, 1] - true)
        inside = (true >= q[:, :, 0]) & (true <= q[:, :, 2])
        res.setdefault(split, {})[vname] = {**{h: float(err[:, k].mean()) for h, k in HZ.items()},
                                            "cover60": float(inside[:, 11].mean() * 100), "ms_per_sample": ms}
        print(split, vname, res[split][vname], flush=True)
        if split == "test":
            np.save(f"{OUT}/task6_c2_{list(VARIANTS).index(vname)}_test_median.npy", q[:, :, 1])
best = min(res["val"], key=lambda k: res["val"][k]["60분"])
res["best_on_val"] = best
json.dump(res, open(f"{OUT}/task6_chronos2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("best on val:", best)
