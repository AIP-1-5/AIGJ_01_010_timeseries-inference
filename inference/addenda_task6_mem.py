"""과제 6 보강: 모델별 프로세스 메모리(RSS) — MacBook CPU 대리 측정.

    PYTHONPATH=. .venv/bin/python addenda_task6_mem.py [--c2-python <py3.11 venv python>]

각 구성을 **새 프로세스**에서 불러온 뒤 한 번 추론하고 RSS(현재)·최대 RSS 를 잰다.
AE(과제 4)는 저장된 파일이 없어 같은 구조(입력 12칸×3, 조건 10)를 임의 가중치로 ONNX 로 내보내 잰다(크기만 의미 있음).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

CODE = {
    "python+numpy+pandas (바탕)": "import numpy, pandas",
    "+ ONNX Runtime + GRU ONNX": """
import numpy as np, pandas, onnxruntime as ort
s = ort.InferenceSession('models/service_gru.onnx')
f = {i.name: np.zeros([1 if isinstance(d, str) or d is None else d for d in i.shape], np.float32) for i in s.get_inputs()}
s.run(None, f)""",
    "+ ONNX Runtime + GRU + AE ONNX + 상태 모델(pkl)": """
import numpy as np, pandas, onnxruntime as ort, pickle, sys
s = ort.InferenceSession('models/service_gru.onnx')
f = {i.name: np.zeros([1 if isinstance(d, str) or d is None else d for d in i.shape], np.float32) for i in s.get_inputs()}
s.run(None, f)
a = ort.InferenceSession(sys.argv[1]); a.run(None, {'x': np.zeros((1, 36), np.float32), 'c': np.zeros((1, 10), np.float32)})
sm = pickle.load(open('models/state_models.pkl', 'rb'))""",
    "torch import 만": "import numpy, pandas, torch",
    "torch + chronos-bolt-tiny (과거 24h 1회 예측)": """
import torch, numpy as np
from chronos import BaseChronosPipeline
p = BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-tiny', device_map='cpu', torch_dtype=torch.float32)
p.predict_quantiles([torch.randn(288) * 50 + 600], prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])""",
    "torch + chronos-bolt-small (과거 24h 1회 예측)": """
import torch, numpy as np
from chronos import BaseChronosPipeline
p = BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-small', device_map='cpu', torch_dtype=torch.float32)
p.predict_quantiles([torch.randn(288) * 50 + 600], prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])""",
}
C2 = {
    "[py3.11] torch import 만": "import numpy, torch",
    "[py3.11] torch + chronos-2 (과거 24h + 공변량 7개 1회 예측)": """
import torch, numpy as np
from chronos import BaseChronosPipeline
p = BaseChronosPipeline.from_pretrained('amazon/chronos-2', device_map='cpu', dtype=torch.float32)
x = {'target': np.random.randn(288).astype('float32'), 'past_covariates': {f'c{i}': np.random.randn(288) for i in range(7)},
     'future_covariates': {f'c{i}': np.random.randn(12) for i in range(5)}}
p.predict_quantiles([x], prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])""",
}
TAIL = """
import psutil, resource, json, sys as _s
r = psutil.Process().memory_info().rss / 2**20
mx = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if _s.platform == 'darwin' else 2**10)
print('RESULT', json.dumps({'rss_mb': r, 'max_rss_mb': mx}))"""


def run(py, code, arg):
    env = dict(os.environ, PYTHONPATH=".", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4")
    o = subprocess.run([py, "-c", code + TAIL, arg], capture_output=True, text=True, env=env)
    line = [l for l in o.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        return {"error": o.stderr[-500:]}
    return json.loads(line[0][7:])


def make_ae(path):
    import torch
    from tsinf.anomaly import AE
    m = AE(36, 10).eval()
    torch.onnx.export(m, (torch.zeros(1, 36), torch.zeros(1, 10)), path, input_names=["x", "c"], output_names=["r"],
                      dynamic_axes={"x": {0: "b"}, "c": {0: "b"}})
    return os.path.getsize(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2-python")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    ae = os.path.join(tempfile.mkdtemp(), "ae.onnx")
    res = {"ae_onnx_bytes": make_ae(ae), "rows": {}}
    jobs = [(sys.executable, k, v) for k, v in CODE.items()]
    if a.c2_python:
        jobs += [(a.c2_python, k, v) for k, v in C2.items()]
    for py, k, v in jobs:
        rs = [run(py, v, ae) for _ in range(a.reps)]
        ok = [r for r in rs if "error" not in r]
        res["rows"][k] = {"rss_mb": [r["rss_mb"] for r in ok], "max_rss_mb": [r["max_rss_mb"] for r in ok],
                          **({"error": rs[0]["error"]} if not ok else {})}
        print(k, res["rows"][k], flush=True)
    for f in ("models/service_gru.onnx", "models/state_models.pkl"):
        res[f] = os.path.getsize(f)
    json.dump(res, open("reports/addenda/task6_mem.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
