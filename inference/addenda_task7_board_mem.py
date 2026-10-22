"""과제 7 보강 — 한 프로세스에 서비스 모델들을 차례로 올리며 RSS 측정 (보드).
    nice -n 19 .venv-bench/bin/python addenda_task7_board_mem.py --extra-site <.venv site-packages> --out ...
.venv-bench 에는 torch/chronos/psutil 이 있고 sklearn 이 없어, .venv 의 site-packages 를 경로 맨 뒤에 붙여 sklearn 만 빌려 쓴다.
"""
import argparse, gc, json, os, subprocess, sys, time
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--extra-site", default=None)
a = ap.parse_args()
if a.extra_site:
    sys.path.append(a.extra_site)


def rss():
    d = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS", "VmHWM")):
            d[line.split(":")[0]] = int(line.split()[1]) / 1024
    return d


def free_m():
    out = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.splitlines()
    h = out[0].split(); v = out[1].split()[1:]
    return dict(zip(h, map(int, v)))


steps = [{"step": "시작 (python)", **rss()}]
res = {"free_before": free_m(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": steps}


def mark(name, t0):
    gc.collect(); r = rss(); r.update(step=name, load_s=round(time.time() - t0, 2)); steps.append(r)
    print(name, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)


import numpy as np
t = time.time(); import onnxruntime as ort; mark("import onnxruntime+numpy", t)
sess = {}
for name in ("service_gru", "forecast_gru", "forecast_lstm", "forecast_tcn"):
    t = time.time()
    sess[name] = ort.InferenceSession(f"models/{name}.onnx", providers=["CPUExecutionProvider"])
    ins = {i.name: np.zeros([1 if (d is None or isinstance(d, str)) else d for d in i.shape], np.float32) for i in sess[name].get_inputs()}
    sess[name].run(None, ins)  # 첫 실행까지 포함 (작업 메모리 할당)
    mark(f"+ {name}.onnx (1회 실행 포함)", t)
t = time.time()
import pickle
import sklearn  # noqa
state = pickle.load(open("models/state_models.pkl", "rb")); mark(f"+ state_models.pkl (sklearn {sklearn.__version__})", t)
ae = [f for f in os.listdir("models") if "ae" in f.lower() or "autoenc" in f.lower()]
res["autoencoder_files"] = ae
try:
    t = time.time(); import torch; torch.set_num_threads(2); mark(f"+ import torch {torch.__version__}", t)
    t = time.time(); from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu", torch_dtype=torch.float32)
    mark("+ chronos-bolt-tiny 로드", t)
    t = time.time()
    ctx = [torch.tensor(500 + 100 * np.sin(np.arange(288) / 20), dtype=torch.float32)] * 4
    for _ in range(5):
        pipe.predict_quantiles(ctx, prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])
    mark("+ chronos 추론 5회 (배치 4, ctx 24h)", t)
except Exception as e:
    res["chronos_error"] = f"{type(e).__name__}: {e}"
res["free_after_loaded"] = free_m()
res["final_rss_mb"] = rss()
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1); print("saved", a.out)
