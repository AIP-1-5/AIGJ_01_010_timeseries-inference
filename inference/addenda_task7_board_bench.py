"""과제 7 보강 — 보드에서 FP32 vs INT8(동적/정적) ONNX Runtime CPU 지연 측정.
    nice -n 19 .venv/bin/python addenda_task7_board_bench.py --out reports/addenda/task7_int8_board_bench.json
tools/bench.py 와 같은 방식: 워밍업 50회 + 500회, 같은 무작위 입력(seed=batch), 기본 SessionOptions, CPU EP.
"""
import argparse, json, os, platform, subprocess, time
import numpy as np
import onnxruntime as ort

WARM, ITERS = 50, 500


def timeit(fn):
    for _ in range(WARM):
        fn()
    ts = []
    for _ in range(ITERS):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {"mean_ms": float(ts.mean()), "p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
            "p99_ms": float(np.percentile(ts, 99))}


def inputs(meta, n_in, b, rng):
    return (rng.standard_normal((b, meta["lookback"], n_in)).astype(np.float32),
            np.eye(len(meta["spaces"]), dtype=np.float32)[np.arange(b) % len(meta["spaces"])],
            rng.random((b, 12)).astype(np.float32))


ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
meta = json.load(open("models/forecast_meta.json", encoding="utf-8"))
res = {"host": platform.node(), "ort": ort.__version__, "providers": ort.get_available_providers(),
       "time": time.strftime("%Y-%m-%d %H:%M:%S"), "nice": os.nice(0), "warmup": WARM, "iters": ITERS,
       "loadavg_before": os.getloadavg(), "rows": []}
for tag in ("GRU", "LSTM", "TCN"):
    g = meta["models"][tag]; k = g["kind"]
    for var, path in (("FP32", f"models/forecast_{k}.onnx"), ("INT8 동적", f"models/int8/forecast_{k}_int8_dyn.onnx"),
                      ("INT8 정적", f"models/int8/forecast_{k}_int8_static.onnx")):
        try:
            s = ort.InferenceSession(path, ort.SessionOptions(), providers=["CPUExecutionProvider"])
        except Exception as e:
            res["rows"].append({"model": tag, "variant": var, "error": str(e)[:200]}); continue
        for b in (1, 4):
            xs = inputs(meta, g["n_in"], b, np.random.default_rng(b))
            feed = {"x_seq": xs[0], "x_static": xs[1], "x_future": xs[2]}
            t = timeit(lambda: s.run(None, feed))
            res["rows"].append({"model": tag, "variant": var, "batch": b, **t})
            print(tag, var, b, {kk: round(v, 3) for kk, v in t.items()}, flush=True)
res["loadavg_after"] = os.getloadavg()
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1); print("saved", a.out)
