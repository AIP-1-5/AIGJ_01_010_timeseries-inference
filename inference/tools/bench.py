"""과제 7(및 과제 6 비용) 벤치마크 — 보드에서 실행.

    .venv-bench/bin/python tools/bench.py --out reports/bench_board.json

측정
- 실행 방식: PyTorch CPU, (있으면) PyTorch GPU, ONNX Runtime CPU, (있으면) ONNX Runtime GPU/TensorRT
- 모델: GRU / LSTM / TCN (과제 3), 배치 1(한 공간) · 4(네 공간 한꺼번에)
- 지연: 워밍업 50회 후 500회 → 평균·p95 (ms)
- 결과 차이: PyTorch 대비 ONNX 출력 차이를 원래 단위(ppm)로 (정규화 값 × CO₂ 표준편차)
- 메모리: 프로세스 최대 RSS, 전력: tegrastats VDD_IN 평균 (측정 구간 동안)
- Chronos-Bolt tiny 1회 추론 시간·메모리 (과제 6 비용)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WARM, ITERS = 50, 500


class Power:
    """tegrastats 를 백그라운드로 돌려 VDD_IN(mW)·RAM·GPU 사용률을 모은다."""

    def __init__(self):
        self.samples, self.proc, self.th = [], None, None

    def start(self):
        self.samples = []
        try:
            self.proc = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=subprocess.PIPE, text=True)
        except FileNotFoundError:
            return
        self.th = threading.Thread(target=self._read, daemon=True)
        self.th.start()

    def _read(self):
        for line in self.proc.stdout:
            m = re.search(r"VDD_IN (\d+)mW", line)
            g = re.search(r"GR3D_FREQ (\d+)%", line)
            r = re.search(r"RAM (\d+)/", line)
            if m:
                self.samples.append((int(m.group(1)), int(g.group(1)) if g else None, int(r.group(1)) if r else None))

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(2)
        if not self.samples:
            return {}
        a = np.array([s[0] for s in self.samples])
        gpu = [s[1] for s in self.samples if s[1] is not None]
        ram = [s[2] for s in self.samples if s[2] is not None]
        return {"power_mw_mean": float(a.mean()), "gpu_util_mean": float(np.mean(gpu)) if gpu else None,
                "ram_mb_max": int(max(ram)) if ram else None}


def timeit(fn):
    for _ in range(WARM):
        fn()
    ts = []
    for _ in range(ITERS):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {"mean_ms": float(ts.mean()), "p95_ms": float(np.percentile(ts, 95)), "p99_ms": float(np.percentile(ts, 99))}


def inputs(meta, n_in, b, rng):
    return (rng.standard_normal((b, meta["lookback"], n_in)).astype(np.float32),
            np.eye(len(meta["spaces"]), dtype=np.float32)[np.arange(b) % len(meta["spaces"])],
            rng.random((b, 12)).astype(np.float32))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/bench_board.json")
    ap.add_argument("--chronos", default="amazon/chronos-bolt-tiny")
    ap.add_argument("--skip-chronos", action="store_true")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    meta = json.load(open("models/forecast_meta.json", encoding="utf-8"))
    co2_sd = float(np.mean([v[1] for k, v in meta["stats"].items() if k.endswith("|co2")]))
    res = {"host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "warmup": WARM, "iters": ITERS, "rows": []}
    try:
        res["nvpmodel"] = open("/etc/nvpmodel.conf").read()[:0] or subprocess.run(
            ["cat", "/var/lib/nvpmodel/status"], capture_output=True, text=True).stdout.strip()
    except Exception:
        res["nvpmodel"] = None
    pw = Power()

    try:
        import torch
        from tsinf import forecast as fc
        torch.set_num_threads(os.cpu_count())
        have_torch = True
        res["torch"] = torch.__version__
        res["cuda"] = torch.cuda.is_available()
    except ImportError:
        have_torch = False
    import onnxruntime as ort
    res["onnxruntime"] = ort.__version__
    res["ort_providers"] = ort.get_available_providers()

    for tag in ("GRU", "LSTM", "TCN"):
        g = meta["models"][tag]
        kind, n_in = g["kind"], g["n_in"]
        ref = {}
        if have_torch:
            m = fc.make_model(kind, n_in, len(meta["spaces"]))
            m.load_state_dict(torch.load(g["file"], map_location="cpu"))
            m.eval()
            devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
            for dev in devices:
                md = m.to(dev)
                for b in (1, 4):
                    x = [torch.from_numpy(a).to(dev) for a in inputs(meta, n_in, b, np.random.default_rng(b))]
                    with torch.no_grad():
                        ref[b] = md(*x).cpu().numpy() if dev == "cpu" else ref.get(b)

                        def f():
                            md(*x)
                            if dev == "cuda":
                                torch.cuda.synchronize()
                        pw.start(); t = timeit(f); p = pw.stop()
                    res["rows"].append({"model": tag, "runtime": f"PyTorch {dev.upper()}", "batch": b, **t, **p})
                    print(tag, f"PyTorch {dev}", b, t, p)
        sess_opts = ort.SessionOptions()
        for prov in [p for p in ("CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider") if p in ort.get_available_providers()]:
            s = ort.InferenceSession(f"models/forecast_{kind}.onnx", sess_opts, providers=[prov])
            for b in (1, 4):
                xs = inputs(meta, n_in, b, np.random.default_rng(b))
                feed = {"x_seq": xs[0], "x_static": xs[1], "x_future": xs[2]}
                out = s.run(None, feed)[0]
                diff = float(np.abs(out[..., 0] - ref[b][..., 0]).max() * co2_sd) if b in ref else None
                pw.start(); t = timeit(lambda: s.run(None, feed)); p = pw.stop()
                name = {"CPUExecutionProvider": "ONNX Runtime CPU", "CUDAExecutionProvider": "ONNX Runtime CUDA",
                        "TensorrtExecutionProvider": "ONNX Runtime TensorRT"}[prov]
                res["rows"].append({"model": tag, "runtime": name, "batch": b, **t, **p, "max_co2_diff_ppm_vs_torch": diff})
                print(tag, name, b, t, p, "diff", diff)

    # 아무것도 안 할 때 전력 (기준선)
    pw.start(); time.sleep(5); res["idle"] = pw.stop()

    if not args.skip_chronos:
        try:
            import torch
            from chronos import BaseChronosPipeline
            rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            pipe = BaseChronosPipeline.from_pretrained(args.chronos, device_map="cpu", torch_dtype=torch.float32)
            ctx = [torch.tensor(500 + 100 * np.sin(np.arange(288) / 20), dtype=torch.float32)]
            for b in (1, 4):
                c = ctx * b
                pw.start()
                t = timeit(lambda: pipe.predict_quantiles(c, prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])) if b == 1 else \
                    timeit(lambda: pipe.predict_quantiles(c, prediction_length=12, quantile_levels=[0.1, 0.5, 0.9]))
                p = pw.stop()
                res["rows"].append({"model": args.chronos.split("/")[-1] + " (ctx 24h)", "runtime": "PyTorch CPU", "batch": b, **t, **p})
                print("chronos", b, t, p)
            res["chronos_rss_increase_mb"] = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 1024
        except Exception as e:
            res["chronos_error"] = f"{type(e).__name__}: {e}"
    res["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
