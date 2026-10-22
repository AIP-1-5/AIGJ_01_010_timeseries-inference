"""과제 7: TensorRT (FP32·FP16) 변환과 측정 — 보드 B, sudo 없이 pip(tensorrt-cu13) 로 내 폴더에만 설치한 환경(.venv-trt).

    nice -n 19 .venv-trt/bin/python addenda_task7_trt.py
같은 합성 시험 주 입력(reports/addenda/task7_trt_inputs.npz)으로 ONNX Runtime CPU 와 결과·속도를 비교한다.
"""
import json
import os
import platform
import time

import numpy as np
import onnxruntime as ort
import tensorrt as trt
from cuda.bindings import runtime as cudart

LOG = trt.Logger(trt.Logger.WARNING)
D = np.load("reports/addenda/task7_trt_inputs.npz")
NAMES = ["x_seq", "x_static", "x_future"]
MODELS = ["forecast_gru", "forecast_lstm", "forecast_tcn"]
os.makedirs("models/trt", exist_ok=True)


def ck(r):
    err = r[0] if isinstance(r, tuple) else r
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return r[1] if isinstance(r, tuple) and len(r) > 1 else None


def build(onnx_path, fp16, max_b=1024):
    b = trt.Builder(LOG)
    data = open(onnx_path, "rb").read()
    flags = 0
    if fp16:
        # TensorRT 11 은 FP16 빌더 플래그가 없어졌다 → ONNX 를 FP16 으로 바꾼 뒤 '강한 타입' 네트워크로 빌드 (입출력은 FP32 유지)
        import onnx
        from onnxconverter_common import float16
        data = float16.convert_float_to_float16(onnx.load(onnx_path), keep_io_types=True).SerializeToString()
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    net = b.create_network(flags)
    p = trt.OnnxParser(net, LOG)
    if not p.parse(data):
        raise RuntimeError([str(p.get_error(i)) for i in range(p.num_errors)])
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)     # 공용 보드: 작업 메모리 256MB 로 제한
    prof = b.create_optimization_profile()
    for i in range(net.num_inputs):
        t = net.get_input(i)
        shp = list(t.shape)
        mk = lambda n: tuple([n] + shp[1:])
        prof.set_shape(t.name, mk(1), mk(4), mk(max_b))
    cfg.add_optimization_profile(prof)
    t0 = time.time()
    ser = b.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError("build failed")
    return bytes(ser), time.time() - t0


class Runner:
    def __init__(self, eng_bytes, max_b=1024):
        self.rt = trt.Runtime(LOG)
        self.eng = self.rt.deserialize_cuda_engine(eng_bytes)
        self.ctx = self.eng.create_execution_context()
        self.stream = ck(cudart.cudaStreamCreate())
        self.names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.out = [n for n in self.names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
        self.buf = {}
        for n in self.names:
            shp = list(self.eng.get_tensor_shape(n)); shp[0] = max_b
            nbytes = int(np.prod(shp)) * 4
            self.buf[n] = ck(cudart.cudaMalloc(nbytes))

    def __call__(self, feeds):
        bsz = len(feeds["x_seq"])
        for n in NAMES:
            a = np.ascontiguousarray(feeds[n], np.float32)
            self.ctx.set_input_shape(n, a.shape)
            ck(cudart.cudaMemcpyAsync(self.buf[n], a.ctypes.data, a.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        for n in self.names:
            self.ctx.set_tensor_address(n, self.buf[n])
        self.ctx.execute_async_v3(self.stream)
        out = np.empty((bsz, 12, 2), np.float32)
        ck(cudart.cudaMemcpyAsync(out.ctypes.data, self.buf[self.out], out.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        ck(cudart.cudaStreamSynchronize(self.stream))
        return out


def all_out(fn):
    return np.concatenate([fn({n: D[n][i:i + 1024] for n in NAMES}) for i in range(0, len(D["x_seq"]), 1024)])


def bench(fn, bsz, warm=50, iters=500):
    feeds = {n: D[n][:bsz] for n in NAMES}
    for _ in range(warm):
        fn(feeds)
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(feeds); ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "mean_ms": float(ts.mean())}


rows = []
ppm = lambda dz: D["base"][:, None, :] + dz * D["sd"][:, None, :]
true = D["true"]
for m in MODELS:
    path = f"models/{m}.onnx"
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    ort_fn = lambda f: s.run(None, f)[0]
    ref = ppm(all_out(ort_fn))
    rows.append({"model": m, "runtime": "ONNX Runtime CPU", "mae60_co2": float(np.abs(ref - true)[:, 11, 0].mean()),
                 "max_diff_ppm_vs_ort": 0.0, **{f"b{b}_{k}": v for b in (1, 4) for k, v in bench(ort_fn, b).items()}})
    for fp16 in (False, True):
        eng, sec = build(path, fp16)
        open(f"models/trt/{m}_{'fp16' if fp16 else 'fp32'}.engine", "wb").write(eng)
        r = Runner(eng)
        out = ppm(all_out(r))
        rows.append({"model": m, "runtime": f"TensorRT {'FP16' if fp16 else 'FP32'}", "build_sec": round(sec, 1),
                     "engine_kb": round(len(eng) / 1024), "mae60_co2": float(np.abs(out - true)[:, 11, 0].mean()),
                     "max_diff_ppm_vs_ort": float(np.abs(out - ref)[:, :, 0].max()),
                     **{f"b{b}_{k}": v for b in (1, 4) for k, v in bench(r, b).items()}})
        print(rows[-1], flush=True)
res = {"host": platform.node(), "trt": trt.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_test": int(len(true)),
       "protocol": "warmup 50 + 500, 입력 복사(H2D)·출력 복사(D2H) 포함, nice 19, ORT intra_op 2스레드", "rows": rows}
json.dump(res, open("reports/addenda/task7_trt_board.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False)[:3000])
