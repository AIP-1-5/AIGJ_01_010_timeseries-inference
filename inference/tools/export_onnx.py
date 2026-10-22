"""과제 3 모델(GRU/LSTM/TCN)을 ONNX 로 내보낸다 (과제 7: ONNX Runtime·TensorRT 변환의 출발점).
    .venv/bin/python tools/export_onnx.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsinf import forecast as fc  # noqa: E402

meta = json.load(open("models/forecast_meta.json", encoding="utf-8"))
n_static = len(meta["spaces"])
for tag in ("GRU", "LSTM", "TCN"):
    g = meta["models"][tag]
    m = fc.make_model(g["kind"], g["n_in"], n_static)
    m.load_state_dict(torch.load(g["file"]))
    m.eval()
    x = (torch.zeros(1, meta["lookback"], g["n_in"]), torch.zeros(1, n_static), torch.zeros(1, 12))
    out = f"models/forecast_{g['kind']}.onnx"
    torch.onnx.export(m, x, out, input_names=["x_seq", "x_static", "x_future"], output_names=["dz"],
                      dynamic_axes={k: {0: "b"} for k in ("x_seq", "x_static", "x_future", "dz")}, opset_version=17, dynamo=False)
    print("saved", out)
