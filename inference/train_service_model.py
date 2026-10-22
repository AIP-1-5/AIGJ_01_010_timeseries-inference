"""과제 8 서비스용 범용 예측 모델 학습 + ONNX 내보내기.

실데이터 공간(실습실·사무실·회의실)은 합성 데이터의 공간과 다르고 일정 자료도 없으므로,
'공간 구분 없음 + 일정 없음' GRU 를 학습한다. 각 공간 값은 그 공간의 최근 통계로 정규화해 넣는다.
    .venv/bin/python train_service_model.py
산출물: models/service_gru.pt, models/service_gru.onnx, models/service_meta.json
"""
import json

import numpy as np
import pandas as pd
import torch

from tsinf import forecast as fc
from tsinf import prep
from tsinf.synth import SPACES

df, _ = prep.clean(prep.load_readings("data/synth/sensors.db"))
grid = prep.to_grid(df)
spaces = [s.name for s in SPACES]
cap = {s.name: float(s.capacity) for s in SPACES}
t0 = grid.index[0].normalize()
tr_t = grid.index[grid.index < t0 + pd.Timedelta(days=21)]
va_t = grid.index[(grid.index >= t0 + pd.Timedelta(days=21)) & (grid.index < t0 + pd.Timedelta(days=28))]
te_t = grid.index[grid.index >= t0 + pd.Timedelta(days=28)]
stats = prep.norm_stats(grid, tr_t)
sched = fc.schedule_grid(None, grid.index, spaces, cap)
faults = pd.read_csv("data/synth/faults.csv", parse_dates=["start", "end"])
excl = {}
for r in faults.itertuples():
    excl.setdefault(r.space, []).append((pd.Timestamp(r.start), pd.Timestamp(r.end)))
spec = fc.DatasetSpec(spaces, 24, False, stats, cap, space_onehot=False)
tr = fc.build_samples(grid, sched, spec, tr_t)
va = fc.build_samples(grid, sched, spec, va_t)
te = fc.build_samples(grid, sched, spec, te_t, excl)
model, best = fc.train("gru", tr, va, epochs=40)
mae = fc.mae_by_h(fc.predict(model, te), te)
base = fc.mae_by_h(np.repeat(te["base"][:, None, :], 12, axis=1), te)
print(f"범용 GRU 60분 CO2 MAE {mae[11, 0]:.1f} (현재값 유지 {base[11, 0]:.1f}), 온도 {mae[11, 1]:.2f} ({base[11, 1]:.2f})")
torch.save(model.state_dict(), "models/service_gru.pt")
model.eval()
n_in = tr["x_seq"].shape[2]
dummy = (torch.zeros(1, 24, n_in), torch.ones(1, 1), torch.zeros(1, 12))
torch.onnx.export(model, dummy, "models/service_gru.onnx", input_names=["x_seq", "x_static", "x_future"],
                  output_names=["dz"], dynamic_axes={"x_seq": {0: "b"}, "x_static": {0: "b"}, "x_future": {0: "b"}, "dz": {0: "b"}},
                  opset_version=17, dynamo=False)
json.dump({"lookback": 24, "horizon": 12, "n_in": int(n_in), "features": list(fc.FEAT), "targets": list(fc.TARGETS),
           "test_mae_60min": {"co2": float(mae[11, 0]), "temperature": float(mae[11, 1])},
           "test_mae_persistence_60min": {"co2": float(base[11, 0]), "temperature": float(base[11, 1])}},
          open("models/service_meta.json", "w"), ensure_ascii=False, indent=1)
print("saved models/service_gru.onnx")
