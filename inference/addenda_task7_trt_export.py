"""과제 7: TensorRT 비교용 합성 시험 주 입력을 npz 로 저장 (보드로 복사해 같은 입력으로 비교)."""
import json
import numpy as np
import pandas as pd
from tsinf import forecast as fc, prep
from tsinf.synth import SPACES

DATA = "data/synth"
meta = json.load(open("models/forecast_meta.json", encoding="utf-8"))
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
te_t = grid.index[grid.index >= t0 + pd.Timedelta(days=28)]
stats = prep.norm_stats(grid, tr_t)
te = fc.build_samples(grid, sched, fc.DatasetSpec(spaces, meta["lookback"], True, stats, cap), te_t, excl)
true = te["base"][:, None, :] + te["y"] * te["sd"][:, None, :]
np.savez_compressed("reports/addenda/task7_trt_inputs.npz", x_seq=te["x_seq"], x_static=te["x_static"], x_future=te["x_future"],
                    base=te["base"], sd=te["sd"], true=true)
print(len(te["y"]))
