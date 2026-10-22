"""내 수집기 데이터(보드 B, 2026-10-20 17:51~)로 과제 3·4·5·6 을 평가한다.

    .venv/bin/python task_own.py --db data/sensors_boardB.db

- 과제 3: 이미 만든 모델(실데이터 학습 / 합성 학습)과 기준선을 내 데이터에서 비교 (내 데이터는 학습에 쓰지 않음)
- 과제 4: 탐지기(규칙)를 내 데이터 1분 원본에 돌려 경보 목록과 원인 확인 (정답이 없으므로 직접 확인)
- 과제 5: 환기 선제 경보를 내 데이터로 평가 (정답 '30분 안 1,000ppm 초과'는 데이터에서 자동으로 만들 수 있음)
         사용 여부는 직접 관찰 기록(data/own_observed.csv)이 있으면 평가
- 과제 6: Chronos-Bolt tiny 를 내 데이터에 바로 적용 (학습 불필요)
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import onnxruntime as ort
import pandas as pd

from tsinf import anomaly as an
from tsinf import forecast as fc
from tsinf import prep

HZ = {"5분": 0, "15분": 2, "30분": 5, "60분": 11}


def md(d, fmt="{:.1f}"):
    d = d.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "-" if pd.isna(v) else fmt.format(v))
    return d.to_markdown()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sensors_boardB.db")
    ap.add_argument("--out", default="reports/own/own_results.md")
    ap.add_argument("--no-chronos", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    raw = prep.load_readings(args.db)
    df, st = prep.clean(raw)
    grid = prep.to_grid(df)
    spaces = sorted({s for s, m in grid.columns if all((s, x) in grid.columns for x in prep.METRICS)})
    t0, t1 = grid.index[0], grid.index[-1]
    hours = (t1 - t0).total_seconds() / 3600
    # 정규화 통계: 앞 절반으로만 (평가 구간의 정보를 미리 쓰지 않도록)
    half = grid.index[: len(grid) // 2]
    stats = prep.norm_stats(grid, half)
    sched = fc.schedule_grid(None, grid.index, spaces, {})
    eval_times = grid.index[24:]
    d = fc.build_samples(grid, sched, fc.DatasetSpec(spaces, 24, False, stats, {}, space_onehot=False), eval_times)
    true = d["base"][:, None, :] + d["y"] * d["sd"][:, None, :]
    lines = [
        "# 내 수집기 데이터로 평가 (과제 3·4·5·6)",
        "",
        f"> 측정 조건: 보드 B 수집기 실데이터 {t0:%m-%d %H:%M} ~ {t1:%m-%d %H:%M} ({hours:.1f}시간) · 공간 {', '.join(spaces)} · 5분 간격 · "
        f"원본 {st.raw:,}건 → 정제 {st.kept:,}건 · 이 데이터는 어떤 모델의 학습에도 쓰지 않았다",
        "",
    ]

    # ------------------------------------------------ 과제 3
    rows, preds = [], {}

    def add(name, pred):
        preds[name] = pred
        err = np.abs(pred - true)
        rows.append({"model": name, **{f"CO₂ {h}": float(np.nanmean(err[:, k, 0])) for h, k in HZ.items()},
                     "온도 60분": float(np.nanmean(err[:, 11, 1]))})

    for name, pred in fc.baseline_preds(grid, d).items():
        add(name, pred)
    for name, path in (("실데이터 학습 GRU (서비스 모델)", "models/service_gru.onnx"),
                       ("합성 데이터 학습 GRU", "models/service_gru_synth.onnx")):
        if os.path.exists(path):
            s = ort.InferenceSession(path)
            dz = s.run(None, {"x_seq": d["x_seq"], "x_static": d["x_static"], "x_future": d["x_future"]})[0]
            add(name, d["base"][:, None, :] + dz * d["sd"][:, None, :])
    t3 = pd.DataFrame(rows).set_index("model")
    p60 = t3.loc["현재값 유지", "CO₂ 60분"]
    t3["60분 개선율(%)"] = (1 - t3["CO₂ 60분"] / p60) * 100
    lines += ["## 과제 3. 60분 예측 — 내 데이터에서 평가 (CO₂ MAE ppm, 온도 ℃)", md(t3), "",
              f"- 샘플 {len(d['y']):,}개. '어제 같은 시각'은 24시간 이전 값이 있는 시점부터 계산된다.", ""]

    # ------------------------------------------------ 과제 6 (Chronos)
    if not args.no_chronos:
        try:
            import torch
            from chronos import BaseChronosPipeline
            pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu", torch_dtype=torch.float32)
            pos = pd.Series(np.arange(len(grid)), index=grid.index)
            sel = np.arange(0, len(d["y"]), 3)               # 15분 간격으로 뽑기
            ctx = []
            for n in sel:
                i = pos[d["time"][n]]
                s = grid[(d["space"][n], "co2")].to_numpy()[max(0, i - 287):i + 1]
                ctx.append(torch.tensor(pd.Series(s).interpolate(limit_direction="both").to_numpy(), dtype=torch.float32))
            q, _ = pipe.predict_quantiles(ctx, prediction_length=12, quantile_levels=[0.1, 0.5, 0.9])
            q = q.numpy()
            tt = true[sel, :, 0]
            cov = ((tt >= q[:, :, 0]) & (tt <= q[:, :, 2])).mean(axis=0) * 100
            per = np.abs(d["base"][sel, None, 0] - tt)
            gru_row = t3.loc["실데이터 학습 GRU (서비스 모델)"] if "실데이터 학습 GRU (서비스 모델)" in t3.index else None
            gp = preds.get("실데이터 학습 GRU (서비스 모델)")
            t6 = pd.DataFrame({
                "현재값 유지": {h: float(per[:, k].mean()) for h, k in HZ.items()},
                **({"실데이터 학습 GRU (같은 시점)": {h: float(np.abs(gp[sel, k, 0] - tt[:, k]).mean()) for h, k in HZ.items()}} if gp is not None else {}),
                "Chronos-Bolt tiny (최대 24h 과거)": {h: float(np.abs(q[:, k, 1] - tt[:, k]).mean()) for h, k in HZ.items()},
                "10~90% 구간 적중률(%)": {h: float(cov[k]) for h, k in HZ.items()},
            }).T
            lines += ["## 과제 6. Chronos — 내 데이터에 바로 적용 (CO₂ MAE ppm)", md(t6), "",
                      f"- 15분 간격 {len(sel)}개 시점. 학습 없이 내 교실 데이터에 적용한 결과.", ""]
        except Exception as e:
            lines += [f"## 과제 6\n- Chronos 실행 실패: {e}", ""]

    # ------------------------------------------------ 과제 5 (환기 선제 경보)
    with open("models/state_models.pkl", "rb") as fh:
        sm = pickle.load(fh)
    from service.infer import state_features
    vrows = []
    for sp in spaces:
        gs = grid[[(sp, m) for m in prep.METRICS]].droplevel(0, axis=1)
        f = state_features(gs, len(sm["spaces"]))
        for k in (0, 1, 3, 6):
            f[f"plan_{k}"] = 0.0
        c = gs["co2"]
        fut = pd.concat([c.shift(-k) for k in range(1, 7)], axis=1).max(axis=1)
        f["y"] = ((c < 1000) & (fut >= 1000)).astype(int)     # 공간 평균 기준
        f["valid"] = (c < 1000) & fut.notna()
        f["space"] = sp
        vrows.append(f)
    V = pd.concat(vrows)
    V = V[V["valid"] & V[sm["vent_features"]].notna().all(axis=1)]
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    y = V["y"].to_numpy()
    pm = (sm["vent"].predict_proba(V[sm["vent_features"]])[:, 1] >= sm["vent_threshold"]).astype(int)
    pr = (V["co2"].to_numpy() >= 850).astype(int)
    t5 = pd.DataFrame({
        name: {"재현율": recall_score(y, p, zero_division=0) * 100, "정밀도": precision_score(y, p, zero_division=0) * 100,
               "macro-F1": f1_score(y, p, average="macro", zero_division=0) * 100}
        for name, p in (("분류 모델 (합성 학습)", pm), ("단순 규칙 (CO₂ ≥ 850)", pr))}).T
    cm = confusion_matrix(y, pm, labels=[0, 1])
    lines += ["## 과제 5. 환기 선제 경보 — 내 데이터에서 평가", f"- 시점 {len(y):,}개 중 양성(30분 안 1,000ppm 초과) {int(y.sum())}개 ({y.mean() * 100:.1f}%)",
              ("- ⚠️ 이 기간 공간 평균 CO₂ 가 1,000ppm 을 넘은 적이 없어 양성이 없다 → 재현율은 계산할 수 없고, '헛경보를 내지 않았는가'만 확인 가능. "
               "데이터가 더 쌓이면(수업이 많은 날) 다시 평가한다." if y.sum() == 0 else ""),
              md(t5), "", f"- 혼동 행렬(모델): 경보 없음→경보 없음 {cm[0, 0]}, 헛경보 {cm[0, 1]}, 놓침 {cm[1, 0]}, 맞힘 {cm[1, 1]}", ""]

    obs_path = "data/own_observed.csv"
    if os.path.exists(obs_path):
        ob = pd.read_csv(obs_path, parse_dates=["start", "end"])
        src = ob["source"].iloc[0] if "source" in ob.columns else "직접 관찰"
        lines += [f"### 사용 여부 (정답: {src})",
                  "- 직접 관찰 기록이 없어, 실습실 강단의 소음 센서(LAeq)로 대리 정답을 만들었다: 5분 최대 소음의 15분 중앙값 ≥ 34.5dB 이면 '사용중' "
                  "(빈 밤 약 32dB, 수업 중 35~37dB). 소음 센서는 모델 입력에 쓰지 않으므로 정답으로 쓸 수 있다. 다만 조용히 자습하는 시간은 '비어있음'으로 잡힐 수 있다."]
        o_rows = []
        for sp in spaces:
            gs = grid[[(sp, m) for m in prep.METRICS]].droplevel(0, axis=1)
            f = state_features(gs, len(sm["spaces"]))
            truth = pd.Series(np.nan, index=grid.index)
            for r in ob[ob.space == sp].itertuples():
                s, e = pd.Timestamp(r.start).tz_localize(prep.TZ), pd.Timestamp(r.end).tz_localize(prep.TZ)
                truth[(truth.index >= s) & (truth.index < e)] = float(r.occupied)
            m = truth.notna() & f[sm["occ_features"]].notna().all(axis=1)
            if m.sum():
                p = (sm["occupancy"].predict_proba(f.loc[m, sm["occ_features"]])[:, 1] >= 0.5).astype(int)
                rule = (f.loc[m, "dco2_3"] >= 30).astype(int).to_numpy()
                o_rows.append({"공간": sp, "칸 수": int(m.sum()), "모델 정확도": (p == truth[m]).mean() * 100,
                               "규칙 정확도": (rule == truth[m]).mean() * 100})
        lines += [md(pd.DataFrame(o_rows).set_index("공간")) if o_rows else "- 관찰 기록과 겹치는 칸 없음", ""]
    else:
        lines += ["- 사용 여부: 직접 관찰 기록(data/own_observed.csv: space,start,end,occupied)이 생기면 평가", ""]

    # ------------------------------------------------ 과제 4 (탐지기 실행)
    raw1 = (df.set_index("time").groupby(["space", "metric"])["value"].resample("1min").mean().unstack(["space", "metric"]))
    ev_rows = []
    for sp in sorted(raw1.columns.get_level_values(0).unique()):
        r = raw1[sp].reindex(columns=[m for m in an.FEAT if m in raw1[sp].columns])
        ra = an.rule_alarms(r)
        for col in ra.columns:
            for (s, e) in an.to_events(ra[col]):
                ev_rows.append({"공간": sp, "경보": col.replace("rule_", ""), "시작": f"{s:%m-%d %H:%M}", "끝": f"{e:%m-%d %H:%M}"})
    lines += ["## 과제 4. 탐지기(규칙)를 내 데이터에 실행 — 정답이 없어 직접 확인",
              (pd.DataFrame(ev_rows).to_markdown(index=False) if ev_rows else "- 규칙 경보 없음 (튄 값·값 고정 없음)"), "",
              "- 설비 고장(예측 오차) 경보는 상시 서비스가 실시간으로 계산한다 → 과제 4 보고서 §9 실데이터 운영 점검 참고", ""]

    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
