"""과제 8 추론 한 주기: 수집기 DB → (과제 2 와 같은) 전처리 → 공간별 예측·이상·상태.

학습 때와 똑같은 전처리(tsinf.prep)를 그대로 쓴다. 전처리가 다르면 모델이 학습 때 본 적 없는 분포를 받아
예측이 조용히 틀어진다 (예: 학습은 5분 평균, 서비스는 1분 원본 → 잡음이 커져 엉뚱한 변화량 예측).
"""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
import pandas as pd

from tsinf import anomaly as an
from tsinf import forecast as fc
from tsinf import prep

SLUG = {"실습실": "lab", "사무실": "office", "회의실": "meeting", "사무실 밖": "office_outside", "복도": "corridor"}
STD_FLOOR = {"co2": 30.0, "temperature": 0.3, "humidity": 1.0}


def slug(space: str) -> str:
    return SLUG.get(space, "".join(ch if ch.isalnum() else "_" for ch in space))


def state_features(gs: pd.DataFrame, n_spaces: int) -> pd.DataFrame:
    """task5_state.state_features 와 같은 특징 (실제 공간은 학습 공간이 아니므로 공간 원-핫은 모두 0)."""
    c, t, h = gs["co2"], gs["temperature"], gs["humidity"]
    f = pd.DataFrame(index=gs.index)
    f["co2"] = c
    for k in (1, 3, 6):
        f[f"dco2_{k}"] = c - c.shift(k)
        f[f"dtemp_{k}"] = t - t.shift(k)
        f[f"dhum_{k}"] = h - h.shift(k)
    f["co2_above_min6h"] = c - c.rolling(72, min_periods=12).min()
    tf = fc.time_features(gs.index)
    for i, n in enumerate(["hs", "hc", "ds", "dc", "we"]):
        f[n] = tf[:, i]
    for j in range(n_spaces):
        f[f"sp{j}"] = 0.0
    return f


@dataclass
class Engine:
    model_dir: str
    sess: ort.InferenceSession = None
    meta: dict = None
    state: dict = None
    history: Dict[str, List[dict]] = field(default_factory=dict)     # 공간별 최근 예측 (예측 오차 탐지용)
    resid_run: Dict[str, int] = field(default_factory=dict)
    last_z: Dict[str, dict] = field(default_factory=dict)

    _mtime: float = 0.0

    def __post_init__(self):
        self._load()
        with open(os.path.join(self.model_dir, "state_models.pkl"), "rb") as fh:
            self.state = pickle.load(fh)

    def _load(self):
        """모델 파일이 바뀌면 다시 읽는다 (재학습한 모델을 서비스 재시작 없이 교체)."""
        path = os.path.join(self.model_dir, "service_gru.onnx")
        mt = os.path.getmtime(path)
        if mt != self._mtime:
            self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.meta = json.load(open(os.path.join(self.model_dir, "service_meta.json"), encoding="utf-8"))
            self._mtime = mt
            self.history.clear()

    # ------------------------------------------------------------------
    def run(self, db_path: str, now: Optional[pd.Timestamp] = None) -> dict:
        t_start = time.time()
        self._load()
        now = now or pd.Timestamp.now(tz=prep.TZ)
        raw = prep.load_readings(db_path, since=(now - pd.Timedelta(hours=26)).timestamp())
        out = {"generated_at": now.isoformat(), "spaces": {}, "errors": []}
        if raw.empty:
            out["errors"].append("최근 26시간 데이터 없음")
            return out
        df, cst = prep.clean(raw)
        grid_all = prep.to_grid(df)
        # 막 시작된 5분 칸은 1~2분치만 들어 있어 평균이 치우친다 → 다 찬 칸까지만 쓴다
        cutoff = now.floor("5min")
        grid = grid_all[grid_all.index < cutoff]
        raw1 = (df.set_index("time").groupby(["space", "metric"])["value"].resample("1min").mean()
                  .unstack(["space", "metric"]))
        last_seen = df.groupby("space")["recv_ts"].max()
        # 공간마다 평소 보고 주기 (회의실 센서는 10분 주기 → 10분 조용한 건 정상)
        intervals = (df.sort_values("recv_ts").groupby(["space", "device"])["recv_ts"]
                       .apply(lambda s: float(np.median(np.diff(s.unique()))) if s.nunique() > 2 else 600.0)
                       .groupby("space").min())
        collector_age = now.timestamp() - float(raw["recv_ts"].max())
        out["collector_last_recv_age_sec"] = round(collector_age, 1)

        spaces = sorted({c[0] for c in grid.columns})
        for sp in spaces:
            try:
                out["spaces"][sp] = self._space(sp, grid, raw1, raw, last_seen.get(sp), now,
                                                collector_age, float(intervals.get(sp, 600.0)))
            except Exception as e:     # 한 공간이 실패해도 다른 공간은 계속
                out["errors"].append(f"{sp}: {type(e).__name__}: {e}")
        out["processing_ms"] = round((time.time() - t_start) * 1000, 1)
        out["clean_stats"] = {"raw": cst.raw, "duplicates": cst.duplicates, "clock_fixed": cst.clock_fixed,
                              "out_of_range": cst.out_of_range}
        return out

    # ------------------------------------------------------------------
    def _space(self, sp, grid, raw1, raw, last_recv, now, collector_age: float = 0.0, interval: float = 600.0) -> dict:
        metrics = [m for m in prep.METRICS if (sp, m) in grid.columns]
        gs = grid[[(sp, m) for m in metrics]].droplevel(0, axis=1)
        latest_t = gs.dropna(how="all").index.max()
        latest = {m: (None if pd.isna(gs[m].get(latest_t)) else round(float(gs[m].get(latest_t)), 2)) for m in metrics}
        age = None if last_recv is None else round(now.timestamp() - float(last_recv), 1)
        res = {
            "space": sp, "slug": slug(sp), "time": None if latest_t is pd.NaT else latest_t.isoformat(),
            "latest": latest, "data_age_sec": age,
            "units": {"co2": "ppm", "temperature": "°C", "humidity": "%"},
        }

        # ---------- 이상 (규칙)
        rules = []
        # 데이터 끊김: (1) 수집기 전체가 3분 넘게 아무것도 못 받음 → 수집기·브로커·네트워크 문제
        #             (2) 이 공간만 평소 주기의 3배(최소 15분) 넘게 조용함 → 센서·게이트웨이 문제
        gap_limit = max(900.0, 3 * interval)
        if collector_age > 180:
            rules.append({"type": "data_gap", "metric": None,
                          "detail": f"수집기 전체 수신 없음 {collector_age:.0f}초 (수집기·브로커·네트워크 확인)"})
        elif age is None or age > gap_limit:
            rules.append({"type": "data_gap", "metric": None,
                          "detail": f"이 공간 마지막 수신 {age if age is not None else '-'}초 전 (평소 주기 {interval:.0f}초)"})
        recent = raw[(raw.space == sp) & (raw.recv_ts >= now.timestamp() - 3600)]
        for m, (lo, hi) in prep.VALID_RANGE.items():
            n_bad = int(((recent.metric == m) & ((recent.value < lo) | (recent.value > hi))).sum())
            if n_bad:
                rules.append({"type": "out_of_range", "metric": m, "detail": f"최근 1시간 {n_bad}건"})
        if sp in raw1.columns.get_level_values(0):
            r1 = raw1[sp].reindex(columns=[m for m in an.FEAT if m in raw1[sp].columns])
            r1 = r1[r1.index >= now - pd.Timedelta(hours=3)]
            ra = an.rule_alarms(r1)
            last10 = ra[ra.index >= now - pd.Timedelta(minutes=10)]
            for col in last10.columns[last10.any()]:
                kind, m = col.replace("rule_", "").split("_", 1)
                rules.append({"type": {"spike": "spike", "stuck": "stuck"}[kind], "metric": m, "detail": "최근 10분"})

        res["forecast"] = {"available": False, "step_min": 5, "horizon_min": 60}
        res["alerts"] = {"co2_over_1000_in_min": None, "ventilation_soon": None, "ventilation_prob": None,
                         "co2_already_over_1000": latest.get("co2") is not None and latest["co2"] >= 1000}
        res["state"] = {"occupied": None, "occupied_prob": None}
        resid_z = None
        if all(m in metrics for m in prep.METRICS):
            # ---------- 예측
            L = self.meta["lookback"]
            win = gs[list(prep.METRICS)].iloc[-L:]
            if len(win) == L and not win.isna().any().any() and win.index[-1] == latest_t:
                hist = gs[gs.index >= latest_t - pd.Timedelta(hours=24)]
                mu = hist.mean()
                sd = pd.Series({m: max(float(hist[m].std() or 0), STD_FLOOR[m]) for m in prep.METRICS})
                z = ((win - mu) / sd).to_numpy(np.float32)
                tf = fc.time_features(win.index)
                x = np.concatenate([z, tf, np.zeros((L, 1), np.float32)], axis=1)[None]
                dz = self.sess.run(None, {"x_seq": x, "x_static": np.ones((1, 1), np.float32),
                                          "x_future": np.zeros((1, 12), np.float32)})[0][0]
                base = win.iloc[-1]
                co2_f = (base["co2"] + dz[:, 0] * sd["co2"]).round(1)
                tmp_f = (base["temperature"] + dz[:, 1] * sd["temperature"]).round(2)
                times = [(latest_t + pd.Timedelta(minutes=5 * (k + 1))).isoformat() for k in range(12)]
                res["forecast"] = {"available": True, "step_min": 5, "horizon_min": 60, "base_time": latest_t.isoformat(),
                                   "times": times, "co2": co2_f.tolist(), "temperature": tmp_f.tolist(),
                                   "model": self.meta.get("model_name", "service_gru") + " (ONNX Runtime)"}
                if base["co2"] < 1000:
                    over = np.where(co2_f >= 1000)[0]
                    res["alerts"]["co2_over_1000_in_min"] = int((over[0] + 1) * 5) if len(over) else None

                # ---------- 예측 오차 (15분 전에 한 예측 vs 지금 실제), 한쪽(예상보다 높음)
                h = self.history.setdefault(sp, [])
                past = [p for p in h if p["base_time"] == latest_t - pd.Timedelta(minutes=15)]
                new_step = not h or h[-1]["base_time"] != latest_t
                if past and new_step:
                    p = past[-1]
                    r = np.array([(base["co2"] - p["co2"][2]) / sd["co2"], (base["temperature"] - p["temperature"][2]) / sd["temperature"]])
                    rs = np.array([self.meta["resid_sd_15min_norm"]["co2"], self.meta["resid_sd_15min_norm"]["temperature"]])
                    zz = np.clip(r / rs, 0, None)
                    resid_z = {"co2": round(float(zz[0]), 2), "temperature": round(float(zz[1]), 2)}
                    self.last_z[sp] = resid_z
                    # 5분 칸 기준 6칸(30분) 연속으로 예상보다 높아야 경보 (회의 시작처럼 잠깐 오르는 건 무시)
                    self.resid_run[sp] = self.resid_run.get(sp, 0) + 1 if zz.max() > 4.0 else 0
                resid_z = resid_z or self.last_z.get(sp)
                if not h or h[-1]["base_time"] != latest_t:
                    h.append({"base_time": latest_t, "co2": co2_f, "temperature": tmp_f})
                    del h[:-6]
            else:
                res["forecast"]["reason"] = "최근 2시간에 빈 칸이 있어 예측 생략"

            # ---------- 상태 (사용 여부, 환기 선제 경보)
            f = state_features(gs[list(prep.METRICS)], len(self.state["spaces"]))
            row = f.iloc[[-1]]
            if row.notna().all(axis=1).iloc[0]:
                po = float(self.state["occupancy"].predict_proba(row[self.state["occ_features"]])[0, 1])
                res["state"] = {"occupied": po >= 0.5, "occupied_prob": round(po, 3)}
                vf = row.copy()
                for k in (0, 1, 3, 6):
                    vf[f"plan_{k}"] = 0.0          # 실데이터에는 일정 자료 없음
                pv = float(self.state["vent"].predict_proba(vf[self.state["vent_features"]])[0, 1])
                res["alerts"]["ventilation_prob"] = round(pv, 3)
                res["alerts"]["ventilation_soon"] = bool(pv >= self.state["vent_threshold"] and latest["co2"] < 1000)

        model_alarm = self.resid_run.get(sp, 0) >= 6            # 5분 칸 6개(30분) 연속
        cause = None
        if model_alarm and resid_z:
            cause = max(resid_z, key=resid_z.get)
        elif rules:
            cause = rules[0]["metric"]
        res["anomaly"] = {
            "is_anomaly": bool(rules) or model_alarm,
            "score": None if resid_z is None else max(resid_z.values()),
            "model_alarm": model_alarm, "residual_z": resid_z, "cause": cause, "rules": rules,
            "data_gap": any(r["type"] == "data_gap" for r in rules),
        }
        return res
