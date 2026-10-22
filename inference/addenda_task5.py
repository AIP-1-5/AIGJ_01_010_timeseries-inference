"""과제 5 보강 (주의: 합성 데이터 11벌을 한꺼번에 올려 메모리 약 1GB — 다른 무거운 작업과 동시에 돌리지 말 것) — 회의실 노쇼·예약 없는 사용 탐지, 오판율 임계값 조정, 공간 하나 빼고 학습(LOSO), 실데이터 시간표 점검.

    .venv/bin/python addenda_task5.py            # → reports/addenda/task5_addendum.md

- 사용 여부 모델은 task5_state.py 가 저장한 models/state_models.pkl 을 그대로 쓴다 (원본 data/synth 1~3주, 일정 기반 정답으로 학습).
- 정답은 생성기의 '실제 재실 인원'(occupancy.csv, 1분) → 직접 관찰 6일보다 훨씬 많은 시간을 평가할 수 있다.
- 노쇼·예약 없는 사용은 원본 1주(검증 주·시험 주) 만으로는 건수가 몇 건뿐이라, 같은 생성기를 씨앗만 바꿔 더 만든다
  (data/synth_extra/s1xx = 검증용, s2xx = 시험용). 모델은 이 데이터를 한 번도 보지 않았다.
- 파라미터(임계값·창 길이 등)는 검증 데이터에서만 고르고, 시험 데이터는 보고만 한다.
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from task5_state import state_features
from tsinf import prep
from tsinf.synth import SPACES

SPACE_NAMES = [s.name for s in SPACES]
VAL_SEEDS = (101, 102, 103, 104)
TEST_SEEDS = (201, 202, 203, 204, 205, 206)
SLOT = pd.Timedelta(minutes=5)


# ----------------------------------------------------------------------------- 데이터
def load_set(d: str, tag: str) -> dict:
    df, _ = prep.clean(prep.load_readings(f"{d}/sensors.db"))
    grid = prep.to_grid(df)
    sched = pd.read_csv(f"{d}/schedule.csv", parse_dates=["start", "end"])
    for c in ("start", "end"):
        sched[c] = pd.to_datetime(sched[c], utc=True).dt.tz_convert(prep.TZ)
    occ = pd.read_csv(f"{d}/occupancy.csv", index_col=0)
    occ.index = pd.to_datetime(occ.index, utc=True).tz_convert(prep.TZ)
    occ5 = (occ.resample("5min").max() > 0).astype(int).reindex(grid.index)
    faults = pd.read_csv(f"{d}/faults.csv", parse_dates=["start", "end"])
    fmask = {sp: np.zeros(len(grid), bool) for sp in SPACE_NAMES}
    for f in faults.itertuples():
        s = pd.Timestamp(f.start).tz_convert(prep.TZ); e = pd.Timestamp(f.end).tz_convert(prep.TZ)
        fmask[f.space] |= (grid.index >= s - pd.Timedelta(minutes=30)) & (grid.index <= e + pd.Timedelta(minutes=60))
    feats = {}
    for si, sp in enumerate(SPACE_NAMES):
        gs = grid[[(sp, m) for m in prep.METRICS]].droplevel(0, axis=1)
        feats[sp] = state_features(gs, si, len(SPACE_NAMES))
    obs_days = set()
    if os.path.exists(f"{d}/observed.csv"):
        ob = pd.read_csv(f"{d}/observed.csv", index_col=0)
        obs_days = set(pd.to_datetime(ob.index, utc=True).tz_convert(prep.TZ).normalize())
    return {"tag": tag, "grid": grid, "sched": sched, "occ": occ, "occ5": occ5, "fmask": fmask, "feats": feats,
            "obs_days": obs_days, "t0": grid.index[0].normalize()}


def ensure_extra(root: str):
    for s in VAL_SEEDS + TEST_SEEDS:
        out = f"{root}/s{s}"
        if not os.path.exists(f"{out}/sensors.db"):
            subprocess.run([sys.executable, "-m", "tsinf.synth", "--out", out, "--seed", str(s)], check=True,
                           stdout=subprocess.DEVNULL)


def week_mask(S, w0, w1):
    idx = S["grid"].index
    return (idx >= S["t0"] + pd.Timedelta(days=7 * w0)) & (idx < S["t0"] + pd.Timedelta(days=7 * w1))


def slot_table(S, model, occ_feats, time_mask) -> pd.DataFrame:
    """평일 8~19시, 고장 구간 제외 칸들의 (공간, 시각, 정답, 모델 확률, 규칙)"""
    idx = S["grid"].index
    base = time_mask & (idx.dayofweek < 5) & (idx.hour >= 8) & (idx.hour < 19)
    rows = []
    for sp in SPACE_NAMES:
        f = S["feats"][sp]
        m = base & ~S["fmask"][sp] & f.notna().all(axis=1).to_numpy() & S["occ5"][sp].notna().to_numpy()
        if not m.any():
            continue
        rows.append(pd.DataFrame({"space": sp, "time": idx[m], "y": S["occ5"][sp].to_numpy()[m].astype(int),
                                  "p": model.predict_proba(f.loc[m, occ_feats])[:, 1],
                                  "rule": (f["dco2_3"].to_numpy()[m] >= 30).astype(int), "set": S["tag"]}))
    return pd.concat(rows, ignore_index=True)


def bin_metrics(y, p) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    tn = int(((y == 0) & (p == 0)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    return {"F1": f1_score(y, p, zero_division=0) * 100, "재현율": recall_score(y, p, zero_division=0) * 100,
            "정밀도": precision_score(y, p, zero_division=0) * 100, "빈 방→사용중 오판율": fp / max(tn + fp, 1) * 100,
            "정확도": (y == p).mean() * 100}


# ----------------------------------------------------------------------------- 노쇼 / 예약 없는 사용
def bookings(S, time_mask) -> pd.DataFrame:
    """회의실 예약별 정답(노쇼 여부)과 시작 후 칸별 모델 확률·CO₂ 변화."""
    idx = S["grid"].index
    occ1 = S["occ"]["회의실"]
    p_all = S["p_meeting"]
    f = S["feats"]["회의실"]
    fm = pd.Series(S["fmask"]["회의실"], index=idx)
    ok_t = pd.Series(time_mask, index=idx)
    rows = []
    for r in S["sched"][S["sched"].space == "회의실"].itertuples():
        st = r.start
        s0 = st.floor("5min")           # 시작 시각이 든 5분 칸. k=0..2 칸은 시작 후 15분 안에 끝난다 → 15분 판정에 쓸 수 있음
        if s0 not in ok_t.index or not ok_t[s0]:
            continue
        if fm[(fm.index >= s0 - SLOT) & (fm.index < s0 + pd.Timedelta(minutes=30))].any():
            continue
        first15 = occ1[(occ1.index >= st) & (occ1.index < st + pd.Timedelta(minutes=15))]
        whole = occ1[(occ1.index >= st) & (occ1.index < r.end)]
        slots = [s0 + k * SLOT for k in range(6)]
        if any(s not in p_all.index or np.isnan(p_all[s]) for s in slots):
            continue
        rows.append({"set": S["tag"], "start": st, "end": r.end, "noshow15": int(first15.max() == 0),
                     "noshow_all": int(whole.max() == 0),
                     **{f"p{k}": float(p_all[s]) for k, s in enumerate(slots)},
                     # 시작 직전 대비 CO₂ 상승 (단순 규칙용)
                     **{f"rise{k}": float(f["co2"][s] - f["co2"][s0 - SLOT]) if (s0 - SLOT) in f.index else np.nan
                        for k, s in enumerate(slots)}})
    return pd.DataFrame(rows)


def noshow_detect(B: pd.DataFrame, kind: str, n_slots: int, thr: float) -> np.ndarray:
    """kind='model': 시작 후 n_slots 칸의 모델 확률이 모두 thr 미만 → 노쇼
       kind='rule' : 시작 후 n_slots 칸 동안 CO₂ 상승(시작 직전 대비)의 최대가 thr ppm 미만 → 노쇼"""
    if kind == "model":
        return (B[[f"p{k}" for k in range(n_slots)]].max(axis=1) < thr).astype(int).to_numpy()
    return (B[[f"rise{k}" for k in range(n_slots)]].max(axis=1) < thr).astype(int).to_numpy()


def booked_mask(S, buffer_min: int) -> pd.Series:
    idx = S["grid"].index
    m = pd.Series(False, index=idx)
    for r in S["sched"][S["sched"].space == "회의실"].itertuples():
        m |= (idx >= r.start.floor("5min") - SLOT) & (idx < r.end + pd.Timedelta(minutes=buffer_min))
    return m


def runs(b: np.ndarray):
    out, i, n = [], 0, len(b)
    while i < n:
        if b[i]:
            j = i
            while j < n and b[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def unbooked_events(S, time_mask, thr, k_slots, buffer_min, p_or_rule="model"):
    """예약 없는 시간(예약 구간 + 끝난 뒤 buffer 분 제외)에 '사용중' 판정이 k_slots 칸 이상 이어지면 사건 1건.
    정답 사건: 실제 재실이 예약 구간(앞뒤 여유 없음)과 겹치지 않는 연속 구간."""
    idx = S["grid"].index
    work = time_mask & (idx.dayofweek < 5) & (idx.hour >= 8) & (idx.hour < 19) & ~S["fmask"]["회의실"]
    work &= S["feats"]["회의실"].notna().all(axis=1).to_numpy()
    y = S["occ5"]["회의실"].fillna(0).to_numpy().astype(bool)
    strict_booked = booked_mask(S, 0).to_numpy()
    # 정답 사건: 실제 재실 연속 구간 중 예약과 하나도 안 겹치는 것
    true_ev = [(a, b) for a, b in runs(y & work) if not strict_booked[a:b].any()]
    free = work & ~booked_mask(S, buffer_min).to_numpy()
    if p_or_rule == "model":
        pred = (S["p_meeting"].to_numpy() >= thr) & free
    else:
        pred = (S["feats"]["회의실"]["dco2_3"].to_numpy() >= thr) & free
    det = [(a, b) for a, b in runs(pred) if b - a >= k_slots]
    hit_true = [any(a < db and da < b for da, db in det) for a, b in true_ev]
    hit_det = [any(a < tb and ta < b for ta, tb in true_ev) for a, b in det]
    delays = []
    for a, b in true_ev:
        ds = [da for da, db in det if a < db and da < b]
        if ds:
            delays.append((max(min(ds), a) + k_slots - a) * 5)     # 판정 시각 = 연속 k칸 확인 시점
    return {"true": len(true_ev), "det": len(det), "tp_true": int(sum(hit_true)), "tp_det": int(sum(hit_det)),
            "delay": delays}


def ev_scores(rs):
    t = sum(r["true"] for r in rs); d = sum(r["det"] for r in rs)
    tpt = sum(r["tp_true"] for r in rs); tpd = sum(r["tp_det"] for r in rs)
    rec = tpt / t * 100 if t else np.nan; prec = tpd / d * 100 if d else np.nan
    f1 = 2 * rec * prec / (rec + prec) if (t and d and rec + prec > 0) else 0.0
    dl = [x for r in rs for x in r["delay"]]
    return {"실제 사건": t, "탐지 사건": d, "헛탐지": d - tpd, "재현율": rec, "정밀도": prec, "F1": f1,
            "판정 지연(분, 중앙값)": float(np.median(dl)) if dl else np.nan}


# ----------------------------------------------------------------------------- 출력 도우미
def md(d, fmt="{:.1f}"):
    d = d.copy()
    for c in d.columns:
        d[c] = d[c].map(lambda v: "-" if (isinstance(v, float) and np.isnan(v)) else
                        (fmt.format(v) if isinstance(v, (float, np.floating)) else v))
    return d.to_markdown()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--extra", default="data/synth_extra")
    ap.add_argument("--out", default="reports/addenda/task5_addendum.md")
    args = ap.parse_args(argv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sm = pickle.load(open("models/state_models.pkl", "rb"))
    model, occ_feats = sm["occupancy"], sm["occ_features"]
    ensure_extra(args.extra)

    main_S = load_set(args.data, "원본")
    extra = {s: load_set(f"{args.extra}/s{s}", f"s{s}") for s in VAL_SEEDS + TEST_SEEDS}
    for S in [main_S, *extra.values()]:
        f = S["feats"]["회의실"]; ok = f.notna().all(axis=1)
        p = pd.Series(np.nan, index=f.index)
        p[ok] = model.predict_proba(f.loc[ok, occ_feats])[:, 1]
        S["p_meeting"] = p

    n = len(main_S["grid"].index)
    all_t = np.ones(n, bool)
    va_t = week_mask(main_S, 3, 4); te_t = week_mask(main_S, 4, 5)
    val_sets = [(main_S, va_t)] + [(extra[s], np.ones(len(extra[s]["grid"]), bool)) for s in VAL_SEEDS]
    test_sets = [(main_S, te_t)] + [(extra[s], np.ones(len(extra[s]["grid"]), bool)) for s in TEST_SEEDS]

    L = ["# 과제 5 보강 — 노쇼·예약 없는 사용 탐지 · 오판율 임계값 · 새 공간 일반화 · 실데이터 시간표 점검", "",
         "> 측정 조건", "> - **합성 데이터**(tsinf.synth, 5분 칸, 평일 8~19시, 고장 구간 앞 30분·뒤 60분 제외). "
         "사용 여부 모델은 과제 5 그대로(models/state_models.pkl: 원본 씨앗 7의 1~3주, 일정 기반 정답, 센서 특징만).",
         "> - **정답 = 생성기의 실제 재실 인원**(occupancy.csv, 1분 → 5분 칸에 1분이라도 사람이 있으면 '사용중'). "
         "과제 5 본 보고서의 '직접 관찰 6일'보다 넓은 구간을 평가할 수 있다.",
         f"> - **검증** = 원본 4주차 + 같은 생성기의 새 씨앗 {len(VAL_SEEDS)}개(101~104, 각 5주) · "
         f"**시험** = 원본 5주차 + 새 씨앗 {len(TEST_SEEDS)}개(201~206, 각 5주). 새 씨앗은 모델이 한 번도 보지 않은 데이터다. "
         "모든 파라미터는 검증에서 고르고 시험은 보고만 했다.",
         "> - 실데이터 점검: 실습실, 보드 B 수집(data/sensors_boardB.db) 10/20 17:50 ~ 10/21 16:15 + 소음 LAeq 대리 정답"
         "(data/own_observed.csv), 교육장 InfluxDB 8주(data/influx_56d.db, 10/1~10/21) · 실제 시간표 평일 09:00~11:30·13:00~18:00 (10/5·10/9 휴일).",
         "> - 장비: 맥북 CPU. 학습은 HistGradientBoosting(과제 5와 같은 설정, 300회)이라 줄인 것 없음.", ""]

    # ===================================================== 1. 노쇼 / 예약 없는 사용
    Bv = pd.concat([bookings(S, t) for S, t in val_sets], ignore_index=True)
    Bt = pd.concat([bookings(S, t) for S, t in test_sets], ignore_index=True)
    Bv = Bv.dropna(subset=[c for c in Bv.columns if c.startswith("rise")]).reset_index(drop=True)
    Bt = Bt.dropna(subset=[c for c in Bt.columns if c.startswith("rise")]).reset_index(drop=True)

    def score_ns(B, pred):
        y = B["noshow15"].to_numpy()
        tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
        rec = tp / max(tp + fn, 1) * 100; prec = tp / (tp + fp) * 100 if tp + fp else np.nan
        f1 = 2 * tp / max(2 * tp + fp + fn, 1) * 100
        return {"예약": len(B), "실제 노쇼": int(y.sum()), "탐지": int(pred.sum()), "맞힘": tp, "헛탐지": fp, "놓침": fn,
                "재현율": rec, "정밀도": prec, "F1": f1}

    best = {}
    grid_rows = []
    for kind, thrs in (("model", list(np.round(np.arange(0.05, 0.96, 0.05), 2)) + [0.97, 0.98, 0.99]), ("rule", [10, 20, 30, 40, 60, 80, 100, 150])):
        cands = []
        for ns in (3, 4, 5, 6):        # 15·20·25·30분 기다린 뒤 판정
            for th in thrs:
                s = score_ns(Bv, noshow_detect(Bv, kind, ns, th))
                # 같은 F1 이면 더 빨리(칸 수 적게) 판정하는 쪽
                cands.append((s["F1"], -ns, s["정밀도"] if not np.isnan(s["정밀도"]) else 0, ns, th, s))
                grid_rows.append({"방법": kind, "판정 시점(분)": ns * 5, "임계값": th, **{k: s[k] for k in ("재현율", "정밀도", "F1")}})
        cands.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        best[kind] = cands[0][3:5]
        # 과제 문구대로 '15분' 에서 판정하는 경우의 최선도 따로
        c15 = [c for c in cands if c[3] == 3]
        best[kind + "15"] = c15[0][3:5]

    ns_rows = {}
    for key, label in (("model15", "모델 확률 · 15분 판정"), ("model", "모델 확률 · 검증 최선"),
                       ("rule15", "CO₂ 상승 규칙 · 15분 판정"), ("rule", "CO₂ 상승 규칙 · 검증 최선")):
        kind = key.replace("15", "")
        ns, th = best[key]
        thr_txt = f"확률 < {th:.2f}" if kind == "model" else f"상승 < {th:.0f}ppm"
        sv = score_ns(Bv, noshow_detect(Bv, kind, ns, th)); st_ = score_ns(Bt, noshow_detect(Bt, kind, ns, th))
        ns_rows[f"{label} ({ns * 5}분, {thr_txt})"] = {"검증 F1": sv["F1"], **{f"시험 {k}": v for k, v in st_.items()
                                                                          if k in ("예약", "실제 노쇼", "맞힘", "헛탐지", "놓침", "재현율", "정밀도", "F1")}}
    ns_tab = pd.DataFrame(ns_rows).T
    for c in ("시험 예약", "시험 실제 노쇼", "시험 맞힘", "시험 헛탐지", "시험 놓침"):
        ns_tab[c] = ns_tab[c].astype(int)
    # 원본 5주차만
    ns, th = best["model15"]
    Bt_main = Bt[Bt.set == "원본"]
    s_main = score_ns(Bt_main, noshow_detect(Bt_main, "model", ns, th)) if len(Bt_main) else None
    Bv_main = Bv[Bv.set == "원본"]

    # 노쇼 정답이 생성기의 '통째 노쇼'와 같은지
    agree_ns = (Bv["noshow15"] == Bv["noshow_all"]).mean() * 100

    L += ["## 1. 회의실 노쇼 · 예약 없는 사용 탐지 (심화)", "",
          "### 정답을 어떻게 만들었나",
          "- 생성기(tsinf/synth.py)는 회의실 예약마다 **15% 확률로 노쇼**(그 예약 시간 동안 아무도 안 옴), "
          "평일마다 **35% 확률로 예약 없는 사용**(10~17시 시작, 20~60분, 2~4명)을 심는다. 다만 '이 예약은 노쇼' 라는 표는 따로 저장하지 않는다.",
          "- 그래서 **실제 재실(occupancy.csv)과 예약표(schedule.csv)를 맞대어 정답을 다시 만들었다**: "
          "예약 시작부터 15분 동안 실제 재실이 한 번도 없으면 '노쇼'. 예약과 전혀 겹치지 않는 실제 재실 연속 구간은 '예약 없는 사용' 1건.",
          f"- 이렇게 만든 '15분 노쇼'가 '예약 시간 내내 아무도 없음'과 일치하는 비율 {agree_ns:.1f}% (생성기의 지각은 0~5분이라 15분 안에는 다 들어온다).",
          f"- 건수가 적다: 원본 검증 주(4주차) 회의실 예약 {len(Bv_main)}건 중 노쇼 {int(Bv_main.noshow15.sum())}건, "
          f"시험 주(5주차) {len(Bt_main)}건 중 노쇼 {int(Bt_main.noshow15.sum())}건 (고장 구간 겹친 예약 제외) → "
          "이것만으로는 임계값을 고를 수도, 정밀도를 말할 수도 없어 **씨앗만 바꾼 합성 데이터를 더 만들어** 검증·시험 건수를 늘렸다.",
          f"- 최종 건수: 검증 예약 {len(Bv)}건(노쇼 {int(Bv.noshow15.sum())}), 시험 예약 {len(Bt)}건(노쇼 {int(Bt.noshow15.sum())}).", "",
          "### 노쇼 판정 규칙",
          "- **모델 확률 규칙**: 예약 시작 후 N분 동안의 5분 칸 모두에서 사용 여부 모델 확률 < θ 이면 '노쇼'.",
          "- **CO₂ 상승 규칙(비교용)**: 예약 시작 후 N분 동안 CO₂ 가 시작 직전보다 θ ppm 이상 오른 적이 없으면 '노쇼'.",
          "- N(15/20/25/30분)과 θ 는 검증 데이터 F1 로 골랐다. 과제 문구대로 15분에 판정하는 경우도 따로 적었다.", "",
          md(ns_tab), ""]
    if s_main:
        L += [f"- 원본 5주차만 떼어 보면(모델·15분): 예약 {s_main['예약']}건, 노쇼 {s_main['실제 노쇼']}건 중 {s_main['맞힘']}건 탐지, 헛탐지 {s_main['헛탐지']}건.", ""]

    # 예약 없는 사용
    ub_cands = []
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99):
        for k in (2, 3, 4, 6):
            for buf in (0, 15, 30, 45):
                sc = ev_scores([unbooked_events(S, t, thr, k, buf) for S, t in val_sets])
                ub_cands.append((sc["F1"], -k, thr, k, buf, sc))
    ub_cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, u_thr, u_k, u_buf, u_val = ub_cands[0]
    ubr = []
    for thr in (20, 30, 50, 80):
        for k in (1, 2, 3):
            for buf in (0, 15, 30, 45):
                sc = ev_scores([unbooked_events(S, t, thr, k, buf, "rule") for S, t in val_sets])
                ubr.append((sc["F1"], -k, thr, k, buf, sc))
    ubr.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, r_thr, r_k, r_buf, r_val = ubr[0]
    u_test = ev_scores([unbooked_events(S, t, u_thr, u_k, u_buf) for S, t in test_sets])
    r_test = ev_scores([unbooked_events(S, t, r_thr, r_k, r_buf, "rule") for S, t in test_sets])
    u_test_nobuf = ev_scores([unbooked_events(S, t, u_thr, u_k, 0) for S, t in test_sets])
    u_main = ev_scores([unbooked_events(main_S, te_t, u_thr, u_k, u_buf)])
    ub_tab = pd.DataFrame({
        f"모델 확률 ≥ {u_thr:.2f} 가 {u_k * 5}분 연속, 예약 끝 {u_buf}분 뒤부터": {"검증 F1": u_val["F1"], **{f"시험 {k}": v for k, v in u_test.items()}},
        f"같은 규칙, 예약 끝 직후부터(여유 0분)": {"검증 F1": np.nan, **{f"시험 {k}": v for k, v in u_test_nobuf.items()}},
        f"CO₂ 15분 상승 ≥ {r_thr}ppm 가 {r_k * 5}분 연속, 예약 끝 {r_buf}분 뒤부터": {"검증 F1": r_val["F1"], **{f"시험 {k}": v for k, v in r_test.items()}},
    }).T
    for c in ("시험 실제 사건", "시험 탐지 사건"):
        ub_tab[c] = ub_tab[c].astype(int)
    L += ["### 예약 없는 사용 판정 규칙",
          "- 예약이 없는 시간(평일 8~19시)에 '사용중' 판정이 K칸 연속이면 사건 1건. 예약이 끝난 직후는 CO₂ 가 천천히 빠져 '사용중'으로 보이기 쉬우므로 "
          "예약 끝 뒤 B분은 보지 않는다. θ·K·B 는 검증 F1 로 골랐다.",
          "- 채점: 탐지 사건이 실제 사건과 조금이라도 겹치면 맞힘. 판정 지연 = 실제 시작 → K칸 연속 확인 시점.", "",
          md(ub_tab), "",
          f"- 원본 5주차만: 실제 {u_main['실제 사건']}건, 탐지 {u_main['탐지 사건']}건, 맞힘(실제 기준) {u_main['재현율']:.0f}%.", ""]

    ns_m = ns_tab.iloc[0]
    # 헛탐지(노쇼라 했는데 실제로는 사람이 옴) 예약의 특징: 직전 30분에 사람이 있었나 / 실제 인원
    ns15, th15 = best["model15"]
    pr_t = noshow_detect(Bt, "model", ns15, th15)
    fp_rows = Bt[(pr_t == 1) & (Bt.noshow15 == 0)]
    allS = {S["tag"]: S for S, _ in test_sets}
    fp_prev, fp_people = [], []
    for r in fp_rows.itertuples():
        o = allS[r.set]["occ"]["회의실"]
        fp_prev.append(bool(o[(o.index >= r.start - pd.Timedelta(minutes=30)) & (o.index < r.start)].max() > 0))
        fp_people.append(float(o[(o.index >= r.start) & (o.index < r.start + pd.Timedelta(minutes=15))].max()))
    tp_people = []
    for r in Bt[(Bt.noshow15 == 0)].itertuples():
        o = allS[r.set]["occ"]["회의실"]
        tp_people.append(float(o[(o.index >= r.start) & (o.index < r.start + pd.Timedelta(minutes=15))].max()))
    L += ["### 해석",
          f"- **왜**: 회의실은 60m³ 로 작아 2~3명만 들어와도 15분이면 CO₂ 가 눈에 띄게 오른다. 그래서 '시작 후 15분 동안 오름이 없다'는 노쇼의 뚜렷한 신호이고, "
          f"15분 판정으로도 시험 재현율 {ns_m['시험 재현율']:.0f}% · 정밀도 {ns_m['시험 정밀도']:.0f}% 가 나온다. "
          f"시험 헛탐지 {len(fp_rows)}건을 열어 보면 직전 30분에 사람이 있던 예약(앞 사용에서 이어짐) {sum(fp_prev)}건, "
          f"첫 15분 실제 인원 중앙값 {np.median(fp_people) if fp_people else float('nan'):.0f}명(정상 예약 전체는 {np.median(tp_people):.0f}명) → "
          "적은 인원이 들어와 CO₂ 상승이 작거나, 이미 높은 CO₂ 에서 시작해 '오름'이 잘 안 보이는 경우가 헛탐지가 된다.",
          "- **그래서**: 노쇼 판정은 '자동 취소'보다 '15분 시점에 예약자에게 확인 알림 → 응답 없으면 30분에 해제' 처럼 두 단계로 쓰면 헛탐지 비용(쓰는 사람을 쫓아냄)을 줄일 수 있다.",
          f"- 예약 없는 사용은 **예약 끝 직후 여유(B)가 핵심**이다: 여유 0분이면 예약 뒤 남은 CO₂ 때문에 헛탐지가 {u_test['헛탐지']}건 → {u_test_nobuf['헛탐지']}건으로 늘어 "
          f"정밀도가 {u_test_nobuf['정밀도']:.0f}% 로 떨어진다. 대신 예약 직후 바로 이어진 무단 사용은 놓친다.",
          "- 모델은 '일정 기반 정답'으로 학습했는데도(노쇼 예약이 학습에서는 '사용중'으로 잘못 표시됨) 입력에 일정을 넣지 않았기 때문에 센서만 보고 노쇼를 가려낸다. "
          "과제 5에서 일정을 입력에서 뺀 결정이 여기서 효과를 낸다.", ""]

    # ===================================================== 2. 임계값 조정
    Pv = pd.concat([slot_table(S, model, occ_feats, t) for S, t in val_sets], ignore_index=True)
    Pt = pd.concat([slot_table(S, model, occ_feats, t) for S, t in test_sets], ignore_index=True)
    # 원본 관찰 6일(과제 5 본 보고서와 같은 조건)로 재현 확인
    obs_t = main_S["grid"].index.normalize().isin(list(main_S["obs_days"]))
    Po = slot_table(main_S, model, occ_feats, obs_t)
    rep_model = bin_metrics(Po.y, (Po.p >= 0.5).astype(int)); rep_rule = bin_metrics(Po.y, Po.rule)
    # 0.5 의 '빈 방→사용중' 오판이 어디서 나오나: 실제 퇴실 후 30분 이내 / 그 외 (검증)
    def after_leave(P, mins=30):
        out = np.zeros(len(P), bool)
        for _, g in P.groupby(["set", "space"]):
            g = g.sort_values("time")
            last_occ = g.time.where(g.y == 1).ffill()
            out[g.index] = ((g.time - last_occ) <= pd.Timedelta(minutes=mins)).fillna(False).to_numpy() & (g.y == 0).to_numpy()
        return out
    al_v = after_leave(Pv)
    fp_v = ((Pv.p >= 0.5) & (Pv.y == 0)).to_numpy()
    fp_after_share = (fp_v & al_v).sum() / max(fp_v.sum(), 1) * 100
    fp_v94 = None

    rule_v = bin_metrics(Pv.y, Pv.rule); rule_t = bin_metrics(Pt.y, Pt.rule)
    sweep = []
    for th in np.round(np.arange(0.30, 0.991, 0.05), 2).tolist() + [0.97, 0.98, 0.99]:
        sv = bin_metrics(Pv.y, (Pv.p >= th).astype(int))
        sweep.append({"임계값": th, **{f"검증 {k}": sv[k] for k in ("빈 방→사용중 오판율", "재현율", "F1")}})
    sweep = pd.DataFrame(sweep).drop_duplicates("임계값").sort_values("임계값").set_index("임계값")
    fine = np.round(np.arange(0.30, 0.995, 0.01), 2)
    fr_v = np.array([bin_metrics(Pv.y, (Pv.p >= th).astype(int))["빈 방→사용중 오판율"] for th in fine])
    th_match = float(fine[np.argmax(fr_v <= rule_v["빈 방→사용중 오판율"])]) if (fr_v <= rule_v["빈 방→사용중 오판율"]).any() else np.nan
    f1_v = np.array([bin_metrics(Pv.y, (Pv.p >= th).astype(int))["F1"] for th in fine])
    th_f1 = float(fine[np.argmax(f1_v)])

    # 끄기 지연(15분 '비어있음' 연속일 때만 끔)과 결합: 켜기는 높은 임계값, 끄기는 늦게
    def hyst(P, th_on, off_slots=3):
        out = []
        for _, g in P.groupby(["set", "space", P.time.dt.date]):
            g = g.sort_values("time"); on = False; cnt = 0; res = []
            for pv in g.p.to_numpy():
                if pv >= th_on:
                    on, cnt = True, 0
                elif on:
                    cnt += 1
                    if cnt >= off_slots:
                        on = False
                res.append(int(on))
            out.append(pd.Series(res, index=g.index))
        return pd.concat(out).reindex(P.index).to_numpy()

    comp = {}
    for name, pv, pt in (("단순 규칙 (15분 CO₂ +30ppm)", Pv.rule, Pt.rule),
                         ("모델, 임계값 0.50 (과제 5 그대로)", (Pv.p >= 0.5).astype(int), (Pt.p >= 0.5).astype(int)),
                         (f"모델, 검증 F1 최대 임계값 {th_f1:.2f}", (Pv.p >= th_f1).astype(int), (Pt.p >= th_f1).astype(int)),
                         (f"모델, 오판율을 규칙과 맞춘 임계값 {th_match:.2f}", (Pv.p >= th_match).astype(int), (Pt.p >= th_match).astype(int)),
                         (f"모델 {th_match:.2f}로 켜고, 15분 연속 '비어있음'일 때 끔", hyst(Pv, th_match), hyst(Pt, th_match))):
        sv = bin_metrics(Pv.y, pv); st_ = bin_metrics(Pt.y, pt)
        comp[name] = {"검증 오판율": sv["빈 방→사용중 오판율"], "시험 오판율": st_["빈 방→사용중 오판율"],
                      "시험 재현율": st_["재현율"], "시험 정밀도": st_["정밀도"], "시험 F1": st_["F1"]}
    comp = pd.DataFrame(comp).T

    # 비용 환산: 시험 데이터의 공간·평일 1개당 빈 방 냉방 시간 / 사람 있는데 꺼진 시간
    n_spdays = Pt.groupby(["set", "space", Pt.time.dt.date]).ngroups
    cost = {}
    for name, pt in (("단순 규칙", Pt.rule.to_numpy()), ("모델 0.50", (Pt.p >= 0.5).astype(int).to_numpy()),
                     (f"모델 {th_match:.2f}", (Pt.p >= th_match).astype(int).to_numpy()),
                     (f"모델 {th_match:.2f} + 15분 끄기 지연", hyst(Pt, th_match))):
        y = Pt.y.to_numpy()
        cost[name] = {"빈 방 냉방(분/공간·일)": ((y == 0) & (pt == 1)).sum() * 5 / n_spdays,
                      "사람 있는데 꺼짐(분/공간·일)": ((y == 1) & (pt == 0)).sum() * 5 / n_spdays}
    cost = pd.DataFrame(cost).T

    L += ["## 2. 빈 방→사용중 오판율 줄이기: 임계값 조정", "",
          f"- 재현 확인(과제 5와 같은 관찰 6일·임계값 0.5): 모델 오판율 {rep_model['빈 방→사용중 오판율']:.1f}% · F1 {rep_model['F1']:.1f}, "
          f"규칙 오판율 {rep_rule['빈 방→사용중 오판율']:.1f}% · F1 {rep_rule['F1']:.1f} → 본 보고서 숫자(8.2% / 1.1%)와 같다.",
          f"- 검증 데이터(평일 8~19시 {len(Pv):,}칸)에서 임계값을 0.30~0.99 로 바꿔 가며 오판율을 쟀다. 규칙의 검증 오판율은 {rule_v['빈 방→사용중 오판율']:.1f}%.", "",
          md(sweep), "",
          f"- 검증 오판율이 규칙({rule_v['빈 방→사용중 오판율']:.1f}%) 이하가 되는 가장 낮은 임계값: **{th_match:.2f}** · 검증 F1 최대 임계값: {th_f1:.2f}", "",
          f"### 시험 결과 (평일 8~19시 {len(Pt):,}칸)", md(comp), "",
          "### 비용으로 바꿔 보면 (시험, 공간 1개·평일 하루 평균)", md(cost), "",
          "### 해석",
          f"- **왜 0.5 에서 오판이 많았나**: 검증에서 0.5 의 '빈 방→사용중' 오판 중 {fp_after_share:.0f}% 가 **실제 퇴실 후 30분 이내**다. "
          "사람이 나가도 CO₂ 는 환기 속도만큼 천천히 빠지므로 확률이 0.5~0.9 사이에 한동안 머문다. "
          "게다가 class_weight='balanced' 로 '사용중'(소수 쪽)에 더 큰 가중을 주어 학습했으니, 0.5 는 '켜는 쪽'으로 기운 임계값이다. 임계값을 올리면 이 애매한 꼬리를 잘라 낸다 "
          "(재현율은 몇 점만 잃는다: 들어온 뒤 CO₂ 가 확실히 오르면 확률이 0.95 이상으로 뛰기 때문).",
          f"- **그래서**: 냉난방을 '켜는' 판단에는 임계값 {th_match:.2f} 를 쓰고(빈 방 냉방이 규칙 수준), "
          "'끄는' 판단은 15분 연속 '비어있음'일 때로 늦추면 사람이 있는데 꺼지는 시간을 줄일 수 있다. 하나의 0.5 대신 '켜기/끄기 두 기준'을 쓰는 것이 운영에 맞다.",
          f"- 다만 끄기 지연은 공짜가 아니다: 시험에서 '사람 있는데 꺼짐'은 {cost.iloc[2, 1]:.0f} → {cost.iloc[3, 1]:.0f}분/일로 줄지만, 퇴실 뒤 15분을 더 켜 두므로 빈 방 냉방이 "
          f"{cost.iloc[2, 0]:.0f} → {cost.iloc[3, 0]:.0f}분/일로 늘어난다(오판율 {comp.iloc[4]['시험 오판율']:.1f}%). 그래도 0.5 하나로 켜고 끄는 것({cost.iloc[1, 0]:.0f}분 / {cost.iloc[1, 1]:.0f}분)보다는 두 비용 모두 작다.",
          "- 빈 방 냉방의 비용은 '에너지 낭비'(몇 분 단위), 사람 있는데 꺼짐은 '불편·민원'이다. 어느 쪽이 더 비싼지는 운영자가 정할 일이라, 위 표처럼 분 단위로 보여 주고 고르게 한다.", ""]

    # ===================================================== 3. LOSO
    S = main_S
    idx = S["grid"].index
    tr_t = week_mask(S, 0, 3) & ~idx.normalize().isin(list(S["obs_days"]))
    sched_grid = {}
    for sp in SPACE_NAMES:
        g = pd.Series(0, index=idx)
        for r in S["sched"][S["sched"].space == sp].itertuples():
            g[(idx >= r.start.floor("5min")) & (idx < r.end)] = 1
        sched_grid[sp] = g.to_numpy()
    has_sched = {sp: sched_grid[sp].sum() > 0 for sp in SPACE_NAMES}

    def train_xy(spaces, label, drop_onehot):
        Xs, ys = [], []
        cols = [c for c in occ_feats if not (drop_onehot and c.startswith("sp"))]
        for sp in spaces:
            f = S["feats"][sp]
            m = tr_t & f.notna().all(axis=1).to_numpy() & ~S["fmask"][sp]
            y = sched_grid[sp] if label == "plan" else S["occ5"][sp].fillna(0).to_numpy().astype(int)
            Xs.append(f.loc[m, cols]); ys.append(y[m])
        return pd.concat(Xs), np.concatenate(ys), cols

    def fit(spaces, label, drop_onehot):
        X, y, cols = train_xy(spaces, label, drop_onehot)
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
        c.fit(X, y)
        return c, cols

    # 시험: 원본 5주차 + 새 씨앗(시험용) 의 해당 공간
    def eval_space(clf, cols, sp, zero_onehot=False, thr=0.5):
        ys, ps = [], []
        for SS, t in test_sets:
            f = SS["feats"][sp].copy()
            if zero_onehot:
                for c in f.columns:
                    if c.startswith("sp"):
                        f[c] = 0.0
            i2 = SS["grid"].index
            m = t & (i2.dayofweek < 5) & (i2.hour >= 8) & (i2.hour < 19) & ~SS["fmask"][sp] & f.notna().all(axis=1).to_numpy()
            ys.append(SS["occ5"][sp].to_numpy()[m].astype(int)); ps.append((clf.predict_proba(f.loc[m, cols])[:, 1] >= thr).astype(int))
        return bin_metrics(np.concatenate(ys), np.concatenate(ps))

    lo_rows = []
    sched_spaces = [sp for sp in SPACE_NAMES if has_sched[sp]]
    clf_in_plan, cols_in_plan = fit(sched_spaces, "plan", False)            # = 과제 5 모델과 같은 설정 (재학습)
    clf_in_true, cols_in_true = fit(SPACE_NAMES, "true", False)
    clf_in_true_noh, cols_in_true_noh = fit(SPACE_NAMES, "true", True)
    for sp in SPACE_NAMES:
        others_plan = [s for s in sched_spaces if s != sp]
        others_all = [s for s in SPACE_NAMES if s != sp]
        r = {"공간": sp}
        # (가) 일정 기반 정답 (과제 5 방식)
        if has_sched[sp]:
            r["일정정답·학습에 포함 F1"] = eval_space(clf_in_plan, cols_in_plan, sp)["F1"]
        else:
            r["일정정답·학습에 포함 F1"] = np.nan
        c1, k1 = fit(others_plan, "plan", True)
        e1 = eval_space(c1, k1, sp)
        r["일정정답·빼고 학습 F1"] = e1["F1"]; r["일정정답·빼고 오판율"] = e1["빈 방→사용중 오판율"]
        c1b, k1b = fit(others_plan, "plan", False)
        r["일정정답·빼고(원-핫 0, 서비스 방식) F1"] = eval_space(c1b, k1b, sp, zero_onehot=True)["F1"]
        # (나) 실제 재실 정답 (라벨 잡음 없이 '공간 차이'만 보기)
        e_in = eval_space(clf_in_true, cols_in_true, sp)
        r["실제정답·포함 F1"] = e_in["F1"]
        c2, k2 = fit(others_all, "true", True)
        e2 = eval_space(c2, k2, sp)
        r["실제정답·빼고 F1"] = e2["F1"]; r["실제정답·빼고 오판율"] = e2["빈 방→사용중 오판율"]; r["실제정답·빼고 재현율"] = e2["재현율"]
        lo_rows.append(r)
    lo = pd.DataFrame(lo_rows).set_index("공간")
    lo["실제정답 F1 차이(빼고−포함)"] = lo["실제정답·빼고 F1"] - lo["실제정답·포함 F1"]

    L += ["## 3. 한 공간을 학습에서 빼고 사용 여부 추정 (심화, leave-one-space-out)", "",
          "- 방법: 4개 공간 중 하나를 학습에서 완전히 빼고(원본 1~3주, 관찰일 제외) 나머지로 학습 → 뺀 공간을 시험(원본 5주차 + 새 씨앗 6개, 평일 8~19시, 임계값 0.5).",
          "- 공간 원-핫은 뺀 공간에 대해 의미가 없으므로 기본은 **원-핫을 입력에서 뺀 모델**. 서비스(service/infer.py)처럼 '원-핫을 둔 채 새 공간은 모두 0'으로 넣는 방식도 따로 적었다.",
          "- 두 가지 정답으로 학습했다: (가) **일정 기반 정답**(과제 5 방식 — 라운지는 일정이 없어 '포함'이 불가능), "
          "(나) **실제 재실 정답**(라벨 잡음을 없애고 '공간이 달라서 생기는 손해'만 보려고. 실제 현장에서는 얻기 어려운 정답이다).", "",
          md(lo), "", "### 판단: 새 공간에 바로 써도 되나?"]
    worst = lo["실제정답 F1 차이(빼고−포함)"].idxmin()
    L += [f"- 실제 재실 정답 기준으로, 공간을 빼도 F1 이 떨어지는 폭은 {lo['실제정답 F1 차이(빼고−포함)'].min():.1f} ~ {lo['실제정답 F1 차이(빼고−포함)'].max():.1f}점 (가장 손해 큰 곳: {worst}).",
          "- **왜 버티나**: 입력이 CO₂ '변화량'과 '최근 6시간 최저 대비 높이' 중심이라, 방 크기가 달라도 '사람이 들어오면 오른다'는 모양은 같다. "
          "**왜 손해가 나나**: 부피·환기율이 다르면 같은 인원에도 오르는 폭이 달라(회의실 60m³ vs 강의실1 210m³), 다른 방에서 배운 '얼마나 올라야 사람인가'의 기준이 어긋난다.",
          f"- **일정 기반 정답의 함정**: 회의실은 '학습에 포함'(F1 {lo.loc['회의실', '일정정답·학습에 포함 F1']:.1f})보다 '빼고 학습'({lo.loc['회의실', '일정정답·빼고 학습 F1']:.1f})이 오히려 낫다. "
          "회의실의 일정 정답에는 노쇼(정답은 '사용중'인데 실제 빈 방)와 예약 없는 사용(정답은 '빈 방'인데 실제 사용)이 섞여 있어, 그 공간을 넣으면 틀린 정답을 배우기 때문이다. "
          "즉 '일정이 실제와 자주 어긋나는 공간'은 그 일정으로 학습시키는 것보다 다른 공간에서 배운 모델이 낫다.",
          f"- **라운지**(일정이 없어 늘 '처음 보는 공간'): 일정 정답 모델로는 F1 {lo.loc['라운지', '일정정답·빼고 학습 F1']:.1f} · 오판율 {lo.loc['라운지', '일정정답·빼고 오판율']:.1f}% 로 약하다. "
          "라운지는 10분짜리 짧은 방문이 많아 CO₂ 가 오르기 전에 사람이 나가는데, 강의실·회의실에서 배운 '길게 머무는 모양'과 다르다.",
          "- **판단**: 학습 공간과 크기·쓰임이 비슷한 공간(강의실1↔강의실2)은 **바로 써도 된다**(실제 재실 정답 기준 F1 손해 2점 이하·오판율 3% 이하. 다만 일정 정답으로 학습한 모델로는 강의실2 오판율 4.5% → 켜기 임계값을 올려 쓴다). "
          "학습 범위 밖(가장 작은 회의실, 짧은 방문 위주의 라운지)은 **바로 자동 제어에 쓰기엔 이르다** — 판정은 보여 주되 냉난방은 켜기 임계값을 올리고(2절), "
          "1~2주 실제 재실(관찰·소음 대리 정답 등)을 모아 그 공간을 넣어 다시 학습한 뒤 쓴다.",
          "- **그래서**: (1) 크기·환기가 학습 공간 범위 안에 있는 새 공간은 '켜기 임계값을 올린 채(2절) 시범 운영' 정도는 가능하다. "
          "(2) 학습 공간보다 훨씬 작거나 큰 공간, 환기 방식이 다른 공간은 바로 자동 제어에 쓰지 말고, 1~2주 일정·관찰 기록을 모아 그 공간을 넣어 다시 학습한다. "
          "(3) 서비스의 '원-핫 0' 방식은 학습에서 본 적 없는 입력이라 결과가 원-핫을 뺀 모델과 다를 수 있다 → 새 공간용으로는 원-핫 없는 모델을 따로 두는 편이 안전하다.", ""]

    # ===================================================== 4. 실데이터
    L += real_check()

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


# ----------------------------------------------------------------------------- 실데이터 (실습실)
def real_check():
    from task_schedule_real import class_days, BLOCKS, START
    L = ["## 4. 실데이터: 실습실 실제 시간표 vs 소음 대리 정답 / CO₂", ""]
    sm = pickle.load(open("models/state_models.pkl", "rb"))
    occ_feats = sm["occupancy"], sm["occ_features"]
    model, feats_cols = occ_feats

    def feats_for(db):
        df, _ = prep.clean(prep.load_readings(db, spaces=["실습실"]))
        grid = prep.to_grid(df)
        gs = grid[[("실습실", m) for m in prep.METRICS]].droplevel(0, axis=1)
        f = state_features(gs, 0, len(sm["spaces"]))
        for c in f.columns:
            if c.startswith("sp"):
                f[c] = 0.0                      # 서비스와 같게: 새 공간 원-핫 0
        ok = f.notna().all(axis=1)
        p = pd.Series(np.nan, index=f.index); p[ok] = model.predict_proba(f.loc[ok, feats_cols])[:, 1]
        return f, p

    # (a) 소음 대리 정답이 있는 10/20~10/21
    ob = pd.read_csv("data/own_observed.csv", parse_dates=["start", "end"])
    ob["start"] = ob["start"].dt.tz_localize(prep.TZ)
    truth = ob.set_index("start")["occupied"]
    t_lo, t_hi = truth.index.min(), truth.index.max() + SLOT
    try:
        fB, pB = feats_for("data/sensors_boardB.db")
    except Exception as e:      # noqa
        fB, pB = None, None
    days = sorted({d for d in truth.index.date})
    blocks = []
    for d in days:
        if pd.Timestamp(d).dayofweek >= 5 or d < START.date():
            continue
        for s, e in BLOCKS:
            st = pd.Timestamp(f"{d} {s}", tz=prep.TZ); en = pd.Timestamp(f"{d} {e}", tz=prep.TZ)
            if st < t_lo or st + pd.Timedelta(minutes=15) > t_hi:
                continue
            first = truth[(truth.index >= st) & (truth.index < st + pd.Timedelta(minutes=15))]
            covered = truth[(truth.index >= st) & (truth.index < en)]
            r = {"수업 블록": f"{d:%m-%d} {s}~{e}", "대리 정답 덮는 칸": f"{len(covered)}/{int((en - st) / SLOT)}",
                 "첫 15분 '사용중' 칸(소음)": f"{int(first.sum())}/{len(first)}",
                 "노쇼처럼 보임(소음)": "예" if first.sum() == 0 else "아니오"}
            if pB is not None:
                pf = pB[(pB.index >= st) & (pB.index < st + pd.Timedelta(minutes=15))]
                r["첫 15분 모델 확률 최대"] = f"{pf.max():.2f}" if pf.notna().any() else "-"
            blocks.append(r)
    # 시간표 밖
    in_tt = pd.Series(False, index=truth.index)
    for d in class_days(truth.index):
        for s, e in BLOCKS:
            in_tt |= (truth.index >= pd.Timestamp(f"{d} {s}", tz=prep.TZ)) & (truth.index < pd.Timestamp(f"{d} {e}", tz=prep.TZ))
    off = truth[~in_tt]
    off_runs = []
    b = off.reindex(pd.date_range(off.index.min(), off.index.max(), freq="5min")).fillna(0).to_numpy().astype(bool)
    rng_idx = pd.date_range(off.index.min(), off.index.max(), freq="5min")
    for a, z in runs(b):
        off_runs.append({"시작": f"{rng_idx[a]:%m-%d %H:%M}", "끝": f"{rng_idx[z - 1] + SLOT:%m-%d %H:%M}", "길이(분)": (z - a) * 5})
    off_runs_df = pd.DataFrame(off_runs)

    L += [f"### (a) 소음 LAeq 대리 정답이 있는 구간 ({t_lo:%m-%d %H:%M} ~ {t_hi:%m-%d %H:%M}, 5분 칸 {len(truth)}개)",
          "- 대리 정답: 강단 소음 센서의 5분 최대 소음 15분 중앙값 ≥ 34.5dB → '사용중' (task_own.py 정의). 조용한 자습은 '비어있음'으로 잡힐 수 있다.",
          f"- 이 구간에 들어오는 수업 블록은 {len(blocks)}개뿐이다 (10/21 오전·오후, 오후는 16:15 까지만 기록).", "",
          md(pd.DataFrame(blocks).set_index("수업 블록")) if blocks else "- 겹치는 수업 블록 없음", "",
          f"- 시간표 밖 칸 {len(off)}개 중 '사용중'(소음) {int(off.sum())}칸 = {off.sum() * 5}분. 연속 구간:", "",
          md(off_runs_df.set_index("시작")) if len(off_runs_df) else "- 없음", ""]
    lunch = truth[(truth.index.hour * 60 + truth.index.minute >= 11 * 60 + 30) & (truth.index.hour < 13)]
    L += [f"- 점심(11:30~13:00) {len(lunch)}칸 중 '사용중' {int(lunch.sum())}칸 → 시간표상 비어야 할 점심에도 사람이 남아 있다(교육생이 실습실에서 점심·자습). "
          "시간표만 믿고 점심에 냉난방을 끄면 사람이 있는데 끄게 된다.", ""]

    # (b) InfluxDB 8주: 정답 없이 CO₂ 모델이 본 '노쇼처럼 보이는 블록' / '시간표 밖 사용' 수
    try:
        fI, pI = feats_for("data/influx_56d.db")
        th = 0.5
        cd = class_days(pI.index)
        rows = []
        for d in cd:
            for s, e in BLOCKS:
                st = pd.Timestamp(f"{d} {s}", tz=prep.TZ)
                pf = pI[(pI.index >= st) & (pI.index < st + pd.Timedelta(minutes=15))]
                pf30 = pI[(pI.index >= st) & (pI.index < st + pd.Timedelta(minutes=30))]
                rise = fI["co2"].reindex([st + pd.Timedelta(minutes=k) for k in (0, 5, 10, 15, 20, 25)]).max() - fI["co2"].get(st - SLOT, np.nan)
                if pf.notna().sum() < 3:
                    continue
                rows.append({"block": f"{d} {s}", "pmax15": pf.max(), "pmax30": pf30.max(), "rise30": rise})
        R = pd.DataFrame(rows)
        i2 = pI.index
        tt = pd.Series(False, index=i2)
        for d in cd:
            for s, e in BLOCKS:
                tt |= (i2 >= pd.Timestamp(f"{d} {s}", tz=prep.TZ)) & (i2 < pd.Timestamp(f"{d} {e}", tz=prep.TZ))
        after = pd.Series(False, index=i2)
        for d in cd:
            for s, e in BLOCKS:
                en = pd.Timestamp(f"{d} {e}", tz=prep.TZ)
                after |= (i2 >= en) & (i2 < en + pd.Timedelta(minutes=45))
        considered = (i2 >= START) & ~tt & ~after & pI.notna().to_numpy()
        occ_off = (pI >= th) & considered
        ev = [(a, z) for a, z in runs(occ_off.to_numpy()) if z - a >= 3]
        ev_df = pd.DataFrame([{"시작": i2[a], "길이(분)": (z - a) * 5} for a, z in ev])
        L += [f"### (b) 교육장 InfluxDB 8주 ({START:%m-%d} ~ {pI.index[-1]:%m-%d %H:%M}) — 정답 없이 CO₂ 모델이 본 것",
              f"- 수업 블록 {len(R)}개(결측 많은 블록 제외). 모델은 합성 데이터로 학습한 것을 원-핫 0 으로 그대로 적용(서비스와 같음).",
              f"- 첫 15분 모델 확률이 모두 {th} 미만('노쇼처럼 보임'): **{int((R.pmax15 < th).sum())}개**, "
              f"30분까지 봐도 미만: {int((R.pmax30 < th).sum())}개, 30분 동안 CO₂ 상승 < 30ppm: {int((R.rise30 < 30).sum())}개.",
              f"- 시간표 밖(수업 끝 뒤 45분 제외)에서 모델 '사용중'이 15분 이상 이어진 구간: **{len(ev)}건**"
              + (f", 그중 점심 시간(11:30~13:00) 시작 {int(sum((t.hour * 60 + t.minute >= 690) and t.hour < 13 for t in ev_df['시작']))}건, "
                 f"평일 18:45 이후·주말 {int(sum((t.hour >= 19 or t.dayofweek >= 5) for t in ev_df['시작']))}건." if len(ev_df) else "."),
              "- ⚠️ 이 숫자는 **정답으로 확인하지 않은 것**이다. 실습실은 학습한 합성 공간과 크기·인원·환기가 다를 수 있어 모델 확률의 눈금이 맞지 않을 수 있다(3절). "
              "'노쇼처럼 보인 블록'은 실제 결강이라기보다 '사람은 있는데 CO₂ 가 거의 안 오른 날'(창문 개방·적은 인원)일 가능성도 크다.",
              "- 그래서: 실데이터의 노쇼·시간표 밖 사용 판정은 소음 대리 정답이 며칠 더 쌓인 뒤 검증해야 한다. 지금 확인된 사실은 (a)의 하루치: "
              "수업 블록 2개 모두 정상 시작(노쇼 아님), 점심 시간 내내 사용중.", ""]
        if len(ev_df):
            L += ["<details><summary>시간표 밖 '사용중' 구간 목록 (모델, 정답 없음)</summary>", "",
                  md(ev_df.assign(시작=ev_df["시작"].dt.strftime("%m-%d(%a) %H:%M")).set_index("시작")), "", "</details>", ""]
    except Exception as e:      # noqa
        L += [f"- InfluxDB 8주 점검은 하지 못했다: {e}", ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
