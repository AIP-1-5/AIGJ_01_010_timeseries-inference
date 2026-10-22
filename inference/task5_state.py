"""과제 5. 상태 추정 — (A) 사용 여부 추정, (B) 환기 선제 경보.

    .venv/bin/python task5_state.py --data data/synth

(A) 사용 여부: 정답은 일정(수업·예약)으로 만들되, 일정은 실제와 다를 수 있으므로 '직접 관찰' 6일(observed.csv)로 평가.
    입력에 일정 정보는 쓰지 않는다 (정답을 일정에서 만들었는데 입력에도 넣으면 모델이 일정을 베껴 정답을 맞히는 척만 함).
(B) 환기 선제 경보: '지금은 1,000ppm 미만인데 30분 안에 1,000ppm 이상이 된다' = 양성.
    이미 넘은 시점은 데이터에서 뺀다 (넣으면 '지금 높다'만 보고 맞히는 쉬운 문제가 되어 '미리' 알리는 능력을 못 잰다).
분할: 1~3주 학습 · 4주 검증(임계값) · 5주 시험 (고장 구간 제외)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from tsinf import forecast as fc
from tsinf import prep
from tsinf.synth import SPACES


def state_features(gs: pd.DataFrame, si: int, n_spaces: int) -> pd.DataFrame:
    """센서값만으로 만든 특징 (일정 정보 없음). gs: time × [co2, temperature, humidity]"""
    c, t, h = gs["co2"], gs["temperature"], gs["humidity"]
    f = pd.DataFrame(index=gs.index)
    f["co2"] = c
    for k in (1, 3, 6):                       # 5·15·30분 변화량
        f[f"dco2_{k}"] = c - c.shift(k)
        f[f"dtemp_{k}"] = t - t.shift(k)
        f[f"dhum_{k}"] = h - h.shift(k)
    f["co2_above_min6h"] = c - c.rolling(72, min_periods=12).min()     # 최근 6시간 최저값 대비
    tf = fc.time_features(gs.index)
    for i, n in enumerate(["hs", "hc", "ds", "dc", "we"]):
        f[n] = tf[:, i]
    for j in range(n_spaces):
        f[f"sp{j}"] = 1.0 if j == si else 0.0
    return f


def metrics(y, p) -> dict:
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"정확도": accuracy_score(y, p) * 100, "F1": f1_score(y, p) * 100, "macro-F1": f1_score(y, p, average="macro") * 100,
            "재현율": recall_score(y, p, zero_division=0) * 100, "정밀도": precision_score(y, p, zero_division=0) * 100,
            "빈 방→사용중 오판율": fp / max(tn + fp, 1) * 100, "cm": cm}


def cm_md(cm, labels) -> str:
    return (f"| 실제 \\ 예측 | {labels[0]} | {labels[1]} |\n| --- | ---: | ---: |\n"
            f"| {labels[0]} | {cm[0, 0]} | {cm[0, 1]} |\n| {labels[1]} | {cm[1, 0]} | {cm[1, 1]} |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    os.makedirs("models", exist_ok=True)

    df, _ = prep.clean(prep.load_readings(f"{args.data}/sensors.db"))
    grid = prep.to_grid(df)
    spaces = [s.name for s in SPACES]
    cap = {s.name: float(s.capacity) for s in SPACES}
    sched_df = pd.read_csv(f"{args.data}/schedule.csv", parse_dates=["start", "end"])
    sched = fc.schedule_grid(sched_df, grid.index, spaces, cap)
    occ = pd.read_csv(f"{args.data}/occupancy.csv", index_col=0, parse_dates=True)
    occ.index = pd.to_datetime(occ.index, utc=True).tz_convert(prep.TZ)
    occ5 = (occ.resample("5min").max() > 0).astype(int).reindex(grid.index)
    obs = pd.read_csv(f"{args.data}/observed.csv", index_col=0, parse_dates=True)
    obs.index = pd.to_datetime(obs.index, utc=True).tz_convert(prep.TZ)
    obs5 = obs.resample("5min").max().dropna(how="all")
    obs_days = set(obs5.index.normalize())
    faults = pd.read_csv(f"{args.data}/faults.csv", parse_dates=["start", "end"])

    t0 = grid.index[0].normalize()
    tr = grid.index < t0 + pd.Timedelta(days=21)
    va = (grid.index >= t0 + pd.Timedelta(days=21)) & (grid.index < t0 + pd.Timedelta(days=28))
    te = grid.index >= t0 + pd.Timedelta(days=28)
    is_obs_day = grid.index.normalize().isin(list(obs_days))
    fault_mask = {sp: np.zeros(len(grid), bool) for sp in spaces}
    for f in faults.itertuples():
        fault_mask[f.space] |= (grid.index >= f.start - pd.Timedelta(minutes=30)) & (grid.index <= f.end + pd.Timedelta(minutes=60))

    # ======================= (A) 사용 여부 ===============================
    Xs, ys_plan, ys_true, split, sp_col, tcol = [], [], [], [], [], []
    for si, sp in enumerate(spaces):
        gs = grid[[(sp, m) for m in prep.METRICS]].droplevel(0, axis=1)
        f = state_features(gs, si, len(spaces))
        plan = (sched[sp] > 0).astype(int)
        Xs.append(f); ys_plan.append(plan); ys_true.append(occ5[sp])
        split.append(np.where(is_obs_day, "obs", np.where(tr, "train", np.where(va, "val", "test"))))
        sp_col += [sp] * len(grid); tcol.append(grid.index)
    X = pd.concat(Xs); y_plan = pd.concat(ys_plan).to_numpy(); y_true = pd.concat(ys_true).to_numpy()
    split = np.concatenate(split); sp_col = np.array(sp_col); tidx = pd.DatetimeIndex(np.concatenate(tcol))
    hours = tidx.hour
    ok = X.notna().all(axis=1).to_numpy()
    work = (hours >= 8) & (hours < 19)                     # 관찰 기록은 8~19시

    # 일정 기반 정답의 정확도 (직접 관찰과 비교)
    m_obs = (split == "obs") & work & ok
    agree = metrics(y_true[m_obs], y_plan[m_obs])

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
    # 라운지는 일정(예약·수업)이 없어 일정 기반 정답이 항상 '비어있음' → 학습에 넣으면 '라운지는 늘 빈 방'을 배운다.
    # 그래서 일정이 있는 공간으로만 학습하고, 라운지는 '학습하지 않은 공간'으로 평가한다 (심화: 새 공간 일반화)
    no_sched = {sp for sp in spaces if (sched[sp] > 0).sum() == 0}
    m_tr = (split == "train") & ok & ~np.isin(sp_col, list(no_sched))
    clf.fit(X[m_tr], y_plan[m_tr])
    pred_obs = clf.predict(X[m_obs])
    res_model = metrics(y_true[m_obs], pred_obs)
    rule_obs = (X["dco2_3"].to_numpy()[m_obs] >= 30).astype(int)                 # 단순 규칙
    res_rule = metrics(y_true[m_obs], rule_obs)

    # 공간별
    per_space = []
    for sp in spaces:
        mm = sp_col[m_obs] == sp
        if mm.sum():
            r = metrics(y_true[m_obs][mm], pred_obs[mm])
            per_space.append({"공간": sp, "정확도": r["정확도"], "F1": r["F1"], "빈 방→사용중 오판율": r["빈 방→사용중 오판율"]})
    per_space = pd.DataFrame(per_space).set_index("공간")

    # 가장 많이 틀리는 상황: 사람이 막 나간 직후 / 막 들어온 직후
    yt, yp = y_true[m_obs], pred_obs
    prev_true = pd.Series(y_true).shift(1).to_numpy()[m_obs]
    after_leave = (yt == 0) & (prev_true == 1)
    left_recent = pd.Series((y_true == 0) & (pd.Series(y_true).shift(1) == 1)).rolling(4, min_periods=1).max().to_numpy()[m_obs] > 0
    just_enter = pd.Series((y_true == 1) & (pd.Series(y_true).shift(1) == 0)).rolling(3, min_periods=1).max().to_numpy()[m_obs] > 0
    err = yt != yp
    situ = pd.DataFrame({
        "상황": ["퇴실 후 20분 이내 (실제 빈 방)", "입실 후 15분 이내 (실제 사용중)", "그 외"],
        "샘플 수": [int(left_recent.sum()), int(just_enter.sum()), int((~left_recent & ~just_enter).sum())],
        "오류율(%)": [err[left_recent].mean() * 100 if left_recent.any() else np.nan,
                   err[just_enter].mean() * 100 if just_enter.any() else np.nan,
                   err[~left_recent & ~just_enter].mean() * 100],
    }).set_index("상황")

    # ======================= (B) 환기 선제 경보 ===========================
    rows = []
    for si, sp in enumerate(spaces):
        gs = grid[[(sp, m) for m in prep.METRICS]].droplevel(0, axis=1)
        f = state_features(gs, si, len(spaces))
        fut = sched[sp]
        for k in (0, 1, 3, 6):                                    # 일정은 써도 됨 (사용 여부 과제와 달리 정답이 일정에서 오지 않음)
            f[f"plan_{k}"] = fut.shift(-k)
        c = gs["co2"]
        future_max = pd.concat([c.shift(-k) for k in range(1, 7)], axis=1).max(axis=1)
        f["y"] = ((c < 1000) & (future_max >= 1000)).astype(int)
        f["valid"] = (c < 1000) & future_max.notna() & ~fault_mask[sp]
        f["split"] = np.where(tr, "train", np.where(va, "val", "test"))
        f["space"] = sp
        rows.append(f)
    V = pd.concat(rows)
    V = V[V["valid"] & V.drop(columns=["y", "valid", "split", "space"]).notna().all(axis=1)]
    feats = [c for c in V.columns if c not in ("y", "valid", "split", "space")]
    pos_rate = V["y"].mean() * 100
    vclf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, class_weight="balanced", random_state=0)
    vclf.fit(V.loc[V.split == "train", feats], V.loc[V.split == "train", "y"])
    pv = vclf.predict_proba(V.loc[V.split == "val", feats])[:, 1]
    yv = V.loc[V.split == "val", "y"].to_numpy()
    # 검증 주에서 재현율 90% 이상을 지키는 가장 높은 임계값 (헛경보를 줄이려고)
    thr = 0.5
    for cand in np.linspace(0.95, 0.05, 91):
        if recall_score(yv, pv >= cand, zero_division=0) >= 0.90:
            thr = float(cand); break
    Te = V[V.split == "test"]
    pt = (vclf.predict_proba(Te[feats])[:, 1] >= thr).astype(int)
    v_model = metrics(Te["y"].to_numpy(), pt)
    v_rule = metrics(Te["y"].to_numpy(), (Te["co2"].to_numpy() >= 850).astype(int))
    v_all0 = metrics(Te["y"].to_numpy(), np.zeros(len(Te), int))

    with open("models/state_models.pkl", "wb") as fh:
        pickle.dump({"occupancy": clf, "vent": vclf, "vent_threshold": thr, "vent_features": feats,
                     "occ_features": list(X.columns), "spaces": spaces}, fh)

    def md(d, fmt="{:.1f}"):
        d = d.copy()
        for c in d.columns:
            if d[c].dtype.kind == "f":
                d[c] = d[c].map(lambda v: "-" if pd.isna(v) else fmt.format(v))
        return d.to_markdown()

    keys = ["정확도", "F1", "macro-F1", "재현율", "정밀도", "빈 방→사용중 오판율"]
    occ_tab = pd.DataFrame({"일정 기반 정답 (참고)": {k: agree[k] for k in keys},
                            "단순 규칙 (15분 CO₂ +30ppm)": {k: res_rule[k] for k in keys},
                            "분류 모델 (센서만)": {k: res_model[k] for k in keys}}).T
    vent_tab = pd.DataFrame({"모두 '경보 없음'": {k: v_all0[k] for k in keys[:5]},
                             "단순 규칙 (CO₂ ≥ 850)": {k: v_rule[k] for k in keys[:5]},
                             "분류 모델": {k: v_model[k] for k in keys[:5]}}).T

    lines = [
        "# 과제 5. 상태 추정",
        "",
        f"> 측정 조건: 합성 데이터 5분 간격 · 학습 1~3주 · 검증 4주 · 시험 5주(고장 구간 제외) · 사용 여부는 직접 관찰 {len(obs_days)}일(8~19시)로 평가",
        "",
        "## A. 사용 여부 추정 (사용중 / 비어있음)",
        "- **정답 정의**: 학습 정답은 일정(수업 시간표·회의실 예약)으로 만든다. 일정은 결석·노쇼·연장·예약 없는 사용 때문에 실제와 다르므로, "
        "직접 관찰한 6일로 평가한다.",
        f"- 일정 기반 정답이 직접 관찰과 일치하는 비율: **{agree['정확도']:.1f}%** (빈 방인데 일정상 '사용중'인 비율 {agree['빈 방→사용중 오판율']:.1f}%)",
        "- **입력에 일정 정보를 쓰지 않는다.** 정답을 일정에서 만들었는데 입력에도 넣으면, 모델은 센서를 보지 않고 일정을 그대로 베껴 "
        "학습 정답과 100% 가깝게 맞히는 척만 하고, 노쇼·예약 없는 사용 같은 진짜 차이는 하나도 못 잡는다.",
        "- 입력: CO₂ 수준·5/15/30분 변화량, 온도·습도 변화량, 최근 6시간 최저 대비 CO₂, 시간대·요일, 공간",
        f"- **{', '.join(sorted(no_sched)) or '-'}** 는 일정이 없어 일정 기반 정답이 늘 '비어있음'이다 → 학습에서 빼고 '처음 보는 공간'으로 평가했다 (아래 공간별 표).",
        "",
        md(occ_tab),
        "",
        "### 혼동 행렬 (분류 모델, 직접 관찰 기준)",
        cm_md(res_model["cm"], ["비어있음", "사용중"]),
        "",
        "### 공간별",
        md(per_space),
        "",
        "### 가장 많이 틀리는 상황",
        md(situ),
        "",
        "- **퇴실 직후 빈 방이 '사용중'으로 오판되기 쉽다**: 사람이 나가도 CO₂ 는 환기 속도만큼 천천히 내려가 한동안 '사람이 있던 수준'에 머문다.",
        "- **입실 직후는 '비어있음'으로 오판**: CO₂ 가 오르기 시작하는 데 몇 분이 걸린다 (센서 반응 지연 + 5분 평균).",
        "- 큰 강의실에 적은 인원이 있으면 부피 대비 발생량이 작아 CO₂ 변화가 작다 → 구별이 어렵다. 작은 회의실은 2~3명도 잘 보인다.",
        "- 냉난방 자동 제어에 쓰려면 '빈 방→사용중' 오판(빈 방에 냉방을 켜는 낭비)과 '사용중→빈 방' 오판(사람이 있는데 끄는 불편)의 "
        "비용이 다르므로, 끄는 결정은 '비어있음'이 15분 이상 이어질 때만 내리는 식으로 늦추는 것이 안전하다.",
        "",
        "## B. 환기 선제 경보 (30분 안에 1,000ppm 초과 예상)",
        f"- **양성 정의**: 지금 CO₂ < 1,000ppm 이고 앞으로 30분(6칸) 안에 1,000ppm 이상이 되는 시점. 양성 비율 {pos_rate:.1f}%",
        "- **이미 넘은 시점은 뺀다**: 넣으면 '지금 1,000 이상이면 경보'만 배워도 점수가 오르지만, 그건 '미리' 알리는 게 아니라 이미 늦은 경보다.",
        "- **불균형**: 양성이 몇 % 뿐이라 '항상 경보 없음'만 해도 정확도가 90%를 넘는다 → 정확도는 의미가 없고 재현율·정밀도·macro-F1 로 본다.",
        f"- 입력: CO₂ 수준·변화량 + 시간대 + **앞으로의 일정 인원**(여기서는 써도 됨, 정답이 일정에서 오지 않으므로). 임계값 {thr:.2f} (검증 주 재현율 90% 기준)",
        "",
        md(vent_tab),
        "",
        "### 혼동 행렬 (분류 모델)",
        cm_md(v_model["cm"], ["경보 없음", "곧 환기 필요"]),
        "",
        f"- 결과: 재현율 **{v_model['재현율']:.1f}%** ({'✅ 85% 이상' if v_model['재현율'] >= 85 else '❌ 85% 미만'}), "
        f"macro-F1 {v_model['macro-F1']:.1f} vs 단순 규칙 {v_rule['macro-F1']:.1f} ({'✅ 규칙보다 높음' if v_model['macro-F1'] > v_rule['macro-F1'] else '❌'})",
        "- 정밀도가 낮으면(헛경보가 많으면) 사람들이 경보를 무시하게 된다. 다만 환기를 늦추는 비용(졸림·두통)이 헛경보로 창문을 한 번 여는 비용보다 크므로 "
        "**재현율을 우선**하고, 같은 공간에 15분 안에 다시 울리지 않게 묶어 체감 헛경보를 줄인다.",
        "- 과제 3 예측 모델로도 '30분 뒤 예측값 ≥ 1,000' 으로 판단할 수 있다. 예측 모델은 값 전체를 맞히려다 보니 임계값 근처의 판단에는 "
        "분류기보다 둔할 수 있고, 분류기는 '넘느냐 안 넘느냐'에만 집중해 경계 근처를 더 잘 가른다. 서비스에서는 둘 다 계산해 하나라도 울리면 알리는 방식도 가능하다.",
    ]
    with open(f"{args.out}/task5_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
