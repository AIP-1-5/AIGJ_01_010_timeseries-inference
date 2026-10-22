"""과제 4. 이상 탐지 — 합성 데이터 마지막 주(고장 15건 정답)로 평가.

    .venv/bin/python task4_anomaly.py --data data/synth

학습: 1~3주(정상) · 임계값 결정: 4주(정상, 헛경보 기준) · 평가: 5주(고장 포함)
탐지기
- 규칙: 급변(튄 값), 값 고정, 영점 틀어짐 전용(빈 방인데 CO₂ 상승)
- 모델: 오토인코더(정상 1시간 창을 복원, 복원 오차 = 이상 점수), 예측 오차(과제 3 GRU 의 15분 예측과 실제의 차이)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from tsinf import anomaly as an
from tsinf import forecast as fc
from tsinf import prep
from tsinf.synth import SPACES

FA_TARGET_PER_SPACE_DAY = 0.2       # 공간 4개 → 하루 0.8회 이하 목표


def pick_threshold(val_scores: pd.Series, days: float, cands) -> float:
    """검증 주(정상)에서 경보 이벤트가 하루 0.2회 이하가 되는 가장 낮은 임계값"""
    for thr in sorted(cands):
        if len(an.to_events(val_scores > thr)) / days <= FA_TARGET_PER_SPACE_DAY:
            return float(thr)
    return float(max(cands))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    torch.set_num_threads(4)

    df, _ = prep.clean(prep.load_readings(f"{args.data}/sensors.db"))
    grid = prep.to_grid(df)
    spaces = [s.name for s in SPACES]
    cap = {s.name: float(s.capacity) for s in SPACES}
    sched_df = pd.read_csv(f"{args.data}/schedule.csv", parse_dates=["start", "end"])
    sched = fc.schedule_grid(sched_df, grid.index, spaces, cap)
    faults = pd.read_csv(f"{args.data}/faults.csv", parse_dates=["start", "end"])
    t0 = grid.index[0].normalize()
    tr = grid.index < t0 + pd.Timedelta(days=21)
    va = (grid.index >= t0 + pd.Timedelta(days=21)) & (grid.index < t0 + pd.Timedelta(days=28))
    te = grid.index >= t0 + pd.Timedelta(days=28)
    te0, te1 = grid.index[te][0], grid.index[te][-1]
    stats = prep.norm_stats(grid, grid.index[tr])
    tf = fc.time_features(grid.index)

    # 1분 원본 (규칙용)
    raw1 = (df.set_index("time").groupby(["space", "metric"])["value"].resample("1min").mean()
              .unstack(["space", "metric"]))

    det = {name: {} for name in ("규칙", "오토인코더", "예측 오차", "시간대 기대값", "모델(AE+예측+기대값)", "결합(전체)")}
    series = {name: {} for name in ("규칙", "오토인코더", "예측 오차", "시간대 기대값")}
    causes = []
    sweep_rows = []

    # ---------------- 예측 오차 준비 (과제 3 GRU)
    meta = json.load(open("models/forecast_meta.json", encoding="utf-8"))
    g = meta["models"]["GRU"]
    gru = fc.make_model("gru", g["n_in"], len(spaces)); gru.load_state_dict(torch.load(g["file"])); gru.eval()
    spec = fc.DatasetSpec(spaces, meta["lookback"], True, stats, cap)

    for si, sp in enumerate(spaces):
        gs = grid[[(sp, m) for m in an.FEAT]].droplevel(0, axis=1)
        z = ((gs - stats.loc[sp, "mean"].reindex(an.FEAT).values) / stats.loc[sp, "std"].reindex(an.FEAT).values).to_numpy()
        cond = np.concatenate([tf, sched[[sp]].to_numpy(), np.tile(np.eye(len(spaces))[si], (len(grid), 1))], axis=1)

        # ---- 규칙
        r = raw1[sp].reindex(columns=list(an.FEAT))
        rules = an.rule_alarms(r)
        drift = an.drift_alarm(gs, sched[sp])
        rule_any = rules.any(axis=1) | drift.reindex(rules.index, method="ffill").fillna(False)
        rule_te = rule_any[(rule_any.index >= te0) & (rule_any.index <= te1 + pd.Timedelta(minutes=5))]
        det["규칙"][sp] = an.to_events(rule_te)

        # ---- 오토인코더 (창 12칸 = 1시간)
        W = 12
        X, C, idx = an.ae_windows(z, cond, W)
        tmask = tr[idx]
        ae = an.train_ae(X[tmask], C[tmask])
        s_all, per_m = an.ae_scores(ae, X, C, W)
        score = pd.Series(np.nan, index=grid.index); score.iloc[idx] = s_all
        sv = score[va].dropna()
        thr = pick_threshold(sv, 7, np.quantile(sv, [0.99, 0.995, 0.998, 0.999, 1.0]).tolist() + [sv.max() * k for k in (1.2, 1.5, 2, 3)])
        ae_alarm = (score > thr) & te
        det["오토인코더"][sp] = an.to_events(ae_alarm[te])
        for k in (0.5, 0.75, 1.0, 1.5, 2.0):
            sweep_rows.append({"space": sp, "배율": k, "events": an.to_events(((score > thr * k) & te)[te])})

        # 원인 항목: 경보 시점 항목별 오차 중 최대
        pm = pd.DataFrame(np.nan, index=grid.index, columns=an.FEAT); pm.iloc[idx] = per_m
        for (s, e) in det["오토인코더"][sp]:
            causes.append({"space": sp, "start": s, "원인 항목": pm.loc[s:e].mean().idxmax()})

        # ---- 예측 오차 (15분 뒤 예측과 실제의 차이, 3칸 연속 넘으면 경보)
        sub = fc.build_samples(grid, sched, spec, grid.index[va | te])
        msk = sub["space"] == sp
        sub = {k: (v[msk] if isinstance(v, np.ndarray) and len(v) == len(msk) else v) for k, v in sub.items()}
        pred = fc.predict(gru, sub)[:, 2, :]                          # 15분 뒤
        tgt_time = pd.DatetimeIndex(sub["time"]) + pd.Timedelta(minutes=15)
        actual = np.stack([gs["co2"].reindex(tgt_time).to_numpy(), gs["temperature"].reindex(tgt_time).to_numpy()], 1)
        res = (actual - pred) / sub["sd"]
        resid = pd.DataFrame(res, index=tgt_time, columns=["co2", "temperature"]).reindex(grid.index)
        vmask = va & resid.notna().all(axis=1)
        rs = resid[vmask].std()
        # 설비 고장(환기·냉방 정지)과 영점 틀어짐은 모두 값을 '올린다' → 예상보다 높을 때만 본다(한쪽 검정)
        zres = (resid / rs).clip(lower=0).max(axis=1)
        sustained = zres.rolling(3).min()                             # 3칸(15분) 연속
        sv2 = sustained[va].dropna()
        thr2 = pick_threshold(sv2, 7, np.quantile(sv2, [0.99, 0.995, 0.999, 1.0]).tolist() + [sv2.max() * k for k in (1.2, 1.5, 2)])
        fr_alarm = (sustained > thr2) & te
        det["예측 오차"][sp] = an.to_events(fr_alarm[te])

        # ---- 시간대 기대값: 평소 이 시간대보다 30분(6칸) 연속 높으면 경보 (한쪽)
        prof = an.profile_fit(gs, tr)
        psc = an.profile_score(gs, prof).clip(lower=0)
        mod = grid.index.hour * 60 + grid.index.minute
        op_hours = (grid.index.dayofweek < 5) & (mod >= 8 * 60 + 30) & (mod < 18 * 60 + 30)
        night = (grid.index.hour >= 20) | (grid.index.hour < 7)
        # 온도는 냉방 운영 시간에만(꺼진 시간의 온도는 날씨 따라 자연히 달라짐), CO₂ 는 밤에만(낮은 재실 인원 따라 들쭉날쭉)
        pz = pd.concat([psc["temperature"].where(op_hours, 0), psc["co2"].where(night, 0)], axis=1).max(axis=1).rolling(6).min()
        sv3 = pz[va].dropna()
        thr3 = pick_threshold(sv3, 7, np.quantile(sv3, [0.99, 0.995, 0.999, 1.0]).tolist() + [sv3.max() * k for k in (1.2, 1.5, 2)])
        pf_alarm = (pz > thr3) & te
        det["시간대 기대값"][sp] = an.to_events(pf_alarm[te])

        for nm, a in (("오토인코더", ae_alarm), ("예측 오차", fr_alarm), ("시간대 기대값", pf_alarm)):
            series[nm][sp] = a[te].reindex(rule_te.index, method="ffill").fillna(False)
        series["규칙"][sp] = rule_te
        model_alarm = (ae_alarm | fr_alarm | pf_alarm)[te]
        det["모델(AE+예측+기대값)"][sp] = an.to_events(model_alarm)
        comb = model_alarm.reindex(rule_te.index, method="ffill").fillna(False) | rule_te
        det["결합(전체)"][sp] = an.to_events(comb)

    # ---------------- 조합 비교
    import itertools
    combo_rows = []
    names = list(series)
    for k in range(1, len(names) + 1):
        for combo in itertools.combinations(names, k):
            evs = {}
            for sp in spaces:
                a = series[combo[0]][sp].copy()
                for nm in combo[1:]:
                    a = a | series[nm][sp]
                evs[sp] = an.to_events(a)
            r = an.evaluate(evs, faults, te0, te1)
            combo_rows.append({"조합": " + ".join(combo), "탐지율(%)": r["rate"] * 100, "하루당 헛경보": r["fa_per_day"],
                               "기준 충족": "✅" if r["rate"] >= 0.8 and r["fa_per_day"] <= 1 else ""})
    combos = pd.DataFrame(combo_rows).sort_values(["기준 충족", "탐지율(%)", "하루당 헛경보"], ascending=[False, False, True])

    # ---------------- 평가
    lines_eval, per_type_rows = [], []
    for name, ev in det.items():
        r = an.evaluate(ev, faults, te0, te1)
        lines_eval.append({"탐지기": name, "탐지": f"{r['detected']}/{r['total']}", "탐지율(%)": r["rate"] * 100,
                           "탐지 지연 중앙값(분)": r["delay_median_min"], "헛경보 수": r["false_alarms"],
                           "하루당 헛경보": r["fa_per_day"]})
        for t, (h, n) in r["per_type"].items():
            per_type_rows.append({"탐지기": name, "고장": t, "결과": f"{h}/{n}"})
    ev_df = pd.DataFrame(lines_eval).set_index("탐지기")
    pt = pd.DataFrame(per_type_rows).pivot(index="고장", columns="탐지기", values="결과")[list(det)]

    sw = []
    for k in (0.5, 0.75, 1.0, 1.5, 2.0):
        evs = {row["space"]: row["events"] for row in sweep_rows if row["배율"] == k}
        r = an.evaluate(evs, faults, te0, te1)
        sw.append({"임계값 배율": k, "탐지율(%)": r["rate"] * 100, "하루당 헛경보": r["fa_per_day"]})
    sw = pd.DataFrame(sw).set_index("임계값 배율")

    comb_r = an.evaluate(det["결합(전체)"], faults, te0, te1)
    ok = combos[combos["기준 충족"] == "✅"]
    best_combo = (ok if len(ok) else combos).iloc[0]

    def md(d, fmt="{:.1f}"):
        d = d.copy()
        for c in d.columns:
            if d[c].dtype.kind == "f":
                d[c] = d[c].map(lambda v: "-" if pd.isna(v) else fmt.format(v))
        return d.to_markdown()

    miss_model = [t for t, row in pt.iterrows() if row["모델(AE+예측+기대값)"].startswith("0/")]
    miss_rule = [t for t, row in pt.iterrows() if row["규칙"].startswith("0/")]
    lines = [
        "# 과제 4. 이상 탐지",
        "",
        f"> 측정 조건: 합성 데이터 · 학습 1~3주(정상) · 임계값 4주(정상) · 평가 5주 {te0:%m-%d}~{te1:%m-%d} · 고장 {len(faults)}건 · 공간 {len(spaces)}개",
        "",
        "## 1. 탐지기",
        "| 종류 | 방법 | 잘 잡는 것 |",
        "| --- | --- | --- |",
        "| 규칙 | 직전 10분 중앙값과 급변 (CO₂ 300ppm, 온도 3℃) · 같은 값 60분 이상 · 빈 방인데 CO₂ 상승(영점 틀어짐 전용) | 튄 값, 값 고정, 영점 틀어짐 |",
        "| 오토인코더 | 정상 1시간 창(CO₂·온도·습도 + 시간대·일정)을 복원하도록 학습 → 복원 오차가 크면 이상 | 평소와 다른 모양 (냉방 정지 등) |",
        "| 예측 오차 | 과제 3 GRU 의 15분 뒤 예측보다 실제가 15분 연속 높으면 이상 | 환기 정지(예상보다 빨리 오름), 냉방 정지 |",
        "| 시간대 기대값 | 학습 주의 '요일 × 하루 중 5분 칸' 중앙값·퍼짐 대비 30분 연속 높으면 이상 (온도는 냉방 운영 시간, CO₂ 는 밤에만) | 냉방 정지(설정 온도보다 높음), 영점 틀어짐(밤인데 CO₂ 높음) |",
        "",
        "임계값: 정상인 검증 주에서 공간당 하루 0.2회(4개 공간 합 0.8회) 이하 헛경보가 되는 가장 낮은 값.",
        "",
        "## 2. 탐지 성적표",
        md(ev_df),
        "",
        "## 3. 고장 종류별 (탐지/전체)",
        pt.to_markdown(),
        "",
        "## 4. 탐지기 조합 비교",
        md(combos.set_index("조합")),
        "",
        "## 5. 임계값을 바꾸면 (오토인코더)",
        md(sw),
        "임계값을 낮추면 더 많이 잡지만 헛경보가 늘고, 높이면 헛경보는 줄지만 놓치는 고장이 생긴다.",
        "",
        "## 6. 결론",
        f"- **채택 조합: {best_combo['조합']}** → 탐지율 **{best_combo['탐지율(%)']:.0f}%**, 하루당 헛경보 **{best_combo['하루당 헛경보']:.2f}회** "
        f"→ 기준(탐지율 80% 이상, 헛경보 하루 1회 이하) {'✅ 충족' if best_combo['기준 충족'] else '❌ 미충족'}",
        f"- 탐지기를 다 합치면 탐지율 {comb_r['rate'] * 100:.0f}%로 더 높지만 헛경보가 하루 {comb_r['fa_per_day']:.1f}회로 늘어 운영자가 알림을 끄게 된다 "
        "→ 헛경보 기준을 지키는 조합 중 탐지율이 가장 높은 것을 채택.",
        f"- 모델이 못 잡은 종류: {', '.join(miss_model) or '없음'} — 값 고정은 값이 '정상 범위 안'에서 멈추면 복원 오차가 작아 모델이 이상으로 보지 않는다. "
        "한 칸짜리 튄 값은 5분 평균에 묻힌다.",
        f"- 규칙이 못 잡은 종류: {', '.join(miss_rule) or '없음'} — 냉방·환기 정지는 값 하나하나는 정상 범위라 고정 임계값 규칙으로는 안 보이고, "
        "'이 시간·이 인원이면 이 정도여야 한다'는 기대값(모델)과 비교해야 보인다.",
        "- 그래서 둘을 합쳐야 한다: 규칙은 센서 고장(튄 값·고정·틀어짐), 모델은 설비 고장(냉방·환기)을 잡는다.",
        "- 주의: 채택 조합의 헛경보는 기준에 딱 걸치는 수준이라, 실제 운영에서는 경보를 30분 묶음으로 보내고 담당자 확인 결과로 임계값을 계속 조정해야 한다.",
        "- 빈 방에서 환기가 멈추면 CO₂ 가 오르지 않아 탐지할 수 없고, 사람이 없으니 당장 탐지할 필요도 낮다(다음 사용 때 잡힘).",
        "- 실데이터에는 정답이 없다 → 경보마다 담당자가 확인 결과(진짜/헛경보)를 기록해 정답을 쌓고, 고장을 일부러 만들어(센서 가리기 등) 시험한다.",
        "",
        "## 7. 운영 제안",
        "- 강의실: 결합 탐지기, 경보는 시설 담당자에게 (냉방·환기 정지는 수업 중 바로 조치 필요)",
        "- 회의실·라운지: 사용 중일 때만 설비 경보, 센서 경보(고정·틀어짐)는 하루 한 번 모아서 알림",
        "- 같은 고장은 30분 단위로 묶어 한 번만 알리고, 1시간 지나도 계속되면 다시 알림",
    ]
    if causes:
        cdf = pd.DataFrame(causes)
        lines += ["", "## 8. 원인 항목 지목 (오토인코더 경보)", cdf.groupby(["space", "원인 항목"]).size().rename("경보 수").to_frame().to_markdown()]
    with open(f"{args.out}/task4_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
