"""과제 4 보강: 영점 틀어짐 전용 탐지 단독 성적 · 채택 조합(규칙+예측 오차) 종류별 지연 · 잔차 임계값 표 · 원인 지목 정확도.

    PYTHONPATH=. .venv/bin/python addenda_task4.py

원본 task4_anomaly.py 와 같은 데이터·분할·탐지기 설정을 그대로 다시 계산한다 (원본 파일은 건드리지 않음).
선택은 모두 4주(검증, 정상)로 하고, 5주(시험, 고장 15건)는 보고만 한다.
영점 틀어짐 검증용으로, 4주 사본에 같은 방식(6~12시간 동안 250~400ppm 선형 상승)의 가짜 틀어짐을 심어 탐지율을 본다.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from tsinf import anomaly as an
from tsinf import forecast as fc
from tsinf import prep
from tsinf.synth import SPACES

DATA = "data/synth"
OUT = "reports/addenda/task4_addenda_tables.json"
FA_PER_SPACE_DAY = 0.2
torch.set_num_threads(4)


def fault_eval(alarm1: dict, cause1: dict, faults, t0, t1, margin="30min"):
    """alarm1/cause1: {space: 1분 bool 시리즈 / 1분 원인 항목 시리즈}. 고장별 탐지·지연·원인 항목 반환 + 헛경보."""
    mg = pd.Timedelta(margin)
    evs = {sp: an.to_events(a) for sp, a in alarm1.items()}
    used = {sp: [False] * len(e) for sp, e in evs.items()}
    rows = []
    for f in faults.itertuples():
        hits = [(k, s, e) for k, (s, e) in enumerate(evs.get(f.space, [])) if s <= f.end + mg and e >= f.start]
        for k, _, _ in hits:
            used[f.space][k] = True
        if hits:
            s0 = min(s for _, s, _ in hits)
            # 원인: 고장 시작 이후(또는 이벤트 시작) 첫 경보 시각의 원인 항목
            a = alarm1[f.space]
            win = a[(a.index >= max(s0, f.start - mg)) & (a.index <= f.end + mg)]
            first_t = win[win].index[0] if win.any() else s0
            cause = cause1[f.space].get(first_t, None)
            rows.append({"type": f.type, "space": f.space, "metric": f.metric, "hit": True,
                         "delay": max(0.0, (s0 - f.start).total_seconds() / 60), "cause": cause})
        else:
            rows.append({"type": f.type, "space": f.space, "metric": f.metric, "hit": False, "delay": np.nan, "cause": None})
    days = (t1 - t0 + pd.Timedelta(minutes=5)).total_seconds() / 86400
    fa = sum(1 for u in used.values() for x in u if not x)
    return pd.DataFrame(rows), fa, fa / days


def fa_only(alarm: dict, days: float):
    n = sum(len(an.to_events(a)) for a in alarm.values())
    return n, n / days


def main():
    df, _ = prep.clean(prep.load_readings(f"{DATA}/sensors.db"))
    grid = prep.to_grid(df)
    spaces = [s.name for s in SPACES]
    cap = {s.name: float(s.capacity) for s in SPACES}
    sched = fc.schedule_grid(pd.read_csv(f"{DATA}/schedule.csv", parse_dates=["start", "end"]), grid.index, spaces, cap)
    faults = pd.read_csv(f"{DATA}/faults.csv", parse_dates=["start", "end"])
    t0 = grid.index[0].normalize()
    tr = grid.index < t0 + pd.Timedelta(days=21)
    va = (grid.index >= t0 + pd.Timedelta(days=21)) & (grid.index < t0 + pd.Timedelta(days=28))
    te = grid.index >= t0 + pd.Timedelta(days=28)
    va0, va1 = grid.index[va][0], grid.index[va][-1]
    te0, te1 = grid.index[te][0], grid.index[te][-1]
    stats = prep.norm_stats(grid, grid.index[tr])
    raw1 = (df.set_index("time").groupby(["space", "metric"])["value"].resample("1min").mean().unstack(["space", "metric"]))
    out = {"period": {"train": f"{grid.index[0]}~{grid.index[tr][-1]}", "val": f"{va0}~{va1}", "test": f"{te0}~{te1}"}}

    # =========================================================== 1) 영점 틀어짐 전용 단독
    drift_f = faults[faults.type == "영점 틀어짐"]
    rng = np.random.default_rng(42)
    # 검증 주에 가짜 틀어짐 심기: 공간당 3건, 서로 다른 날, 시작 시각은 아무 때나(원본 합성과 같은 방식)
    inj, gval = [], grid.copy()
    vidx = grid.index[va]
    for sp in spaces:
        days_pick = rng.choice(np.arange(7), 3, replace=False)
        for d in sorted(days_pick):
            st = vidx[0] + pd.Timedelta(days=int(d)) + pd.Timedelta(minutes=int(rng.integers(0, 24 * 12)) * 5)
            dur = int(rng.integers(6 * 12, 12 * 12))           # 5분 칸 (6~12시간)
            amp = rng.uniform(250, 400)
            ii = np.searchsorted(grid.index, st)
            ii1 = min(ii + dur, np.searchsorted(grid.index, vidx[-1]))
            gval.iloc[ii:ii1, gval.columns.get_loc((sp, "co2"))] += np.linspace(0, amp, ii1 - ii)
            inj.append({"type": "영점 틀어짐", "space": sp, "metric": "co2", "start": grid.index[ii], "end": grid.index[ii1]})
    inj = pd.DataFrame(inj)
    out["injected_val_drifts"] = [{"space": r.space, "start": str(r.start), "hours": (r.end - r.start).total_seconds() / 3600} for r in inj.itertuples()]

    def drift_run(g, mask, flist, **kw):
        al, ca = {}, {}
        for sp in spaces:
            gs = g[[(sp, m) for m in an.FEAT]].droplevel(0, axis=1)
            a = an.drift_alarm(gs, sched[sp], **kw)[mask]
            al[sp] = a
            ca[sp] = pd.Series("co2", index=a.index)
        idx = g.index[mask]
        return fault_eval(al, ca, flist, idx[0], idx[-1])

    drift_rows = []
    for c_out in (420.0, 400.0, 450.0):          # above = c_out + 80
        for rise in (10, 15, 25, 40, 60):
            for emp in (1.0, 1.5, 3.0):
                kw = dict(c_out=c_out, rise_ppm=rise, min_empty_h=emp)
                vclean = drift_run(grid, va, inj.iloc[0:0], **kw)            # 깨끗한 검증 주 → 헛경보
                vinj = drift_run(gval, va, inj, **kw)                        # 가짜 틀어짐 심은 검증 주 → 탐지
                tst = drift_run(grid, te, drift_f, **kw)                     # 시험 주(진짜 고장 2건)
                tst_all, fa_te, fa_te_d = drift_run(grid, te, faults, **kw)  # 시험 주 헛경보 (어떤 고장과도 안 겹친 경보)
                drift_rows.append({"c_out": c_out, "above": c_out + 80, "rise_ppm": rise, "min_empty_h": emp,
                                   "val_det": int(vinj[0].hit.sum()), "val_tot": len(inj),
                                   "val_delay_med": float(vinj[0].delay.median()),
                                   "val_fa_day": vclean[2],
                                   "test_det": int(tst[0].hit.sum()), "test_tot": len(drift_f),
                                   "test_delays": tst[0].delay.round(0).tolist(),
                                   "test_fa_day": fa_te_d})
    dr = pd.DataFrame(drift_rows)
    # 기본/선택 설정 시험 주 헛경보 사건
    for kw in ({}, dict(c_out=400.0, rise_ppm=10, min_empty_h=1.0)):
        evl = []
        for sp in spaces:
            gs = grid[[(sp, m) for m in an.FEAT]].droplevel(0, axis=1)
            for (a0, a1) in an.to_events(an.drift_alarm(gs, sched[sp], **kw)[te]):
                ff = faults[(faults.space == sp) & (faults.start <= a1) & (faults.end + pd.Timedelta("30min") >= a0)]
                evl.append({"space": sp, "start": str(a0), "end": str(a1), "fault": ",".join(ff.type) or "헛경보"})
        out[f"drift_test_events_{'pick' if kw else 'default'}"] = evl
        print(evl)
    out["drift_sweep"] = dr.to_dict("records")
    # 검증으로 고르기: 검증 헛경보 0.8/일(4공간) 이하 중 검증 탐지 최대 → 지연 짧은 것
    ok = dr[dr.val_fa_day <= FA_PER_SPACE_DAY * 4].sort_values(["val_det", "val_delay_med"], ascending=[False, True])
    out["drift_pick"] = ok.iloc[0].to_dict() if len(ok) else None
    # 시작 시각별 지연 (기본 설정, 검증 주입)
    v_def = drift_run(gval, va, inj)[0]
    out["drift_default_val_detail"] = [{"space": r.space, "start": str(st), "hit": bool(r.hit), "delay": r.delay}
                                       for r, st in zip(v_def.itertuples(), inj.start)]
    print(dr.to_string())

    # =========================================================== 2) 채택 조합: 규칙 + 예측 오차
    meta = json.load(open("models/forecast_meta.json", encoding="utf-8"))
    g = meta["models"]["GRU"]
    gru = fc.make_model("gru", g["n_in"], len(spaces)); gru.load_state_dict(torch.load(g["file"])); gru.eval()
    spec = fc.DatasetSpec(spaces, meta["lookback"], True, stats, cap)

    rule_al = {"va": {}, "te": {}}
    rule_ca = {"va": {}, "te": {}}
    zres_all, zarg_all, rs_all = {}, {}, {}

    def residual(gX, sp):
        gs = gX[[(sp, m) for m in an.FEAT]].droplevel(0, axis=1)
        sub = fc.build_samples(gX, sched, spec, gX.index[va | te])
        msk = sub["space"] == sp
        sub = {k: (v[msk] if isinstance(v, np.ndarray) and len(v) == len(msk) else v) for k, v in sub.items()}
        pred = fc.predict(gru, sub)[:, 2, :]
        tgt = pd.DatetimeIndex(sub["time"]) + pd.Timedelta(minutes=15)
        actual = np.stack([gs["co2"].reindex(tgt).to_numpy(), gs["temperature"].reindex(tgt).to_numpy()], 1)
        return pd.DataFrame((actual - pred) / sub["sd"], index=tgt, columns=["co2", "temperature"]).reindex(gX.index)
    for sp in spaces:
        gs = grid[[(sp, m) for m in an.FEAT]].droplevel(0, axis=1)
        r = raw1[sp].reindex(columns=list(an.FEAT))
        rules = an.rule_alarms(r)
        drift = an.drift_alarm(gs, sched[sp]).reindex(rules.index, method="ffill").fillna(False)
        rr = rules.copy(); rr["drift_co2"] = drift
        any_ = rr.any(axis=1)
        # 원인 항목: 울린 규칙 열 이름의 항목 (여럿이면 첫 번째)
        cause = rr.idxmax(axis=1).str.split("_").str[-1].where(any_)
        for nm, (a0, a1) in (("va", (va0, va1)), ("te", (te0, te1))):
            m = (any_.index >= a0) & (any_.index <= a1 + pd.Timedelta(minutes=5))
            rule_al[nm][sp] = any_[m]; rule_ca[nm][sp] = cause[m]

        resid = residual(grid, sp)
        rs = resid[va & resid.notna().all(axis=1)].std()
        rs_all[sp] = rs
        zr = (resid / rs).clip(lower=0)
        zres_all[sp] = zr.max(axis=1)
        zarg_all[sp] = zr.fillna(-1).idxmax(axis=1)

    def resid_alarm(sp, z, k, nm):
        mask = va if nm == "va" else te
        s = zres_all[sp].rolling(k).min()
        a = (s > z)[mask]
        idx1 = rule_al[nm][sp].index
        a1 = a.reindex(idx1, method="ffill").fillna(False).astype(bool)
        c1 = zarg_all[sp][mask].reindex(idx1, method="ffill")
        return a1, c1

    # 원본 방식 임계값 (공간별: 검증 0.2회/일 이하 가장 낮은 값, 3칸) 재현
    orig_thr = {}
    for sp in spaces:
        s = zres_all[sp].rolling(3).min()
        sv = s[va].dropna()
        cands = np.quantile(sv, [0.99, 0.995, 0.999, 1.0]).tolist() + [sv.max() * k for k in (1.2, 1.5, 2)]
        thr = max(cands)
        for c in sorted(cands):
            if len(an.to_events(sv > c)) / 7 <= FA_PER_SPACE_DAY:
                thr = c; break
        orig_thr[sp] = float(thr)
    out["orig_resid_thr"] = orig_thr

    def combo(nm, thr_fn, k):
        al, ca = {}, {}
        for sp in spaces:
            ra, rc = resid_alarm(sp, thr_fn(sp), k, nm)
            al[sp] = rule_al[nm][sp] | ra
            # 규칙이 울린 시각은 규칙 원인, 아니면 잔차 원인
            ca[sp] = rule_ca[nm][sp].where(rule_al[nm][sp], rc.where(ra))
        return al, ca

    # 채택 조합 (원본 임계값) 시험 성적
    al, ca = combo("te", lambda sp: orig_thr[sp], 3)
    per, fa, fad = fault_eval(al, ca, faults, te0, te1)
    out["adopted_test"] = {"detected": int(per.hit.sum()), "total": len(per), "fa": fa, "fa_day": fad}
    pt = per.groupby("type").agg(탐지=("hit", "sum"), 전체=("hit", "size"), 지연평균=("delay", "mean"), 지연중앙값=("delay", "median"))
    out["adopted_per_type"] = pt.reset_index().to_dict("records")
    out["adopted_per_fault"] = per.to_dict("records")
    alv, _ = combo("va", lambda sp: orig_thr[sp], 3)
    out["adopted_val_fa_day"] = fa_only(alv, 7)[1]
    out["rules_only_val_fa_day"] = fa_only(rule_al["va"], 7)[1]
    # 원인 지목 정확도
    hit = per[per.hit]
    out["cause_acc"] = {"match": int((hit.cause == hit.metric).sum()), "n": len(hit),
                        "by_type": hit.assign(ok=hit.cause == hit.metric).groupby("type").ok.agg(["sum", "size"]).reset_index().to_dict("records")}
    print(per.to_string()); print(pt)

    # 검증 주에 설비 고장 근사 주입: 냉방 정지(수업 중 온도 3~5℃ 선형 상승), 환기 정지(창 시작 대비 CO₂ 상승 폭 1.8배)
    gh = gval.copy()                 # 틀어짐 12건이 이미 심긴 사본에 추가
    hv = []
    plan = [("강의실1", "2026-10-06", (13, 17), "냉방 정지"), ("강의실1", "2026-10-08", (9, 12), "환기 정지"),
            ("강의실2", "2026-10-05", (9, 12), "냉방 정지"), ("강의실2", "2026-10-06", (13, 17), "환기 정지"),
            ("회의실", "2026-10-07", (13, 17), "환기 정지"), ("회의실", "2026-10-09", (9, 12), "냉방 정지")]
    for sp, day, (h0, h1), kind in plan:
        a = pd.Timestamp(f"{day} {h0:02d}:00", tz=grid.index.tz); b = pd.Timestamp(f"{day} {h1:02d}:00", tz=grid.index.tz)
        m = (gh.index >= a) & (gh.index < b)
        if kind == "냉방 정지":
            gh.loc[m, (sp, "temperature")] += np.linspace(0, rng.uniform(3, 5), m.sum())
            met = "temperature"
        else:
            c = gh.loc[m, (sp, "co2")]; c0 = c.iloc[0]
            gh.loc[m, (sp, "co2")] = np.maximum(c, c0 + 1.8 * (c - c0))
            met = "co2"
        hv.append({"type": kind, "space": sp, "metric": met, "start": a, "end": b})
    inj_all = pd.concat([inj, pd.DataFrame(hv)], ignore_index=True)
    zres_inj = {}
    for sp in spaces:
        zr = (residual(gh, sp) / rs_all[sp]).clip(lower=0)
        zres_inj[sp] = zr.max(axis=1)
    out["injected_val_hvac"] = [{k: str(v) for k, v in r.items()} for r in hv]

    # 잔차 임계값 표: z(공통 값) × 연속 칸 → 검증 헛경보 / 시험 탐지
    sweep = []
    for k in (1, 2, 3, 4, 6, 8):
        for z in (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            alv, _ = combo("va", lambda sp: z, k)
            alt, cat = combo("te", lambda sp: z, k)
            pr, fa_t, fad_t = fault_eval(alt, cat, faults, te0, te1)
            # 잔차 단독
            ra_v = {sp: resid_alarm(sp, z, k, "va")[0] for sp in spaces}
            ra_t = {sp: resid_alarm(sp, z, k, "te")[0] for sp in spaces}
            pr_r, _, fad_r = fault_eval(ra_t, {sp: pd.Series(dtype=object) for sp in spaces}, faults, te0, te1)
            ra_inj = {sp: (zres_inj[sp].rolling(k).min() > z)[va] for sp in spaces}
            pv, _, _ = fault_eval(ra_inj, {sp: pd.Series(dtype=object) for sp in spaces}, inj_all, va0, va1)
            vt = pv.groupby("type").hit.sum().to_dict()
            sweep.append({"k": k, "z": z, "val_inj_det": int(pv.hit.sum()), "val_inj_tot": len(pv),
                          "val_inj_hvac": int(vt.get("냉방 정지", 0) + vt.get("환기 정지", 0)), "val_inj_types": vt, "val_fa_day_combo": fa_only(alv, 7)[1], "val_fa_day_resid": fa_only(ra_v, 7)[1],
                          "test_det_combo": int(pr.hit.sum()), "test_fa_day_combo": fad_t,
                          "test_det_resid": int(pr_r.hit.sum()), "test_fa_day_resid": fad_r,
                          "test_resid_types": pr_r[pr_r.hit].type.value_counts().to_dict()})
    sw = pd.DataFrame(sweep)
    out["resid_sweep"] = sw.to_dict("records")
    print(sw.to_string())
    # 검증으로 고르기: 검증 헛경보(조합, 4공간 합) 0.8/일 이하 중 주입 설비 고장 탐지 → 주입 전체 탐지 → 헛경보 적은 순
    cand = sw[sw.val_fa_day_combo <= FA_PER_SPACE_DAY * 4].sort_values(["val_inj_hvac", "val_inj_det", "val_fa_day_combo"], ascending=[False, False, True])
    pk = cand.iloc[0]
    out["resid_pick"] = {"k": int(pk.k), "z": float(pk.z)}
    al, ca = combo("te", lambda sp: float(pk.z), int(pk.k))
    per2, fa2, fad2 = fault_eval(al, ca, faults, te0, te1)
    out["pick_test"] = {"detected": int(per2.hit.sum()), "total": len(per2), "fa": fa2, "fa_day": fad2}
    out["pick_per_type"] = per2.groupby("type").agg(det=("hit", "sum"), tot=("hit", "size"), mean=("delay", "mean"), med=("delay", "median")).reset_index().to_dict("records")
    out["pick_per_fault"] = per2.to_dict("records")
    h2 = per2[per2.hit]
    out["pick_cause_acc"] = {"match": int((h2.cause == h2.metric).sum()), "n": len(h2)}
    mg = pd.Timedelta("30min")
    fa_list = []
    for sp in spaces:
        for (a0, a1) in an.to_events(al[sp]):
            ff = faults[(faults.space == sp) & (faults.start <= a1) & (faults.end + mg >= a0)]
            if ff.empty:
                fa_list.append({"space": sp, "start": str(a0), "end": str(a1), "cause": ca[sp].get(a0)})
    out["pick_fa_events"] = fa_list
    print(fa_list)
    print("PICK", out["resid_pick"], out["pick_test"]); print(per2.to_string())
    # 조합 중 원본 공간별 임계값의 검증 주입 탐지
    ra_inj = {sp: (zres_inj[sp].rolling(3).min() > orig_thr[sp])[va] for sp in spaces}
    pv, _, _ = fault_eval(ra_inj, {sp: pd.Series(dtype=object) for sp in spaces}, inj_all, va0, va1)
    out["orig_thr_val_inj"] = pv.groupby("type").hit.agg(["sum", "size"]).reset_index().to_dict("records")
    print(out["orig_thr_val_inj"])
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
