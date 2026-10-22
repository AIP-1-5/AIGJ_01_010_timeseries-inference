"""과제 2. 데이터 이해와 기준선.

    .venv/bin/python task2_baseline.py --db data/sensors_boardB.db

산출물: reports/task2_report.md, reports/fig_*.png
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from tsinf import prep  # noqa: E402

HORIZONS = {"5분": 1, "15분": 3, "30분": 6, "60분": 12}   # 5분 칸 기준 몇 칸 뒤
TARGETS = ("co2", "temperature")


def setup_font():
    for name in ("AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Malgun Gothic"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def baselines(series: pd.Series, h: int) -> dict:
    """각 시점 t 에서 t+h 를 예측. 반환: {이름: 예측 Series(타깃 시점 기준 정렬)}"""
    target = series.shift(-h)
    return {
        "target": target,
        "현재값 유지": series,                                        # y(t+h) = y(t)
        "최근 1시간 평균": series.rolling(12, min_periods=6).mean(),     # 최근 12칸 평균
        "어제 같은 시각": series.shift(288 - h),                         # y(t+h) = y(t+h-24h)
    }


def evaluate(grid: pd.DataFrame, test_idx: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for (space, metric) in grid.columns:
        if metric not in TARGETS:
            continue
        s = grid[(space, metric)]
        for hname, h in HORIZONS.items():
            b = baselines(s, h)
            tgt = b.pop("target").loc[test_idx]
            # 같은 시점·같은 조건: 세 기준선 모두 값이 있는 시점만 비교 (어제 값이 없으면 그 기준선은 따로 표시)
            common = tgt.notna() & b["현재값 유지"].loc[test_idx].notna() & b["최근 1시간 평균"].loc[test_idx].notna()
            for name, pred in b.items():
                p = pred.loc[test_idx]
                mask = common & p.notna()
                mae = float((p[mask] - tgt[mask]).abs().mean()) if mask.sum() else np.nan
                rows.append({"space": space, "metric": metric, "horizon": hname, "baseline": name,
                             "MAE": mae, "n": int(mask.sum())})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sensors_boardB.db")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    setup_font()

    raw = prep.load_readings(args.db)
    df, st = prep.clean(raw)
    grid_raw = prep.to_grid(df, interpolate=False)
    grid = prep.to_grid(df, interpolate=True)
    idx = grid.index
    splits = prep.split_time(idx)
    stats = prep.norm_stats(grid, splits["train"])
    hours = (idx[-1] - idx[0]).total_seconds() / 3600

    # --- 진단 ---------------------------------------------------------------
    diag = []
    for (space, metric) in grid.columns:
        sub = df[(df.space == space) & (df.metric == metric)]
        diag.append({
            "공간": space, "항목": metric, "원본 건수": len(sub), "기기 수": sub.device.nunique(),
            "결측률(보간 전)": grid_raw[(space, metric)].isna().mean() * 100,
            "결측률(보간 후)": grid[(space, metric)].isna().mean() * 100,
            "평균": grid[(space, metric)].mean(), "최소": grid[(space, metric)].min(),
            "최대": grid[(space, metric)].max(),
        })
    diag = pd.DataFrame(diag)

    # 기기별 보고 주기
    per_dev = (df.sort_values("ts").groupby(["space", "point", "device"])["ts"]
                 .apply(lambda s: np.median(np.diff(np.unique(s))) / 60 if s.nunique() > 1 else np.nan)
                 .reset_index(name="주기(분)"))

    # --- 그림 ---------------------------------------------------------------
    spaces = sorted({c[0] for c in grid.columns})
    colors = {sp: f"C{i}" for i, sp in enumerate(spaces)}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, metric in zip(axes, prep.METRICS):
        for space in spaces:
            if (space, metric) in grid.columns:
                ax.plot(grid.index, grid[(space, metric)], label=space, lw=1, color=colors[space])
        ax.legend(ncol=5, fontsize=8, loc="upper right")
        ax.set_ylabel({"co2": "CO₂ (ppm)", "temperature": "온도 (℃)", "humidity": "습도 (%)"}[metric])
        ax.grid(alpha=0.3)
        for name, sidx in splits.items():
            if name != "train" and len(sidx):
                ax.axvline(sidx[0], color="k", ls="--", lw=0.8)
    axes[0].set_title(f"공간별 5분 평균 (점선: 학습 | 검증 | 시험 경계) · {idx[0]:%m-%d %H:%M} ~ {idx[-1]:%m-%d %H:%M}")
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_timeseries.png", dpi=120)
    plt.close(fig)

    hourly = grid.groupby(grid.index.hour).mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    for space in sorted({c[0] for c in grid.columns}):
        if (space, "co2") in hourly.columns:
            ax.plot(hourly.index, hourly[(space, "co2")], marker="o", label=space, color=colors[space])
    ax.axhline(1000, color="r", ls=":", lw=1, label="환기 권고 1,000ppm")
    ax.set_xlabel("시각 (시)"); ax.set_ylabel("CO₂ (ppm)"); ax.set_xticks(range(24)); ax.grid(alpha=0.3)
    ax.legend(fontsize=8); ax.set_title("시간대별 평균 CO₂")
    fig.tight_layout(); fig.savefig(f"{args.out}/fig_hourly_co2.png", dpi=120); plt.close(fig)

    # --- 기준선 -------------------------------------------------------------
    res = evaluate(grid, splits["test"])
    table = res.pivot_table(index=["metric", "space", "baseline"], columns="horizon", values="MAE")
    table = table[list(HORIZONS)]
    overall = res.groupby(["metric", "baseline", "horizon"])["MAE"].mean().unstack("horizon")[list(HORIZONS)]

    # --- 보고서 -------------------------------------------------------------
    def md(dfx: pd.DataFrame, fmt="{:.1f}") -> str:
        d = dfx.copy()
        for c in d.columns:
            if d[c].dtype.kind == "f":
                d[c] = d[c].map(lambda v: "-" if pd.isna(v) else fmt.format(v))
        return d.to_markdown(index=not isinstance(d.index, pd.RangeIndex))

    yday_ok = hours >= 24 + 1 and res[res.baseline == "어제 같은 시각"]["n"].sum() > 0
    lines = [
        "# 과제 2. 데이터 이해와 기준선",
        "",
        f"> 측정 조건: 실데이터 (교육장 MQTT, 보드 B 수집기) · 기간 {idx[0]:%Y-%m-%d %H:%M} ~ {idx[-1]:%Y-%m-%d %H:%M} "
        f"({hours:.1f}시간) · 분석 간격 5분 · 생성 {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## 1. 정제 규칙",
        "| 상황 | 처리 | 이유 | 이번 데이터 |",
        "| --- | --- | --- | ---: |",
        f"| 같은 업링크가 두 번 (게이트웨이 2대) | 5초 안 같은 기기·항목·값은 버림 | 같은 측정이 5분 간격 두 값처럼 보이는 가짜 데이터 | {st.duplicates:,}건 제거 |",
        f"| 게이트웨이 시계가 약 5분 빠름 | 센서 시각이 미래면 받은 시각 사용 | 시각이 틀리면 순서·간격이 무너짐 | {st.clock_fixed:,}건 보정 |",
        f"| 물리적으로 불가능한 값 (0ppm, 60℃ 등) | 제거 (CO₂ 300~5000, 온도 -10~50, 습도 0~100) | 센서·통신 오류, 학습을 망침 | "
        f"CO₂ {st.out_of_range['co2']} · 온도 {st.out_of_range['temperature']} · 습도 {st.out_of_range['humidity']}건 |",
        "| 센서마다 다른 기록 주기 (1·2·10분) | 기기별 5분 평균 → 공간 내 기기 평균 | 공통 시간축 필요, 예측도 5분 간격 | - |",
        f"| 짧은 결측 (≤15분) | 선형 보간 | CO₂·온도는 15분 안에 크게 안 변함 | - |",
        f"| 긴 결측 (>15분) | 비워 둠, 학습·평가 창에서 제외 | 긴 구간을 지어내면 가짜 패턴 학습 | - |",
        "| 위치 태그 오류 (실습실 태그인데 설치 위치 '복도') | 공간을 '복도'로 분리 | 복도 공기(18℃·83%)가 실습실 평균을 왜곡 | 센서 1대 |",
        "",
        f"원본 {st.raw:,}건 → 정제 후 {st.kept:,}건",
        "",
        "## 2. 분석 간격: 5분",
        "- 센서 보고 주기가 기기마다 1분·2분·10분으로 달라 1분 칸은 빈칸이 많다 (회의실 센서는 10분 주기).",
        "- 과제 3 예측이 '60분 앞까지 5분 간격'이므로 같은 간격이 자연스럽다.",
        "- 5분 평균은 LoRa 수신 지터와 측정 잡음을 줄인다. 대신 5분보다 짧은 급변(튄 값)은 평균에 묻히므로 과제 4 이상 탐지는 원본(1분)도 함께 본다.",
        "",
        "### 기기별 보고 주기",
        md(per_dev.set_index(["space", "point", "device"])),
        "",
        "## 3. 학습·검증·시험 분할",
        "| 구간 | 시작 | 끝 | 칸 수 |",
        "| --- | --- | --- | ---: |",
    ]
    for name, sidx in splits.items():
        if len(sidx):
            lines.append(f"| {name} | {sidx[0]:%m-%d %H:%M} | {sidx[-1]:%m-%d %H:%M} | {len(sidx)} |")
    lines += [
        "",
        "- **시간순**으로 70/15/15 분할. 무작위로 섞으면 바로 옆 5분 값이 학습과 시험에 동시에 들어가 "
        "'미래를 보고 맞히는' 누수가 생겨 성능이 실제보다 좋게 나온다.",
        "- 정규화 평균·표준편차는 **학습 구간으로만** 계산 (아래 표).",
        "",
        md(stats.reset_index().rename(columns={"level_0": "space", "level_1": "metric"}).set_index(["space", "metric"])),
        "",
        "## 4. 데이터 진단",
        md(diag.set_index(["공간", "항목"])),
        "",
        "![시계열](fig_timeseries.png)",
        "",
        "![시간대별 CO2](fig_hourly_co2.png)",
        "",
        "## 5. 기준선 성능 (시험 구간 MAE)",
        "같은 시점·같은 조건: 세 기준선 모두 시험 구간의 같은 시점에서, 정답이 있는 칸만 비교.",
        "",
        "### 전체 평균",
        md(overall),
        "",
        "### 공간별",
        md(table),
        "",
    ]
    if not yday_ok:
        lines += [
            f"> ⚠️ 현재 데이터가 {hours:.1f}시간뿐이라 **'어제 같은 시각'** 기준선은 계산할 수 없다 (24시간 이전 값이 필요). "
            "데이터가 하루 이상 쌓이면 이 스크립트를 다시 돌리면 자동으로 채워진다.",
            "",
        ]
    lines += [
        "## 6. 해석 (초안)",
        "- **가장 강한 기준선은 '현재값 유지'**일 가능성이 높다. CO₂·온도는 5~60분 사이에 천천히 변하므로 '지금 값'이 가장 좋은 추정이다. "
        "예측 시점이 길어질수록(60분) 오차가 커지는데, 수업 시작·쉬는 시간처럼 사람 수가 바뀌는 순간을 못 따라가기 때문이다.",
        "- **'최근 1시간 평균'**은 변화에 한 박자 늦게 따라가 상승·하강 구간에서 오차가 커진다.",
        "- **'어제 같은 시각'**은 수업 시간표가 매일 같다면 강하겠지만, 요일·결석·환기 여부에 따라 날마다 달라 가장 약할 것으로 예상한다. "
        "매일 같은 일정으로 쓰이는 공간(예: 고정 수업 강의실)이라면 이 방법이 강해질 수 있다.",
        "- **과제 3 목표**: 60분 뒤 CO₂ 오차를 '현재값 유지'보다 30% 이상 줄이기.",
        "- **데이터 한계**: 일정(수업·예약) 정보가 없어 '일정 변화 후 CO₂ 반응 시간' 분석은 제공 키트 또는 강사님 일정 자료가 필요하다.",
    ]
    with open(f"{args.out}/task2_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print("...\n전체 평균 MAE:\n", overall.round(2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
