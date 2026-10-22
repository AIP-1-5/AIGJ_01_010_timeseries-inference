"""수집 현황: 공간·항목별 건수, 마지막 수신 시각, 커버율, 빠진 구간.

    python tools/status.py                 # 최근 24시간
    python tools/status.py --hours 6       # 최근 6시간
    python tools/status.py --markdown > report.md   # 보고서용 표

커버율 = (기간 안에서 데이터가 1건 이상 있는 1분 칸 수) / (기간의 전체 1분 칸 수)
기간이 수집 시작 전부터라면 '수집 시작 이후 커버율'도 따로 보여준다.
빠진 구간이 수집기 끊김(events 테이블)과 겹치면 원인으로 표시한다.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from collector.config import load_config  # noqa: E402
from collector.store import connect  # noqa: E402

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def fmt_ts(ts: Optional[float], tz: str) -> str:
    if ts is None:
        return "-"
    zi = ZoneInfo(tz) if ZoneInfo else None
    return datetime.fromtimestamp(ts, zi).strftime("%m-%d %H:%M:%S")


def fmt_age(sec: float) -> str:
    if sec < 120:
        return f"{int(sec)}초 전"
    if sec < 7200:
        return f"{int(sec // 60)}분 전"
    return f"{sec / 3600:.1f}시간 전"


def find_gaps(minutes: List[int], start_min: int, end_min: int, min_gap: int) -> List[Tuple[int, int]]:
    """데이터가 있는 분 목록에서 min_gap 분 이상 비어 있는 구간 [시작, 끝) 을 찾는다."""
    gaps = []
    prev = start_min - 1
    for m in minutes + [end_min]:
        if m - prev - 1 >= min_gap:
            gaps.append((prev + 1, m))
        prev = m
    return gaps


def explain_gap(conn: sqlite3.Connection, g_start: float, g_end: float) -> str:
    rows = conn.execute(
        "SELECT ts, event, detail FROM events WHERE ts BETWEEN ? AND ? ORDER BY ts",
        (g_start - 120, g_end + 120),
    ).fetchall()
    events = [r["event"] for r in rows]
    if not rows:
        last = conn.execute("SELECT event FROM events WHERE ts < ? ORDER BY ts DESC LIMIT 1", (g_start,)).fetchone()
        if last and last["event"] in ("stop", "disconnected"):
            return f"수집기 {last['event']} 상태였음"
        return "수집기는 연결 중 → 센서/게이트웨이 쪽 미전송 추정"
    if "stop" in events or "start" in events:
        return "수집기 재시작/정지 (" + ", ".join(events) + ")"
    if "disconnected" in events:
        return "브로커 연결 끊김 (" + ", ".join(events) + ")"
    return ", ".join(events)


def build_report(db: str, hours: float, min_gap: int, tz: str, now: Optional[float] = None) -> dict:
    conn = connect(db, readonly=True)
    now = time.time() if now is None else now
    start = now - hours * 3600
    # 지금 진행 중인 1분 칸은 아직 덜 찼으므로 커버율 계산에서 뺀다
    start_min, end_min = int(start // 60), int(now // 60)
    total_min = max(end_min - start_min, 1)

    series = conn.execute(
        "SELECT space, metric, COUNT(*) AS n, MAX(recv_ts) AS last_recv, MIN(ts) AS first_ts "
        "FROM readings WHERE ts >= ? GROUP BY space, metric ORDER BY space, metric",
        (start,),
    ).fetchall()

    rows = []
    for s in series:
        mins = [r[0] for r in conn.execute(
            "SELECT DISTINCT CAST(ts / 60 AS INTEGER) FROM readings "
            "WHERE space = ? AND metric = ? AND ts >= ? AND ts < ? ORDER BY 1",
            (s["space"], s["metric"], start_min * 60, end_min * 60),
        )]
        first_min = int(s["first_ts"] // 60)
        since_total = max(end_min - first_min, 1)
        gaps = find_gaps(mins, start_min, end_min, min_gap)
        rows.append({
            "space": s["space"],
            "metric": s["metric"],
            "count": s["n"],
            "last_recv": s["last_recv"],
            "coverage": len(mins) / total_min * 100,
            "coverage_since_start": len(mins) / since_total * 100,
            "gaps": [(a * 60, b * 60, explain_gap(conn, a * 60, b * 60)) for a, b in gaps
                     if a >= first_min],  # 수집 시작 전 구간은 결측으로 치지 않음
        })

    # 기기별: 센서는 보통 1분이 아니라 자기 주기(예: 2분)로 보낸다 → 주기를 추정해 '주기 기준 커버율'을 따로 계산
    devices = []
    for d in conn.execute(
        "SELECT device, space, point, GROUP_CONCAT(DISTINCT metric) AS metrics, MAX(recv_ts) AS last_recv "
        "FROM readings WHERE ts >= ? AND device != '' GROUP BY device, space ORDER BY space, point",
        (start,),
    ).fetchall():
        tss = [r[0] for r in conn.execute(
            "SELECT DISTINCT ts FROM readings WHERE device = ? AND space = ? AND ts >= ? ORDER BY ts",
            (d["device"], d["space"], start),
        )]
        diffs = sorted(b - a for a, b in zip(tss, tss[1:]) if b - a > 1)
        interval = diffs[len(diffs) // 2] if diffs else None
        cov = None
        if interval and len(tss) >= 2:
            span_start = max(start, tss[0])
            expected = max((now - span_start) / interval, 1)
            slots = len({int((t - span_start) // interval) for t in tss})
            cov = min(slots / expected * 100, 100.0)
        max_gap = max(diffs) if diffs else None
        devices.append({
            "device": d["device"], "space": d["space"], "point": d["point"], "metrics": d["metrics"],
            "count": len(tss), "last_recv": d["last_recv"], "interval": interval,
            "coverage": cov, "max_gap": max_gap,
        })

    counters = {r["name"]: r["value"] for r in conn.execute("SELECT name, value FROM counters")}
    last_event = conn.execute("SELECT ts, event, detail FROM events ORDER BY ts DESC LIMIT 1").fetchone()
    unmatched_topics = conn.execute(
        "SELECT topic, COUNT(*) AS n FROM unmatched WHERE recv_ts >= ? GROUP BY topic ORDER BY n DESC LIMIT 10",
        (start,),
    ).fetchall()
    conn.close()
    return {
        "now": now, "hours": hours, "rows": rows, "devices": devices, "counters": counters,
        "last_event": dict(last_event) if last_event else None,
        "unmatched_topics": [dict(r) for r in unmatched_topics], "tz": tz,
    }


def print_text(rep: dict) -> None:
    tz, now = rep["tz"], rep["now"]
    print(f"=== 수집 현황 (최근 {rep['hours']:g}시간, 기준 {fmt_ts(now, tz)}) ===")
    if rep["last_event"]:
        e = rep["last_event"]
        print(f"수집기 마지막 상태: {e['event']} ({fmt_ts(e['ts'], tz)}) {e['detail'] or ''}")
    c = rep["counters"]
    print(f"누적: 저장 {c.get('readings', 0)}건 · 중복 제거 {c.get('duplicates', 0)}건 · "
          f"대상 항목 없음 {c.get('ignored', 0)}건 · 규칙 불일치 {c.get('unmatched', 0)}건\n")
    if not rep["rows"]:
        print("이 기간에 저장된 데이터가 없습니다.")
    else:
        print(f"{'공간':<14}{'항목':<14}{'건수':>8}  {'마지막 수신':<16}{'경과':<10}{'커버율':>8}{'시작후':>8}")
        for r in rep["rows"]:
            age = fmt_age(now - r["last_recv"])
            print(f"{r['space']:<14}{r['metric']:<14}{r['count']:>8}  {fmt_ts(r['last_recv'], tz):<16}"
                  f"{age:<10}{r['coverage']:>7.1f}%{r['coverage_since_start']:>7.1f}%")
        print()
        for r in rep["rows"]:
            for a, b, why in r["gaps"]:
                print(f"  빈 구간 {r['space']}/{r['metric']}: {fmt_ts(a, tz)} ~ {fmt_ts(b, tz)} "
                      f"({int((b - a) // 60)}분) → {why}")
    if rep.get("devices"):
        print("\n[기기별] 센서 자기 주기 기준 커버율")
        print(f"{'공간':<10}{'위치':<14}{'기기':<18}{'건수':>6}  {'주기':>6}  {'커버율':>7}  {'최대공백':>8}  마지막 수신")
        for d in rep["devices"]:
            iv = f"{d['interval'] / 60:.1f}분" if d["interval"] else "-"
            cov = f"{d['coverage']:.1f}%" if d["coverage"] is not None else "-"
            mg = f"{d['max_gap'] / 60:.0f}분" if d["max_gap"] else "-"
            print(f"{d['space']:<10}{(d['point'] or '-'):<14}{d['device'][-16:]:<18}{d['count']:>6}  {iv:>6}  {cov:>7}  {mg:>8}  "
                  f"{fmt_ts(d['last_recv'], tz)} ({fmt_age(now - d['last_recv'])})")
    if rep["unmatched_topics"]:
        print("\n규칙에 안 맞아 저장 못 한 토픽 (config.yaml rules 확인):")
        for u in rep["unmatched_topics"]:
            print(f"  {u['topic']}  ({u['n']}건)")


def print_markdown(rep: dict) -> None:
    tz, now = rep["tz"], rep["now"]
    print(f"### 수집 현황 (최근 {rep['hours']:g}시간, 기준 {fmt_ts(now, tz)})\n")
    print("| 공간 | 항목 | 건수 | 마지막 수신 | 커버율 | 수집 시작 후 커버율 |")
    print("| --- | --- | ---: | --- | ---: | ---: |")
    for r in rep["rows"]:
        print(f"| {r['space']} | {r['metric']} | {r['count']} | {fmt_ts(r['last_recv'], tz)} | "
              f"{r['coverage']:.1f}% | {r['coverage_since_start']:.1f}% |")
    print("\n### 결측 구간과 추정 원인\n")
    print("| 공간 | 항목 | 시작 | 끝 | 길이(분) | 추정 원인 |")
    print("| --- | --- | --- | --- | ---: | --- |")
    any_gap = False
    for r in rep["rows"]:
        for a, b, why in r["gaps"]:
            any_gap = True
            print(f"| {r['space']} | {r['metric']} | {fmt_ts(a, tz)} | {fmt_ts(b, tz)} | {int((b - a) // 60)} | {why} |")
    if not any_gap:
        print("| - | - | - | - | - | 결측 구간 없음 |")
    if rep.get("devices"):
        print("\n### 기기별 (센서 자기 주기 기준)\n")
        print("| 공간 | 위치 | 기기 | 항목 | 건수 | 주기(분) | 주기 기준 커버율 | 최대 공백(분) |")
        print("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
        for d in rep["devices"]:
            iv = f"{d['interval'] / 60:.1f}" if d["interval"] else "-"
            cov = f"{d['coverage']:.1f}%" if d["coverage"] is not None else "-"
            mg = f"{d['max_gap'] / 60:.0f}" if d["max_gap"] else "-"
            print(f"| {d['space']} | {d['point'] or '-'} | {d['device']} | {d['metrics']} | {d['count']} | {iv} | {cov} | {mg} |")
    c = rep["counters"]
    print(f"\n누적 저장 {c.get('readings', 0)}건 · 중복 제거 {c.get('duplicates', 0)}건 · 규칙 불일치 {c.get('unmatched', 0)}건")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="수집 현황 보기")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--db", help="DB 경로 (기본: config.yaml 의 storage.path)")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--gap-min", type=int, default=15,
                    help="이 분 수 이상 비면 빈 구간으로 표시 (센서 주기가 1~2분이라 기본 15분 = 여러 번 연속 누락)")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)

    tz = "Asia/Seoul"
    db = args.db
    if not db or os.path.exists(args.config):
        try:
            cfg = load_config(args.config, env_file=None)
            tz = cfg.timezone
            db = db or cfg.storage.path
        except (FileNotFoundError, ValueError):
            pass
    if not db or not os.path.exists(db):
        print(f"DB 파일이 없습니다: {db}. 수집기를 먼저 실행하세요.")
        return 1
    rep = build_report(db, args.hours, args.gap_min, tz)
    (print_markdown if args.markdown else print_text)(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
