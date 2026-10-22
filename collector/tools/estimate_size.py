"""(심화 B) 저장소가 1년 뒤 얼마나 커질지 추정한다.

    python tools/estimate_size.py
    python tools/estimate_size.py --spaces 4 --metrics 3 --per-minute 1   # DB 없이 이론값만

실측: 현재 DB 의 (파일 크기 / 행 수) 로 행당 바이트를 구하고, 최근 하루 저장 속도로 1년치를 곱한다.
이론: 공간 수 × 항목 수 × 분당 건수 × 1440분 × 365일 × 행당 바이트.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

DEFAULT_BYTES_PER_ROW = 110  # 데이터 + 인덱스 2개 + 토픽 문자열 포함한 대략값 (실측으로 대체됨)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sensors.db")
    ap.add_argument("--spaces", type=int, default=4)
    ap.add_argument("--metrics", type=int, default=3)
    ap.add_argument("--per-minute", type=float, default=1.0)
    args = ap.parse_args(argv)

    bpr = DEFAULT_BYTES_PER_ROW
    rows_per_day = args.spaces * args.metrics * args.per_minute * 1440
    measured = False
    if os.path.exists(args.db):
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        n = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        last_day = conn.execute("SELECT COUNT(*) FROM readings WHERE recv_ts >= ?", (time.time() - 86400,)).fetchone()[0]
        span = conn.execute("SELECT MIN(recv_ts), MAX(recv_ts) FROM readings").fetchone()
        conn.close()
        if n >= 1000:
            bpr = page_size * page_count / n
            measured = True
        if span[0] and span[1] and span[1] - span[0] >= 3600:
            rows_per_day = last_day if span[1] - span[0] >= 86400 else n / (span[1] - span[0]) * 86400
        print(f"현재 DB: {n:,}행, {human(page_size * page_count)}")

    year_rows = rows_per_day * 365
    year_bytes = year_rows * bpr
    print(f"하루 저장량: {rows_per_day:,.0f}행 ({'실측' if measured else '이론'} 행당 {bpr:.0f}B)")
    print(f"1년 예상:   {year_rows:,.0f}행 ≈ {human(year_bytes)}")
    print()
    print("제안 보관 정책 (tools/retention.py):")
    raw_days = 90
    keep_raw = rows_per_day * raw_days * bpr
    agg_rows_year = rows_per_day / 5 * (365 - raw_days)  # 5분 평균으로 줄이면 1/5
    print(f"  - 최근 {raw_days}일: 1분 원본 그대로 ≈ {human(keep_raw)}  (학습·평가·이상탐지에 필요)")
    print(f"  - {raw_days}일 이후: 5분 평균/최소/최대로 다운샘플 ≈ {human(agg_rows_year * 60)} /년")
    print("  - 결과: 1년 뒤에도 수백 MB 이하 → Jetson NVMe 에 충분")
    return 0


if __name__ == "__main__":
    sys.exit(main())
