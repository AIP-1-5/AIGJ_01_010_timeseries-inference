"""(심화 B) 오래된 1분 원본을 5분 집계(평균·최소·최대·건수)로 줄인다.

    python tools/retention.py --keep-days 90            # 미리보기 (아무것도 안 바꿈)
    python tools/retention.py --keep-days 90 --apply    # 실제 적용

수집기가 돌고 있어도 실행할 수 있다 (WAL). 하루 한 번 cron/systemd timer 로 돌리면 된다.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time

AGG_SCHEMA = """
CREATE TABLE IF NOT EXISTS readings_5m (
    space TEXT NOT NULL, metric TEXT NOT NULL, bucket_ts REAL NOT NULL,
    avg REAL, min REAL, max REAL, n INTEGER,
    PRIMARY KEY (space, metric, bucket_ts)
);
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sensors.db")
    ap.add_argument("--keep-days", type=int, default=90)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    cutoff = time.time() - args.keep_days * 86400
    conn = sqlite3.connect(args.db, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    n_old = conn.execute("SELECT COUNT(*) FROM readings WHERE ts < ?", (cutoff,)).fetchone()[0]
    print(f"{args.keep_days}일보다 오래된 원본: {n_old:,}행")
    if not args.apply or n_old == 0:
        print("(미리보기) 적용하려면 --apply" if n_old else "줄일 데이터가 없습니다.")
        return 0
    conn.executescript(AGG_SCHEMA)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO readings_5m(space, metric, bucket_ts, avg, min, max, n) "
            "SELECT space, metric, CAST(ts / 300 AS INTEGER) * 300, AVG(value), MIN(value), MAX(value), COUNT(*) "
            "FROM readings WHERE ts < ? GROUP BY space, metric, CAST(ts / 300 AS INTEGER)",
            (cutoff,),
        )
        conn.execute("DELETE FROM readings WHERE ts < ?", (cutoff,))
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print(f"{n_old:,}행을 5분 집계로 옮겼습니다 (readings_5m).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
