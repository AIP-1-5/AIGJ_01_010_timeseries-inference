"""교육장 InfluxDB(내부망)의 과거 센서 데이터를 수집기와 같은 형식의 SQLite 로 내보낸다.

보드 B 처럼 교육장 내부망(10.x)에 있는 기기에서 실행:
    python3 tools/export_influx.py --days 56 --out data/influx_56d.db
접속 정보는 ~/ts-inference/.env_influx (INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG_ID, INFLUX_BUCKET) — 토큰은 출력하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone


def load_env(path):
    for line in open(os.path.expanduser(path), encoding="utf-8"):
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def query(flux: str) -> str:
    url = f"{os.environ['INFLUX_URL']}/api/v2/query?orgID={os.environ['INFLUX_ORG_ID']}"
    req = urllib.request.Request(url, data=flux.encode(), method="POST", headers={
        "Authorization": f"Token {os.environ['INFLUX_TOKEN']}",
        "Content-Type": "application/vnd.flux", "Accept": "application/csv"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read().decode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=56)
    ap.add_argument("--out", default="data/influx.db")
    ap.add_argument("--env", default="~/ts-inference/.env_influx")
    args = ap.parse_args(argv)
    load_env(args.env)
    bucket = os.environ.get("INFLUX_BUCKET", "iot-bucket")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)
    con = sqlite3.connect(args.out)
    con.execute("CREATE TABLE readings (space TEXT, metric TEXT, device TEXT, point TEXT, value REAL, "
                "ts REAL, sensor_ts REAL, recv_ts REAL, topic TEXT, rule TEXT)")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    total = 0
    day = start
    while day < end:
        nxt = min(day + timedelta(days=1), end)
        flux = (f'from(bucket:"{bucket}") |> range(start: {day.strftime("%Y-%m-%dT%H:%M:%SZ")}, stop: {nxt.strftime("%Y-%m-%dT%H:%M:%SZ")}) '
                '|> filter(fn:(r)=> (r._measurement=="co2" or r._measurement=="temperature" or r._measurement=="humidity") and r._field=="value") '
                '|> keep(columns:["_time","_value","_measurement","device_id","location","point"])')
        for attempt in range(3):
            try:
                text = query(flux)
                break
            except Exception as e:     # 네트워크가 잠깐 끊겨도 다시
                print("retry", day.date(), e, file=sys.stderr)
                time.sleep(5)
        else:
            day = nxt
            continue
        rows = []
        for rec in csv.DictReader(io.StringIO(text)):
            try:
                t = datetime.fromisoformat(rec["_time"].replace("Z", "+00:00")).timestamp()
                v = float(rec["_value"])
            except (KeyError, ValueError, TypeError):
                continue
            rows.append((rec.get("location") or "", rec["_measurement"], rec.get("device_id") or "", rec.get("point") or "",
                         v, t, t, t, "influx", "influx"))
        con.executemany("INSERT INTO readings VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        con.commit()
        total += len(rows)
        print(f"{day.date()} {len(rows):6d}행 (누적 {total:,})", flush=True)
        day = nxt
    con.execute("CREATE INDEX idx ON readings(space, metric, ts)")
    con.commit()
    con.close()
    print("saved", args.out, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
