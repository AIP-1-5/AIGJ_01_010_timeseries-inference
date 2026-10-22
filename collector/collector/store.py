"""SQLite 저장소.

- WAL 모드: 수집기가 쓰는 동안에도 학습·추론·현황 도구가 동시에 읽을 수 있다.
- 시각은 두 개를 저장한다: sensor_ts(센서가 찍은 시각), recv_ts(수집기가 받은 시각).
  분석 기준 시각 ts 는 sensor_ts 가 있으면 그것, 없으면 recv_ts.
- (device, space, metric, ts) 가 같으면 재전송된 중복으로 보고 한 번만 저장한다.
  한 공간에 센서가 여러 대일 수 있어 기기(device)와 설치 위치(point)도 함께 남긴다.
- events 테이블에 접속/끊김/시작/종료를 남겨, 빈 구간의 원인을 나중에 설명할 수 있게 한다.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

from .parser import Reading

CLOCK_VALID_AFTER = 1_577_836_800.0      # 2020-01-01. 이보다 이전 시각이면 보드 시계가 아직 동기화 전

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id        INTEGER PRIMARY KEY,
    space     TEXT NOT NULL,
    metric    TEXT NOT NULL,
    device    TEXT NOT NULL DEFAULT '',
    point     TEXT NOT NULL DEFAULT '',
    seq       INTEGER,
    seq_hour  INTEGER,
    value     REAL NOT NULL,
    ts        REAL NOT NULL,
    sensor_ts REAL,
    recv_ts   REAL NOT NULL,
    topic     TEXT,
    rule      TEXT,
    UNIQUE (device, space, metric, ts)
);
CREATE INDEX IF NOT EXISTS idx_readings_series ON readings (space, metric, ts);
CREATE INDEX IF NOT EXISTS idx_readings_recv ON readings (recv_ts);
CREATE INDEX IF NOT EXISTS idx_readings_device ON readings (device, metric, ts);

CREATE TABLE IF NOT EXISTS unmatched (
    id      INTEGER PRIMARY KEY,
    recv_ts REAL NOT NULL,
    topic   TEXT NOT NULL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id     INTEGER PRIMARY KEY,
    ts     REAL NOT NULL,
    event  TEXT NOT NULL,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);
"""


def connect(path: str, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


class Store:
    def __init__(self, path: str, keep_unmatched: bool = True, unmatched_limit: int = 5000,
                 max_future_sec: float = 30.0):
        self.path = path
        self.max_future_sec = max_future_sec
        self.keep_unmatched = keep_unmatched
        self.unmatched_limit = unmatched_limit
        self.conn = connect(path)
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()
        self._unmatched_since_trim = 0

    def _migrate(self) -> None:
        """예전 DB 에 seq 열이 없으면 추가하고, (기기, 항목, 일련번호) 중복 방지 인덱스를 만든다."""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(readings)")}
        if "seq" not in cols:
            self.conn.execute("ALTER TABLE readings ADD COLUMN seq INTEGER")
        if "seq_hour" not in cols:
            self.conn.execute("ALTER TABLE readings ADD COLUMN seq_hour INTEGER")
        # 일련번호(fCnt)는 센서가 재가입하면 0부터 다시 시작할 수 있다 → 같은 시간대(1시간) 안에서만 중복으로 본다
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_readings_seq ON readings (device, metric, seq, seq_hour) "
            "WHERE seq IS NOT NULL")

    def close(self) -> None:
        self.conn.close()

    def _bump(self, name: str, n: int = 1) -> None:
        self.conn.execute(
            "INSERT INTO counters(name, value) VALUES(?, ?) "
            "ON CONFLICT(name) DO UPDATE SET value = value + excluded.value",
            (name, n),
        )

    def add_readings(self, readings: Iterable[Reading], recv_ts: Optional[float] = None) -> int:
        """저장한 개수를 돌려준다. 중복(재전송)은 건너뛰고 duplicates 카운터를 올린다."""
        recv_ts = time.time() if recv_ts is None else recv_ts
        inserted = 0
        dup = 0
        skew = 0
        for r in readings:
            ts = r.sensor_ts if r.sensor_ts is not None else recv_ts
            # 보드 시계가 아직 안 맞음 (재부팅 직후 NTP 동기화 전에는 1970년으로 시작) → 센서 시각을 쓴다
            if recv_ts < CLOCK_VALID_AFTER:
                if r.sensor_ts is not None:
                    ts = r.sensor_ts
                self._bump("board_clock_unset")
            # 센서(게이트웨이) 시각이 받은 시각보다 미래면 시계가 틀린 것 → 받은 시각을 기준으로 쓴다
            elif r.sensor_ts is not None and r.sensor_ts - recv_ts > self.max_future_sec:
                ts = recv_ts
                skew += 1
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO readings(space, metric, device, point, seq, seq_hour, value, ts, sensor_ts, recv_ts, "
                "topic, rule) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (r.space, r.metric, r.device or "", r.point or "", r.seq,
                 int(recv_ts // 3600) if r.seq is not None else None, r.value, ts, r.sensor_ts, recv_ts,
                 r.topic, r.rule),
            )
            if cur.rowcount == 1:
                inserted += 1
            else:
                dup += 1
        if inserted:
            self._bump("readings", inserted)
        if dup:
            self._bump("duplicates", dup)
        if skew:
            self._bump("clock_skew", skew)
        self.conn.commit()
        return inserted

    def count_ignored(self) -> None:
        self._bump("ignored")
        self.conn.commit()

    def add_unmatched(self, topic: str, payload: bytes, recv_ts: Optional[float] = None) -> None:
        self._bump("unmatched")
        if self.keep_unmatched:
            self.conn.execute(
                "INSERT INTO unmatched(recv_ts, topic, payload) VALUES (?, ?, ?)",
                (time.time() if recv_ts is None else recv_ts, topic,
                 payload[:2000].decode("utf-8", errors="replace")),
            )
            self._unmatched_since_trim += 1
            if self._unmatched_since_trim >= 500:
                self.conn.execute(
                    "DELETE FROM unmatched WHERE id <= (SELECT MAX(id) FROM unmatched) - ?",
                    (self.unmatched_limit,),
                )
                self._unmatched_since_trim = 0
        self.conn.commit()

    def add_event(self, event: str, detail: str = "", ts: Optional[float] = None) -> None:
        self.conn.execute(
            "INSERT INTO events(ts, event, detail) VALUES (?, ?, ?)",
            (time.time() if ts is None else ts, event, detail),
        )
        self.conn.commit()
