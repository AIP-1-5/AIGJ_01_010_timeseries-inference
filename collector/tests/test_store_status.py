import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collector.parser import Reading  # noqa: E402
from collector.store import Store  # noqa: E402
from tools.status import build_report, find_gaps  # noqa: E402


def R(space, metric, value, ts):
    return Reading(space, metric, value, ts, "t", "r")


def test_dedup_and_both_timestamps(tmp_path):
    db = str(tmp_path / "s.db")
    st = Store(db)
    T = 1_792_000_000.0
    assert st.add_readings([R("a", "co2", 1, T)], recv_ts=T + 30) == 1
    # 같은 센서 시각으로 재전송 → 무시
    assert st.add_readings([R("a", "co2", 1, T)], recv_ts=T + 31) == 0
    row = st.conn.execute("SELECT ts, sensor_ts, recv_ts FROM readings").fetchone()
    assert (row["ts"], row["sensor_ts"], row["recv_ts"]) == (T, T, T + 30)
    c = dict(st.conn.execute("SELECT name, value FROM counters").fetchall())
    assert c == {"readings": 1, "duplicates": 1}
    st.close()


def test_find_gaps():
    assert find_gaps([0, 1, 2, 10, 11], 0, 12, 5) == [(3, 10)]
    assert find_gaps([0, 1, 2], 0, 10, 5) == [(3, 10)]
    assert find_gaps([], 0, 10, 5) == [(0, 10)]
    assert find_gaps([0, 1, 2, 3], 0, 4, 1) == []


def test_report_coverage_and_gap_cause(tmp_path):
    db = str(tmp_path / "s.db")
    st = Store(db)
    now = 1_000_000 * 60.0
    start = now - 60 * 60  # 1시간
    # 60분 중 50분만 데이터, 20~30분 구간 비움 + 그 사이 브로커 끊김 이벤트
    readings = [R("lecture1", "co2", 500, start + m * 60 + 5) for m in range(60) if not (20 <= m < 30)]
    st.add_readings(readings, recv_ts=now - 1)
    st.add_event("connected", ts=start - 10)
    st.add_event("disconnected", "broker down", ts=start + 20 * 60 + 2)
    st.add_event("connected", ts=start + 30 * 60)
    st.close()

    rep = build_report(db, hours=1, min_gap=5, tz="Asia/Seoul", now=now)
    row = rep["rows"][0]
    assert row["count"] == 50
    assert abs(row["coverage"] - 50 / 60 * 100) < 0.01
    assert len(row["gaps"]) == 1
    a, b, why = row["gaps"][0]
    assert (b - a) == 10 * 60
    assert "브로커 연결 끊김" in why


def test_seq_dedup_and_clock_skew(tmp_path):
    """같은 업링크가 게이트웨이 2대를 거쳐 시각만 295초 다르게 두 번 와도 fCnt 로 한 번만 저장.
    게이트웨이 시계가 미래면 받은 시각을 기준으로 쓴다."""
    st = Store(str(tmp_path / "s.db"))
    recv = 1_792_000_000.0
    a = Reading("실습실", "co2", 900, recv + 0.2, "t", "r", device="d1", seq=42)     # 정상 게이트웨이
    b = Reading("실습실", "co2", 900, recv + 295.5, "t", "r", device="d1", seq=42)   # 시계 5분 빠른 게이트웨이
    assert st.add_readings([b], recv_ts=recv + 0.01) == 1
    assert st.add_readings([a], recv_ts=recv + 0.02) == 0
    row = st.conn.execute("SELECT ts, sensor_ts FROM readings").fetchone()
    assert row["ts"] == recv + 0.01 and row["sensor_ts"] == recv + 295.5
    c = dict(st.conn.execute("SELECT name, value FROM counters").fetchall())
    assert c["duplicates"] == 1 and c["clock_skew"] == 1
    # 센서가 재가입해 fCnt 가 다시 42 가 돼도, 다른 시간대면 저장된다
    later = Reading("실습실", "co2", 910, recv + 7200, "t", "r", device="d1", seq=42)
    assert st.add_readings([later], recv_ts=recv + 7200) == 1
    st.close()


def test_migrate_old_db(tmp_path):
    import sqlite3
    db = str(tmp_path / "old.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE readings (id INTEGER PRIMARY KEY, space TEXT NOT NULL, metric TEXT NOT NULL, "
                "device TEXT NOT NULL DEFAULT '', point TEXT NOT NULL DEFAULT '', value REAL NOT NULL, "
                "ts REAL NOT NULL, sensor_ts REAL, recv_ts REAL NOT NULL, topic TEXT, rule TEXT, "
                "UNIQUE (device, space, metric, ts))")
    con.execute("INSERT INTO readings(space, metric, value, ts, recv_ts) VALUES ('a','co2',1,1,1)")
    con.commit(); con.close()
    st = Store(db)
    assert st.add_readings([Reading("a", "co2", 2, 5.0, "t", "r", device="x", seq=1)], recv_ts=5.0) == 1
    assert st.conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == 2
    st.close()


def test_board_clock_unset_after_reboot(tmp_path):
    """재부팅 직후 NTP 동기화 전에는 보드 시계가 1970년 → 센서 시각으로 저장"""
    st = Store(str(tmp_path / "s.db"))
    st.add_readings([Reading("실습실", "co2", 800, 1792556000.0, "t", "r", device="d", seq=1)], recv_ts=48.5)
    row = st.conn.execute("SELECT ts, recv_ts FROM readings").fetchone()
    assert row["ts"] == 1792556000.0 and row["recv_ts"] == 48.5
    assert dict(st.conn.execute("SELECT name, value FROM counters").fetchall())["board_clock_unset"] == 1
    st.close()
