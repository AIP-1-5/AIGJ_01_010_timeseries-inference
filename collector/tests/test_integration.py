"""로컬 mosquitto 로 '브로커 재시작 후 1분 안에 재수집'과 '중복 제거'를 실제로 시험한다.

mosquitto 가 설치돼 있지 않으면 건너뛴다 (맥: brew install mosquitto, 우분투: sudo apt install mosquitto).
"""
import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time

import paho.mqtt.client as mqtt
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
MOSQUITTO = shutil.which("mosquitto") or next(
    (p for p in ("/opt/homebrew/sbin/mosquitto", "/usr/local/sbin/mosquitto", "/usr/sbin/mosquitto")
     if os.path.exists(p)), None)

pytestmark = pytest.mark.skipif(MOSQUITTO is None, reason="mosquitto 미설치")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_broker(tmp_path, port):
    conf = tmp_path / "mosquitto.conf"
    conf.write_text(f"listener {port} 127.0.0.1\nallow_anonymous true\n")
    p = subprocess.Popen([MOSQUITTO, "-c", str(conf)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return p
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("broker did not start")


def publish(port, topic, payload):
    c = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    c.connect("127.0.0.1", port)
    c.loop_start()
    c.publish(topic, payload, qos=1).wait_for_publish(5)
    c.loop_stop()
    c.disconnect()


def q(db, sql):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def wait_for(fn, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if fn():
            return True
        time.sleep(0.2)
    return False


def test_reconnect_after_broker_restart_and_dedup(tmp_path):
    port = free_port()
    db = str(tmp_path / "s.db")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"""
mqtt:
  host: 127.0.0.1
  port: {port}
  client_id: test-collector
  topics: ["academy/#"]
  reconnect_min_delay: 1
  reconnect_max_delay: 5
storage:
  path: {db}
rules:
  - name: auto
    topic: "#"
    payload: auto
    only_metrics: [co2, temperature, humidity]
""")
    broker = start_broker(tmp_path, port)
    col = subprocess.Popen([sys.executable, "-m", "collector.main", "--config", str(cfg), "--env", "/nonexistent"],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        assert wait_for(lambda: os.path.exists(db) and
                        q(db, "SELECT COUNT(*) FROM events WHERE event='connected'")[0][0] >= 1)
        base = 1760925600
        for i in range(3):
            publish(port, "academy/lecture1", json.dumps({"co2": 600 + i, "temp": 24, "ts": base + 60 * i}))
        assert wait_for(lambda: q(db, "SELECT COUNT(*) FROM readings")[0][0] == 6)

        # 재전송된 같은 메시지 → 중복 제거
        publish(port, "academy/lecture1", json.dumps({"co2": 600, "temp": 24, "ts": base}))
        assert wait_for(lambda: q(db, "SELECT value FROM counters WHERE name='duplicates'") == [(2,)])

        # 브로커 재시작
        broker.terminate()
        broker.wait(5)
        assert wait_for(lambda: q(db, "SELECT COUNT(*) FROM events WHERE event='disconnected'")[0][0] >= 1)
        time.sleep(2)
        t_restart = time.time()
        broker = start_broker(tmp_path, port)
        assert wait_for(lambda: q(db, "SELECT COUNT(*) FROM events WHERE event='connected'")[0][0] >= 2, timeout=60)
        reconnect_sec = time.time() - t_restart
        assert reconnect_sec < 60, reconnect_sec

        publish(port, "academy/meeting/co2", b"1100")
        assert wait_for(lambda: q(db, "SELECT COUNT(*) FROM readings WHERE space='meeting'")[0][0] == 1)
        print(f"\n브로커 재시작 후 재접속까지 {reconnect_sec:.1f}초")
    finally:
        col.send_signal(signal.SIGTERM)
        try:
            col.wait(10)
        except subprocess.TimeoutExpired:
            col.kill()
        broker.terminate()
        broker.wait(5)
    assert q(db, "SELECT COUNT(*) FROM events WHERE event='stop'")[0][0] == 1
