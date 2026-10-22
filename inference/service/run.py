"""과제 8 상시 추론 서비스.

    python -m service.run --config service/config.yaml

- 매분: 수집기 DB → 공간별 예측·이상·상태 계산 → MQTT 발행(retain) + 결과 DB 저장
- 조회 API (HTTP, 기본 8080 포트):
    GET /health                       서비스 상태
    GET /spaces                       공간 목록과 최신 요약
    GET /spaces/<공간 slug 또는 이름>    한 공간의 최신 결과 전체
    GET /spaces/<공간>/forecast?minutes=30   N분 뒤 예측값 (5분 단위로 반올림)
    GET /alarms?hours=12              최근 경보 내역
    GET /log/summary?hours=12         주기별 처리 시간·성공률 요약 (12시간 가동 기록)
- 한 주기가 실패해도 다음 주기는 정상으로 돌아온다 (예외를 잡고 기록만 함)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sqlite3
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.infer import Engine, slug  # noqa: E402

log = logging.getLogger("service")
ALARM_REPEAT_SEC = 3600          # 같은 경보는 1시간 안에 다시 보내지 않음
MONITOR_WINDOW_SEC = 6 * 3600     # 최근 6시간 채점 결과로 판단
MONITOR_MIN_N = 36                # 60분 예측 36개(3시간치) 이상 채점됐을 때만
MONITOR_RATIO = 1.2               # 모델 오차가 '현재값 유지'의 1.2배를 넘으면 경보
MONITOR_MIN_DIFF = 5.0            # 단, 차이가 5ppm 이상일 때만 (조용한 밤에 4 vs 3.5ppm 같은 작은 차이로 울리지 않게)


def load_env(path):
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.lock = threading.Lock()
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS latest (space TEXT PRIMARY KEY, slug TEXT, ts REAL, json TEXT);
        CREATE TABLE IF NOT EXISTS cycles (ts REAL, ok INTEGER, duration_ms REAL, n_spaces INTEGER, error TEXT);
        CREATE TABLE IF NOT EXISTS alarms (ts REAL, space TEXT, kind TEXT, metric TEXT, detail TEXT, sent INTEGER);
        CREATE TABLE IF NOT EXISTS forecasts (space TEXT, base_time TEXT, target_time TEXT, horizon INTEGER,
                                              co2_pred REAL, co2_base REAL, graded INTEGER DEFAULT 0,
                                              UNIQUE(space, base_time, horizon));
        CREATE INDEX IF NOT EXISTS idx_fc_target ON forecasts(space, target_time, graded);
        CREATE TABLE IF NOT EXISTS scores (ts REAL, space TEXT, target_time TEXT, horizon INTEGER,
                                           model_err REAL, persist_err REAL);
        CREATE INDEX IF NOT EXISTS idx_scores_ts ON scores(ts);
        CREATE INDEX IF NOT EXISTS idx_cycles_ts ON cycles(ts);
        CREATE INDEX IF NOT EXISTS idx_alarms_ts ON alarms(ts);
        """)

    def q(self, sql, args=()):
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def x(self, sql, args=()):
        with self.lock:
            self.conn.execute(sql, args)
            self.conn.commit()


class Service:
    def __init__(self, cfg):
        self.cfg = cfg
        self.engine = Engine(cfg["model_dir"])
        self.store = Store(cfg["results_db"])
        self.last_alarm = {}
        self.mqtt = None
        self.started = time.time()
        self.last_ok = None
        self.stop_flag = threading.Event()
        if cfg["mqtt"].get("enabled"):
            self._mqtt_connect()

    # ---------------------------------------------------------------- MQTT
    def _mqtt_connect(self):
        import paho.mqtt.client as mqtt
        m = self.cfg["mqtt"]
        c = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                        client_id=m.get("client_id", "team5b-inference"), clean_session=True)
        if os.environ.get("MQTT_USERNAME"):
            c.username_pw_set(os.environ["MQTT_USERNAME"], os.environ.get("MQTT_PASSWORD"))
        status_topic = f"{m['prefix']}/_status"
        c.will_set(status_topic, json.dumps({"online": False}), qos=1, retain=True)    # 서비스가 죽으면 브로커가 대신 알림
        c.reconnect_delay_set(1, 10)
        c.connect_async(m["host"], m.get("port", 1883), keepalive=30)
        c.loop_start()
        self.mqtt = c

    def _publish(self, topic, payload):
        if self.mqtt is None:
            return
        self.mqtt.publish(topic, json.dumps(payload, ensure_ascii=False, default=str), qos=1, retain=True)

    # ---------------------------------------------------------------- 주기
    def cycle(self):
        t0 = time.time()
        try:
            out = self.engine.run(self.cfg["collector_db"])
            prefix = self.cfg["mqtt"].get("prefix", "team5b/inference")
            for sp, res in out["spaces"].items():
                res["monitor"] = self._self_check(sp, res, t0)
                self.store.x("INSERT OR REPLACE INTO latest(space, slug, ts, json) VALUES (?,?,?,?)",
                             (sp, res["slug"], t0, json.dumps(res, ensure_ascii=False, default=str)))
                self._publish(f"{prefix}/{res['slug']}", res)
                self._alarms(sp, res, prefix)
            status = {"online": True, "generated_at": out["generated_at"], "spaces": list(out["spaces"]),
                      "processing_ms": out.get("processing_ms"), "errors": out["errors"],
                      "uptime_sec": round(time.time() - self.started)}
            self._publish(f"{prefix}/_status", status)
            dur = (time.time() - t0) * 1000
            self.store.x("INSERT INTO cycles VALUES (?,?,?,?,?)", (t0, 1, dur, len(out["spaces"]), "; ".join(out["errors"]) or None))
            self.last_ok = t0
            log.info("주기 완료: 공간 %d개, %.0fms%s", len(out["spaces"]), dur, f", 오류 {out['errors']}" if out["errors"] else "")
        except Exception as e:
            log.exception("주기 실패 (다음 주기에 다시 시도)")
            self.store.x("INSERT INTO cycles VALUES (?,?,?,?,?)", (t0, 0, (time.time() - t0) * 1000, 0, f"{type(e).__name__}: {e}"))

    # ---------------------------------------------------------------- 자기 채점 (과제 8 심화)
    def _self_check(self, sp, res, now):
        """예측을 저장해 두었다가 그 시각이 오면 실제 값과 비교해 채점한다.
        같은 시점의 '현재값 유지' 오차와 비교해, 모델이 기준선보다 나빠지면 알린다."""
        f = res.get("forecast", {})
        if f.get("available") and f.get("base_co2") is not None:
            bt = f["base_time"]
            for k, (t, c) in enumerate(zip(f["times"], f["co2"])):
                self.store.x("INSERT OR IGNORE INTO forecasts(space, base_time, target_time, horizon, co2_pred, co2_base) "
                             "VALUES (?,?,?,?,?,?)", (sp, bt, t, (k + 1) * 5, c, f["base_co2"]))
            # 지금 칸(bt)을 목표로 했던 과거 예측들을 채점
            due = self.store.q("SELECT rowid, horizon, co2_pred, co2_base FROM forecasts WHERE space=? AND target_time=? AND graded=0",
                               (sp, bt))
            for rid, hz, pred, base in due:
                self.store.x("INSERT INTO scores VALUES (?,?,?,?,?,?)",
                             (now, sp, bt, hz, abs(pred - f["base_co2"]), abs(base - f["base_co2"])))
                self.store.x("UPDATE forecasts SET graded=1 WHERE rowid=?", (rid,))
            self.store.x("DELETE FROM forecasts WHERE space=? AND target_time < ? AND graded=0 AND base_time < ?",
                         (sp, bt, (pd.Timestamp(bt) - pd.Timedelta(hours=3)).isoformat()))   # 끊긴 동안 못 채점한 것 정리
        rows = self.store.q("SELECT model_err, persist_err FROM scores WHERE space=? AND horizon=60 AND ts >= ?",
                            (sp, now - MONITOR_WINDOW_SEC))
        n = len(rows)
        if n == 0:
            return {"graded_60min": 0, "window_hours": MONITOR_WINDOW_SEC / 3600}
        m = float(np.mean([r[0] for r in rows]))
        p = float(np.mean([r[1] for r in rows]))
        degraded = n >= MONITOR_MIN_N and m > p * MONITOR_RATIO and m - p >= MONITOR_MIN_DIFF
        return {"window_hours": MONITOR_WINDOW_SEC / 3600, "graded_60min": n,
                "mae_60min_model": round(m, 1), "mae_60min_persistence": round(p, 1),
                "skill_pct": round((1 - m / p) * 100, 1) if p > 0 else None, "degraded": degraded}

    def _alarms(self, sp, res, prefix):
        items = []
        a = res.get("anomaly", {})
        for r in a.get("rules", []):
            items.append((r["type"], r.get("metric"), r.get("detail")))
        if a.get("model_alarm"):
            items.append(("model_residual", a.get("cause"), f"예상보다 높음 z={a.get('score')}"))
        mo = res.get("monitor", {})
        if mo.get("degraded"):
            items.append(("model_degraded", "co2", f"최근 {mo['window_hours']:.0f}시간 60분 예측 오차 {mo['mae_60min_model']}ppm > "
                                                   f"현재값 유지 {mo['mae_60min_persistence']}ppm (재학습 필요)"))
        al = res.get("alerts", {})
        if al.get("ventilation_soon"):
            items.append(("ventilation_soon", "co2", f"30분 안에 1,000ppm 초과 확률 {al.get('ventilation_prob')}"))
        if al.get("co2_over_1000_in_min"):
            items.append(("co2_over_1000_forecast", "co2", f"{al['co2_over_1000_in_min']}분 뒤 1,000ppm 초과 예상"))
        now = time.time()
        for kind, metric, detail in items:
            key = (sp, kind, metric)
            sent = 0
            if now - self.last_alarm.get(key, 0) >= ALARM_REPEAT_SEC:      # 같은 경보는 1시간에 한 번만 보냄
                self.last_alarm[key] = now
                sent = 1
                self._publish(f"{prefix}/{res['slug']}/alarm",
                              {"space": sp, "kind": kind, "metric": metric, "detail": detail,
                               "time": pd.Timestamp.now(tz="Asia/Seoul").isoformat()})
            self.store.x("INSERT INTO alarms VALUES (?,?,?,?,?,?)", (now, sp, kind, metric, detail, sent))

    def loop(self):
        interval = self.cfg.get("interval_sec", 60)
        while not self.stop_flag.is_set():
            start = time.time()
            self.cycle()
            # 매분 0초 근처에 맞춰 돈다
            self.stop_flag.wait(max(1.0, interval - (time.time() - start) % interval))

    # ---------------------------------------------------------------- HTTP
    def http_handler(self):
        svc = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj, ensure_ascii=False, default=str, indent=1).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urlparse(self.path)
                parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
                qs = parse_qs(u.query)
                try:
                    if not parts or parts[0] == "health":
                        age = None if svc.last_ok is None else round(time.time() - svc.last_ok, 1)
                        return self._send(200 if age is not None and age < 180 else 503,
                                          {"ok": age is not None and age < 180, "last_cycle_age_sec": age,
                                           "uptime_sec": round(time.time() - svc.started),
                                           "mqtt": bool(svc.mqtt and svc.mqtt.is_connected())})
                    if parts[0] == "spaces":
                        rows = svc.store.q("SELECT space, slug, json FROM latest")
                        data = {r[0]: json.loads(r[2]) for r in rows}
                        if len(parts) == 1:
                            return self._send(200, [{"space": s, "slug": d["slug"], "time": d.get("time"), "latest": d.get("latest"),
                                                     "occupied": d.get("state", {}).get("occupied"),
                                                     "is_anomaly": d.get("anomaly", {}).get("is_anomaly"),
                                                     "co2_over_1000_in_min": d.get("alerts", {}).get("co2_over_1000_in_min")}
                                                    for s, d in data.items()])
                        key = parts[1]
                        d = next((v for s, v in data.items() if key in (s, v["slug"])), None)
                        if d is None:
                            return self._send(404, {"error": f"공간 없음: {key}", "spaces": [v["slug"] for v in data.values()]})
                        if len(parts) >= 3 and parts[2] == "forecast":
                            mins = int(qs.get("minutes", ["30"])[0])
                            f = d.get("forecast", {})
                            if not f.get("available"):
                                return self._send(503, {"error": "예측 불가", "reason": f.get("reason")})
                            k = min(max(round(mins / 5), 1), 12) - 1
                            return self._send(200, {"space": d["space"], "minutes": (k + 1) * 5, "time": f["times"][k],
                                                    "co2": f["co2"][k], "temperature": f["temperature"][k],
                                                    "base_time": f["base_time"], "units": d["units"]})
                        return self._send(200, d)
                    if parts[0] == "alarms":
                        hours = float(qs.get("hours", ["12"])[0])
                        rows = svc.store.q("SELECT ts, space, kind, metric, detail, sent FROM alarms WHERE ts >= ? AND sent = 1 ORDER BY ts DESC",
                                           (time.time() - hours * 3600,))
                        return self._send(200, [{"time": pd.Timestamp(r[0], unit="s", tz="UTC").tz_convert("Asia/Seoul").isoformat(),
                                                 "space": r[1], "kind": r[2], "metric": r[3], "detail": r[4]} for r in rows])
                    if parts[0] == "monitor":
                        hours = float(qs.get("hours", ["24"])[0])
                        rows = svc.store.q("SELECT space, horizon, AVG(model_err), AVG(persist_err), COUNT(*) FROM scores "
                                           "WHERE ts >= ? GROUP BY space, horizon ORDER BY space, horizon", (time.time() - hours * 3600,))
                        out = {}
                        for sp, hz, me, pe, n in rows:
                            out.setdefault(sp, []).append({"horizon_min": hz, "n": n, "mae_model": round(me, 1), "mae_persistence": round(pe, 1),
                                                           "skill_pct": round((1 - me / pe) * 100, 1) if pe else None})
                        return self._send(200, {"hours": hours, "unit": "ppm", "spaces": out})
                    if parts[0] == "log" and len(parts) > 1 and parts[1] == "summary":
                        hours = float(qs.get("hours", ["12"])[0])
                        return self._send(200, summary(svc.store, hours))
                    return self._send(404, {"error": "없는 경로"})
                except Exception as e:
                    return self._send(500, {"error": f"{type(e).__name__}: {e}"})

        return H


def summary(store, hours):
    since = time.time() - hours * 3600
    rows = store.q("SELECT ts, ok, duration_ms, error FROM cycles WHERE ts >= ? ORDER BY ts", (since,))
    if not rows:
        return {"cycles": 0}
    df = pd.DataFrame(rows, columns=["ts", "ok", "ms", "error"])
    gaps = df["ts"].diff().dropna()
    alarms = store.q("SELECT kind, COUNT(*) FROM alarms WHERE ts >= ? AND sent = 1 GROUP BY kind", (since,))
    return {
        "from": pd.Timestamp(df.ts.iloc[0], unit="s", tz="UTC").tz_convert("Asia/Seoul").isoformat(),
        "to": pd.Timestamp(df.ts.iloc[-1], unit="s", tz="UTC").tz_convert("Asia/Seoul").isoformat(),
        "hours_covered": round((df.ts.iloc[-1] - df.ts.iloc[0]) / 3600, 2),
        "cycles": int(len(df)), "ok": int(df.ok.sum()), "failed": int((df.ok == 0).sum()),
        "processing_ms": {"mean": round(df.ms.mean(), 1), "p95": round(df.ms.quantile(0.95), 1), "max": round(df.ms.max(), 1)},
        "max_gap_between_cycles_sec": round(float(gaps.max()), 1) if len(gaps) else None,
        "alarms_sent": {k: n for k, n in alarms},
        "errors": df.error.dropna().value_counts().head(5).to_dict(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="service/config.yaml")
    ap.add_argument("--once", action="store_true", help="한 주기만 돌고 결과 출력")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    load_env(cfg.get("env_file"))
    svc = Service(cfg)
    if args.once:
        out = svc.engine.run(cfg["collector_db"])
        print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
        return 0
    httpd = ThreadingHTTPServer((cfg.get("http_host", "0.0.0.0"), cfg.get("http_port", 8080)), svc.http_handler())
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    log.info("조회 API: http://%s:%s  MQTT 발행: %s", cfg.get("http_host", "0.0.0.0"), cfg.get("http_port", 8080),
             "켜짐 (" + cfg["mqtt"]["prefix"] + "/…)" if cfg["mqtt"].get("enabled") else "꺼짐")

    def _stop(sig, frame):
        log.info("종료 신호 %s", sig)
        svc.stop_flag.set()
        httpd.shutdown()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    svc.loop()
    if svc.mqtt:
        svc._publish(f"{cfg['mqtt']['prefix']}/_status", {"online": False})
        time.sleep(0.5)
        svc.mqtt.loop_stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
