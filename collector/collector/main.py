"""수집기 본체: MQTT 구독 → 파싱 → SQLite 저장.

실행:  python -m collector.main --config config.yaml

끊김 대응
- 브로커가 꺼지거나 네트워크가 끊기면 paho 가 1초 → 최대 reconnect_max_delay(기본 5초) 간격으로
  계속 재접속한다. 다시 붙으면 on_connect 에서 구독을 새로 건다 → 브로커 재시작 후 1분 안에 재수집.
- clean_session=False + QoS 1: 짧게 끊긴 동안 브로커가 쌓아 둔 메시지를 재접속 후 받는다.
- 프로세스가 죽거나 보드가 재부팅되면 systemd(Restart=always, enable)가 다시 띄운다.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

import paho.mqtt.client as mqtt

from .config import Config, load_config
from .parser import Parser
from .store import Store

log = logging.getLogger("collector")


class Collector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = Store(cfg.storage.path, cfg.storage.keep_unmatched, cfg.storage.unmatched_limit)
        self.parser = Parser(cfg.rules, cfg.timezone, cfg.metric_aliases, cfg.space_aliases)
        m = cfg.mqtt
        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=m.client_id,
            clean_session=m.clean_session,
        )
        if m.username:
            self.client.username_pw_set(m.username, m.password)
        self.client.reconnect_delay_set(min_delay=m.reconnect_min_delay, max_delay=m.reconnect_max_delay)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.msg_count = 0
        self.saved_count = 0
        self._last_stat = time.time()

    # --- MQTT 콜백 -------------------------------------------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("브로커 접속 실패: %s", reason_code)
            self.store.add_event("connect_failed", str(reason_code))
            return
        log.info("브로커 접속 성공 (%s:%s), 세션 유지=%s", self.cfg.mqtt.host, self.cfg.mqtt.port,
                 flags.session_present)
        self.store.add_event("connected", f"{self.cfg.mqtt.host}:{self.cfg.mqtt.port}")
        for t in self.cfg.mqtt.topics:
            client.subscribe(t, qos=self.cfg.mqtt.qos)
            log.info("구독: %s (QoS %d)", t, self.cfg.mqtt.qos)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        log.warning("브로커 연결 끊김: %s → 자동 재접속 시도", reason_code)
        self.store.add_event("disconnected", str(reason_code))

    def _on_message(self, client, userdata, msg):
        recv_ts = time.time()
        self.msg_count += 1
        try:
            readings = self.parser.parse(msg.topic, msg.payload)
            if readings is None:
                self.store.add_unmatched(msg.topic, msg.payload, recv_ts)
            elif readings:
                self.saved_count += self.store.add_readings(readings, recv_ts)
            else:
                # 규칙은 맞았지만 저장할 항목이 없음 (예: 소음 센서 → co2/온도/습도 없음)
                self.store.count_ignored()
        except Exception:  # 한 메시지가 이상해도 수집은 멈추지 않는다
            log.exception("메시지 처리 실패: topic=%s", msg.topic)
        if recv_ts - self._last_stat >= 60:
            log.info("최근 1분: 메시지 %d건 수신, %d건 저장", self.msg_count, self.saved_count)
            self.msg_count = self.saved_count = 0
            self._last_stat = recv_ts

    # --- 실행 -------------------------------------------------------------
    def run(self) -> None:
        m = self.cfg.mqtt
        self.store.add_event("start", f"client_id={m.client_id}")
        log.info("수집기 시작: %s:%s, 저장소=%s", m.host, m.port, self.cfg.storage.path)
        self.client.connect_async(m.host, m.port, keepalive=m.keepalive)
        # 첫 접속이 실패해도(브로커가 꺼져 있어도) 계속 재시도한다
        self.client.loop_forever(retry_first_connection=True)

    def stop(self, reason: str = "") -> None:
        log.info("수집기 종료 (%s)", reason)
        try:
            self.store.add_event("stop", reason)
        finally:
            self.client.disconnect()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MQTT 센서 수집기")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--env", default=".env", help="비밀값 파일 (없으면 무시)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    cfg = load_config(args.config, args.env)
    collector = Collector(cfg)

    def _handle(sig, frame):
        collector.stop(signal.Signals(sig).name)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    collector.run()
    collector.store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
