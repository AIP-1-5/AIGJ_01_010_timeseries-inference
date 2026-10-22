"""브로커에 어떤 토픽이 어떤 형식으로 오는지 살펴보고 config.yaml 규칙 초안을 만든다.

    python tools/discover.py --seconds 90

비밀번호는 .env 의 MQTT_PASSWORD 에서 읽는다 (화면에 출력하지 않음).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import paho.mqtt.client as mqtt  # noqa: E402

from collector.config import load_config  # noqa: E402


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):
            yield from flatten(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1], obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MQTT 토픽·형식 살펴보기")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--seconds", type=int, default=90)
    ap.add_argument("--topic", default=None, help="구독할 토픽 (기본: config.yaml 의 topics)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.env)
    m = cfg.mqtt
    topics = [args.topic] if args.topic else m.topics
    seen: "OrderedDict[str, dict]" = OrderedDict()

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                         client_id=f"{m.client_id}-discover-{int(time.time())}")
    if m.username:
        client.username_pw_set(m.username, m.password)

    def on_connect(c, u, flags, rc, props):
        if rc.is_failure:
            print(f"접속 실패: {rc} (아이디/비밀번호/주소 확인)")
            return
        print(f"접속 성공 → {topics} 구독, {args.seconds}초 동안 수집합니다...\n")
        for t in topics:
            c.subscribe(t, qos=0)

    def on_message(c, u, msg):
        info = seen.setdefault(msg.topic, {"count": 0, "sample": None})
        info["count"] += 1
        if info["sample"] is None:
            info["sample"] = msg.payload[:500].decode("utf-8", errors="replace")
            print(f"[새 토픽] {msg.topic}\n   {info['sample']}\n")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(m.host, m.port, keepalive=30)
    client.loop_start()
    time.sleep(args.seconds)
    client.loop_stop()
    client.disconnect()

    print("=" * 60)
    print(f"토픽 {len(seen)}개 발견 ({args.seconds}초 동안)\n")
    for topic, info in seen.items():
        print(f"- {topic}  ({info['count']}건)")
        try:
            data = json.loads(info["sample"])
            for path, val in flatten(data):
                print(f"    {path} = {val!r}")
        except (json.JSONDecodeError, TypeError):
            print(f"    (JSON 아님) {info['sample']!r}")
    print("\n위 결과를 보고 config.yaml 의 rules 를 채우세요 (README '규칙 쓰는 법' 참고).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
