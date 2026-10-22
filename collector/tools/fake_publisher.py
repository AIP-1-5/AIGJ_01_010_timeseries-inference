"""로컬 테스트용 가짜 센서. 진짜 브로커가 아니라 내 컴퓨터의 mosquitto 로만 보낸다.

    mosquitto -p 1884 &                      # 로컬 브로커
    python tools/fake_publisher.py --port 1884 --interval 1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import paho.mqtt.client as mqtt

SPACES = ["lecture1", "lecture2", "meeting", "lounge"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1884)
    ap.add_argument("--interval", type=float, default=60, help="초 (실제 센서는 60)")
    ap.add_argument("--count", type=int, default=0, help="0 이면 무한")
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost"):
        raise SystemExit("가짜 데이터는 로컬 브로커에만 보냅니다.")

    c = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    c.connect(args.host, args.port)
    c.loop_start()
    i = 0
    while args.count == 0 or i < args.count:
        now = time.time()
        for k, s in enumerate(SPACES):
            co2 = 450 + 400 * max(0.0, math.sin(now / 900 + k)) + random.gauss(0, 15)
            payload = {"co2": round(co2), "temp": round(23 + random.gauss(0, 0.3), 1),
                       "hum": round(40 + random.gauss(0, 1), 1), "ts": int(now)}
            c.publish(f"academy/{s}", json.dumps(payload), qos=1)
        i += 1
        time.sleep(args.interval)
    c.loop_stop()
    c.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
