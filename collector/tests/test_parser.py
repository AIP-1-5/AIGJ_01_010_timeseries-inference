import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collector.parser import Parser, match_topic, parse_timestamp  # noqa: E402

ALIASES = {"co2": "co2", "temp": "temperature", "hum": "humidity", "temperature": "temperature"}


def test_match_topic():
    assert match_topic("academy/{space}/{metric}", "academy/lecture1/co2") == {"space": "lecture1", "metric": "co2"}
    assert match_topic("academy/{space}/{metric}", "academy/lecture1") is None
    assert match_topic("academy/+/co2", "academy/x/co2") == {}
    assert match_topic("academy/#", "academy/a/b/c") == {"#": "a/b/c"}
    assert match_topic("#", "anything/at/all") == {"#": "anything/at/all"}
    assert match_topic("other/{space}", "academy/x") is None


def test_parse_timestamp():
    assert parse_timestamp(1760925600) == 1760925600
    assert parse_timestamp(1760925600123) == 1760925600.123
    assert parse_timestamp("2025-10-20T02:00:00Z") == 1760925600
    # 시간대 없는 문자열은 Asia/Seoul 로 해석 (UTC+9)
    assert parse_timestamp("2025-10-20 11:00:00") == 1760925600
    assert parse_timestamp("garbage") is None
    assert parse_timestamp(None) is None


def test_per_metric_json_rule():
    p = Parser([{"name": "r", "topic": "academy/{space}/{metric}", "payload": "json",
                 "value": "value", "timestamp": "ts"}], metric_aliases=ALIASES)
    out = p.parse("academy/lecture1/co2", json.dumps({"value": 812, "ts": 1760925600}).encode())
    assert len(out) == 1
    r = out[0]
    assert (r.space, r.metric, r.value, r.sensor_ts) == ("lecture1", "co2", 812.0, 1760925600)


def test_multi_field_json_rule_with_aliases():
    p = Parser([{"name": "r", "topic": "academy/{space}", "payload": "json",
                 "fields": {"co2": "co2", "temp": "t", "hum": "h"}, "timestamp": "time"}],
               metric_aliases=ALIASES)
    out = p.parse("academy/meeting", json.dumps({"co2": 900, "t": 24.5, "h": "41", "time": "2025-10-20 11:00:00"}).encode())
    got = {(r.metric, r.value) for r in out}
    assert got == {("co2", 900.0), ("temperature", 24.5), ("humidity", 41.0)}
    assert all(r.sensor_ts == 1760925600 for r in out)


def test_number_payload_and_bad_values():
    p = Parser([{"name": "r", "topic": "a/{space}/{metric}", "payload": "number"}])
    assert p.parse("a/lounge/co2", b"650")[0].value == 650.0
    assert p.parse("a/lounge/co2", b"NaN") == []
    assert p.parse("a/lounge/co2", b"not-a-number") == []
    assert p.parse("zzz/lounge/co2", b"1") is None  # 규칙 불일치


def test_broken_json_does_not_crash():
    p = Parser([{"name": "r", "topic": "a/{space}", "payload": "json", "fields": {"co2": "co2"}}])
    assert p.parse("a/x", b"{not json") == []


def test_space_from_payload_path():
    p = Parser([{"name": "r", "topic": "sensors/+", "payload": "json", "space_path": "room",
                 "fields": {"co2": "data.co2"}}])
    out = p.parse("sensors/dev7", json.dumps({"room": "meeting", "data": {"co2": 1001}}).encode())
    assert out[0].space == "meeting" and out[0].value == 1001.0


def test_auto_rule_variants():
    rule = {"name": "auto", "topic": "#", "payload": "auto", "only_metrics": ["co2", "temperature", "humidity"]}
    p = Parser([rule], metric_aliases=ALIASES)
    # 숫자만
    out = p.parse("academy/lecture2/co2", b"700")
    assert [(r.space, r.metric, r.value) for r in out] == [("lecture2", "co2", 700.0)]
    # 여러 필드 JSON, 공간은 토픽에서
    out = p.parse("academy/lounge", json.dumps({"co2": 500, "temp": 23, "battery": 90, "ts": 1760925600}).encode())
    assert {(r.space, r.metric) for r in out} == {("lounge", "co2"), ("lounge", "temperature")}
    assert all(r.sensor_ts == 1760925600 for r in out)
    # 공간이 페이로드에 있음
    out = p.parse("x/y", json.dumps({"room": "meeting", "humidity": 40}).encode())
    assert [(r.space, r.metric) for r in out] == [("meeting", "humidity")]
    # value 형식
    out = p.parse("academy/lecture1/temperature", json.dumps({"value": 25.1}).encode())
    assert [(r.space, r.metric, r.value) for r in out] == [("lecture1", "temperature", 25.1)]
    # 다른 팀의 예측값 같은 무관한 숫자는 저장하지 않음
    assert p.parse("team3/prediction", json.dumps({"score": 0.9}).encode()) == []


def test_format_change_needs_only_config():
    """형식 변경 시나리오: 같은 데이터가 새 형식으로 와도 규칙만 추가하면 같은 레코드가 된다."""
    old = Parser([{"name": "v1", "topic": "academy/{space}/{metric}", "payload": "number"}])
    new = Parser([{"name": "v2", "topic": "academy/v2/{space}", "payload": "json",
                   "fields": {"co2": "measurements.co2_ppm"}, "timestamp": "measured_at"}])
    a = old.parse("academy/lecture1/co2", b"812")[0]
    b = new.parse("academy/v2/lecture1",
                  json.dumps({"measurements": {"co2_ppm": 812}, "measured_at": 1760925600}).encode())[0]
    assert (a.space, a.metric, a.value) == (b.space, b.metric, b.value)


CHIRP = {
    "deduplicationId": "x", "time": "2026-10-20T07:48:42.752+00:00",
    "deviceInfo": {"devEui": "24e124128c067999", "deviceName": "AM107-067999",
                   "tags": {"location": "실습실", "point": "후면 오른쪽"}},
    "object": {"activity": 0, "tvoc": 284, "humidity": 61.5, "temperature": 24.3, "co2": 974},
}


def test_chirpstack_uplink_rule():
    rule = {"name": "chirp", "topic": "application/+/device/{device}/event/up", "payload": "json",
            "space_path": "deviceInfo.tags.location", "point_path": "deviceInfo.tags.point",
            "timestamp": "time", "fields_from": "object", "only_metrics": ["co2", "temperature", "humidity"]}
    p = Parser([rule])
    out = p.parse("application/96b4/device/24e124128c067999/event/up", json.dumps(CHIRP).encode())
    assert {(r.metric, r.value) for r in out} == {("co2", 974.0), ("temperature", 24.3), ("humidity", 61.5)}
    r = out[0]
    assert (r.space, r.device, r.point) == ("실습실", "24e124128c067999", "후면 오른쪽")
    assert abs(r.sensor_ts - 1792482522.752) < 1e-3


def test_hvac_status_rule():
    rule = {"name": "hvac", "topic": "xflow/hvacr/{unit_type}/{device}/status", "payload": "json",
            "timestamp": "ts", "fields_from": "values.*", "metric_prefix": "hvac_", "bool_as_number": True,
            "only_metrics": ["hvac_power", "hvac_current_temperature"]}
    p = Parser([rule])
    msg = {"ts": 1792482080491, "values": {"개발실-실내기": {"current_humidity": 48, "current_temperature": 22.3,
                                                       "power": False, "mode": 0}}}
    out = p.parse("xflow/hvacr/idu/5442/status", json.dumps(msg).encode())
    assert {(r.space, r.metric, r.value) for r in out} == {
        ("개발실-실내기", "hvac_current_temperature", 22.3), ("개발실-실내기", "hvac_power", 0.0)}
    assert out[0].device == "5442" and out[0].sensor_ts == 1792482080.491


def test_chirpstack_v3_format_change():
    """형식 변경 기록(docs/format_change_example.md): 규칙만 추가하면 v3 형식도 같은 레코드가 된다."""
    v4 = {"name": "chirpstack-uplink", "topic": "application/+/device/{device}/event/up", "payload": "json",
          "space_path": "deviceInfo.tags.location", "point_path": "deviceInfo.tags.point",
          "timestamp": "time", "fields_from": "object", "only_metrics": ["co2", "temperature", "humidity"]}
    v3 = {"name": "chirpstack-v3-rx", "topic": "application/+/device/{device}/rx", "payload": "json",
          "space_path": "tags.location", "point_path": "tags.point",
          "timestamp": "publishedAt", "fields_from": "object", "only_metrics": ["co2", "temperature", "humidity"]}
    p = Parser([v4, v3])
    a = p.parse("application/96b4/device/24e124128c067999/event/up", json.dumps(CHIRP).encode())
    v3_msg = {"applicationName": "광주 캠퍼스", "devEUI": "24e124128c067999",
              "tags": {"location": "실습실", "point": "후면 오른쪽"},
              "publishedAt": "2026-10-20T07:48:42.752Z",
              "object": {"co2": 974, "temperature": 24.3, "humidity": 61.5}}
    b = p.parse("application/96b4/device/24e124128c067999/rx", json.dumps(v3_msg).encode())
    key = lambda rs: sorted((r.space, r.device, r.point, r.metric, r.value, round(r.sensor_ts, 3)) for r in rs)
    assert key(a) == key(b) and len(a) == 3


def test_chirpstack_seq():
    rule = {"name": "chirp", "topic": "application/+/device/{device}/event/up", "payload": "json",
            "space_path": "deviceInfo.tags.location", "timestamp": "time", "seq_path": "fCnt",
            "fields_from": "object", "only_metrics": ["co2"]}
    msg = dict(CHIRP, fCnt=106633)
    out = Parser([rule]).parse("application/x/device/24e124128c067999/event/up", json.dumps(msg).encode())
    assert out[0].seq == 106633
