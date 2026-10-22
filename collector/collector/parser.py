"""MQTT 메시지를 (공간, 항목, 값, 센서시각) 레코드로 바꾼다.

어떤 토픽이 어떤 공간·항목인지, 페이로드의 어느 필드가 값·시각인지는
전부 config.yaml 의 rules 로 정한다. 형식이 바뀌면 코드가 아니라 설정만 고친다.

규칙 예시 (config.yaml):

    rules:
      - name: per-metric-json
        topic: "academy/{space}/{metric}"   # {이름} = 토픽 한 단계를 캡처, + / # 도 사용 가능
        payload: json                       # json | number | csv
        value: value                        # 값이 들어 있는 필드 (점으로 중첩: data.value)
        timestamp: ts                       # 센서가 찍은 시각 필드 (없으면 생략)

      - name: multi-field-json
        topic: "sensors/{space}"
        payload: json
        fields: {co2: co2, temperature: temp, humidity: hum}   # 항목 이름: 필드
        timestamp: time
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


@dataclass
class Reading:
    space: str
    metric: str
    value: float
    sensor_ts: Optional[float]  # UTC epoch 초, 센서가 찍은 시각 (없으면 None)
    topic: str
    rule: str
    device: str = ""            # 센서 기기 ID (한 공간에 여러 대일 수 있음)
    point: str = ""             # 공간 안 설치 위치 (예: 전방 좌측)
    seq: Optional[int] = None   # 메시지 일련번호 (LoRaWAN fCnt) → 같은 번호면 재전송 중복


DEFAULT_METRIC_ALIASES = {
    "co2": "co2", "carbon_dioxide": "co2",
    "temp": "temperature", "tmp": "temperature", "temperature": "temperature",
    "hum": "humidity", "humi": "humidity", "rh": "humidity", "humidity": "humidity",
}

_PLACEHOLDER = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def match_topic(pattern: str, topic: str) -> Optional[Dict[str, str]]:
    """토픽이 패턴과 맞으면 캡처한 값들을, 아니면 None 을 돌려준다."""
    p_levels = pattern.split("/")
    t_levels = topic.split("/")
    captures: Dict[str, str] = {}
    for i, p in enumerate(p_levels):
        if p == "#":
            captures["#"] = "/".join(t_levels[i:])
            return captures
        if i >= len(t_levels):
            return None
        t = t_levels[i]
        m = _PLACEHOLDER.match(p)
        if m:
            if t == "":
                return None
            captures[m.group(1)] = t
        elif p == "+":
            continue
        elif p != t:
            return None
    if len(t_levels) != len(p_levels):
        return None
    return captures


def get_path(obj: Any, path: str) -> Any:
    """'a.b.0.c' 같은 점 경로로 중첩 값을 꺼낸다. 없으면 None."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.lstrip("-").isdigit():
            idx = int(part)
            cur = cur[idx] if -len(cur) <= idx < len(cur) else None
        else:
            return None
    return cur


def to_float(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def parse_timestamp(v: Any, fmt: str = "auto", tz: str = "Asia/Seoul") -> Optional[float]:
    """센서 시각을 UTC epoch 초로 바꾼다. 해석 못 하면 None."""
    if v is None or v == "":
        return None
    if fmt in ("auto", "epoch_s", "epoch_ms"):
        f = to_float(v)
        if f is not None:
            if fmt == "epoch_ms" or (fmt == "auto" and f > 1e11):
                f /= 1000.0
            return f
        if fmt != "auto":
            return None
    if isinstance(v, str):
        s = v.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s.replace(" ", "T", 1) if "T" not in s and " " in s else s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            if ZoneInfo is not None:
                dt = dt.replace(tzinfo=ZoneInfo(tz))
            else:  # pragma: no cover
                dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def _decode_payload(payload: bytes, kind: str) -> Any:
    text = payload.decode("utf-8", errors="replace").strip()
    if kind == "json":
        return json.loads(text)
    if kind == "auto":
        try:
            return json.loads(text)
        except ValueError:
            return text
    if kind == "number":
        return text
    if kind == "csv":
        return [x.strip() for x in text.split(",")]
    raise ValueError(f"알 수 없는 payload 종류: {kind}")


def _resolve(rule: Dict[str, Any], key: str, captures: Dict[str, str], data: Any) -> Optional[str]:
    """공간/항목 이름을 정한다: 토픽 캡처 > 페이로드 경로(<key>_path) > 고정값(<key>)."""
    if key in captures:
        return captures[key]
    path = rule.get(f"{key}_path")
    if path and isinstance(data, (dict, list)):
        v = get_path(data, path)
        if v is not None:
            return str(v)
    const = rule.get(key)
    if const is not None:
        return str(const).format(**captures) if "{" in str(const) else str(const)
    return None


class Parser:
    def __init__(self, rules: List[Dict[str, Any]], timezone_name: str = "Asia/Seoul",
                 metric_aliases: Optional[Dict[str, str]] = None,
                 space_aliases: Optional[Dict[str, str]] = None):
        self.rules = rules
        self.tz = timezone_name
        self.metric_aliases = dict(DEFAULT_METRIC_ALIASES)
        self.metric_aliases.update(metric_aliases or {})
        self.space_aliases = space_aliases or {}

    def parse(self, topic: str, payload: bytes) -> Optional[List[Reading]]:
        """맞는 규칙이 없으면 None, 규칙은 맞았지만 값이 없으면 빈 리스트."""
        for rule in self.rules:
            captures = match_topic(rule["topic"], topic)
            if captures is None:
                continue
            try:
                data = _decode_payload(payload, rule.get("payload", "json"))
            except (ValueError, UnicodeDecodeError):
                return []
            return self._extract(rule, captures, data, topic)
        return None

    def _extract(self, rule: Dict[str, Any], captures: Dict[str, str], data: Any, topic: str) -> List[Reading]:
        if rule.get("payload") == "auto":
            return self._extract_auto(rule, captures, data, topic)

        ts_field = rule.get("timestamp")
        sensor_ts = None
        if ts_field is not None:
            raw_ts = get_path(data, ts_field) if isinstance(data, (dict, list)) else None
            sensor_ts = parse_timestamp(raw_ts, rule.get("timestamp_format", "auto"), self.tz)
        device = _resolve(rule, "device", captures, data) or ""
        point = _resolve(rule, "point", captures, data) or ""
        seq = None
        if rule.get("seq_path") and isinstance(data, dict):
            sv = to_float(get_path(data, rule["seq_path"]))
            seq = int(sv) if sv is not None else None

        fields_from = rule.get("fields_from")
        if fields_from and str(fields_from).endswith(".*"):
            # {"values": {"회의실": {...}, "사무실": {...}}} → 키 하나하나가 공간
            parent = get_path(data, str(fields_from)[:-2]) if isinstance(data, dict) else None
            out: List[Reading] = []
            if isinstance(parent, dict):
                for key, child in parent.items():
                    if isinstance(child, dict):
                        space = self.space_aliases.get(str(key), str(key))
                        out += self._readings(rule, space, list(child.items()), sensor_ts, topic, device, point, seq)
            return out

        space = _resolve(rule, "space", captures, data)
        if not space:
            return []
        space = self.space_aliases.get(space, space)

        pairs: List[tuple] = []
        fields = rule.get("fields")
        if fields_from:
            src = get_path(data, str(fields_from)) if isinstance(data, (dict, list)) else None
            if isinstance(src, dict):
                pairs = list(src.items())
        elif fields:
            for metric, path in fields.items():
                pairs.append((metric, get_path(data, str(path)) if isinstance(data, (dict, list)) else None))
        else:
            metric = _resolve(rule, "metric", captures, data)
            if not metric:
                return []
            if rule.get("payload") == "number":
                value = data
            elif isinstance(data, (dict, list)):
                value = get_path(data, rule.get("value", "value"))
            else:
                value = data
            pairs.append((metric, value))

        return self._readings(rule, space, pairs, sensor_ts, topic, device, point, seq)

    def _readings(self, rule, space, pairs, sensor_ts, topic, device: str = "", point: str = "",
                  seq: Optional[int] = None) -> List[Reading]:
        only = rule.get("only_metrics")
        prefix = rule.get("metric_prefix", "")
        bool_as_number = rule.get("bool_as_number", False)
        out: List[Reading] = []
        for metric, value in pairs:
            if isinstance(value, bool):
                if not bool_as_number:
                    continue
                value = 1.0 if value else 0.0
            f = to_float(value)
            if f is None:
                continue
            metric = self.metric_aliases.get(metric, self.metric_aliases.get(str(metric).lower(), metric))
            metric = f"{prefix}{metric}"
            if only and metric not in only:
                continue
            out.append(Reading(space=space, metric=metric, value=f, sensor_ts=sensor_ts,
                               topic=topic, rule=rule["name"], device=device, point=point, seq=seq))
        return out

    def _extract_auto(self, rule: Dict[str, Any], captures: Dict[str, str], data: Any, topic: str) -> List[Reading]:
        """형식을 모를 때 쓰는 자동 해석.

        - 숫자 페이로드: 공간 = 토픽의 space_level 번째 단계(기본 -2), 항목 = metric_level(기본 -1)
        - JSON 객체: 숫자 필드 하나하나를 항목으로 저장. 공간은 space_keys 중 있는 필드,
          없으면 토픽 단계. {"value": 숫자} 꼴이면 항목 이름은 토픽 마지막 단계(또는 metric/type 필드).
        """
        levels = topic.split("/")

        def level(idx: int) -> Optional[str]:
            try:
                return levels[idx] or None
            except IndexError:
                return None

        space_level = int(rule.get("space_level", -2))
        metric_level = int(rule.get("metric_level", -1))
        ts_keys = rule.get("timestamp_keys", ["ts", "timestamp", "time", "datetime", "date", "created_at", "measured_at"])
        space_keys = rule.get("space_keys", ["space", "room", "location", "place", "zone", "area"])
        metric_keys = rule.get("metric_keys", ["metric", "type", "sensor_type", "measurement", "name"])
        ignore = set(rule.get("ignore_keys", ["id", "seq", "rssi", "battery", "snr", "port", "fcnt"]))
        ignore |= set(ts_keys) | set(space_keys) | set(metric_keys)

        if not isinstance(data, dict):
            space = captures.get("space") or (level(space_level) if len(levels) >= 2 else None) or rule.get("space")
            metric = captures.get("metric") or level(metric_level)
            if not space or not metric:
                return []
            space = self.space_aliases.get(space, space)
            return self._readings(rule, space, [(metric, data)], None, topic)

        sensor_ts = None
        for k in ts_keys:
            if k in data:
                sensor_ts = parse_timestamp(data[k], rule.get("timestamp_format", "auto"), self.tz)
                if sensor_ts is not None:
                    break

        value_form = "value" in data and to_float(data.get("value")) is not None
        space = captures.get("space")
        if not space:
            for k in space_keys:
                if data.get(k) not in (None, ""):
                    space = str(data[k])
                    break
        if not space:
            # {"value": ..} 꼴이면 토픽 = …/공간/항목, 여러 필드 꼴이면 토픽 = …/공간
            if value_form and len(levels) >= 2:
                space = level(space_level)
            else:
                space = level(int(rule.get("multi_space_level", -1)))
        if not space:
            return []
        space = self.space_aliases.get(space, space)

        pairs: List[tuple] = []
        if value_form:
            metric = captures.get("metric")
            if not metric:
                for k in metric_keys:
                    if data.get(k) not in (None, ""):
                        metric = str(data[k])
                        break
            metric = metric or level(metric_level)
            if metric:
                pairs.append((metric, data["value"]))
        else:
            src = data
            for nested in ("data", "values", "payload", "sensors", "fields"):
                if isinstance(data.get(nested), dict):
                    src = data[nested]
                    break
            for k, v in src.items():
                if k in ignore or isinstance(v, (dict, list)):
                    continue
                if to_float(v) is not None:
                    pairs.append((k, v))
        return self._readings(rule, space, pairs, sensor_ts, topic)
