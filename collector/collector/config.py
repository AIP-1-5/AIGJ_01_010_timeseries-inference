"""설정 파일(config.yaml)과 비밀값(.env)을 읽는다.

비밀번호 같은 비밀값은 config.yaml에 쓰지 않고 환경변수(.env)로만 받는다.
"""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """아주 단순한 .env 로더. 이미 설정된 환경변수는 덮어쓰지 않는다."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass
class MqttConfig:
    host: str
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    client_id: str = ""
    keepalive: int = 30
    reconnect_min_delay: int = 1
    reconnect_max_delay: int = 5
    topics: List[str] = field(default_factory=lambda: ["#"])
    qos: int = 1
    clean_session: bool = False


@dataclass
class StorageConfig:
    path: str = "data/sensors.db"
    keep_unmatched: bool = True
    unmatched_limit: int = 5000


@dataclass
class Config:
    mqtt: MqttConfig
    storage: StorageConfig
    rules: List[Dict[str, Any]]
    timezone: str = "Asia/Seoul"
    metric_aliases: Dict[str, str] = field(default_factory=dict)
    space_aliases: Dict[str, str] = field(default_factory=dict)
    source_path: str = ""


def load_config(path: str | os.PathLike = "config.yaml", env_file: str | os.PathLike | None = ".env") -> Config:
    if env_file:
        load_dotenv(env_file)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    m = raw.get("mqtt", {})
    username = os.environ.get(m.get("username_env", "MQTT_USERNAME")) or m.get("username")
    password = os.environ.get(m.get("password_env", "MQTT_PASSWORD"))
    host = os.environ.get("MQTT_HOST") or m.get("host")
    if not host:
        raise ValueError("mqtt.host 가 설정되지 않았습니다 (config.yaml 또는 MQTT_HOST)")
    client_id = m.get("client_id") or f"ts-collector-{socket.gethostname()}"

    mqtt_cfg = MqttConfig(
        host=host,
        port=int(os.environ.get("MQTT_PORT") or m.get("port", 1883)),
        username=username,
        password=password,
        client_id=client_id,
        keepalive=int(m.get("keepalive", 30)),
        reconnect_min_delay=int(m.get("reconnect_min_delay", 1)),
        reconnect_max_delay=int(m.get("reconnect_max_delay", 5)),
        topics=list(m.get("topics", ["#"])),
        qos=int(m.get("qos", 1)),
        clean_session=bool(m.get("clean_session", False)),
    )

    s = raw.get("storage", {})
    storage_cfg = StorageConfig(
        path=os.environ.get("TS_DB_PATH") or s.get("path", "data/sensors.db"),
        keep_unmatched=bool(s.get("keep_unmatched", True)),
        unmatched_limit=int(s.get("unmatched_limit", 5000)),
    )

    rules = raw.get("rules") or []
    for i, r in enumerate(rules):
        if "topic" not in r:
            raise ValueError(f"rules[{i}] 에 topic 이 없습니다")
        r.setdefault("name", f"rule{i}")
        r.setdefault("payload", "json")

    return Config(
        mqtt=mqtt_cfg,
        storage=storage_cfg,
        rules=rules,
        timezone=raw.get("timezone", "Asia/Seoul"),
        metric_aliases=raw.get("metric_aliases") or {},
        space_aliases=raw.get("space_aliases") or {},
        source_path=str(path),
    )
