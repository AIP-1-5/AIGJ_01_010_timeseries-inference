# ts-collector — 과제 1. 데이터 수집 체계

MQTT 로 들어오는 센서 메시지(CO₂·온도·습도)를 받아 **공간·항목·시각별로 SQLite 에 저장**하는 상시 수집기.

| 요구사항 | 어떻게 해결했나 |
| --- | --- |
| 1. 공간·항목·시각별 상시 저장 | `collector/` → `readings(space, metric, device, point, value, ts, sensor_ts, recv_ts)` |
| 2. 형식 변경·새 센서는 설정만으로 | `config.yaml` 의 `rules` (토픽 패턴·필드 경로·별칭). 코드 수정 없음 → [예시](docs/format_change_example.md) |
| 3. 브로커 재시작·끊김·재부팅 자동 복구 | paho 자동 재접속(1→최대 5초) + 재접속 시 재구독 + `clean_session=false`/QoS1 + systemd `Restart=always`, `enable` |
| 4. 수집 현황 한눈에 | `tools/status.py` → 공간·항목별 건수·마지막 수신·커버율·빈 구간과 원인 |
| 확인: 브로커 재시작 1분 내 | 로컬 시험에서 **약 1초** 만에 재접속 (`tests/test_integration.py`) |
| 심화: 1년 용량·보관 정책 | `tools/estimate_size.py`, `tools/retention.py` |
| 심화: 중복(재전송) 처리 | `(device, space, metric, ts)` UNIQUE → 중복은 버리고 카운트 |

## 실제 데이터 형식 (2026-10-20 `tools/discover.py` 로 확인)
- **센서**: ChirpStack(LoRaWAN) 업링크 — 토픽 `application/<앱ID>/device/<기기EUI>/event/up`
  - 공간 = `deviceInfo.tags.location` (실습실 · 사무실 · 회의실 · 사무실 밖 …), 설치 위치 = `deviceInfo.tags.point`
  - 값 = `object` (`co2`, `temperature`, `humidity` + 기종에 따라 `tvoc`, `pressure`, `illumination` …)
  - 시각 = `time` (센서 게이트웨이가 찍은 UTC 시각)
  - **한 공간에 센서가 여러 대** (실습실은 9대 이상) → 기기(`device`)·위치(`point`)도 저장
  - 센서마다 주기가 다름 (첫 7분 관측: 대부분 1분, 일부 CO₂·온습도 센서는 2분) → 커버율을 1분 기준과 기기별 주기 기준 두 가지로 보여줌
- **냉난방**: `xflow/hvacr/{idu|odu}/<ID>/status` — `values.<공간이름>.{power, current_temperature, …}` → `hvac_*` 항목으로 저장 (과제 4 '냉방 정지' 분석용)

## 폴더 구조
```
config.yaml            # ★ 형식·토픽 설정 (여기만 고치면 됨)
.env.example           # 비밀값 예시 → .env 로 복사 (깃에 안 올라감)
collector/             # 수집기 본체 (main / parser / store / config)
tools/discover.py      # 브로커에 어떤 토픽·형식이 오는지 살펴보기
tools/status.py        # 수집 현황 (커버율·빈 구간)
tools/estimate_size.py # 1년 용량 추정 (심화)
tools/retention.py     # 오래된 데이터 5분 집계로 줄이기 (심화)
tools/fake_publisher.py# 로컬 테스트용 가짜 센서
deploy/                # systemd 서비스 + 설치 스크립트
docs/                  # 형식 변경 기록, 생각해 볼 점 답변, 보고서 템플릿
tests/                 # 단위 테스트 + 브로커 재시작 통합 테스트
```

## 1) 처음 한 번: 준비
```bash
cd ts-collector
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # 그리고 .env 에 MQTT_PASSWORD 를 이슈 #6 값으로 채우기
```

## 2) 메시지 형식 확인 (90초)
```bash
.venv/bin/python tools/discover.py --seconds 90
```
→ 토픽과 페이로드 예시가 나온다. 지금 `config.yaml` 에는 위 실제 형식에 맞는 규칙(`chirpstack-uplink`, `hvac-status`)이 들어 있다.
새 형식이 생기면 정확한 규칙을 추가하고, 그 전까지는 예비 규칙(`auto`)이 숫자 필드를 자동으로 찾아 저장한다.

## 3) 실행해 보기 (내 노트북에서)
```bash
.venv/bin/python -m collector.main          # Ctrl+C 로 종료
.venv/bin/python tools/status.py --hours 1  # 다른 터미널에서 현황 확인
```

## 4) 보드(Jetson, team5-b)에 상시 서비스로 설치
```bash
# 노트북 → 보드로 복사 (또는 팀 GitHub 에서 git clone)
scp -r ts-collector <계정>@<보드B주소>:~/
ssh <계정>@<보드B주소>
cd ~/ts-collector && cp .env.example .env && nano .env   # 비밀번호 채우기
bash deploy/install.sh                                   # venv + systemd 등록 + 시작
```
- 로그: `journalctl -u ts-collector -f`
- 현황: `.venv/bin/python tools/status.py`
- 재시작(설정 바꾼 뒤): `sudo systemctl restart ts-collector`

## 5) 확인 기준 시험 (보고서용)
| 시험 | 방법 | 기대 결과 |
| --- | --- | --- |
| 브로커 재시작 | 로컬: `pytest -s tests/test_integration.py` | "재접속까지 n초" (60초 미만) |
| 네트워크 끊김 | 보드 랜선/와이파이 30초 뽑았다 꽂기 → `journalctl` | 끊김 → 자동 재접속 로그 |
| 보드 재부팅 | `sudo reboot` 후 `systemctl status ts-collector` | active (running), 수집 재개 |
| 프로세스 죽음 | `sudo kill -9 $(pgrep -f collector.main)` | 5초 뒤 systemd 가 재시작 |
| 24시간 커버율 | 하루 뒤 `tools/status.py --hours 24 --markdown` | 95% 이상, 빈 구간은 원인 표시 |

> 커버율 두 가지: **공간·항목별 1분 커버율**(과제 정의 그대로, 같은 공간 센서 여러 대를 합쳐 1분 칸에 데이터가 있는 비율)과
> **기기별 주기 기준 커버율**(그 센서가 보내는 주기의 칸 기준). 2분 주기 센서 1대뿐인 항목은 1분 커버율이 최대 50%에 그치므로,
> 보고서에는 "센서 주기가 1분보다 길면 1분 커버율 95%는 원리상 불가 → 주기 기준 커버율로 판단"이라고 해석을 붙인다.

## 테스트
```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest -q -s     # mosquitto 가 있으면 브로커 재시작 시험도 실행
```

## 규칙 쓰는 법 (config.yaml)
- `topic`: MQTT 토픽 패턴. `{space}`, `{metric}` 은 그 단계를 캡처, `+` 한 단계, `#` 나머지 전부
- `payload`: `json` | `number` | `csv` | `auto`
- `value`: 값 필드 경로 (점으로 중첩, 예: `data.value`)
- `fields`: 한 메시지에 여러 항목일 때 `{항목: 필드경로}`
- `timestamp`: 센서 시각 필드 (epoch 초/밀리초, ISO 문자열 자동 인식, 시간대 없으면 `timezone` 기준)
- `space_path` / `metric_path` / `device_path` / `point_path`: 공간·항목·기기·위치 이름이 페이로드 안에 있을 때 (토픽의 `{device}` 캡처도 가능)
- `fields_from`: 이 경로 아래의 숫자 필드를 모두 항목으로 (`object`). `values.*` 처럼 끝이 `.*` 면 그 아래 키 하나하나가 공간
- `metric_prefix`: 항목 이름 앞에 붙일 말 (예: `hvac_`), `bool_as_number`: true/false 를 1/0 으로
- `only_metrics`: 이 항목만 저장 (공용 브로커의 다른 데이터 거르기)
- `metric_aliases` / `space_aliases`: 이름 통일 (예: `temp` → `temperature`)

규칙에 안 맞은 메시지는 `unmatched` 테이블에 원문이 남고 `status.py` 에 토픽 목록이 표시된다.

## 비밀값 주의
MQTT 비밀번호·InfluxDB 토큰은 `.env` 에만. `.env` 는 `.gitignore` 에 있어 깃에 올라가지 않는다.
