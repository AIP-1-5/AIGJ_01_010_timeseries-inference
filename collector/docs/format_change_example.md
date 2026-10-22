# 산출물 ② 메시지 형식 변경 대응 기록 (가정 시나리오)

## 지금 형식 (2026-10-20 실제 확인, ChirpStack v4)
- 토픽: `application/<앱ID>/device/<기기EUI>/event/up`
- 페이로드(필요한 부분만):
```json
{"time": "2026-10-20T07:48:42.752+00:00",
 "deviceInfo": {"devEui": "24e124128c067999", "tags": {"location": "실습실", "point": "후면 오른쪽"}},
 "object": {"co2": 974, "temperature": 24.3, "humidity": 61.5}}
```
- 규칙: `config.yaml` 의 `chirpstack-uplink`

## 상황 (가정)
게이트웨이 서버를 다른 버전(ChirpStack v3 방식)으로 바꾸면서 메시지 형식이 바뀌었다.

| | 지금 (v4) | 바뀐 뒤 (v3 방식) |
| --- | --- | --- |
| 토픽 | `application/<앱ID>/device/<EUI>/event/up` | `application/<앱ID>/device/<EUI>/rx` |
| 공간 위치 | `deviceInfo.tags.location` | `tags.location` (최상위) |
| 시각 | `time` (ISO 문자열) | `publishedAt` (ISO 문자열) |
| 값 | `object.co2` … | `object.co2` … (같음) |

```json
{"applicationName": "광주 캠퍼스", "deviceName": "AM107-067999", "devEUI": "24e124128c067999",
 "tags": {"location": "실습실", "point": "후면 오른쪽"},
 "publishedAt": "2026-10-20T07:48:42.752Z",
 "object": {"co2": 974, "temperature": 24.3, "humidity": 61.5}}
```

## 대응: 코드 수정 없이 `config.yaml` 에 규칙 하나 추가

```yaml
mqtt:
  topics: ["application/+/device/+/event/up",
           "application/+/device/+/rx",            # ← 추가
           "xflow/hvacr/+/+/status"]

rules:
  - name: chirpstack-uplink          # 기존 그대로 (v4 센서가 남아 있을 수 있음)
    ...

  - name: chirpstack-v3-rx           # ← 새로 추가
    topic: "application/+/device/{device}/rx"
    payload: json
    space_path: tags.location
    point_path: tags.point
    timestamp: publishedAt
    fields_from: object
    only_metrics: [co2, temperature, humidity]
```

적용: `sudo systemctl restart ts-collector` (설정 다시 읽기)

## 확인
1. `tools/status.py` → 같은 공간·항목(`실습실/co2`)으로 계속 쌓이는지, 기기별 커버율이 떨어지지 않는지
2. "규칙에 안 맞아 저장 못 한 토픽" 목록에 `.../rx` 가 없는지
3. 자동 테스트: `tests/test_parser.py::test_chirpstack_v3_format_change` — v4·v3 메시지가 같은 레코드(공간·기기·항목·값·시각)가 됨을 검증

## 정리
- 바꾼 것: `config.yaml` 에 토픽 1줄 + 규칙 1개 (코드 변경 0줄)
- 바뀌지 않은 것: 저장 형식(공간·항목·기기·시각·값) → 이후 과제(학습·추론)는 영향 없음
- 만약 규칙을 깜빡해도 메시지는 버려지지 않고 `unmatched` 테이블에 원문이 남아, 규칙을 추가한 뒤 원인 파악이 가능하다
