# team5-b 시계열 추론 서비스 — 결과 형식 문서

교육장 센서(CO₂·온도·습도)로 **매분** 공간별 60분 예측·이상 여부·사용 여부를 계산해 MQTT 와 HTTP 로 제공한다.
이 문서만 보고 대시보드·LLM 에이전트·냉난방 제어에서 가져다 쓸 수 있게 정리했다.

## 1. 공간 목록
| 공간 | slug | 측정 항목 | 예측·상태 |
| --- | --- | --- | --- |
| 실습실 | `lab` | CO₂·온도·습도 (센서 여러 대 평균) | ✅ |
| 사무실 | `office` | CO₂·온도·습도 | ✅ |
| 회의실 | `meeting` | CO₂·온도·습도 (약 10분 주기 센서) | ✅ |
| 사무실 밖 | `office_outside` | 온도·습도 | 최신값·이상만 |
| 복도 | `corridor` | 온도·습도 | 최신값·이상만 |

## 2. MQTT
- 브로커: 교육장 브로커 (`<교육장-MQTT-브로커>:1883`, 교육용 공용 계정)
- 모든 메시지는 **retain** → 새로 구독해도 최신값을 바로 받는다. QoS 1.

| 토픽 | 내용 | 주기 |
| --- | --- | --- |
| `team5b/inference/<slug>` | 공간 최신 결과 전체 (아래 JSON) | 매분 |
| `team5b/inference/<slug>/alarm` | 경보 한 건 (같은 경보는 1시간에 한 번만) | 발생 시 |
| `team5b/inference/_status` | 서비스 상태 (`online`, 처리 시간, 오류). 서비스가 죽으면 브로커가 `{"online": false}` 를 대신 발행(LWT) | 매분 |

구독 예: `mosquitto_sub -h <교육장-MQTT-브로커> -u <ID> -P <PW> -t 'team5b/inference/#' -v`

## 3. HTTP 조회 API
- 주소: **`http://<보드B-Tailscale-IP>:8080`** (보드 B `team5-b`, Tailscale 망 안에서 접속. MagicDNS 를 켠 기기는 `http://team5-b:8080`)
- 예: `curl http://<보드B-Tailscale-IP>:8080/spaces/lab/forecast?minutes=30`

| 요청 | 응답 |
| --- | --- |
| `GET /health` | `{"ok": true, "last_cycle_age_sec": 12.3, "uptime_sec": 3600, "mqtt": true}` (마지막 주기가 3분 넘게 없으면 503) |
| `GET /spaces` | 공간별 요약 목록 |
| `GET /spaces/meeting` | 회의실 최신 결과 전체 (slug 또는 한글 이름) |
| `GET /spaces/meeting/forecast?minutes=30` | "회의실 30분 뒤 CO₂는?" → `{"minutes": 30, "co2": 812.4, "temperature": 21.3, "time": "...", "base_time": "..."}` (5분 단위) |
| `GET /alarms?hours=12` | 최근 경보 내역 |
| `GET /log/summary?hours=12` | 주기 수·성공·실패·처리 시간(평균·p95·최대)·최대 공백 |
| `GET /monitor?hours=24` | 자기 채점: 공간·예측 길이(5~60분)별 모델 오차 vs '현재값 유지' 오차, 채점 수, 개선율(%) |

## 4. 공간 결과 JSON (`team5b/inference/<slug>`, `GET /spaces/<slug>`)
```json
{
  "space": "회의실", "slug": "meeting",
  "time": "2026-10-21T10:05:00+09:00",          // 기준 시각 = 마지막으로 '다 찬' 5분 칸의 시작
  "latest": {"co2": 539.0, "temperature": 20.5, "humidity": 67.5},
  "data_age_sec": 41.2,                           // 이 공간 마지막 수신 후 경과 초
  "units": {"co2": "ppm", "temperature": "°C", "humidity": "%"},
  "forecast": {
    "available": true, "step_min": 5, "horizon_min": 60,
    "base_time": "2026-10-21T10:05:00+09:00",
    "times": ["...10:10", "...10:15", "…", "...11:05"],   // 12개
    "co2": [545.1, 551.0, …, 667.3],
    "temperature": [20.5, 20.5, …, 20.6],
    "base_co2": 539.0,                           // 기준 칸의 실제 CO₂ (채점·현재값 유지 비교용)
    "model": "service_gru (ONNX Runtime)"
  },
  "alerts": {
    "co2_over_1000_in_min": null,                // 예: 25 → 25분 뒤 1,000ppm 넘을 것으로 예측 (지금 1,000 미만일 때만)
    "co2_already_over_1000": false,
    "ventilation_soon": false,                   // 30분 안에 1,000ppm 초과 가능성 높음 (분류 모델)
    "ventilation_prob": 0.012
  },
  "state": {"occupied": false, "occupied_prob": 0.046},  // 센서만으로 추정한 사용 여부
  "anomaly": {
    "is_anomaly": false,
    "model_alarm": false,                       // 예측보다 계속 높음(설비 이상 의심), 5분 칸 6개(30분) 연속일 때 true
    "score": 0.4, "residual_z": {"co2": 0.4, "temperature": 0.0},
    "cause": null,                               // 의심 항목 (co2 / temperature / humidity)
    "rules": [                                   // 규칙 경보 목록
      {"type": "data_gap|out_of_range|spike|stuck", "metric": "co2", "detail": "..."}
    ],
    "data_gap": false                            // 수집기 전체 3분 이상 수신 없음, 또는 이 공간이 평소 주기의 3배(최소 15분) 이상 조용함
  },
  "monitor": {                                   // 자기 채점: 60분 전에 한 예측을 지금 실제 값으로 채점 (최근 6시간)
    "window_hours": 6.0, "graded_60min": 72,
    "mae_60min_model": 31.2, "mae_60min_persistence": 38.5, "skill_pct": 19.0,
    "degraded": false                            // 모델 오차가 '현재값 유지'의 1.2배 이상 + 5ppm 이상 차이 (채점 36개 이상일 때) → 재학습 필요
  }
}
```
- 값이 없으면 `null` (예: 예측에 필요한 최근 2시간에 빈 칸이 있으면 `forecast.available=false`, `reason` 에 이유).

## 5. 경보 JSON (`team5b/inference/<slug>/alarm`)
```json
{"space": "사무실", "kind": "co2_over_1000_forecast", "metric": "co2", "detail": "25분 뒤 1,000ppm 초과 예상", "time": "2026-10-21T10:06:00+09:00"}
```
| kind | 뜻 | 권장 조치 |
| --- | --- | --- |
| `co2_over_1000_forecast` | 60분 예측에서 1,000ppm 초과 예정 | 미리 환기 |
| `ventilation_soon` | 30분 안 1,000ppm 초과 가능성 높음 | 환기 |
| `model_residual` | 예상보다 계속 높음 (환기·냉방 정지 의심) | 설비 확인 |
| `data_gap` | 수집기 전체 3분 이상 수신 없음 또는 공간별 평소 주기의 3배 이상 조용함 | 수집기·게이트웨이 확인 |
| `stuck` / `spike` / `out_of_range` | 센서 값 고정 / 튄 값 / 범위 밖 | 센서 확인 |
| `model_degraded` | 자기 채점 결과 모델이 '현재값 유지'보다 나빠짐 | 최근 데이터로 재학습 후 모델 파일 교체(무중단) |

## 6. 계산 방식 요약
- 전처리는 학습과 **같은 코드**(`tsinf/prep.py`): 재전송 중복 제거 → 게이트웨이 시계 보정 → 범위 밖 제거 → 기기별 5분 평균 → 공간 평균 → 15분 이하 결측 보간
- 예측: 범용 GRU (교육장 실데이터 2026-09-27~ 로 학습, 공간·일정 정보 없이 공간별 최근 24시간 통계로 정규화) → ONNX Runtime. 모델 파일을 바꾸면 재시작 없이 다음 주기부터 반영
- 사용 여부·환기 경보: 과제 5 분류 모델 (일정 정보 없음)
- 실데이터에는 정답(일정·고장 기록)이 없어 성능은 합성 데이터 기준이다 (보고서 과제 3~5)
