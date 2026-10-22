# 과제 1 수집 현황 보고 (템플릿)

> 측정 조건: 기간 ____ ~ ____ (24시간) · 실데이터 · 보드: team5-b (JetPack 7.x, 전력 모드 ____)

## 1. 수집 체계
- 구성: MQTT(<교육장-MQTT-브로커>) → 수집기(Python, paho-mqtt) → SQLite(WAL) · systemd 상시 서비스
- 저장 단위: 공간 · 항목 · 시각(센서 시각 + 수신 시각) · 값
- 형식 대응: `config.yaml` rules (코드 수정 없이 설정만 변경)

## 2. 공간·항목별 커버율
`python tools/status.py --hours 24 --markdown` 결과를 붙인다.

## 3. 결측 구간과 추정 원인
(위 명령 결과의 "결측 구간" 표 + 해석)
- 수집기 끊김(events 에 disconnected/stop) 과 겹침 → 수집 측 원인
- 수집기는 연결 중인데 비어 있음 → 센서·게이트웨이 측 미전송

## 4. 장애 시험
| 시험 | 방법 | 결과 |
| --- | --- | --- |
| 브로커 재시작 | (로컬 시험) mosquitto 종료 후 재시작 | 재접속까지 __초 (기준 60초) |
| 네트워크 끊김 | 랜선/와이파이 30초 끊기 | 재접속 __초, 결측 __분 |
| 보드 재부팅 | `sudo reboot` | 부팅 후 __초 만에 수집 재개 |
| 프로세스 강제 종료 | `sudo kill -9 <PID>` | systemd 가 __초 뒤 재시작 |

## 5. 형식 변경 대응 기록
`docs/format_change_example.md`

## 6. 생각해 볼 점
`docs/answers.md`
