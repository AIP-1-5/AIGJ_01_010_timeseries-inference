# AIGJ_01_010_timeseries-inference

교육장 센서(CO₂·온도·습도) 데이터를 MQTT로 모아서 **60분 뒤를 예측하고, 이상과 공간 상태(사용 여부·환기 필요)를 판단해, Jetson Orin Nano(보드 B, team5-b)에서 상시 추론 서비스로 운영**하는 3.2 시계열 데이터 추론 실습 저장소 (과제 1~8, team5 · suyeon040610).

- 결과 보고서: [`inference/reports/final_report.md`](inference/reports/final_report.md) (과제별 세부는 `inference/reports/task*_report.md`)
- 서비스 결과 형식(API·MQTT): [`inference/docs/interface.md`](inference/docs/interface.md)
- 과제를 처음부터 이해하기 위한 공부 문서: [`study/`](study/) (0 큰 그림 → 1~8 과제별 → 9 발표 핵심 정리)
- 과제 안내 원문: 팀 Private 저장소 `AIP-1-5/docs`의 `assignment.md`, 환경은 `environment.md`

**데이터**: ① 내 수집기로 직접 모은 데이터(10/20~) ② 교육장 InfluxDB 과거 데이터(사용 허락 확인, 10/22) ③ 합성 데이터 — 안내문에 '생성기를 쓴다'고 되어 있으나 제공된 생성기가 없어 안내문 설명대로 직접 만듦(`inference/tsinf/synth.py`, 공간 4곳·5주·일정에 따른 물리 변화·마지막 주 고장 5종 15건). 실습실 실제 시간표(평일 09:00~11:30 · 13:00~18:00)로 실데이터 일정 실험도 함.

## 결과 한눈에

| 과제 | 확인 기준 | 결과 | |
|---|---|---|---|
| 1 수집 | 브로커 재시작 1분 내 재수집 · 재부팅 자동 시작 · 24시간 커버율 95% | 재접속 약 1초 · 강제 종료 후 5초 자동 재시작 · **실제 보드 재부팅(10-21 18:54) 후 2분 만에 자동 재개** · 실습실 98.4%(CO₂ 95.1%) · 사무실 97.8% | ✅ (회의실·사무실 밖은 센서 1대라 미달, 원인 기록) |
| 2 기준선 | 학습 기간으로만 정규화 · 같은 조건 비교 | 시간순 70/15/15, 기준선 3종 5/15/30/60분 MAE (내 수집 데이터 + InfluxDB 8주) | ✅ |
| 3 예측 | 60분 CO₂ 오차 '현재값 유지'보다 30% 이상 감소 | 합성(일정 포함) **GRU 63% 감소** (40.9 → 15.0ppm) · **실데이터 + 실제 실습실 시간표: 41% 감소** (50.0 → 29.3ppm) | ✅ 합성 / ✅ 실데이터 |
| 4 이상 탐지 | 탐지율 80% · 헛경보 하루 1회 이하 | **규칙 + 예측 오차: 80%, 하루 1.0회** | ✅ |
| 5 상태 | 규칙보다 좋음 · 환기 경보 재현율 85% | 사용 여부 F1 85.8 vs 규칙 79.1 · **환기 재현율 97.4%** | ✅ |
| 6 사전학습 | 정확도·구간·시간·메모리로 판단 | Chronos-Bolt: 합성에선 최하위, **내 실데이터에선 1등(36.0 vs GRU 39.3 vs 현재값 42.0ppm)**, 보드 92ms | ✅ |
| 7 엣지 | 변환 후 CO₂ 차이 1ppm 이하 · 측정 조건 기록 | ONNX Runtime **0.22ms** (PyTorch CPU 62ms, 약 280배), 차이 0.0001ppm | ✅ (TensorRT GPU FP32 최대 차이 0.4~0.5ppm, FP16 1.1~1.4ppm — 이 작은 모델은 GPU 0.9ms로 CPU보다 느림) |
| 8 서비스 | 12시간 무중단 · 끊김 경보 · 조회 API · 형식 문서 | **12시간 713주기 실패 0**, 평균 0.95초 · 수집기 정지 시험에서 3.5분 만에 '데이터 끊김' 경보 · 헛경보 0 | ✅ (MQTT 발행 가동, 10-22 11:20~) |

## 구성

| 경로 | 내용 |
|---|---|
| `collector/` | 과제 1 수집기 (보드 `~/ts-collector`, systemd `ts-collector`) — 자세한 설명은 [`collector/README.md`](collector/README.md) |
| `inference/tsinf/` | 과제 2~8 공용 코드: `prep.py` 전처리★, `forecast.py` 예측 모델, `anomaly.py` 이상 탐지, `synth.py` 합성 데이터 |
| `inference/task*.py` | 과제별 실험 스크립트 (실행하면 `inference/reports/`에 보고서가 생김) |
| `inference/service/` | 과제 8 상시 추론 서비스 (보드 `~/ts-inference`, systemd `ts-inference`, HTTP 8080) |
| `inference/tools/` | ONNX 변환, 보드 벤치마크, InfluxDB 내보내기, 장애 시험 |
| `inference/reports/` | 결과 보고서·그림·로그 (`real56/` = InfluxDB 8주, `own/` = 내 수집기 데이터) |
| `inference/docs/interface.md` | 서비스 결과 형식 |
| `study/` | 과제 이해용 공부 문서 10편 |

★ `prep.py`는 맥(학습)과 보드(서비스)가 **같은 파일**을 쓴다. 학습과 서비스의 전처리가 어긋나지 않게 하기 위해서다.

## 어디서 무엇을 하나

| 장비 | 하는 일 |
|---|---|
| 맥북 | 코드 작성, 분석, 모델 학습(PyTorch), ONNX 변환 |
| 보드 B (team5-b, Jetson Orin Nano Super 8GB, JetPack 7 / L4T R39.2.1) | 수집기 상시 실행, InfluxDB(내부망) 내려받기, 추론 서비스(ONNX Runtime), 보드 측정 |
| 연결 | Tailscale로 보드 B에 SSH (계정 `suyeon`) |

보드는 팀 공용이라 **학습은 맥에서 하고 보드에는 작은 ONNX 모델만 올린다**. 보드에서 한 번 예측하는 데 0.22ms, 매분 한 번 돈다.

## 시작하기

```bash
git clone https://github.com/AIP-1-5/AIGJ_01_010_timeseries-inference.git
git clone https://github.com/AIP-1-5/docs.git          # 과제 안내 (나란히 둔다)
cd AIGJ_01_010_timeseries-inference
cp collector/.env.example collector/.env              # MQTT 계정은 과제 이슈/단톡방에서 받아 .env 에만 적는다
```

## 과제 1. 수집기

MQTT(`<교육장-MQTT-브로커>:1883`)로 들어오는 ChirpStack LoRaWAN 업링크와 냉난방 상태를 받아 SQLite(WAL)에 저장한다. 형식이 바뀌면 `config.yaml`의 규칙만 고친다.

```bash
cd collector
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                         # 단위·통합 시험 (20개)
.venv/bin/python tools/discover.py --seconds 90       # 브로커에 어떤 토픽·형식이 오는지 보기
.venv/bin/python -m collector.main                    # 앞에서 실행 (Ctrl+C 종료)
bash deploy/install.sh                                # 보드: venv + systemd 등록 (재부팅 후 자동 시작)
.venv/bin/python tools/status.py --hours 24 --markdown   # 커버율·빈 구간과 원인
.venv/bin/python tools/estimate_size.py               # 1년 뒤 용량 추정 (심화)
.venv/bin/python tools/retention.py --keep-days 90    # 오래된 데이터 5분 집계로 줄이기 (미리보기, --apply 로 실행)
```

| 파일 | 역할 |
|---|---|
| `config.yaml` | 브로커, 구독 토픽, 해석 규칙(`chirpstack-uplink`, `hvac-status`, 예비 `auto`) |
| `collector/main.py` | 접속·자동 재접속(1→5초)·재구독, 1분마다 수신 현황 로그 |
| `collector/parser.py` | 규칙 엔진: 토픽 패턴, 필드 경로, 공간·기기·위치 추출, 이름 통일 |
| `collector/store.py` | 저장, 중복 제거(UNIQUE + LoRa 프레임 번호 fCnt), 게이트웨이 시계 오류 보정, 1970년 시각 보정, 이벤트 기록 |
| `tools/status.py` | 공간·항목별 커버율, 기기별 자기 주기 기준 커버율, 빈 구간과 추정 원인 |
| `docs/format_change_example.md` | 형식 변경 대응 기록 (ChirpStack v3 형식 → 규칙 1개 추가, 코드 0줄) |
| `docs/answers.md` | 과제 1 '생각해 볼 점' 답변 |

결과: [`inference/reports/task1_status_24h.md`](inference/reports/task1_status_24h.md) — 누적 54,251건, 중복 제거 1,092건, 규칙 불일치 0건.

## 과제 2. 데이터 이해와 기준선

```bash
cd inference
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# 보드 B에서: 교육장 InfluxDB(내부망) 8주 내보내기 → 맥으로 복사
.venv/bin/python tools/export_influx.py --days 56 --out data/influx_56d.db
.venv/bin/python task2_baseline.py --db data/sensors_boardB.db   # 내 수집기 데이터
```

- 정제: 재전송 중복 제거 · 시계 보정 · 범위 밖 제거(CO₂ 300~5000, 온도 −10~50, 습도 0~100) · 기기별 5분 평균 → 공간 평균 · 15분 이하 결측만 보간 · 위치 태그 보정
- 분할: 시간순 70/15/15, 정규화 값은 학습 구간으로만
- 60분 CO₂ 기준선(InfluxDB 8주): 현재값 유지 49.0 · 최근 1시간 평균 64.6 · 어제 같은 시각 124.0ppm

보고서: [`task2_report.md`](inference/reports/task2_report.md), [`real56/task2_report.md`](inference/reports/real56/task2_report.md)

## 과제 3. 60분 예측

```bash
.venv/bin/python -m tsinf.synth                       # 합성 데이터 5주 생성 (data/synth)
.venv/bin/python task3_forecast.py --models gru lstm tcn   # 합성: 일정 포함/제외, 모델 3종
.venv/bin/python task3_real.py --db data/influx_56d.db     # 실데이터 8주 검증
.venv/bin/python task3_advanced.py                    # 심화: 과거 길이 × 모델, 처음 보는 공간
.venv/bin/python task_schedule_real.py --db data/influx_56d.db   # 실데이터 + 실제 시간표: 일정 포함/제외
```

- 입력: 과거 2시간(24칸) CO₂·온도·습도 + 시간대·요일 + 공간 + 앞으로 60분 일정 인원 → 출력: 60분 12칸을 한 번에(지금 값 대비 변화량)
- 실데이터에서 9월(전 기수, 평균 1,281ppm)과 10월(560ppm)의 사용 패턴이 크게 달라(개념 변화) 9/27 이후 데이터로 재학습
- 실제 실습실 시간표(평일 09:00~11:30 · 13:00~18:00)를 넣으면 실데이터 60분 오차 39.4 → 29.3ppm. 좋아지는 곳은 수업 시작·점심·종료 앞뒤(107 → 53ppm) — [`real56/task_schedule_real_report.md`](inference/reports/real56/task_schedule_real_report.md)

보고서: [`task3_report.md`](inference/reports/task3_report.md), [`task3_advanced_report.md`](inference/reports/task3_advanced_report.md)

## 과제 4. 이상 탐지

```bash
.venv/bin/python task4_anomaly.py                     # 고장 15건(5종)에 탐지기 조합 비교
```

규칙(튄 값·값 고정·영점 틀어짐) + 예측 오차(z > 4가 30분 연속) 조합 채택. 운영 중 오인 경보를 점검해 기준 조정. 보고서: [`task4_report.md`](inference/reports/task4_report.md)

## 과제 5. 상태 판단

```bash
.venv/bin/python task5_state.py                       # 사용 여부 분류 + 30분 안 1,000ppm 초과 선제 경보
.venv/bin/python task_own.py --db data/sensors_boardB.db   # 내 수집기 데이터로 다시 평가 (소음 센서 대리 정답)
```

보고서: [`task5_report.md`](inference/reports/task5_report.md), [`own/own_results.md`](inference/reports/own/own_results.md)

## 과제 6. 사전학습 모델

```bash
.venv/bin/python task6_pretrained.py --models amazon/chronos-bolt-tiny amazon/chronos-bolt-small
```

보고서: [`task6_report.md`](inference/reports/task6_report.md)

## 과제 7. 엣지 최적화

```bash
.venv/bin/python tools/export_onnx.py                 # 맥: PyTorch → ONNX
.venv/bin/python tools/bench.py --out reports/bench_board.json   # 보드: 워밍업 50 + 측정 500회, tegrastats 전력
.venv/bin/python task7_report.py
```

측정 조건: 전력 모드 pmode 0002, 클럭 고정 안 함. TensorRT는 sudo 없이 내 가상환경(`.venv-trt`, pip `tensorrt-cu13`)으로 설치해 GPU에서 FP32·FP16 측정 (`addenda_task7_trt.py`). 보고서: [`task7_report.md`](inference/reports/task7_report.md)

## 과제 8. 상시 추론 서비스

```bash
.venv/bin/python train_service_model.py && .venv/bin/python tools/export_onnx.py   # 서비스 모델 만들기
.venv/bin/python -m service.run --once                # 한 주기만 돌려 보기
bash deploy/install_service.sh                        # 보드: systemd 등록
curl localhost:8080/health
curl localhost:8080/spaces
curl "localhost:8080/spaces/<공간>/forecast?minutes=30"
curl "localhost:8080/alarms?hours=12"
curl "localhost:8080/log/summary?hours=12"
bash tools/gap_test.sh                                # 장애 시험: 수집기 10분 일시정지 → 끊김 경보 확인
.venv/bin/python task8_control.py                     # 심화: 예측 기반 환기 제어 시뮬레이션
```

- 매분: 5분 칸 전처리(학습과 같은 `prep.py`, 막 시작된 칸은 안 씀) → 공간별 60분 예측·"N분 뒤 1,000ppm 초과" 경보 → 이상·사용 여부·환기 판단 → `results.db` 저장
- 모델 파일이 바뀌면 재시작 없이 다시 읽음 (주기적 재학습 모델을 무중단 교체)
- 같은 경보는 1시간에 한 번 · MQTT retain 발행 + LWT (10-22 11:20 가동, 수신 확인 `inference/reports/mqtt_publish_check.txt`)
- 심화 제어: 1,000ppm 초과 시간 288 → 159분/주 (환기 시간은 1,350 → 2,410분)

보고서: [`final_report.md` §8](inference/reports/final_report.md), [`task8_control_report.md`](inference/reports/task8_control_report.md), 기록 [`service_12h_summary.json`](inference/reports/service_12h_summary.json), [`gap_test.log`](inference/reports/gap_test.log)

## 올리지 않은 것

비밀번호·토큰(`.env`), 수집 DB·내려받은 데이터(`data/`), 모델 파일(`.pt .onnx .pkl`)은 저장소에 없다(팀 규칙). 모델은 위 명령으로 다시 만들 수 있다.
