# 시계열 데이터 추론 실습 (3.2) — suyeon040610 · team5-b

교육장 센서(CO₂·온도·습도)를 MQTT로 모아서, 60분 뒤를 예측하고 이상과 상태를 판단해, Jetson 보드에서 상시 서비스로 돌리는 과제 1~8 코드와 결과 보고서.

| 폴더 | 내용 | 보드에서 |
| --- | --- | --- |
| [`collector/`](collector/) | 과제 1: MQTT 수집기 (설정만으로 형식 대응, 자동 재접속, SQLite) | `~/ts-collector`, systemd `ts-collector` |
| [`inference/`](inference/) | 과제 2~8: 전처리·예측·이상 탐지·상태 분류·Chronos·ONNX·상시 추론 서비스 | `~/ts-inference`, systemd `ts-inference` |

- **결과 보고서**: [`inference/reports/final_report.md`](inference/reports/final_report.md) (과제별 세부: `inference/reports/task*_report.md`)
- **서비스 결과 형식**: [`inference/docs/interface.md`](inference/docs/interface.md)
- 과제 안내: `AIP-1-5/docs` 의 `assignment.md`

## 따라 하기
```bash
# 수집기 (보드)
cd collector && cp .env.example .env   # MQTT 계정은 단톡방에서 받아 .env 에만 적는다
bash deploy/install.sh                 # venv 생성 + systemd 등록
.venv/bin/python tools/status.py --hours 24 --markdown   # 수집 현황

# 학습·평가 (맥)
cd inference && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python task2_baseline.py      # task3_forecast.py … task8_control.py 순서
.venv/bin/python train_service_model.py && .venv/bin/python tools/export_onnx.py

# 상시 추론 서비스 (보드)
bash deploy/install_service.sh
curl localhost:8080/spaces/실습실/forecast?minutes=30
```

## 올리지 않은 것
비밀번호·토큰(`.env`), 수집 DB, 모델 파일(`.pt .onnx .pkl`)은 저장소에 없다. 모델은 위 명령으로 다시 만들 수 있다.
