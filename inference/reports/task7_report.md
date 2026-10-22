# 과제 7. 엣지 최적화

## 측정 조건
- 보드: Jetson Orin Nano Super 8GB (team5-b), JetPack 7 (L4T R39.2.1), 전력 모드 `pmode:0002`, 클럭 고정 안 함(기본 DVFS)
- Python 3.12.3, PyTorch 2.14.1+cpu (CPU 빌드), ONNX Runtime 1.31.0 (제공자 AzureExecutionProvider, CPUExecutionProvider)
- 반복: 워밍업 50회 후 500회, 평균·p95 · 전력: tegrastats VDD_IN 200ms 간격 평균 · 아무것도 안 할 때 6709mW
- 다른 팀원 작업(로컬 LLM 등)과 수집기·추론 서비스가 같은 보드에서 돌고 있는 실제 운영 상태에서 측정 → p95 에 다른 작업의 간섭이 섞임

## 1. 실행 방식 × 모델 (과제 3 GRU/LSTM/TCN, 입력 2시간·출력 60분)
|                              |   평균 (배치 1) ms |   p95 (배치 1) ms |   평균 (배치 4) ms |   p95 (배치 4) ms |   평균 전력 mW | PyTorch 대비 최대 CO₂ 차이 ppm   |
|:-----------------------------|---------------:|----------------:|---------------:|----------------:|-----------:|:---------------------------|
| ('GRU', 'ONNX Runtime CPU')  |           0.22 |            0.22 |           1.01 |            4.52 |       9844 | 0.000109                   |
| ('GRU', 'PyTorch CPU')       |          61.96 |           71.68 |          39.72 |           49.78 |       8601 | -                          |
| ('LSTM', 'ONNX Runtime CPU') |           0.48 |            4.26 |           0.64 |            4.32 |       9607 | 0.000076                   |
| ('LSTM', 'PyTorch CPU')      |          23.23 |           26.62 |          41.45 |           47.07 |       8573 | -                          |
| ('TCN', 'ONNX Runtime CPU')  |           0.26 |            0.15 |           0.79 |            4.41 |       9764 | 0.000087                   |
| ('TCN', 'PyTorch CPU')       |           3.44 |            4.25 |           8.53 |           10.23 |       8707 | -                          |

## 2. 사전학습 모델 비용 (과제 6, Chronos-Bolt tiny, 과거 24시간)
| model                       | runtime     |   batch |   mean_ms |   p95_ms |   power_mw_mean |
|:----------------------------|:------------|--------:|----------:|---------:|----------------:|
| chronos-bolt-tiny (ctx 24h) | PyTorch CPU |       1 |      91.9 |    106   |          8614   |
| chronos-bolt-tiny (ctx 24h) | PyTorch CPU |       4 |      88.6 |     99.1 |          8619.6 |
- 메모리 증가: 약 68MB (모델 로드 후 최대 RSS 증가분)

## 3. TensorRT (FP16)
> ⏳ 보드에 CUDA 툴킷·TensorRT(JetPack 구성요소)가 설치되어 있지 않아 아직 측정하지 못함. 팀 공용 보드라 시스템 설치는 팀 확인 후 진행 예정.
> 설치되면 `tools/build_trt.sh`(ONNX → TensorRT FP16 엔진)와 `tools/bench.py` 로 같은 표에 'TensorRT FP16' 행과 변환 오차를 채운다.

## 4. 해석 (CPU 결과 기준)
- **ONNX Runtime 이 PyTorch 보다 수십~수백 배 빠르다** (GRU 배치 1: PyTorch CPU 약 60ms → ONNX Runtime 약 0.2ms). PyTorch 는 연산 하나하나를 파이썬에서 호출하는 비용과 스레드 관리 비용이 작은 모델에서는 계산보다 훨씬 크다. ONNX Runtime 은 그래프 전체를 한 번에 최적화(연산 합치기)해 이 비용이 거의 없다.
- **변환해도 결과가 같다**: ONNX 출력과 PyTorch 출력의 CO₂ 차이가 0.0001ppm 수준 → 기준(1ppm 이하) 충족.
- **평균보다 p95 가 중요**: ONNX Runtime 평균은 1ms 미만이지만 p95 는 수 ms 로 튄다. 같은 보드의 다른 작업(LLM, 수집기)이 CPU 를 잡으면 가끔 늦어진다. 상시 서비스는 '가끔 늦는 순간'에 마감(매분)을 놓치므로 꼬리 지연을 봐야 한다. 그래도 수 ms 는 매분 주기에 비해 충분히 작다.
- **RNN(GRU/LSTM) vs TCN**: PyTorch 에서는 TCN 이 훨씬 빠르다(합성곱은 시간축을 한 번에 병렬 계산, RNN 은 24칸을 순서대로). ONNX Runtime 에서는 둘 다 1ms 미만이라 차이가 작다. GPU/TensorRT 에서는 병렬화가 쉬운 TCN 이 더 크게 빨라질 것으로 예상한다.
- **전력**: 추론 중 약 8.5~10W, 쉴 때 약 6.7W. 모델이 1ms 안에 끝나므로 매분 한 번 돌리는 서비스의 추가 전력은 무시할 만하다.
- **Chronos tiny**: 한 번에 약 90ms, 직접 학습한 모델(ONNX 0.2ms)의 수백 배. 같은 보드에 로컬 LLM(수 GB)을 올려야 하므로 상시 기본 모델은 작은 GRU/TCN + ONNX Runtime 이 맞다.
- **CPU 로도 충분한데 왜 변환하나?** 지금은 공간 5개·매분이라 CPU 로 충분하지만, 같은 보드에 임베딩 모델·LLM 이 함께 돌면 CPU 가 부족해지고, 공간이 늘어나거나 매초 추론이 필요해지면 GPU(TensorRT)로 옮겨 CPU 를 비워 두는 것이 전체 시스템에 유리하다.
