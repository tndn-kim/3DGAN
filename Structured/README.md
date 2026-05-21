# Structured — GAN 기반 데이터 증강 파이프라인

본 모듈은 3D GAN을 활용한 네트워크 트래픽 데이터 증강 파이프라인입니다.  
전처리 → GAN 학습 → 데이터 증강 → 분류 성능 비교 → 시각화까지 단일 진입점(`pipeline.py`)으로 실행할 수 있습니다.

> 기반 논문: [Intrusion Detection Using 3DGAN](https://www.mdpi.com/3793174)

---

## 폴더 구조

```
Structured/
├── pipeline.py        # 전체 파이프라인 진입점 (여기서 실행)
├── Preprocess.py      # 데이터 전처리 (NaN / 이상치 / 정규화 / DataLoader)
├── GAN.py             # 모델 아키텍처 (Generator + Discriminator x3)
├── Train.py           # GAN 학습 루프
├── Augmentation.py    # 학습된 Generator로 클래스별 데이터 생성
├── Classify.py        # 분류 모델 학습·평가 (RF / XGBoost / BiGRU)
├── Visualization.py   # 증강 전/후 성능 비교 그래프 생성
└── utils.py           # 체크포인트 저장/복원, 스케줄러, LR 조회
```

---

## 실행 환경

```
Python >= 3.8
torch
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
```

```bash
pip install torch numpy pandas scikit-learn xgboost matplotlib seaborn
```

---

## 빠른 시작

### 1. 데이터 준비

CSV 파일 2개가 필요합니다.

| 파일 | 내용 |
|---|---|
| `features.csv` | 피처 행렬 (헤더 포함, 레이블 열 없음) |
| `labels.csv`   | 레이블 열 1개 (컬럼명 기본값 `label`) |

두 파일의 행 순서가 일치해야 합니다.

### 2. 경로 및 설정 수정

`pipeline.py` 하단의 `DEFAULT_CONFIG`에서 경로를 수정합니다.

```python
DEFAULT_CONFIG = {
    "base_dir"   : "./output",           # 출력 루트 폴더
    "data_path"  : "./data/features.csv",
    "label_path" : "./data/labels.csv",
    "label_col"  : "label",
    ...
}
```

### 3. 실행

```bash
cd Structured
python pipeline.py
```

---

## 파이프라인 흐름

```
[Step 1] 전처리 (Preprocess.py)
         CSV 로드 → NaN 처리 → 이상치 처리 → 레이블 축소 → 정규화 [-1,1]
              ↓
[Step 2] 분류 - 증강 전 (Classify.py)
         RF / XGBoost / BiGRU 학습·평가 → results_before
              ↓
[Step 3] GAN 학습 (Train.py)
         Generator + DiscriminatorAE / CNN / LSTM 동시 학습
         epoch 마다 체크포인트 저장 → output/checkpoints/
              ↓
[Step 4] 데이터 증강 (Augmentation.py)
         클래스별 부족 샘플을 Generator로 생성 → 투표 필터 통과 시 저장
         결과 → output/augmented/aug_label{N}.npy
              ↓
[Step 5] 분류 - 증강 후 (Classify.py)
         원본 + 증강 데이터 병합 → RF / XGBoost / BiGRU 재학습·평가 → results_after
              ↓
[Step 6] 시각화 (Visualization.py)
         증강 전/후 성능 비교 그래프 3종 → output/visualization/
```

### 출력 폴더

```
output/
├── checkpoints/
│   ├── checkpoint_1.pth
│   ├── checkpoint_2.pth
│   └── ...
├── augmented/
│   ├── aug_label0.npy
│   ├── aug_label1.npy
│   └── ...
└── visualization/
    ├── metric_comparison.png   # Accuracy / F1 / FNR 막대 비교
    ├── confusion_matrices.png  # 증강 전/후 Confusion Matrix
    └── summary_radar.png       # 종합 성능 레이더 차트
```

---

## 모듈별 설명

### `Preprocess.py`

| 함수 | 역할 |
|---|---|
| `preprocess(config)` | 전체 전처리 실행. DataLoader + 정규화 파라미터 반환 |

**설정 키**

| 키 | 기본값 | 설명 |
|---|---|---|
| `nan_method` | `"mean"` | NaN 대체 방식: `mean` / `median` / `mode` |
| `outlier_action` | `"clip"` | 이상치 처리: `drop` / `clip` / `none` |
| `outlier_method` | `"iqr"` | 이상치 감지: `iqr` / `zscore` / `both` |
| `label_groups` | `None` | 레이블 병합 그룹 ex) `[[0,1],[2]]`. `None` 이면 런타임 질문 |
| `batch_size` | `64` | DataLoader 배치 크기 |
| `num_workers` | `0` | Windows에서 `0` 권장 |

---

### `GAN.py`

GAN 모델 아키텍처를 정의합니다.

| 클래스 | 설명 |
|---|---|
| `Generator` | Conditional Generator. `z + label_embedding` → 샘플 생성 |
| `DiscriminatorAE` | 오토인코더 기반 판별기 (BEGAN). 재구성 오차로 real/fake 판별 |
| `DiscriminatorCNN` | 1D-CNN 기반 판별기. Spectral Normalization 적용 |
| `DiscriminatorLSTM` | Bidirectional LSTM 기반 판별기 |
| `build_models(...)` | 4개 모델을 일괄 생성하는 팩토리 함수 |

---

### `Train.py`

| 함수 | 역할 |
|---|---|
| `train(config, dataloader)` | GAN 학습 루프 실행. 학습 완료된 모델 4개 반환 |

**설정 키**

| 키 | 기본값 | 설명 |
|---|---|---|
| `n_epochs` | `200` | 학습 epoch 수 |
| `latent_dim` | `100` | 노이즈 벡터 차원 |
| `lr` | `0.0002` | 학습률 |
| `b1` / `b2` | `0.5` / `0.999` | Adam optimizer β 파라미터 |
| `gamma` | `0.5` | BEGAN 균형 계수 |
| `lambda_k` | `0.001` | BEGAN k 업데이트 속도 |
| `resume_epoch` | `-1` | 이어 학습할 epoch 번호. `-1` 이면 처음부터 |

---

### `Augmentation.py`

| 함수 | 역할 |
|---|---|
| `augment(config, generator, disc_ae, disc_cnn, disc_lstm, x_min, x_max)` | 클래스별 증강 샘플 생성. 3-Discriminator 투표 필터 통과 시 저장 |

**설정 키**

| 키 | 기본값 | 설명 |
|---|---|---|
| `target_counts` | 자동 균등화 | `{레이블: 목표샘플수}`. `None` 이면 최대 클래스 기준 균등화 |
| `ae_threshold` | `0.3` | AE 재구성 오차 통과 기준 (낮을수록 real) |
| `disc_threshold` | `0.5` | CNN / LSTM 통과 기준 확률 |
| `min_votes` | `2` | 최소 통과 투표 수 (3개 중) |
| `aug_batch_size` | `64` | 한 번에 생성할 샘플 수 |

---

### `Classify.py`

| 함수 | 역할 |
|---|---|
| `classify(config, X, y)` | RF / XGBoost / BiGRU 학습·평가. 각 모델의 지표와 객체 반환 |

입력 `X`는 **원본 스케일** 데이터여야 합니다. 내부에서 `MinMaxScaler`로 재정규화합니다.

**설정 키**

| 키 | 기본값 | 설명 |
|---|---|---|
| `models` | `["rf","xgb","bigru"]` | 실행할 모델 선택 |
| `test_size` | `0.3` | Train / Test 분할 비율 |
| `max_per_class` | `None` | 클래스별 최대 샘플 수 제한 |
| `gru_epochs` | `10` | BiGRU 학습 epoch 수 |

---

### `Visualization.py`

| 함수 | 역할 |
|---|---|
| `visualize(config, results_before, results_after)` | 증강 전/후 결과 비교 그래프 3종 저장 |

생성 파일:
- `metric_comparison.png` — Accuracy / F1 / FNR 막대 비교
- `confusion_matrices.png` — 모델별 Confusion Matrix 히트맵
- `summary_radar.png` — 종합 성능 레이더 차트

---

### `utils.py`

Train.py 내부에서 사용하는 공통 유틸입니다.

| 함수 | 역할 |
|---|---|
| `save_checkpoint(...)` | 모델·옵티마이저 상태를 `.pth`로 저장 |
| `load_checkpoint(...)` | 체크포인트에서 상태 복원 |
| `build_schedulers(...)` | 4개 옵티마이저에 `ReduceLROnPlateau` 일괄 적용 |
| `get_lr(optimizer)` | 현재 학습률 조회 |

---

## 이어 학습 (Resume)

학습이 중단된 경우 `resume_epoch`를 설정하면 해당 epoch의 체크포인트부터 이어 학습합니다.

```python
cfg["resume_epoch"] = 50    # output/checkpoints/checkpoint_50.pth 에서 복원
```

---

## 일부 단계만 실행

`pipeline.py`를 직접 수정하지 않고 각 모듈 함수를 개별 호출할 수 있습니다.

```python
import sys
sys.path.insert(0, "./Structured")

from Preprocess import preprocess
from Classify import classify

config = { "data_path": "...", "label_path": "...", ... }
prep = preprocess(config)

X = prep["dataset"].data.numpy()
y = prep["dataset"].labels.numpy()
results = classify(config, X, y)
```

---

## 주의사항

- **Windows 환경**: `num_workers`를 `0`으로 설정하세요. 양수로 설정 시 DataLoader 다중 프로세스 오류가 발생할 수 있습니다.
- **GPU 사용**: CUDA가 감지되면 자동으로 GPU를 사용합니다. 별도 설정이 필요 없습니다.
- **레이블**: 반드시 `0`부터 시작하는 연속 정수여야 합니다. `label_groups`로 병합하여 맞춰주세요.
