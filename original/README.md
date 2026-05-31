
# original — 초기 프로토타입 코드

본 폴더는 3D GAN 실험에 가장 먼저 작성·적용한 **원본(prototype) 코드**입니다.  
이후 모듈화·버그 수정된 버전은 `../Structured/` 폴더를 참조하세요.

---

## 파일 목록

```
original/
├── 앙상블.py   # 원본 단일 스크립트 (모델 정의 + 학습 루프 통합)
└── Readme.md
```

---

## 앙상블.py 개요

argparse 기반 단일 스크립트로, GAN 모델 4종의 정의와 학습 흐름을 하나의 파일에 담은 초기 구현입니다.

### 구성 모델

| 클래스 | 역할 |
|---|---|
| `Generator` | Conditional Generator. 노이즈 `z`에 레이블 임베딩을 **덧셈**으로 조건 부여 |
| `Discriminator_autoencoder` | 오토인코더 기반 판별기. 재구성 MSE를 real/fake 점수로 사용 |
| `Discriminator_LSTM` | Bidirectional LSTM 기반 판별기. 입력 `[B, 76]` → unsqueeze → `[B, 1, 76]` |
| `Discriminator_CNN` | 1D-CNN 기반 판별기. `Conv1d` 2층 + Sigmoid 출력 |

### 실행 인자 (argparse)

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--n_epochs` | `200` | 학습 epoch 수 |
| `--batch_size` | `64` | 배치 크기 |
| `--lr` | `0.0002` | Adam 학습률 |
| `--b1` | `0.5` | Adam β₁ |
| `--b2` | `0.999` | Adam β₂ |
| `--n_cpu` | `8` | DataLoader worker 수 |
| `--latent_dim` | `100` | 노이즈 벡터 차원 |
| `--label_type` | `0` | 처리할 레이블 인덱스 |

### 실행 방법

```bash
cd original
python 앙상블.py --n_epochs 200 --label_type 0
```

---

## 원본 코드와 Structured 버전 차이점

| 항목 | original/앙상블.py | Structured/ |
|---|---|---|
| 구조 | 단일 스크립트 | 모듈 분리 (6개 파일 + pipeline.py) |
| 설정 방식 | argparse CLI 인자 | config 딕셔너리 |
| 레이블 조건 부여 | `z = z + label_emb(labels)` (덧셈) | `proj(cat([z, emb]))` (concat + projection) |
| Spectral Normalization | 미적용 | Discriminator 전체 적용 |
| BEGAN k 업데이트 | 원본 shape 불일치 버그 존재 | 수정됨 |
| 역정규화 저장 | 정규화 값 저장 버그 존재 | 역정규화된 원본 스케일로 저장 |
| 전처리 | 하드코딩 | NaN / 이상치 / 레이블 축소 파이프라인 |
| 분류 평가 | 미포함 | RF / XGBoost / BiGRU 비교 |
| 시각화 | 미포함 | 증강 전/후 성능 비교 그래프 3종 |
| 체크포인트 | 미포함 | epoch 단위 저장 / 복원 |

---

## 참고

- 개선된 전체 파이프라인 실행: `../Structured/pipeline.py`
- 기반 논문: [Intrusion Detection Using 3DGAN](https://www.mdpi.com/3793174)
=======


## 가장 처음 작성하고 완성하여 실험 적용한 모델

## CIC UNSW NB15를 사용한 결과 대시보드 시각화

