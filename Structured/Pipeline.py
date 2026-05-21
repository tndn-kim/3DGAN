"""
pipeline.py
-----------
전체 파이프라인 진입점.

실행 순서:
  Step 1 : 전처리          (Preprocess.py)
  Step 2 : 분류 - 증강 전  (Classify.py)
  Step 3 : GAN 학습        (Train.py)
  Step 4 : 데이터 증강     (Augmentation.py)
  Step 5 : 분류 - 증강 후  (Classify.py)
  Step 6 : 시각화          (Visualization.py)

데이터 스케일 흐름:
  preprocess()  → 정규화 [-1, 1]    (DataLoader 학습용)
  augment()     → 역정규화 후 저장  (원본 스케일)
  classify()    → 내부에서 재정규화 → 원본 스케일 데이터를 받아야 함
  따라서 pipeline 내에서 dataset 데이터를 역정규화 후 classify()에 전달.
"""

import sys
import os
import copy
import numpy as np
from collections import Counter

# Structured 폴더 내 모듈 간 임포트 (Train.py → GAN.py, utils.py) 동작을 위해 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Preprocess import preprocess
from Train import train
from Augmentation import augment
from Classify import classify
from Visualization import visualize


# ────────────────────────────────────────────────
# 내부 유틸
# ────────────────────────────────────────────────

def _inverse_normalize(X_norm: np.ndarray, x_min, x_max) -> np.ndarray:
    """
    Preprocess.py 의 Min-Max 정규화 역변환.
    [-1, 1] → 원본 스케일.
    augment() 가 원본 스케일을 반환하므로 dataset 데이터를 동일 스케일로 맞출 때 사용.
    """
    x_min_arr = x_min.values.reshape(1, -1)
    x_max_arr = x_max.values.reshape(1, -1)
    return ((X_norm + 1) / 2) * (x_max_arr - x_min_arr + 1e-8) + x_min_arr


# ────────────────────────────────────────────────
# 메인 파이프라인
# ────────────────────────────────────────────────

def run_pipeline(config: dict) -> dict:
    """
    전체 파이프라인 실행.

    Args:
        config : 파이프라인 설정 딕셔너리. DEFAULT_CONFIG 참고.

    Returns:
        {
            'preprocess'    : preprocess() 반환값
            'results_before': 증강 전 classify() 반환값
            'results_after' : 증강 후 classify() 반환값
            'augmented'     : augment() 반환값 {label: np.ndarray}
            'models'        : {'generator', 'disc_ae', 'disc_cnn', 'disc_lstm'}
        }
    """

    base_dir = config.get("base_dir", "./output")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    aug_dir  = os.path.join(base_dir, "augmented")
    viz_dir  = os.path.join(base_dir, "visualization")

    for d in [ckpt_dir, aug_dir, viz_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("파이프라인 시작")
    print("=" * 60)

    # ── Step 1: 전처리 ─────────────────────────────────────────
    print("\n[Step 1/6] 데이터 전처리")
    prep = preprocess(config)

    dataloader  = prep["dataloader"]
    dataset     = prep["dataset"]
    x_min       = prep["x_min"]
    x_max       = prep["x_max"]
    num_classes = prep["num_classes"]
    feature_dim = prep["feature_dim"]

    # 후속 단계에 필요한 값을 config에 주입
    config["num_classes"] = num_classes
    config["feature_dim"] = feature_dim

    # dataset 은 [-1, 1] 정규화 상태 → classify() 는 원본 스케일을 기대하므로 역변환
    X_orig = _inverse_normalize(dataset.data.numpy(), x_min, x_max)
    y_orig = dataset.labels.numpy()

    # ── Step 2: 분류 (증강 전) ─────────────────────────────────
    print("\n[Step 2/6] 분류 모델 학습 - 증강 전")
    results_before = classify(copy.copy(config), X_orig, y_orig)

    # ── Step 3: GAN 학습 ───────────────────────────────────────
    print("\n[Step 3/6] GAN 학습")
    train_config             = copy.copy(config)
    train_config["save_dir"] = ckpt_dir
    generator, disc_ae, disc_cnn, disc_lstm = train(train_config, dataloader)

    # ── Step 4: 데이터 증강 ────────────────────────────────────
    print("\n[Step 4/6] 데이터 증강")
    label_counter = Counter(y_orig.tolist())

    aug_config                   = copy.copy(config)
    aug_config["save_dir"]       = aug_dir
    aug_config["current_counts"] = {int(k): v for k, v in label_counter.items()}

    # target_counts 미설정 시 가장 많은 클래스 수에 맞춰 균등화
    if not aug_config.get("target_counts"):
        max_count = max(label_counter.values())
        aug_config["target_counts"] = {
            int(lbl): max_count for lbl in label_counter
        }

    augmented = augment(
        aug_config,
        generator, disc_ae, disc_cnn, disc_lstm,
        x_min, x_max,
    )

    # ── Step 5: 원본 + 증강 데이터 병합 후 분류 ───────────────
    print("\n[Step 5/6] 분류 모델 학습 - 증강 후")
    X_list = [X_orig]
    y_list = [y_orig]

    for label, arr in augmented.items():
        # augment() 는 목표 달성 시 np.empty((0,)) 를 반환 → ndim/size 체크
        if arr is not None and arr.ndim == 2 and arr.size > 0:
            X_list.append(arr)
            y_list.append(np.full(len(arr), int(label), dtype=np.int64))

    X_aug = np.concatenate(X_list, axis=0)
    y_aug = np.concatenate(y_list, axis=0)

    print(f"  병합 완료: {len(X_orig):,}개 → {len(X_aug):,}개")
    for cls in np.unique(y_aug):
        print(f"  Label {int(cls)}: {(y_aug == cls).sum():,}개")

    results_after = classify(copy.copy(config), X_aug, y_aug)

    # ── Step 6: 시각화 ────────────────────────────────────────
    print("\n[Step 6/6] 결과 시각화")
    viz_config             = copy.copy(config)
    viz_config["save_dir"] = viz_dir
    visualize(viz_config, results_before, results_after)

    print("\n" + "=" * 60)
    print("파이프라인 완료")
    print(f"  체크포인트  : {ckpt_dir}")
    print(f"  증강 데이터 : {aug_dir}")
    print(f"  시각화 결과 : {viz_dir}")
    print("=" * 60)

    return {
        "preprocess"    : prep,
        "results_before": results_before,
        "results_after" : results_after,
        "augmented"     : augmented,
        "models": {
            "generator" : generator,
            "disc_ae"   : disc_ae,
            "disc_cnn"  : disc_cnn,
            "disc_lstm" : disc_lstm,
        },
    }


# ────────────────────────────────────────────────
# 기본 설정
# ────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # ── 경로 ──────────────────────────────────────────────────
    "base_dir"        : "./output",          # 체크포인트 / 증강 / 시각화 루트
    "data_path"       : "./data/features.csv",
    "label_path"      : "./data/labels.csv",
    "label_col"       : "label",

    # ── 전처리 ────────────────────────────────────────────────
    "label_groups"    : None,               # None → 런타임에 질문
    "nan_method"      : "mean",             # 'mean' | 'median' | 'mode'
    "outlier_method"  : "iqr",              # 'iqr' | 'zscore' | 'both'
    "outlier_action"  : "clip",             # 'drop' | 'clip' | 'none'
    "batch_size"      : 64,
    "num_workers"     : 0,                  # Windows 에서 0 권장

    # ── GAN 학습 ──────────────────────────────────────────────
    "n_epochs"        : 200,
    "latent_dim"      : 100,
    "lr"              : 0.0002,
    "b1"              : 0.5,
    "b2"              : 0.999,
    "emb_dim"         : 32,
    "gamma"           : 0.5,
    "lambda_k"        : 0.001,
    "lr_factor"       : 0.5,
    "lr_patience"     : 5,
    "resume_epoch"    : -1,                 # -1 → 체크포인트 없음 (처음부터)

    # ── 증강 ──────────────────────────────────────────────────
    "target_counts"   : None,               # None → 자동 균등화 (최대 클래스 수 기준)
    "ae_threshold"    : 0.3,
    "disc_threshold"  : 0.5,
    "min_votes"       : 2,
    "aug_batch_size"  : 64,
    "log_interval"    : 500,

    # ── 분류 ──────────────────────────────────────────────────
    "test_size"       : 0.3,
    "random_state"    : 42,
    "max_per_class"   : None,               # None → 제한 없음
    "models"          : ["rf", "xgb", "bigru"],
    "rf_n_estimators" : 100,
    "xgb_n_estimators": 200,
    "xgb_lr"          : 0.1,
    "xgb_max_depth"   : 6,
    "gru_hidden"      : 64,
    "gru_layers"      : 2,
    "gru_epochs"      : 10,
    "gru_lr"          : 0.001,
    "gru_dropout"     : 0.2,
    "clf_batch_size"  : 64,
}


# ────────────────────────────────────────────────
# 진입점
# ────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # 데이터 경로를 실제 경로로 수정하세요
    # cfg["data_path"]  = "path/to/features.csv"
    # cfg["label_path"] = "path/to/labels.csv"

    run_pipeline(cfg)
