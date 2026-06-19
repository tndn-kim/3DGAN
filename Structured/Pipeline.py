"""
pipeline.py
-----------
전체 파이프라인 진입점 (학습 + 증강을 한 번에 실행하고 싶을 때 사용).

내부적으로 train_pipeline.run_train_pipeline() → augment_pipeline.run_augment_pipeline()
을 순서대로 호출한다. GAN 학습은 시간이 오래 걸리고, 증강 파라미터
(ae_threshold, target_counts 등)는 재학습 없이 여러 번 바꿔보는 경우가 많으므로,
실제 실험에서는 두 파이프라인을 따로 실행하는 것을 권장한다.

  - 학습만:  python train_pipeline.py     → output/models/gan_final.pth 저장
  - 증강만:  python augment_pipeline.py   → 저장된 gan_final.pth 를 읽어 증강 진행

데이터 스케일 흐름:
  preprocess()  → 정규화 [-1, 1]    (DataLoader 학습용)
  augment()     → 역정규화 후 저장  (원본 스케일)
  classify()    → 내부에서 재정규화 → 원본 스케일 데이터를 받아야 함
"""

import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_pipeline import run_train_pipeline
from augment_pipeline import run_augment_pipeline


# ────────────────────────────────────────────────
# 메인 파이프라인
# ────────────────────────────────────────────────

def run_pipeline(config: dict) -> dict:
    """
    학습 + 증강 파이프라인을 순서대로 실행.

    Args:
        config : 파이프라인 설정 딕셔너리. DEFAULT_CONFIG 참고.

    Returns:
        {
            'preprocess'    : preprocess() 반환값
            'results_before': 증강 전 classify() 반환값
            'results_after' : 증강 후 classify() 반환값
            'augmented'     : augment() 반환값 {label: np.ndarray}
            'model_path'    : 저장된 모델 번들 경로
            'models'        : {'generator', 'disc_ae', 'disc_cnn', 'disc_lstm'}
        }
    """
    train_result = run_train_pipeline(config)

    augment_result = run_augment_pipeline(
        config,
        model_path           = train_result["model_path"],
        results_before_path  = train_result["results_before_path"],
    )

    return {
        "preprocess"    : train_result["preprocess"],
        "results_before": train_result["results_before"],
        "results_after" : augment_result["results_after"],
        "augmented"     : augment_result["augmented"],
        "model_path"    : train_result["model_path"],
        "models"        : train_result["models"],
    }


# ────────────────────────────────────────────────
# 기본 설정
# ────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # ── 경로 ──────────────────────────────────────────────────
    "base_dir"        : "./output",          # 체크포인트 / 모델 / 증강 / 시각화 루트
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
