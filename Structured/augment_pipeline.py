"""
augment_pipeline.py
--------------------
GAN 증강 전용 진입점.

train_pipeline.py 가 저장한 모델 번들(output/models/gan_final.pth)을 읽어
GAN을 다시 학습하지 않고 곧바로 아래 단계를 실행한다.

실행 순서:
  Step 1 : 데이터 증강     (Augmentation.py)
  Step 2 : 분류 - 증강 후 (Classify.py)
  Step 3 : 시각화          (Visualization.py)
"""

import sys
import os
import copy
import pickle
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Preprocess import preprocess
from Augmentation import augment
from Classify import classify
from Visualization import visualize
from Utils import load_model_bundle, inverse_normalize


def run_augment_pipeline(config: dict,
                         model_path: str,
                         results_before_path: str = None) -> dict:
    """
    증강 파이프라인 실행.

    Args:
        config              : base_dir 등 출력 경로 / classify·visualize 설정.
                               data_path 등은 model_path 번들에 저장된 학습 시
                               설정값이 우선 적용된다.
        model_path           : train_pipeline.py 가 저장한 모델 번들(.pth) 경로
        results_before_path : train_pipeline.py 가 저장한 증강 전 분류 결과(.pkl) 경로
                               None 이면 base_dir/models/results_before.pkl 사용

    Returns:
        {
            'augmented'     : augment() 반환값 {label: np.ndarray}
            'results_after' : 증강 후 classify() 반환값
            'results_before': 증강 전 분류 결과 (metrics만 포함)
        }
    """

    base_dir  = config.get("base_dir", "./output")
    aug_dir   = os.path.join(base_dir, "augmented")
    viz_dir   = os.path.join(base_dir, "visualization")
    model_dir = os.path.join(base_dir, "models")

    for d in [aug_dir, viz_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("증강 파이프라인 시작")
    print("=" * 60)

    # ── Step 0: 모델 번들 로드 ──────────────────────────────────
    print(f"\n[모델 로드] {model_path}")
    generator, disc_ae, disc_cnn, disc_lstm, x_min, x_max, train_config = \
        load_model_bundle(model_path)

    # 데이터/모델 정체성과 직결된 키는 학습 시 값을 강제 (재현성 보장).
    # ae_threshold / target_counts / classify 설정 등 나머지는 호출 시 config가 우선
    # → 재학습 없이 증강·분류 파라미터만 바꿔서 재실행 가능.
    identity_keys = [
        "data_path", "label_path", "label_col",
        "nan_method", "outlier_method", "outlier_action", "label_groups",
        "feature_dim", "num_classes", "latent_dim", "emb_dim",
        "batch_size", "num_workers",
    ]
    merged_config = copy.copy(config)
    for k in identity_keys:
        if k in train_config:
            merged_config[k] = train_config[k]

    # 원본 데이터 재구성 (학습 때와 동일한 data_path / label_groups 사용)
    print("\n[원본 데이터 로드] 학습 시 설정으로 전처리 재실행")
    prep = preprocess(merged_config)
    dataset = prep["dataset"]
    X_orig = inverse_normalize(dataset.data.numpy(), x_min, x_max)
    y_orig = dataset.labels.numpy()

    # ── Step 1: 데이터 증강 ────────────────────────────────────
    print("\n[Step 1/3] 데이터 증강")
    label_counter = Counter(y_orig.tolist())

    aug_config                   = copy.copy(merged_config)
    aug_config["save_dir"]       = aug_dir
    aug_config["current_counts"] = {int(k): v for k, v in label_counter.items()}

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

    # ── Step 2: 원본 + 증강 데이터 병합 후 분류 ───────────────
    print("\n[Step 2/3] 분류 모델 학습 - 증강 후")
    X_list = [X_orig]
    y_list = [y_orig]

    for label, arr in augmented.items():
        if arr is not None and arr.ndim == 2 and arr.size > 0:
            X_list.append(arr)
            y_list.append(np.full(len(arr), int(label), dtype=np.int64))

    X_aug = np.concatenate(X_list, axis=0)
    y_aug = np.concatenate(y_list, axis=0)

    print(f"  병합 완료: {len(X_orig):,}개 → {len(X_aug):,}개")
    for cls in np.unique(y_aug):
        print(f"  Label {int(cls)}: {(y_aug == cls).sum():,}개")

    results_after = classify(copy.copy(merged_config), X_aug, y_aug)

    # ── Step 3: 시각화 ────────────────────────────────────────
    print("\n[Step 3/3] 결과 시각화")
    results_before_path = results_before_path or os.path.join(model_dir, "results_before.pkl")
    with open(results_before_path, "rb") as f:
        results_before_metrics = pickle.load(f)
    # visualize() 는 {'metrics': {...}} 형태를 기대하므로 동일하게 래핑
    results_before = {k: {"metrics": v} for k, v in results_before_metrics.items()}

    viz_config             = copy.copy(merged_config)
    viz_config["save_dir"] = viz_dir
    visualize(viz_config, results_before, results_after)

    print("\n" + "=" * 60)
    print("증강 파이프라인 완료")
    print(f"  증강 데이터 : {aug_dir}")
    print(f"  시각화 결과 : {viz_dir}")
    print("=" * 60)

    return {
        "augmented"     : augmented,
        "results_after" : results_after,
        "results_before": results_before,
    }


# ────────────────────────────────────────────────
# 진입점
# ────────────────────────────────────────────────

if __name__ == "__main__":
    from Pipeline import DEFAULT_CONFIG

    cfg = copy.deepcopy(DEFAULT_CONFIG)

    model_file = os.path.join(cfg.get("base_dir", "./output"), "models", "gan_final.pth")
    run_augment_pipeline(cfg, model_path=model_file)
