"""
train_pipeline.py
------------------
GAN 학습 전용 진입점.

실행 순서:
  Step 1 : 전처리          (Preprocess.py)
  Step 2 : 분류 - 증강 전  (Classify.py)
  Step 3 : GAN 학습        (Train.py)

학습이 끝나면 모델 가중치 + x_min/x_max + 설정(config)을 하나의 파일로 묶어
output/models/gan_final.pth 에 저장한다. augment_pipeline.py 가 이 파일을 읽어
GAN을 다시 학습하지 않고 바로 증강 단계로 넘어갈 수 있다.
"""

import sys
import os
import copy
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Preprocess import preprocess
from Train import train
from Classify import classify
from Utils import save_model_bundle, inverse_normalize


def run_train_pipeline(config: dict) -> dict:
    """
    학습 파이프라인 실행.

    Args:
        config : 파이프라인 설정 딕셔너리. DEFAULT_CONFIG 참고.

    Returns:
        {
            'preprocess'         : preprocess() 반환값
            'results_before'     : 증강 전 classify() 반환값
            'model_path'         : 저장된 모델 번들 경로
            'results_before_path': 저장된 증강 전 분류 결과 경로
            'models'              : {'generator', 'disc_ae', 'disc_cnn', 'disc_lstm'}
        }
    """

    base_dir  = config.get("base_dir", "./output")
    ckpt_dir  = os.path.join(base_dir, "checkpoints")
    model_dir = os.path.join(base_dir, "models")

    for d in [ckpt_dir, model_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("학습 파이프라인 시작")
    print("=" * 60)

    # ── Step 1: 전처리 ─────────────────────────────────────────
    print("\n[Step 1/3] 데이터 전처리")
    prep = preprocess(config)

    dataloader  = prep["dataloader"]
    dataset     = prep["dataset"]
    x_min       = prep["x_min"]
    x_max       = prep["x_max"]
    num_classes = prep["num_classes"]
    feature_dim = prep["feature_dim"]

    # augment_pipeline.py 가 동일한 전처리 결과를 재현할 수 있도록 config에 기록
    config["num_classes"]  = num_classes
    config["feature_dim"]  = feature_dim
    config["label_groups"] = prep["label_groups"]

    # dataset 은 [-1, 1] 정규화 상태 → classify() 는 원본 스케일을 기대하므로 역변환
    X_orig = inverse_normalize(dataset.data.numpy(), x_min, x_max)
    y_orig = dataset.labels.numpy()

    # ── Step 2: 분류 (증강 전) ─────────────────────────────────
    print("\n[Step 2/3] 분류 모델 학습 - 증강 전")
    results_before = classify(copy.copy(config), X_orig, y_orig)

    results_before_path = os.path.join(model_dir, "results_before.pkl")
    with open(results_before_path, "wb") as f:
        pickle.dump({k: v["metrics"] for k, v in results_before.items()}, f)
    print(f"✅ 증강 전 분류 결과 저장됨: {results_before_path}")

    # ── Step 3: GAN 학습 ───────────────────────────────────────
    print("\n[Step 3/3] GAN 학습")
    train_config             = copy.copy(config)
    train_config["save_dir"] = ckpt_dir
    generator, disc_ae, disc_cnn, disc_lstm = train(train_config, dataloader)

    model_path = os.path.join(model_dir, "gan_final.pth")
    save_model_bundle(
        generator, disc_ae, disc_cnn, disc_lstm,
        x_min, x_max, config,
        model_path,
    )

    print("\n" + "=" * 60)
    print("학습 파이프라인 완료")
    print(f"  체크포인트   : {ckpt_dir}")
    print(f"  모델 번들    : {model_path}")
    print(f"  증강 전 결과 : {results_before_path}")
    print("=" * 60)

    return {
        "preprocess"         : prep,
        "results_before"     : results_before,
        "model_path"         : model_path,
        "results_before_path": results_before_path,
        "models": {
            "generator" : generator,
            "disc_ae"   : disc_ae,
            "disc_cnn"  : disc_cnn,
            "disc_lstm" : disc_lstm,
        },
    }


# ────────────────────────────────────────────────
# 기본 설정 / 진입점
# ────────────────────────────────────────────────

if __name__ == "__main__":
    from Pipeline import DEFAULT_CONFIG

    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # 데이터 경로를 실제 경로로 수정하세요
    # cfg["data_path"]  = "path/to/features.csv"
    # cfg["label_path"] = "path/to/labels.csv"

    run_train_pipeline(cfg)
