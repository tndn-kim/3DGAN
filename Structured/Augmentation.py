"""
augmentation.py
---------------
GAN 기반 데이터 증강 모듈.

원본 코드 개선사항:
  1. gen_samples_inverse 계산 후 gen_samples(정규화값) 저장하던 버그 수정
     → 역정규화된 값을 저장
  2. x_min / x_max CSV 재로드 제거 → preprocess.py 에서 전달받음
  3. label_type별 하드코딩된 num_generate 제거
     → config["target_counts"] 딕셔너리로 일반화
  4. 샘플 1개씩 생성 → 배치 단위 생성으로 효율화
"""

import os
import numpy as np
import pandas as pd
import torch


# ────────────────────────────────────────────────
# 내부 유틸
# ────────────────────────────────────────────────

def _inverse_normalize(samples: torch.Tensor,
                       x_min: pd.Series,
                       x_max: pd.Series) -> np.ndarray:
    """
    Min-Max 정규화 역변환 [-1,1] → 원본 스케일.
    preprocess.py 의 정규화와 대칭.
    """
    x_min_t = torch.tensor(x_min.values, dtype=torch.float32, device=samples.device)
    x_max_t = torch.tensor(x_max.values, dtype=torch.float32, device=samples.device)
    restored = ((samples + 1) / 2) * (x_max_t - x_min_t + 1e-8) + x_min_t
    return restored.cpu().numpy()


def _vote(val_ae: float, val_cnn: float, val_lstm: float,
          ae_threshold: float, disc_threshold: float,
          min_votes: int) -> bool:
    """
    3개 Discriminator 투표.
    AE  : 재구성 오차가 낮을수록 real → threshold 미만이면 통과
    CNN : sigmoid 출력 → threshold 초과이면 통과
    LSTM: sigmoid 출력 → threshold 초과이면 통과
    """
    votes = 0
    if val_ae   < ae_threshold:    votes += 1
    if val_cnn  > disc_threshold:  votes += 1
    if val_lstm > disc_threshold:  votes += 1
    return votes >= min_votes


# ────────────────────────────────────────────────
# 메인 증강 함수
# ────────────────────────────────────────────────

def augment(config: dict,
            generator,
            disc_ae,
            disc_cnn,
            disc_lstm,
            x_min: pd.Series,
            x_max: pd.Series) -> dict:
    """
    학습된 Generator로 클래스별 데이터 증강 수행.

    Args:
        config:
            - latent_dim      : 노이즈 벡터 차원
            - target_counts   : 클래스별 목표 샘플 수 딕셔너리
                                ex) {0: 50000, 1: 50000, 2: 50000}
            - current_counts  : 클래스별 현재 샘플 수 딕셔너리
                                ex) {0: 17120, 1: 38383, 2: 34080}
            - save_dir        : 증강 결과 저장 경로
            - ae_threshold    : AE 통과 기준 재구성 오차 (default 0.3)
            - disc_threshold  : CNN/LSTM 통과 기준 확률 (default 0.5)
            - min_votes       : 통과 기준 최소 투표 수 (default 2)
            - batch_size      : 한 번에 생성할 샘플 수 (default 64)
            - log_interval    : 진행 로그 출력 간격 (default 500)
        generator  : 학습 완료된 Generator
        disc_ae    : 학습 완료된 DiscriminatorAE
        disc_cnn   : 학습 완료된 DiscriminatorCNN
        disc_lstm  : 학습 완료된 DiscriminatorLSTM
        x_min      : preprocess.py 에서 전달받은 정규화 기준 최솟값
        x_max      : preprocess.py 에서 전달받은 정규화 기준 최댓값

    Returns:
        dict: {label(int): np.ndarray [N, feature_dim]}  역정규화된 증강 샘플
    """

    # ── 설정값 파싱 ────────────────────────────────
    cuda           = torch.cuda.is_available()
    device         = torch.device("cuda" if cuda else "cpu")
    Tensor         = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    latent_dim     = config["latent_dim"]
    target_counts  = config["target_counts"]   # {label: target_n}
    current_counts = config["current_counts"]  # {label: current_n}
    save_dir       = config["save_dir"]
    ae_threshold   = config.get("ae_threshold",   0.3)
    disc_threshold = config.get("disc_threshold", 0.5)
    min_votes      = config.get("min_votes",      2)
    batch_size     = config.get("aug_batch_size", 64)
    log_interval   = config.get("log_interval",   500)

    os.makedirs(save_dir, exist_ok=True)

    # ── 모델 eval 모드 ─────────────────────────────
    for m in [generator, disc_ae, disc_cnn, disc_lstm]:
        m.eval()
        m.to(device)

    augmented_result = {}   # {label: np.ndarray}

    # ── 클래스별 증강 ──────────────────────────────
    for label, target_n in target_counts.items():

        current_n   = current_counts.get(label, 0)
        num_generate = target_n - current_n

        if num_generate <= 0:
            print(f"\n[Label {label}] 이미 목표치 달성 "
                  f"(현재 {current_n} ≥ 목표 {target_n}), 건너뜀")
            augmented_result[label] = np.empty((0,), dtype=np.float32)
            continue

        print(f"\n[Label {label}] 증강 시작")
        print(f"  현재: {current_n}개 | 목표: {target_n}개 | "
              f"생성 필요: {num_generate}개")

        label_tensor  = torch.tensor([label], device=device)
        saved_count   = 0
        total_trials  = 0
        all_samples   = []

        with torch.no_grad():
            while saved_count < num_generate:

                # ── 배치 단위 생성 ──────────────────
                remaining  = num_generate - saved_count
                cur_batch  = min(batch_size, remaining * 5)  # 여유있게 생성

                z      = torch.randn(cur_batch, latent_dim).type(Tensor)
                labels = label_tensor.expand(cur_batch)
                gen    = generator(z, labels)               # [B, feature_dim]

                # ── 판별 ────────────────────────────
                val_ae   = disc_ae(gen)    # [B, 1]
                val_cnn  = disc_cnn(gen)   # [B, 1]
                val_lstm = disc_lstm(gen)  # [B, 1]

                # ── 샘플별 투표 필터링 ───────────────
                for idx in range(cur_batch):
                    passed = _vote(
                        val_ae[idx].item(),
                        val_cnn[idx].item(),
                        val_lstm[idx].item(),
                        ae_threshold, disc_threshold, min_votes,
                    )
                    if passed:
                        # [수정] 역정규화된 값 저장 (원본은 정규화 값 저장하는 버그)
                        inv = _inverse_normalize(
                            gen[idx].unsqueeze(0), x_min, x_max
                        )
                        all_samples.append(inv)
                        saved_count += 1

                    if saved_count >= num_generate:
                        break

                total_trials += cur_batch

                if total_trials % log_interval < cur_batch:
                    accept_rate = saved_count / total_trials * 100
                    print(f"  시도: {total_trials:,}회 | "
                          f"저장: {saved_count:,}/{num_generate:,}개 | "
                          f"통과율: {accept_rate:.1f}%")

        # ── 결과 저장 ──────────────────────────────
        result_np  = np.concatenate(all_samples, axis=0)  # [N, feature_dim]
        save_path  = os.path.join(save_dir, f"aug_label{label}.npy")
        np.save(save_path, result_np)

        augmented_result[label] = result_np

        print(f"  ✅ Label {label} 완료: {len(result_np):,}개 저장 → {save_path}")

    # ── 전체 요약 ──────────────────────────────────
    print("\n" + "=" * 50)
    print("증강 완료 요약")
    print("=" * 50)
    for label, arr in augmented_result.items():
        total = current_counts.get(label, 0) + len(arr)
        print(f"  Label {label}: 원본 {current_counts.get(label,0):,}개 "
              f"+ 증강 {len(arr):,}개 = 총 {total:,}개")

    return augmented_result