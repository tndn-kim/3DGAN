"""
train.py
--------
GAN 학습 루프 모듈.
pipeline.py 에서 train(config, dataloader) 형태로 호출하거나
단독 실행도 가능.
"""

import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from GAN import build_models
from utils import save_checkpoint, load_checkpoint, build_schedulers, get_lr


# ────────────────────────────────────────────────
# 학습 함수
# ────────────────────────────────────────────────

def train(config: dict, dataloader):
    """
    GAN 학습 메인 함수.

    Args:
        config     : 하이퍼파라미터 딕셔너리 (pipeline.py 에서 전달)
        dataloader : preprocess.py 에서 생성한 DataLoader

    Returns:
        generator     : 학습 완료된 Generator
        disc_ae       : 학습 완료된 DiscriminatorAE
        disc_cnn      : 학습 완료된 DiscriminatorCNN
        disc_lstm     : 학습 완료된 DiscriminatorLSTM
    """

    # ── 디바이스 설정 ──────────────────────────────
    cuda   = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ── config 파싱 ────────────────────────────────
    n_epochs    = config["n_epochs"]
    latent_dim  = config["latent_dim"]
    lr          = config["lr"]
    b1          = config["b1"]
    b2          = config["b2"]
    num_classes = config["num_classes"]
    save_dir    = config["save_dir"]
    gamma       = config.get("gamma",    0.5)
    lambda_k    = config.get("lambda_k", 0.001)

    os.makedirs(save_dir, exist_ok=True)

    # ── 모델 생성 ──────────────────────────────────
    generator, disc_ae, disc_cnn, disc_lstm = build_models(
        latent_dim  = latent_dim,
        feature_dim = config["feature_dim"],
        num_classes = num_classes,
        emb_dim     = config.get("emb_dim", 32),
    )
    for m in [generator, disc_ae, disc_cnn, disc_lstm]:
        m.to(device)

    # ── Loss ───────────────────────────────────────
    adversarial_loss = nn.MSELoss().to(device)

    # ── Optimizer ─────────────────────────────────
    optimizer_G      = torch.optim.Adam(generator.parameters(),  lr=lr, betas=(b1, b2))
    optimizer_D_ae   = torch.optim.Adam(disc_ae.parameters(),    lr=lr, betas=(b1, b2))
    optimizer_D_cnn  = torch.optim.Adam(disc_cnn.parameters(),   lr=lr, betas=(b1, b2))
    optimizer_D_lstm = torch.optim.Adam(disc_lstm.parameters(),  lr=lr, betas=(b1, b2))

    # ── Scheduler ─────────────────────────────────
    schedulers = build_schedulers(
        optimizer_G, optimizer_D_ae, optimizer_D_cnn, optimizer_D_lstm,
        factor   = config.get("lr_factor",   0.5),
        patience = config.get("lr_patience", 5),
    )

    # ── BEGAN k 초기화 ─────────────────────────────
    k = 0.0

    # ── 체크포인트 복원 ────────────────────────────
    # 체크포인트 없으면 start_epoch=0 으로 시작
    start_epoch    = 0
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{config.get('resume_epoch', -1)}.pth")

    if os.path.exists(checkpoint_path):
        start_epoch, _, _, _, _ = load_checkpoint(
            checkpoint_path,
            generator, disc_ae, disc_cnn, disc_lstm,
            optimizer_G, optimizer_D_ae, optimizer_D_cnn, optimizer_D_lstm,
        )
        print(f"✅ {start_epoch} epoch 체크포인트 복원 완료")

    # ── 학습 루프 ──────────────────────────────────
    for epoch in range(start_epoch + 1, n_epochs):
        for i, (real_samples, real_labels) in enumerate(dataloader):

            real_samples = real_samples.to(device)
            real_labels  = real_labels.to(device)

            batch_size = real_samples.size(0)
            valid = torch.ones (batch_size, 1, device=device)
            fake  = torch.zeros(batch_size, 1, device=device)

            real_sample = Variable(real_samples.type(Tensor))

            # ── Generator 학습 ──────────────────────
            optimizer_G.zero_grad()

            z          = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            gen_samples = generator(z, gen_labels)

            g_loss_a = torch.mean(disc_ae(gen_samples))           # AE: 낮은 재구성 오차 유도
            g_loss_c = adversarial_loss(disc_cnn(gen_samples),  valid)
            g_loss_l = adversarial_loss(disc_lstm(gen_samples), valid)
            g_loss   = (g_loss_a + g_loss_c + g_loss_l) / 3

            g_loss.backward()
            optimizer_G.step()
            g_detach = gen_samples.detach()

            # ── AE Discriminator 학습 (BEGAN) ────────
            optimizer_D_ae.zero_grad()

            d_real = disc_ae(real_sample)   # [B, 1] 재구성 MSE
            d_fake = disc_ae(g_detach)      # [B, 1] 재구성 MSE

            # [수정] 원본: torch.mean(d_real - real_sample) → shape 불일치 ([B,1] vs [B,76])
            # BEGAN 에너지 기반 loss: 실제 재구성 오차 평균끼리 비교
            real_a  = torch.mean(d_real)    # scalar: real 평균 재구성 오차
            fake_a  = torch.mean(d_fake)    # scalar: fake 평균 재구성 오차
            d_loss_a = real_a - k * fake_a

            d_loss_a.backward()
            optimizer_D_ae.step()

            # k 업데이트 (BEGAN 균형 제어)
            diff = torch.mean(gamma * real_a - fake_a)
            k    = k + lambda_k * diff.item()
            k    = min(max(k, 0), 1)

            M = (real_a + torch.abs(diff)).item()   # 수렴 측도

            # ── CNN Discriminator 학습 ───────────────
            optimizer_D_cnn.zero_grad()

            real_c  = adversarial_loss(disc_cnn(real_sample), valid)
            fake_c  = adversarial_loss(disc_cnn(g_detach),    fake)
            d_loss_c = 0.5 * (real_c + fake_c)

            d_loss_c.backward()
            optimizer_D_cnn.step()

            # ── LSTM Discriminator 학습 ──────────────
            optimizer_D_lstm.zero_grad()

            real_l  = adversarial_loss(disc_lstm(real_sample), valid)
            fake_l  = adversarial_loss(disc_lstm(g_detach),    fake)
            d_loss_l = 0.5 * (real_l + fake_l)

            d_loss_l.backward()
            optimizer_D_lstm.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] "
                "[D_AE: %.4f] [k: %.4f] [M: %.4f] "
                "[D_CNN: %.4f] [D_LSTM: %.4f] [G: %.4f] "
                "[lr_G: %.6f]"
                % (
                    epoch, n_epochs, i, len(dataloader),
                    d_loss_a.item(), k, M,
                    d_loss_c.item(), d_loss_l.item(), g_loss.item(),
                    get_lr(optimizer_G),
                )
            )

        # ── epoch 종료: 스케줄러 step ────────────────
        schedulers['G'].step(g_loss.item())
        schedulers['ae'].step(d_loss_a.item())
        schedulers['cnn'].step(d_loss_c.item())
        schedulers['lstm'].step(d_loss_l.item())

        # ── 체크포인트 저장 ──────────────────────────
        epoch_ckpt_path = os.path.join(save_dir, f"checkpoint_{epoch}.pth")
        save_checkpoint(
            epoch,
            generator, disc_ae, disc_cnn, disc_lstm,
            optimizer_G, optimizer_D_ae, optimizer_D_cnn, optimizer_D_lstm,
            epoch_ckpt_path,
            g_loss   = g_loss.item(),
            d_loss_a = d_loss_a.item(),
            d_loss_c = d_loss_c.item(),
            d_loss_l = d_loss_l.item(),
        )

    return generator, disc_ae, disc_cnn, disc_lstm