"""
utils.py
--------
학습 공통 유틸리티 모듈.
- 체크포인트 저장 / 불러오기
- 스케줄러 일괄 생성
- 현재 lr 조회
"""

import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ────────────────────────────────────────────────
# 체크포인트
# ────────────────────────────────────────────────

def save_checkpoint(epoch: int,
                    generator, disc_ae, disc_cnn, disc_lstm,
                    optimizer_G, optimizer_D_ae, optimizer_D_cnn, optimizer_D_lstm,
                    path: str,
                    g_loss: float, d_loss_a: float,
                    d_loss_c: float, d_loss_l: float) -> None:
    """
    모델 + 옵티마이저 상태 + 손실값을 체크포인트로 저장.

    Args:
        epoch           : 현재 epoch
        generator       : Generator 모델
        disc_ae         : DiscriminatorAE 모델
        disc_cnn        : DiscriminatorCNN 모델
        disc_lstm       : DiscriminatorLSTM 모델
        optimizer_G     : Generator 옵티마이저
        optimizer_D_ae  : AE Discriminator 옵티마이저
        optimizer_D_cnn : CNN Discriminator 옵티마이저
        optimizer_D_lstm: LSTM Discriminator 옵티마이저
        path            : 저장 경로 (.pth)
        g_loss          : Generator 손실값
        d_loss_a        : AE Discriminator 손실값
        d_loss_c        : CNN Discriminator 손실값
        d_loss_l        : LSTM Discriminator 손실값
    """
    checkpoint = {
        'epoch'          : epoch,
        'generator'      : generator.state_dict(),
        'disc_ae'        : disc_ae.state_dict(),
        'disc_cnn'       : disc_cnn.state_dict(),
        'disc_lstm'      : disc_lstm.state_dict(),
        'optimizer_G'    : optimizer_G.state_dict(),
        'optimizer_D_ae' : optimizer_D_ae.state_dict(),
        'optimizer_D_cnn': optimizer_D_cnn.state_dict(),
        'optimizer_D_lstm': optimizer_D_lstm.state_dict(),
        'g_loss'         : g_loss,
        'd_loss_a'       : d_loss_a,
        'd_loss_c'       : d_loss_c,
        'd_loss_l'       : d_loss_l,
    }
    torch.save(checkpoint, path)
    print(f"✅ 체크포인트 저장됨: {path}")


def load_checkpoint(path: str,
                    generator, disc_ae, disc_cnn, disc_lstm,
                    optimizer_G, optimizer_D_ae, optimizer_D_cnn, optimizer_D_lstm):
    """
    체크포인트에서 모델 + 옵티마이저 상태 복원.

    Returns:
        epoch   : 마지막으로 저장된 epoch
        g_loss  : 저장 시점 Generator 손실값
        d_loss_a: 저장 시점 AE Discriminator 손실값
        d_loss_c: 저장 시점 CNN Discriminator 손실값
        d_loss_l: 저장 시점 LSTM Discriminator 손실값
    """
    checkpoint = torch.load(path)

    generator.load_state_dict(checkpoint['generator'])
    disc_ae.load_state_dict(checkpoint['disc_ae'])
    disc_cnn.load_state_dict(checkpoint['disc_cnn'])
    disc_lstm.load_state_dict(checkpoint['disc_lstm'])

    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D_ae.load_state_dict(checkpoint['optimizer_D_ae'])
    optimizer_D_cnn.load_state_dict(checkpoint['optimizer_D_cnn'])
    optimizer_D_lstm.load_state_dict(checkpoint['optimizer_D_lstm'])

    return (
        checkpoint['epoch'],
        checkpoint['g_loss'],
        checkpoint['d_loss_a'],
        checkpoint['d_loss_c'],
        checkpoint['d_loss_l'],
    )


# ────────────────────────────────────────────────
# 스케줄러
# ────────────────────────────────────────────────

def build_schedulers(optimizer_G,
                     optimizer_D_ae,
                     optimizer_D_cnn,
                     optimizer_D_lstm,
                     factor: float = 0.5,
                     patience: int = 5):
    """
    옵티마이저 4개에 대한 ReduceLROnPlateau 스케줄러를 일괄 생성.

    train.py 사용 예시:
        schedulers = build_schedulers(opt_G, opt_ae, opt_cnn, opt_lstm)
        # 매 epoch 끝에
        schedulers['G'].step(g_loss)
        schedulers['ae'].step(d_loss_a)
        schedulers['cnn'].step(d_loss_c)
        schedulers['lstm'].step(d_loss_l)

    Args:
        optimizer_*  : 각 모델 옵티마이저
        factor       : lr 감소 비율 (default 0.5 → 절반으로)
        patience     : 개선 없을 시 대기 epoch 수 (default 5)

    Returns:
        dict: {'G', 'ae', 'cnn', 'lstm'} 키를 가진 스케줄러 딕셔너리
    """
    _cfg = dict(mode='min', factor=factor, patience=patience, verbose=True)

    return {
        'G'   : ReduceLROnPlateau(optimizer_G,       **_cfg),
        'ae'  : ReduceLROnPlateau(optimizer_D_ae,    **_cfg),
        'cnn' : ReduceLROnPlateau(optimizer_D_cnn,   **_cfg),
        'lstm': ReduceLROnPlateau(optimizer_D_lstm,  **_cfg),
    }


# ────────────────────────────────────────────────
# LR 조회
# ────────────────────────────────────────────────

def get_lr(optimizer) -> float:
    """
    옵티마이저의 현재 learning rate 반환.

    train.py 사용 예시:
        print(f"현재 G lr: {get_lr(optimizer_G):.6f}")
    """
    return optimizer.param_groups[0]['lr']