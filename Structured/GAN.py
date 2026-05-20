"""
GAN 모델 아키텍처 정의 모듈.

"""

import torch
import torch.nn as nn

SN = nn.utils.spectral_norm   # 편의용 alias


# ────────────────────────────────────────────────
# 공통 유틸
# ────────────────────────────────────────────────

def _init_weights(module: nn.Module) -> None:
    """LeakyReLU 계열에 맞는 He 초기화."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# ────────────────────────────────────────────────
# Generator
# ────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Conditional Generator.

    [변경] 레이블 조건부 방식: addition → concat + projection
      - 이전: z = z + label_emb(labels)
        → 두 벡터가 같은 공간에 있다는 보장 없음, 조건 신호 약함
      - 현재: z = proj(cat([z, label_emb(labels)], dim=1))
        → 레이블이 독립적인 차원으로 들어와 projection 후 혼합,
           조건 신호가 명시적으로 유지됨

    Args:
        latent_dim  : 노이즈 벡터 차원  (default 100)
        feature_dim : 생성할 피처 차원  (default 76)
        num_classes : 클래스 수         (default 4)
        emb_dim     : 레이블 임베딩 차원 (default 32)
    """

    def __init__(self, latent_dim: int = 100,
                 feature_dim: int = 76,
                 num_classes: int = 4,
                 emb_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        # 레이블 임베딩: num_classes → emb_dim
        self.label_emb = nn.Embedding(num_classes, emb_dim)

        # concat(z, emb) → latent_dim 으로 projection
        self.proj = nn.Sequential(
            nn.Linear(latent_dim + emb_dim, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, feature_dim),
            nn.Tanh(),
        )
        _init_weights(self)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z      : [B, latent_dim]  랜덤 노이즈
            labels : [B]              클래스 인덱스
        Returns:
            [B, feature_dim]  생성된 샘플 (-1 ~ 1)
        """
        emb = self.label_emb(labels)                  # [B, emb_dim]
        z   = self.proj(torch.cat([z, emb], dim=1))   # [B, latent_dim]
        return self.model(z)


# ────────────────────────────────────────────────
# Discriminator 1 : Autoencoder (재구성 오차 기반)
# ────────────────────────────────────────────────

class DiscriminatorAE(nn.Module):
    """
    오토인코더 기반 Discriminator (BEGAN 에너지 방식).
    출력 : 샘플별 MSE 재구성 오차 [B, 1] (낮을수록 real 에 가까움)

    [변경] encoder Linear 에 Spectral Normalization 적용
      → 가중치 스펙트럼 반경을 1로 고정, BEGAN k 값 안정화
      decoder 는 SN 미적용 (reconstruction 품질 유지)

    Args:
        input_dim  : 입력 피처 차원 (default 76)
        latent_dim : 병목 차원     (default 32)
    """

    def __init__(self, input_dim: int = 76, latent_dim: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            SN(nn.Linear(input_dim, 128)),
            nn.ReLU(),
            nn.Dropout(0.3),
            SN(nn.Linear(128, latent_dim)),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, input_dim),
        )
        _init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, input_dim]
        Returns:
            [B, 1]  샘플별 재구성 MSE
        """
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        return torch.mean((recon - x) ** 2, dim=1, keepdim=True)


# ────────────────────────────────────────────────
# Discriminator 2 : CNN (1D 합성곱)
# ────────────────────────────────────────────────

class DiscriminatorCNN(nn.Module):
    """
    1D-CNN 기반 Discriminator.
    출력 : real 확률 [B, 1] ∈ (0, 1)

    [변경] Conv1d / Linear 전체에 Spectral Normalization 적용
      → gradient 폭발 방지, 판별 경계면 안정화

    Args:
        input_dim : 입력 피처 차원 (default 76)
    """

    def __init__(self, input_dim: int = 76):
        super().__init__()

        self.model = nn.Sequential(
            SN(nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),

            SN(nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            SN(nn.Linear(32 * input_dim, 1)),
            nn.Sigmoid(),
        )
        _init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, input_dim]
        Returns:
            [B, 1]  real 확률
        """
        return self.model(x.unsqueeze(1))


# ────────────────────────────────────────────────
# Discriminator 3 : Bidirectional LSTM
# ────────────────────────────────────────────────

class DiscriminatorLSTM(nn.Module):


    def __init__(self, input_dim: int = 76,
                 hidden_dim: int = 128,
                 num_layers: int = 3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout    = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            SN(nn.Linear(hidden_dim * 2, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # [B, input_dim] → [B, input_dim, 1]  (seq_len=76, input_size=1)
        x = x.unsqueeze(2)

        _, (h_n, _) = self.lstm(x)
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)      # [B, hidden_dim*2]
        return self.classifier(self.dropout(h_last))


# ────────────────────────────────────────────────
# 팩토리 함수
# ────────────────────────────────────────────────

def build_models(latent_dim: int = 100,
                 feature_dim: int = 76,
                 num_classes: int = 4,
                 emb_dim: int = 32,
                 ae_latent_dim: int = 32,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 3):

    generator = Generator(
        latent_dim=latent_dim,
        feature_dim=feature_dim,
        num_classes=num_classes,
        emb_dim=emb_dim,
    )
    disc_ae = DiscriminatorAE(
        input_dim=feature_dim,
        latent_dim=ae_latent_dim,
    )
    disc_cnn = DiscriminatorCNN(
        input_dim=feature_dim,
    )
    disc_lstm = DiscriminatorLSTM(
        input_dim=feature_dim,
        hidden_dim=lstm_hidden,
        num_layers=lstm_layers,
    )
    return generator, disc_ae, disc_cnn, disc_lstm