import argparse
import os
import numpy as np
import math
import pandas as pd

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=200, help="number of image channels")
parser.add_argument("--label_type", type=int, default=0)
opt = parser.parse_args()
print(opt)

n=opt.label_type

save_dir = ""  
os.makedirs(save_dir, exist_ok=True) 

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

feature_dim = 76

class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)

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
                nn.Tanh()
            )

    def forward(self, z, labels):
        z = z + self.label_emb(labels)
        return self.model(z)

class Discriminator_autoencoder(nn.Module):
    def __init__(self, input_dim=76, latent_dim=32):
        super(Discriminator_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, input_dim)
        )
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        # Reconstruction error (MSE per sample)
        return torch.mean((reconstructed - x) ** 2, dim=1, keepdim=True)

class Discriminator_LSTM(nn.Module):
    def __init__(self, input_dim=76, hidden_dim=128, num_layers=3):
        super(Discriminator_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional = True
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, 76] → [B, 76, 1] for LSTM
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)  # Output shape: [B, 76, hidden_dim]
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers*2, B, hidden_dim]
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [B, hidden_dim*2]
        h_last = self.dropout(h_last)
        return self.classifier(h_last)

class Discriminator_CNN(nn.Module):
    def __init__(self, input_dim=76):
        super(Discriminator_CNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32 * input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.model(x)


        