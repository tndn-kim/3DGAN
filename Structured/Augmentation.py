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
parser.add_argument("--label_type", type=int, default=1)
opt = parser.parse_args()
print(opt)

n=opt.label_type

save_dir = f"C:/Users/milab_8/Desktop/3dgan/models_label"  # 여기에 위치시켜야 함
os.makedirs(save_dir, exist_ok=True) 
start_epoch = 199
checkpoint_path = os.path.join(save_dir, f"checkpoint_{start_epoch}.pth")

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

feature_dim = 76

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)  # 전체 데이터 개수 반환

    def __getitem__(self, idx):
        return self.data[idx] , self.labels[idx]

def load_data():
    x = pd.read_csv('C:/Users/milab_8/Desktop/GAN/Data.csv')
    label=pd.read_csv('C:/Users/milab_8/Desktop/GAN/label_3way.csv')
    x = 2*(x - x.min()) / (x.max() - x.min() + 1e-8)-1
    return x.values, label['label'].values

data, labels = load_data()
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=opt.batch_size , shuffle=True, pin_memory = True)


class Generator(nn.Module):
    def __init__(self, num_classes=4):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, opt.latent_dim)

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 256),
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

def save_checkpoint(epoch, generator, disc_auto, disc_cnn, disc_lstm, optimizer_G,
                    optimizer_D_auto, optimizer_D_cnn, optimizer_D_lstm, path,
                    g_loss, d_loss_a, d_loss_c, d_loss_l):
    checkpoint = {
        'epoch': epoch,
        'generator': generator.state_dict(),
        'disc_auto': disc_auto.state_dict(),
        'disc_cnn': disc_cnn.state_dict(),
        'disc_lstm': disc_lstm.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D_auto': optimizer_D_auto.state_dict(),
        'optimizer_D_cnn': optimizer_D_cnn.state_dict(),
        'optimizer_D_lstm': optimizer_D_lstm.state_dict(),
        'g_loss': g_loss,
        'd_loss_a': d_loss_a,
        'd_loss_c': d_loss_c,
        'd_loss_l': d_loss_l
    }
    torch.save(checkpoint, path)
    print(f"✅ 체크포인트 저장됨: {path}")

def load_checkpoint(path, generator, disc_auto, disc_cnn, disc_lstm, optimizer_G,
                    optimizer_D_auto, optimizer_D_cnn, optimizer_D_lstm):
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint['generator'])
    disc_auto.load_state_dict(checkpoint['disc_auto'])
    disc_cnn.load_state_dict(checkpoint['disc_cnn'])
    disc_lstm.load_state_dict(checkpoint['disc_lstm'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D_auto.load_state_dict(checkpoint['optimizer_D_auto'])
    optimizer_D_cnn.load_state_dict(checkpoint['optimizer_D_cnn'])
    optimizer_D_lstm.load_state_dict(checkpoint['optimizer_D_lstm'])
    return checkpoint['epoch'], checkpoint['g_loss'], checkpoint['d_loss_a'], checkpoint['d_loss_c'], checkpoint['d_loss_l']

        
adversarial_loss = torch.nn.MSELoss()
bce_loss = torch.nn.BCELoss()

generator = Generator()
disc_autoencoder = Discriminator_autoencoder()
disc_cnn = Discriminator_CNN()
disc_lstm = Discriminator_LSTM()

if cuda:
    generator.cuda()
    disc_autoencoder.cuda()
    disc_cnn.cuda()
    disc_lstm.cuda()
    adversarial_loss.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_auto = torch.optim.Adam(disc_autoencoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_cnn = torch.optim.Adam(disc_cnn.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_lstm = torch.optim.Adam(disc_lstm.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ------
gamma = 0.5
lambda_k = 0.001
k = 0.0
# ------

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']
scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)
scheduler_D_auto = ReduceLROnPlateau(optimizer_D_auto, mode='min', factor=0.5, patience=5, verbose=True)
scheduler_D_cnn = ReduceLROnPlateau(optimizer_D_cnn, mode='min', factor=0.5, patience=5, verbose=True)
scheduler_D_lstm = ReduceLROnPlateau(optimizer_D_lstm, mode='min', factor=0.5, patience
                                     =5, verbose=True)



if os.path.exists(checkpoint_path):
    start_epoch,_,_,_,_ = load_checkpoint(checkpoint_path, generator, disc_autoencoder, disc_cnn, disc_lstm, optimizer_G,
                    optimizer_D_auto, optimizer_D_cnn, optimizer_D_lstm)

for epoch in range(start_epoch+1, opt.n_epochs):
    for i, (real_samples, real_labels) in enumerate(dataloader):
        
        real_samples = real_samples.to(device)
        real_labels = real_labels.to(device)

        valid = torch.full((real_samples.size(0), 1),1.0, device=device)
        fake = torch.full((real_samples.size(0), 1),0.0, device=device)

        real_sample = Variable(real_samples.type(Tensor)).to(device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(real_samples.size(0), opt.latent_dim, device=device)
        gen_labels = torch.randint(0, 4, (real_samples.size(0),), device=device)
        gen_samples = generator(z, gen_labels)



        g_loss_a = torch.mean(disc_autoencoder(gen_samples))
        g_loss_c = adversarial_loss(disc_cnn(gen_samples),valid)
        g_loss_l = adversarial_loss(disc_lstm(gen_samples),valid)
        g_loss = (g_loss_a + g_loss_c + g_loss_l) / 3


        g_loss.backward()
        optimizer_G.step()
        g_detach = gen_samples.detach()

        # Train Autoencoder Discriminator
        optimizer_D_auto.zero_grad()
        d_real = disc_autoencoder(real_sample)
        d_fake = disc_autoencoder(g_detach)

        real_a = torch.mean(d_real - real_sample)
        fake_a = torch.mean(d_fake-g_detach)
        d_loss_a = real_a - k * fake_a
        d_loss_a.backward()
        optimizer_D_auto.step()

        diff = torch.mean(gamma * real_a - fake_a)
        k = k + lambda_k * diff.item()
        k = min(max(k, 0), 1)  # Clamp k to [0, 1]

        M = (real_a + torch.abs(diff)).item()

        # Train CNN Discriminator
        optimizer_D_cnn.zero_grad()
        real_c = adversarial_loss(disc_cnn(real_sample), valid)
        fake_c = adversarial_loss(disc_cnn(g_detach), fake)
        d_loss_c = 0.5 * (real_c + fake_c)
        d_loss_c.backward()
        optimizer_D_cnn.step()

        # Train LSTM Discriminator
        optimizer_D_lstm.zero_grad()
        real_l = adversarial_loss(disc_lstm(real_sample), valid)
        fake_l = adversarial_loss(disc_lstm(g_detach), fake)
        d_loss_l = 0.5 * (real_l + fake_l)
        d_loss_l.backward()
        optimizer_D_lstm.step()

        print(

            "[Epoch %d/%d] [Batch %d/%d] [D loss AE: %f] [k_value: %f] [D loss CNN: %f] [D loss LSTM: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader),
               d_loss_a.item(), k, d_loss_c.item(), d_loss_l.item(), g_loss.item())
        )

    scheduler_G.step(g_loss.item())
    scheduler_D_auto.step(d_loss_a.item())
    scheduler_D_cnn.step(d_loss_c.item())
    scheduler_D_lstm.step(d_loss_l.item())
    epoch_checkpoint_path = os.path.join(save_dir, f"checkpoint_{epoch}.pth")
    save_checkpoint(epoch, generator, disc_autoencoder, disc_cnn, disc_lstm,
        optimizer_G, optimizer_D_auto, optimizer_D_cnn, optimizer_D_lstm, epoch_checkpoint_path,
        g_loss=g_loss.item(), d_loss_a=d_loss_a.item(), d_loss_c=d_loss_c.item(), d_loss_l=d_loss_l.item())

# 모델 불러오기
generator.eval()
disc_autoencoder.eval()
disc_cnn.eval()
disc_lstm.eval()

if cuda:
    generator.cuda()
    disc_autoencoder.cuda()
    disc_cnn.cuda()
    disc_lstm.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

generate_sample = int(input())

if opt.label_type==1:
    num_generate = generate_sample-17120

elif opt.label_type==2:
    num_generate = generate_sample-38383
    
elif opt.label_type==3:
    num_generate = generate_sample-34080

saved_count = 0
total_trials = 0

all_samples = []
label_to_generate = opt.label_type


x = pd.read_csv('C:/Users/milab_8/Desktop/GAN/Data.csv')
x_min = x.min()
x_max = x.max()
with torch.no_grad():
    while saved_count < num_generate:
        # 1개 샘플 생성
        z = torch.randn(1, opt.latent_dim).type(Tensor)
        label = torch.tensor([label_to_generate], device=device)
        gen_samples = generator(z, label)

        # 판별 결과
        val_ae = disc_autoencoder(gen_samples)
        val_cnn = disc_cnn(gen_samples)
        val_lstm = disc_lstm(gen_samples)

        # 평가 기준 설정
        fooled = 0
        if val_ae.item() < 0.3:  # Autoencoder → 낮은 reconstruction error
            fooled += 1
        if val_cnn.item() > 0.5:  # CNN → output > 0.5
            fooled += 1
        if val_lstm.item() > 0.5:  # LSTM → output > 0.5
            fooled += 1

        if fooled >= 2:  # 2개 이상 통과
            x_min_tensor = torch.tensor(x_min.values, dtype=torch.float32, device=gen_samples.device)
            x_max_tensor = torch.tensor(x_max.values, dtype=torch.float32, device=gen_samples.device)
            gen_samples_inverse = ((gen_samples + 1) / 2) * (x_max_tensor - x_min_tensor + 1e-8) + x_min_tensor
            all_samples.append(gen_samples.cpu().numpy())
            saved_count += 1

        total_trials += 1
        if total_trials % 100 == 0:
            print(f"시도: {total_trials}회, 저장됨: {saved_count}개")

all_samples_np = np.concatenate(all_samples, axis=0) 

np.save(f"C:/Users/milab_8/Desktop/g_data_3dgan{n}.npy", all_samples_np)