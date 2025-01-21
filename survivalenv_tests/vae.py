import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torch.nn.functional as F

import torchvision
from torchvision.models.vision_transformer import VisionTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class VAE(nn.Module):
    def __init__(self, imgChannels=3, zDim=128):
        super(VAE, self).__init__()

        featureDim=384
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.featureDim = featureDim
        self.encConv1 = nn.Conv2d(imgChannels,   imgChannels*2,  3, stride=2)
        self.encConv2 = nn.Conv2d(imgChannels*2, imgChannels*4,  3, stride=2)
        self.encConv3 = nn.Conv2d(imgChannels*4, imgChannels*8,  3, stride=2)
        self.encConv4 = nn.Conv2d(imgChannels*8, imgChannels*8,  3, stride=2)
        self.encConv5 = nn.Conv2d(imgChannels*8, imgChannels*8,  3, stride=2)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(imgChannels*8, imgChannels*8, 3, stride=2)
        self.decConv2 = nn.ConvTranspose2d(imgChannels*8, imgChannels*8, 3, stride=2)
        self.decConv3 = nn.ConvTranspose2d(imgChannels*8, imgChannels*4, 3, stride=2)
        self.decConv4 = nn.ConvTranspose2d(imgChannels*4, imgChannels*2, 3, stride=2)
        self.decConv5 = nn.ConvTranspose2d(imgChannels*2, imgChannels,   4, stride=2)

    def encoder(self, x):
        print(f'INPUT {x.shape=}')
        x = F.silu(self.encConv1(x))
        print(f'A1 {x.shape=}')
        x = F.silu(self.encConv2(x))
        print(f'A2 {x.shape=}')
        x = F.silu(self.encConv3(x))
        print(f'A3 {x.shape=}')
        x = F.silu(self.encConv4(x))
        print(f'A4 {x.shape=}')
        x = F.silu(self.encConv5(x))
        print(f'A5 {x.shape=}')
        x = x.view(x.shape[0], -1)
        print(f'A5 {x.shape=}')
        mu = self.encFC1(x)
        print(f'mu {mu.shape=}')
        logVar = self.encFC2(x)
        print(f'logvar {mu.shape=}')
        return mu, logVar

    def reparametrize(self, mu, logVar, mle=False):
        if mle:
            return mu
        else:
            std = torch.exp(logVar/2)
            eps = torch.randn_like(std)
            return mu + std * eps

    def decoder(self, z):
        print(f'DECZ {z.shape=}')
        x = F.silu(self.decFC1(z))
        x = x.view(-1, 24, 4, 4)
        print(f'DEC0 {x.shape=}')
        x = F.silu(self.decConv1(x))
        print(f'AD1 {x.shape=}')
        x = F.silu(self.decConv2(x))
        print(f'AD2 {x.shape=}')
        x = F.silu(self.decConv3(x))
        print(f'AD3 {x.shape=}')
        x = F.silu(self.decConv4(x))
        print(f'AD4 {x.shape=}')
        x = torch.sigmoid(self.decConv5(x))
        print(f'AD5 {x.shape=}')

        return x

    def get_z(self, x, mle=False):
        mu, logVar = self.encoder(x)
        return self.reparametrize(mu, logVar, mle)

    def get_deterministic_z(self, x):
        mu, _ = self.encoder(x)
        return mu

    def forward(self, x, mle=False):
        mu, logVar = self.encoder(x)
        z = self.reparametrize(mu, logVar, mle)

        out = self.decoder(z)

        return out, mu, logVar



class VAE_OLD(nn.Module):
    def __init__(self, imgChannels=3, zDim=128):
        super(VAE_OLD, self).__init__()

        featureDim=127896  # 77976
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.featureDim = featureDim
        self.encConv1 = nn.Conv2d(imgChannels, 16, 3)
        self.encConv2 = nn.Conv2d(16, 24, 3)
        self.encConv3 = nn.Conv2d(24, 24, 3)
        self.encConv4 = nn.Conv2d(24, 24, 3)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(24, 24, 3)
        self.decConv2 = nn.ConvTranspose2d(24, 24, 3)
        self.decConv3 = nn.ConvTranspose2d(24, 16, 3)
        self.decConv4 = nn.ConvTranspose2d(16, 8, 3, stride=2)
        self.decConv5 = nn.ConvTranspose2d(8, imgChannels, 3)

    def encoder(self, x):
        x = F.relu(self.encConv1(x))
        x = self.maxpool(x)
        x = F.relu(self.encConv2(x))
        x = F.relu(self.encConv3(x))
        x = F.tanh(self.encConv4(x))
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar, mle=False):
        std = torch.exp(logVar/2)
        if mle:
            eps = torch.zeros_like(std)
        else:
            eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 24, 73, 73)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = F.relu(self.decConv3(x))
        x = F.relu(self.decConv4(x))
        x = torch.sigmoid(self.decConv5(x))[:,:,0:160,0:160]
        return x

    def get_z(self, x, mle=False):
        mu, logVar = self.encoder(x)
        return self.reparameterize(mu, logVar, mle)

    def get_deterministic_z(self, x):
        mu, _ = self.encoder(x)
        return mu

    def forward(self, x, mle=False):
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar, mle)
        out = self.decoder(z)

        return out, mu, logVar

class ViTVAE(nn.Module):
    def __init__(self, zdim, vit_num_classes, vit_mlp_dim, vit_hidden_dim, vit_num_heads, vit_num_layers, imgChannels=3):
        super(ViTVAE, self).__init__()
        featureDim = 77976

        # num_classes = 7*zDim
        # mlp_dim = 17
        # hidden_dim = 64 # num_heads * x
        # num_heads = 8
        # num_layers = 4

        patch_size = 8

        self.resize = torchvision.transforms.Resize((80,80))
        self.vit = VisionTransformer(image_size=80,
                                        patch_size=patch_size,
                                        num_layers=vit_num_layers,
                                        num_heads=vit_num_heads,
                                        hidden_dim=vit_hidden_dim,
                                        mlp_dim=vit_mlp_dim,
                                        num_classes=vit_num_classes,
                                        representation_size=zdim)
        self.encFC1 = nn.Linear(vit_num_classes, zdim)
        self.encFC2 = nn.Linear(vit_num_classes, zdim)
        self.decFC1 = nn.Linear(zdim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(24, 24, 3)
        self.decConv2 = nn.ConvTranspose2d(24, 24, 3)
        self.decConv3 = nn.ConvTranspose2d(24, 16, 3)
        self.decConv4 = nn.ConvTranspose2d(16, imgChannels, 3)

    def encoder(self, x):
        x = self.resize(x)
        x = F.tanh(self.vit(x))
        x = x.view(x.shape[0], -1)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 24, 128-8, 128-8)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = F.relu(self.decConv3(x))
        x = torch.sigmoid(self.decConv4(x))
        return x

    def get_z(self, x):
        mu, logVar = self.encoder(x)
        return self.reparameterize(mu, logVar)

    def forward(self, x):
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)

        return out, mu, logVar




