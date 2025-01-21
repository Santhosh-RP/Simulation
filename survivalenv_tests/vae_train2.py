import random
import torch
import torch.nn.functional as F
import numpy as np

from vae import VAE
from survivalimagedataset import SurvivalImageDataset
from utils import saliency

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PUBLISH = False
if PUBLISH:
    import wandb
    wandb.init(project="VAE")

batch_size = 64 
learning_rate = 1e-4
num_epochs = 1000
max_patience = 50
ZDIM = 75
d = 30000

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

if PUBLISH:
    wandb.config = {"batch_size": batch_size, "learning_rate": learning_rate,
                    "num_epochs": num_epochs, "max_patience": max_patience,
                    "dataset_length": d,      "zdim": ZDIM}


a, b, c = 0, int(d*0.9), int(d*0.95)
print('Loading datasets')
train_loader = torch.utils.data.DataLoader(SurvivalImageDataset(a, b, directory='images_randomised'), batch_size=batch_size, shuffle=True)
val_loader =   torch.utils.data.DataLoader(SurvivalImageDataset(b, c, directory='images_randomised'), batch_size=batch_size)
test_loader =  torch.utils.data.DataLoader(SurvivalImageDataset(c, d, directory='images_randomised'), batch_size=1)
print('Datasets loaded')

net = VAE(zDim=ZDIM).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# rows = 60
# M = torch.concatenate((torch.arange(1,rows//2+1), torch.arange(rows//2, 0, -1)),0).to(device)/(rows//2)
# M.requires_grad_(False)
# M = (M*M).view(rows, 1)
# M[M<0.5] = 0.5


sep = np.ones((128,3,3), dtype=np.uint8)*255

import cv2
import time
t = int(time.time())
save = True
def evaluate(net, dataset_loader, epoch=None):
    with torch.no_grad():
        batch_loss = []
        samples = 0
        for batch_idx, data in enumerate(dataset_loader):
            imgs, _ = data
            samples += imgs.shape[0]
            imgs = imgs.to(device)
            out, mu, logVar = net(imgs)
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
            batch_loss.append(loss.item())
            if save and batch_idx < 3:
                outimg = (np.transpose(out[0].cpu().numpy(), [1,2,0])*255).astype(np.uint8)
                inpimg = (np.transpose(imgs[0].cpu().numpy(), [1,2,0])*255).astype(np.uint8)
                print(f'{inpimg.shape=}, {sep.shape=} {outimg.shape=}')
                concatenate = np.concatenate((inpimg, sep, outimg), axis=1)
                concatenate = cv2.cvtColor(concatenate, cv2.COLOR_BGR2RGB)
                fname = f"log_{__file__.split('/')[-1]}_t{t}_e{str(epoch).zfill(6)}_b{batch_idx}.png"
                cv2.imwrite(fname, concatenate)
        ret_loss = np.absolute(np.array(batch_loss)).sum()/samples
    return ret_loss




best_state_dict = None
patience = max_patience
min_val_loss = None
min_epoch = None
for epoch in range(num_epochs):
    batch_loss = []
    samples = 0
    for idx, data in enumerate(train_loader, 0):
        imgs, _ = data
        samples += imgs.shape[0]
        imgs = imgs.to(device)
        out, mu, logVar = net(imgs)
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.item())
    train_loss = np.absolute(np.array(batch_loss)).sum()/samples

    val_loss = evaluate(net, val_loader, epoch)
    test_loss = evaluate(net, test_loader, epoch)
    print(f'Epoch {str(epoch).zfill(4)}: train({str(train_loss).rjust(12)}) val({str(val_loss).rjust(12)}) test({str(test_loss).rjust(12)})')
    
    if PUBLISH:
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss})


    patience -= 1
    if min_val_loss is None or val_loss < min_val_loss:
        prev_min_val_loss = min_val_loss
        min_val_loss = val_loss
        min_epoch = epoch
        patience = max_patience
        torch.save(net.state_dict(), f"vae2.pth")
        if prev_min_val_loss is not None:
            print(f'  save! ({min_val_loss-prev_min_val_loss})')
        best_state_dict = net.state_dict()
    elif patience == 0:
        break
if min_val_loss is not None:
    torch.save(best_state_dict, f"vae2_{str(epoch).zfill(3)}_{min_val_loss}.pth")


if PUBLISH:
    wandb.finish()
