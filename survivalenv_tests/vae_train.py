import PIL
import cv2
import sys
import random
import torch
import torch.nn.functional as F
import numpy as np

from vae import VAE
from survivalimagedataset import SurvivalImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

task_id = sys.argv[1]
batch_size = int(sys.argv[2])
learning_rate = float(sys.argv[3])
num_epochs = int(sys.argv[4])*2
max_patience = int(sys.argv[5])*2
ZDIM = int(sys.argv[6])
d = 22000

# PUBLISH = True
PUBLISH = False

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

if PUBLISH:
    import wandb
    wandb.init(project="VAE", notes=f"r{task_id}", name=f"r{task_id}")
    wandb.config = {"task_id": task_id, "batch_size": batch_size, "learning_rate": learning_rate,
                    "num_epochs": num_epochs, "max_patience": max_patience,
                    "dataset_length": d,      "zdim": ZDIM
                   }


a, b, c = 0, int(d*0.9), int(d*0.95)
print('Loading datasets')
train_loader = torch.utils.data.DataLoader(SurvivalImageDataset(a, b, directory='images_randomised'), batch_size=batch_size, shuffle=True)
val_loader =   torch.utils.data.DataLoader(SurvivalImageDataset(b, c, directory='images_randomised'), batch_size=batch_size)
test_dataset = SurvivalImageDataset(c, d, directory='images_randomised')
test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=1)
print('Datasets loaded')

net = VAE(zDim=ZDIM).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

def evaluate(net, dataset_loader):
    with torch.no_grad():
        batch_loss = []
        samples = 0
        for data in dataset_loader:
            imgs, _ = data
            samples += imgs.shape[0]
            imgs = imgs.to(device)
            out, mu, logVar = net(imgs)
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
            batch_loss.append(loss.item())
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

    val_loss = evaluate(net, val_loader)
    test_loss = evaluate(net, test_loader)
    print(f'Epoch {str(epoch).zfill(4)}: train({str(train_loss).rjust(12)}) val({str(val_loss).rjust(12)}) test({str(test_loss).rjust(12)})')
    

    if PUBLISH:
        with torch.no_grad():
            collage = None
            for i in [0,50,100,150,200,250,300,350]:
                imgs, _ = test_dataset[i]
                imgs = imgs[None, :, :, :].to(device)
                img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
                image_to_show = np.squeeze(img*255).astype(np.uint8)
                out, mu, logVAR = net(imgs, mle=True)
                outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])*255
                outimg = outimg.astype(np.uint8)
                hsep = np.ones((outimg.shape[0],3,3), dtype=np.uint8)*255
                vsep = np.ones((3,outimg.shape[1]*2+3,3), dtype=np.uint8)*255
                concatenate = np.concatenate((image_to_show, hsep, outimg), axis=1)
                # concatenate = cv2.cvtColor(concatenate, cv2.COLOR_BGR2RGB)
                if collage is None:
                    collage = concatenate
                else:
                    collage = np.concatenate((collage, vsep, concatenate), axis=0)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss, "sample": wandb.Image(collage)})


    patience -= 1
    if min_val_loss is None or val_loss < min_val_loss:
        prev_min_val_loss = min_val_loss
        min_val_loss = val_loss
        min_epoch = epoch
        patience = max_patience
        torch.save(net.state_dict(), f"vae_{task_id}_.pth")
        if prev_min_val_loss is not None:
            print(f'  save! ({min_val_loss-prev_min_val_loss})')
        best_state_dict = net.state_dict()
    elif patience == 0:
        break
if min_val_loss is not None:
    torch.save(best_state_dict, f"vae_{task_id}_{str(epoch).zfill(3)}_{min_val_loss}.pth")


if PUBLISH:
    wandb.finish()


sys.exit(0)


