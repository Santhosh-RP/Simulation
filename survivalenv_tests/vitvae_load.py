import torch
import torch.nn.functional as F
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from vae import ViTVAE
from survivalimagedataset import SurvivalImageDataset


batch_size = 64
learning_rate = 2e-4
num_epochs = 500
max_patience = 25
ZDIM = 50

d = 20000
a, b, c = 0, int(d*0.9), int(d*0.95)
test_loader =  torch.utils.data.DataLoader(SurvivalImageDataset(c, d, directory='images_randomised'), batch_size=1)


print('Dataset loaded')




rows = 60
M = torch.concatenate((torch.arange(1,rows//2+1), torch.arange(rows//2, 0, -1)),0).to(device)/(rows//2)
M.requires_grad_(False)


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


net = ViTVAE(zDim=ZDIM).to(device)
net.load_state_dict(torch.load(f"vitvae.pth"))


net.eval()
i = 0
import cv2
with torch.no_grad():
    for idx, data in enumerate(test_loader, 0):
        imgs, _ = data
        imgs = imgs.to(device)
        img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
        image_to_show = np.squeeze(img*255).astype(np.uint8)
        out, mu, logVAR = net(imgs)
        outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])*255
        outimg = outimg.astype(np.uint8)
        sep = np.ones((outimg.shape[0],3,3), dtype=np.uint8)*255
        concatenate = np.concatenate((image_to_show, sep, outimg), axis=1)
        concatenate = cv2.cvtColor(concatenate, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"vision_{str(i).zfill(6)}_vit.png", concatenate)
        i+= 1




