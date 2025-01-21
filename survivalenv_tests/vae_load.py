import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from vae import VAE
from survivalimagedataset import SurvivalImageDataset
# from utils import saliency


id = sys.argv[1].split("_")[1]
print(id)
with open('tasks.json') as f:
    tasks = json.load(f)
    
config = None
for task, config in tasks:
    if task == id:
        break

print(config)

sys.exit(0)


ZDIM = 50
d = 20000
a, b, c = 0, int(d*0.9), int(d*0.95)
print('Loading dataset')
test_loader =  torch.utils.data.DataLoader(SurvivalImageDataset(c, d, directory='images_randomised'), batch_size=1)
print('Dataset loaded')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = VAE(zDim=ZDIM).to(device)
net.load_state_dict(torch.load(f"vae.pth"))
net.eval()
i = 0
with torch.no_grad():
    for idx, data in enumerate(test_loader, 0):
        if idx>120:
            break
        imgs, _ = data
        imgs = imgs.to(device)
        saliency_values = saliency(imgs)
        att = (saliency_values[0].unsqueeze(dim=0).detach().cpu().numpy()*255).astype(np.uint8)[0]
        att = np.transpose(att, [1,2,0])
        cv2.imwrite(f"att_{str(i).zfill(6)}.png", att)
        print(att.shape)
        img = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
        image_to_show = np.squeeze(img*255).astype(np.uint8)
        out, mu, logVAR = net(imgs)
        outimg = np.transpose(out[0].cpu().numpy(), [1,2,0])*255
        outimg = outimg.astype(np.uint8)
        sep = np.ones((outimg.shape[0],3,3), dtype=np.uint8)*255
        concatenate = np.concatenate((att, sep, image_to_show, sep, outimg), axis=1)
        concatenate = cv2.cvtColor(concatenate, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"vision_{str(i).zfill(6)}.png", concatenate)
        i+= 1
