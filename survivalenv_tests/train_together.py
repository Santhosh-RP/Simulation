import os
import sys

VAE_BATCH_SIZE = 32
VAE_LEARNING_RATE  = 0.000005
RSSM_LEARNING_RATE = 0.0001
EPISODE_BATCH_SIZE = 4

# Do not store mu+logVar, but only z

if True:
#    threads = 29
#    threads = str(int(threads))
#    os.environ["MKL_NUM_THREADS"] = threads
#    os.environ["NUMEXPR_NUM_THREADS"] = threads
#    os.environ["OMP_NUM_THREADS"] = threads
    SAVE_PATH = "/users/m/mansol/"
    NUM_EPOCHS = 20000
    MAX_PATIENCE = 20
    VAE_ZDIM = 250
    STEPS = 200
    RSSM_HIDDEN_SIZE = 2048
    RSSM_NUM_LAYERS = 3
    LAST_DATA_INDEX = 45
    a, b, c, d = 0, int(LAST_DATA_INDEX*0.90), int(LAST_DATA_INDEX*0.95), LAST_DATA_INDEX
    EPISODES_PATH = "/users/m/mansol/episodes/"
else:
    SAVE_PATH = "/home/ljmanso/Dropbox/"
    EPISODE_BATCH_SIZE = 1
    NUM_EPOCHS = 20000
    MAX_PATIENCE = 20
    VAE_ZDIM = 250
    STEPS = 70
    RSSM_HIDDEN_SIZE = 1000
    RSSM_NUM_LAYERS = 3
    # a, b, c, d = 0, 2, 3, 4
    LAST_DATA_INDEX = 45
    a, b, c, d = 0, int(LAST_DATA_INDEX*0.90), int(LAST_DATA_INDEX*0.95), LAST_DATA_INDEX
    EPISODES_PATH = "/dl"


import os
import random
import itertools
from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np

import cv2

from vae import VAE, VAE_OLD
from rssm_train import RSSM

from survivalimagedataset import SurvivalDataset, SurvivalDatasetEpisodeImages

try:
    TASK_ID = sys.argv[1]
except Exception:
    print('If you want to upload data to wandb, you need to provide an experiment identifier as the first parameter.')
    print('If you want to upload data to wandb, you need to provide an experiment identifier as the first parameter.')
    print('If you want to upload data to wandb, you need to provide an experiment identifier as the first parameter.')
    TASK_ID = None

if TASK_ID is not None:
    import wandb
    PUBLISH = True
else:
    PUBLISH = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


#val_data = None
#train_data =   SurvivalDataset(a, b, directory=EPISODES_PATH, limit=STEPS)
#val_data =     SurvivalDataset(b, c, directory=EPISODES_PATH, limit=STEPS)
# test_data =    SurvivalDataset(c, d, directory=EPISODES_PATH, limit=STEPS)
#train_loader = torch.utils.data.DataLoader(train_data, batch_size=EPISODE_BATCH_SIZE, shuffle=True)
#val_loader =   torch.utils.data.DataLoader(  val_data, batch_size=EPISODE_BATCH_SIZE)
# test_loader =  torch.utils.data.DataLoader( test_data, batch_size=1)

#if val_data is None:
#    print(f'Datasets loaded: {len(train_loader)=}')
#else:
#    print(f'Datasets loaded: ({len(train_loader)=}/{len(val_loader)=})')
#    # print(f'Datasets loaded: ({len(train_loader)=}/{len(val_loader)=}/{len(test_loader)=})')

vae = VAE_OLD(zDim=VAE_ZDIM).to(DEVICE)
rssm = RSSM(17+2*VAE_ZDIM, RSSM_HIDDEN_SIZE, 17+2*VAE_ZDIM-2, num_layers=RSSM_NUM_LAYERS).to(DEVICE)


#               observation,          reward,    done,  terminated,    action   
# joint_velocity(2), gyro(3), accelerometer(3), magnetometer(3), health(1), view(160,210,3)
RSSM_INPUT = 2+3+3+3+1+VAE_ZDIM*2   +   1    +    1    +    1     +      2


vae_optimizer  = torch.optim.Adam(vae.parameters(),  lr=VAE_LEARNING_RATE)
combined_params = itertools.chain(*[vae.parameters(), rssm.parameters()])
combined_optimizer = torch.optim.Adam(combined_params, lr=RSSM_LEARNING_RATE)

mse = nn.MSELoss()

def reward_proc(reward):
    if reward is None:
        ret = [0]
    else:
        ret = [reward]
    return ret


def action_proc(action):
    if action is None:
        ret = [0,0]
    else:
        ret = action
    return ret


def setup_wandb():
    global train_data
    wandb.init(project="RSSM", notes=f"r{TASK_ID}", name=f"r{TASK_ID}")
    wandb.config = {"task_id": TASK_ID, "batch_size": EPISODE_BATCH_SIZE, "learning_rate": RSSM_LEARNING_RATE,
                    "num_epochs": NUM_EPOCHS, "max_patience": MAX_PATIENCE,
                    "dataset_length": len(train_data),  "zdim": VAE_ZDIM, "rssm_dim": RSSM_HIDDEN_SIZE
                   }


def draw_input_images(batch_data, future, out, epoch, prefix):
    video = None
    line = np.zeros((160,2,3), dtype=np.uint8)
    for r in range(future.shape[1]):
        xx = batch_data[0][r][0:160*160*3].detach()
        show0 = np.ascontiguousarray((xx.reshape(3,160,160).flip(0).cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8))
        show1 = np.ascontiguousarray((future[0][r].detach().flip(0).cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8))
        show2 = np.ascontiguousarray((   out[0][r].detach().flip(0).cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8))
        for img in [show0, show1, show2]:
            cv2.line(img, (0,  53), (160,  53), (255, 0, 0), thickness=1)
            cv2.line(img, (0, 107), (160, 107), (255, 0, 0), thickness=1)
            cv2.line(img, ( 53, 0), ( 53, 160), (255, 0, 0), thickness=1)
            cv2.line(img, (107, 0), (107, 160), (255, 0, 0), thickness=1)
        show = np.concatenate((show0, line, show1, line, show2), 1)
        show = cv2.resize(show, (0,0), fx=4, fy=4)#, interpolation=cv2.INTER_NEAREST)
        if video is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video = cv2.VideoWriter(f'{SAVE_PATH}/{prefix}_{str(epoch).zfill(5)}.avi', fourcc, 2, (show.shape[1], show.shape[0]))
        # print(show.shape)
        # cv2.imshow("draw input window", show)
        # cv2.waitKey(5)
        video.write(show)
    video.release()


resize = torchvision.transforms.Resize((160, 160))



# torch.autograd.set_detect_anomaly(True)


def run_epoch(vae, rssm, data, prefix, train=True):
    if train:
        vae.train()
        rssm.train()
    else:
        vae.eval()
        rssm.eval()

    # vae_episode_losses = []
    # for episode in range(len(data)):
    #     episode_image_data = SurvivalDatasetEpisodeImages(data[episode])
    #     episode_image_data_loader = torch.utils.data.DataLoader(episode_image_data, batch_size=VAE_BATCH_SIZE, shuffle=True)
    #     vae_batch_losses = []
    #     for data in episode_image_data_loader:
    #         imgs, _ = data
    #         imgs = imgs.to(DEVICE)
    #         out, mu, logVar = vae(imgs)
    #         kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
    #         vae_loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
    #         if train:
    #             vae_optimizer.zero_grad()
    #             vae_loss.backward()
    #             vae_optimizer.step()
    #         vae_batch_losses.append(vae_loss.item()/VAE_BATCH_SIZE)
    #     vae_episode_loss = torch.mean(torch.tensor(vae_batch_losses))
    #     vae_episode_losses.append(vae_episode_loss)
    # vae_epoch_loss = torch.mean(torch.tensor(vae_episode_losses))
    # vae_epoch_loss = 0
    # print(f'  VAE epoch loss {vae_epoch_loss}')


    combined_batch_losses = []
    perm = torch.randperm(len(data))
    number_of_batches = int(floor(len(data)/EPISODE_BATCH_SIZE))
    for batch_number in range(number_of_batches):
        batch_idxs = perm[batch_number*EPISODE_BATCH_SIZE:(batch_number+1)*EPISODE_BATCH_SIZE]
        # `batch_raw_images`: contains images, from              TIME: t=0 to t=T-1
        batch_raw_images  = torch.zeros((len(batch_idxs), STEPS,             3, 160, 160), device=DEVICE)
        # `batch_obs_wo_img`: contains observations w/o images   TIME: t_0 to t=T-1
        batch_obs_wo_img  = torch.zeros((len(batch_idxs), STEPS, RSSM_INPUT-2*VAE_ZDIM-2), device=DEVICE)
        # `batch_actions`: contains the actions by the agent     TIME: t_0 to t=T-1
        batch_actions     = torch.zeros((len(batch_idxs), STEPS,                       2), device=DEVICE)
        # `batch_imagination`: contains dreamed states
        batch_imagination = torch.zeros((len(batch_idxs), STEPS, RSSM_INPUT-2),            device=DEVICE)
        vae_lv = None
        vae_mu = None
        for inbatch_idx, episode_idx in enumerate(batch_idxs):
            episode = data[episode_idx]
            for step_idx, step in enumerate(episode[0:STEPS]):
                obs = step['observation']
                # raw images
                image = torch.tensor(obs["view"], dtype=torch.float32, device=DEVICE).permute(2, 0, 1)/255
                image = resize(image)
                image[image>1] = 1
                batch_raw_images[inbatch_idx, step_idx] = image
                # z IF FIRST STEP
                if step_idx == 0:
                    vae_mu, vae_lv = vae.encoder(image.unsqueeze(0))
                # observations without images
                obs_wo_img_chunks = (
                    torch.tensor(obs['joint_velocity'],       dtype=torch.float32, device=DEVICE),
                    torch.tensor(obs['gyro'],                 dtype=torch.float32, device=DEVICE),
                    torch.tensor(obs['accelerometer'],        dtype=torch.float32, device=DEVICE),
                    torch.tensor(obs['magnetometer'],         dtype=torch.float32, device=DEVICE),
                    torch.tensor(obs['health'],               dtype=torch.float32, device=DEVICE),
                    torch.tensor(reward_proc(step['reward']), dtype=torch.float32, device=DEVICE),
                    torch.tensor([step['done']],              dtype=torch.float32, device=DEVICE),
                    torch.tensor([step['terminated']],        dtype=torch.float32, device=DEVICE),
                )
                batch_obs_wo_img[inbatch_idx, step_idx] = torch.concat(obs_wo_img_chunks).to(DEVICE)
                # actions
                batch_actions[inbatch_idx, step_idx] = torch.tensor(action_proc(step['action']), dtype=torch.float32, device=DEVICE)
                # imagination
                rssm_step_input = torch.concat((vae_mu.view(-1), # Variables `vae_mu` and `vae_lv` refer to the "current" time step, and
                                                vae_lv.view(-1), # can come from either "real" (t=0) or "dreamed" images (t>0)
                                                batch_obs_wo_img[inbatch_idx, step_idx].view(-1),
                                                batch_actions[   inbatch_idx, step_idx].view(-1))).view(1,1,-1)
                rssm_out = rssm(rssm_step_input, online=step_idx!=0).squeeze().clone()
                batch_imagination[inbatch_idx, step_idx] = rssm_out
                # SET Z FOR THE NEXT STEP!
                vae_mu = rssm_out[       0:  VAE_ZDIM]
                vae_lv = rssm_out[VAE_ZDIM:2*VAE_ZDIM]
 

        vae_mus = batch_imagination[:, :,        0:VAE_ZDIM]
        vae_logVars = batch_imagination[:, :, VAE_ZDIM:VAE_ZDIM*2]
        z = vae.reparameterize(vae_mus, vae_logVars)
        # `batch_drm_images`: contains dreamed images            TIME: t=1 to t=T
        batch_drm_images = vae.decoder(z).view(len(batch_idxs), STEPS,             3, 160, 160)

        # rssm_loss = mse(batch_drm_images[:, :-1], batch_raw_images[:, 1:])

    # #     # <<< VAE THROUGH RSSM
        kl_divergence = 0.5 * torch.sum(-1 - vae_logVars + vae_mus.pow(2) + vae_logVars.exp())
        vae_loss = F.binary_cross_entropy(batch_drm_images[:, :-1], batch_raw_images[:, 1:], size_average=False) + kl_divergence
    # #     # VAE THROUGH RSSM >>>

        # combined_loss = rssm_loss
        combined_loss = vae_loss
        
        if train:
            combined_optimizer.zero_grad()
            combined_loss.backward()
            combined_optimizer.step()
        combined_batch_losses.append(combined_loss.item()/batch_drm_images.shape[1])

    rssm_epoch_loss = torch.mean(torch.tensor(combined_batch_losses))
    print(f'  RSSM epoch loss {rssm_epoch_loss}')

    # draw_input_images(batch_data, future, out, epoch, prefix)
    return rssm_epoch_loss



for epoch in range(NUM_EPOCHS):
    global train_data

    if epoch % 5 == 0:
        train_data =   None
        train_data =   SurvivalDataset(a, b, directory=EPISODES_PATH, limit=STEPS)
        val_data =     None
        val_data =     SurvivalDataset(b, c, directory=EPISODES_PATH, limit=STEPS)
        train_loader = None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=EPISODE_BATCH_SIZE, shuffle=True)
        val_loader =   None
        val_loader =   torch.utils.data.DataLoader(  val_data, batch_size=EPISODE_BATCH_SIZE)

    if PUBLISH:
        setup_wandb()

    print(f'EPOCH {epoch}')
    train_loss = run_epoch(vae=vae, rssm=rssm, data=train_data, prefix="train", train=True)
    dev_loss   = run_epoch(vae=vae, rssm=rssm, data=  val_data, prefix="val", train=False)

    torch.save(vae.state_dict(), f"vae.pth")

    torch.save(rssm.state_dict(), f"rssm.pth")

    if PUBLISH:
        wandb.log({"epoch": epoch,
                   "train_loss": train_loss,
                   "val_loss": dev_loss,
                     }) #  "sample": wandb.Image(collage)


