import os
import sys
import cv2
import math
import numpy as np

import survivalenv
import gymnasium as gym

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from vae import VAE_OLD

ENV_NAME = 'survivalenv'

RSSM_HIDDEN_SIZE = 2048
RSSM_NUM_LAYERS = 3
RSSM_FILE = 'rssm.pth'

VAE_ZDIM = 250
vae = VAE_OLD(zDim=VAE_ZDIM).to(DEVICE)
vae.load_state_dict(torch.load('vae.pth'))

STEPS = 100
PREDICTION_STEPS = 1

# from here import RSSM
from rssm_train import RSSM

from survivalimagedataset import SurvivalImageDataset, SurvivalDataset, SurvivalDatasetEpisodeImages

resize = torchvision.transforms.Resize((160, 160))

def interleave_in_list(l, sep):
    return [item for sublist in [[elem, sep] for elem in l[:-1]] + [[l[-1]]] for item in sublist]


def preprocess_observation(obs, action, reward, done, terminated):
    image = torch.tensor(obs["view"], dtype=torch.float32, device=DEVICE)
    image = resize(image.permute(2, 0, 1)).unsqueeze(0)/255
    mu, lv = vae.encoder(image)
    chunks = (
        mu.squeeze(0),
        lv.squeeze(0),
        torch.tensor(obs['joint_velocity'], dtype=torch.float32, device=DEVICE),
        torch.tensor(obs['gyro'], dtype=torch.float32, device=DEVICE),
        torch.tensor(obs['accelerometer'], dtype=torch.float32, device=DEVICE),
        torch.tensor(obs['magnetometer'], dtype=torch.float32, device=DEVICE),
        torch.tensor(obs['health'], dtype=torch.float32, device=DEVICE),
        torch.tensor(reward_proc(reward), dtype=torch.float32, device=DEVICE),
        torch.tensor([done], dtype=torch.float32, device=DEVICE),
        torch.tensor([terminated], dtype=torch.float32, device=DEVICE),
        torch.tensor(action_proc(action), dtype=torch.float32, device=DEVICE)
    )
    return torch.concat(chunks), image


test_data =    SurvivalDataset(fname=sys.argv[1], limit=STEPS)
test_loader =  torch.utils.data.DataLoader( test_data, batch_size=1)

# Instantiation of the RSSM. The observation size is 162, the (reward+done) would be 2.
# The action however is not predicted, therefore the input is 162+2+2 but the output is
# smaller (without the action) i.e. 162+2.
rssm = RSSM(17+2*VAE_ZDIM, RSSM_HIDDEN_SIZE, 17+2*VAE_ZDIM-2, num_layers=RSSM_NUM_LAYERS).to(DEVICE)
rssm.load_state_dict(torch.load(RSSM_FILE))

if __name__ == '__main__':
    # env = gym.make('survivalenv/SurvivalEnv-v0', render_cameras=True, render_view=True)
    # obs = env.reset()
    reward = None
    done = False
    terminated = False
    total_reward = 0
    step = 0
    black_row = np.zeros((160,2,3), dtype=np.uint8)

    # obs = preprocess_observation(obs[0], None, reward, done, terminated)
    print(f"Sequences: {len(test_data)} {type(test_data)}")
    for seq in range(len(test_data)):
        print(f"Sequence {seq}")
        for datum_idx in range(len(test_data[seq])):
            print(datum_idx, '/', len(test_data[seq]))
            datum = test_data[seq][datum_idx]
            # env.render()
            # Action
            # np_action = np.array([0.7,0.3])
            # Step
            # obs, reward, done, terminated, info = env.step(np_action)
            np_action = datum["action"]
            obs = datum["observation"]
            reward = datum["reward"]
            terminated = datum["terminated"]
            done = datum["done"]
            tch_action = torch.from_numpy(np_action).type(torch.float32)
            obs, image = preprocess_observation(obs, tch_action, reward, done, terminated)

            # GENERATE ENCODED-DECODED IMAGE (INSTANTANEOUS)
            z_gpu = obs[0:VAE_ZDIM].type(torch.float32).to(DEVICE)
            outimg1 = vae.decoder(z_gpu).detach().squeeze(0).movedim(0,2).cpu()
            outimg1 = cv2.cvtColor((outimg1.numpy()*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            # RSSM input
            rssm_input = obs.unsqueeze(0).type(torch.float32).to(DEVICE)
            # RSSM output
            rssm_output = rssm(rssm_input, online=False)
            # Dream
            mu_rssm = rssm_output[:, 0:VAE_ZDIM]
            lvar_rssm = rssm_output[:, VAE_ZDIM:2*VAE_ZDIM]
            z_rssm = vae.reparameterize(mu_rssm, lvar_rssm)
            decoded = vae.decoder(z_rssm)
            # decoded = vae.decoder(mu_rssm)
            outimg2 = cv2.cvtColor((decoded.squeeze().movedim(0,2).detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_BGR2RGB)

            # FUTURE!
            good_h = rssm.h.clone().detach()
            rssm_future_output = rssm_output.clone().detach()
            for step in range(PREDICTION_STEPS):
                rssm_input = torch.concatenate((rssm_future_output,tch_action.to(DEVICE).unsqueeze(0)),1)
                rssm_future_output = rssm(rssm_input, online=False)
            rssm.h = good_h.clone().detach()
            # Dream further
            mu_rssm_f = rssm_future_output[:, 0:VAE_ZDIM]
            lvar_rssm_f = rssm_future_output[:, VAE_ZDIM:2*VAE_ZDIM]
            z_rssm_f = vae.reparameterize(mu_rssm_f, lvar_rssm_f)
            decoded = vae.decoder(z_rssm_f)
            outimg3 = cv2.cvtColor((decoded.squeeze().movedim(0,2).detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_BGR2RGB)

            # Draw
            I = np.ascontiguousarray((image.squeeze(0).permute(1, 2, 0).detach()*255).flip(2).cpu().numpy().astype(np.uint8))
            image_list = [
                I
                , outimg1
                , outimg2
                , outimg3
                ] 
            for img in image_list:
                cv2.line(img, (0,  53), (160,  53), (255, 0, 0), thickness=1)
                cv2.line(img, (0, 107), (160, 107), (255, 0, 0), thickness=1)
                cv2.line(img, ( 53, 0), ( 53, 160), (255, 0, 0), thickness=1)
                cv2.line(img, (107, 0), (107, 160), (255, 0, 0), thickness=1)
            image_list_wsep = interleave_in_list(l=image_list, sep=black_row)
            
            cv2.imshow("window", cv2.resize(np.concatenate(image_list_wsep, 1), (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
            k = cv2.waitKey(100)
            if k == 112:
                k = -1
                while True:
                    k = cv2.waitKey(100)
                    if k == 112:
                        break

            if done:
                print('Done??')
                break



