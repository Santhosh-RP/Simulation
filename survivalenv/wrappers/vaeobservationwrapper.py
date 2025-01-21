import time
import copy
import torch
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium import spaces
from pathlib import Path
import gymnasium as gym
import urllib
import pygame

from .vae import VAE

from torchvision.transforms import Resize
from survivalenv.envs.survivalenv import SurvivalEnv, default_base
import cv2

import survivalenv.envs.survivalenv as survival

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE(zDim=128).to(device)

final_path = Path(__file__).parents[1] / "assets" / "vae.pth"

print(f'VAEObservationWrapper: Using VAE pth file: {final_path}')

if not final_path.is_file():
    # urllib.request.urlretrieve("https://github.com/ljmanso/survivalenv/releases/download/vae/vae.pth", "survival/assets/vae.pth")
    urllib.request.urlretrieve("https://www.dropbox.com/scl/fi/laulwuekxmztcfrs4wi5r/vae.pth?rlkey=k49ja8op7kxqp41z5872jh0nv&dl=1",
                               str(final_path))


vae.load_state_dict(torch.load(str(final_path)))
vae = vae.to('cpu')
device = 'cpu'

MAX_VAE = 4.


class VAEObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.resize = Resize(size=(128,128), antialias=True) 
        self.observation_space["view"] = spaces.Box(-10., 10., shape=(1, 32), dtype=float)
        self.window_initialised = False

    def observation(self, obs):
        with torch.no_grad():
            image = cv2.resize(obs["view"], (128, 128))
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)/255

            image.unsqueeze_(0)
            z_gpu = vae.get_z(image.to(device))
            z = z_gpu.cpu().numpy()
            z_gpu[z_gpu < -MAX_VAE] = -MAX_VAE
            z_gpu[z_gpu > +MAX_VAE] = +MAX_VAE
            rec = (vae.decoder(z_gpu).squeeze().movedim(0,2).cpu().numpy()*255).astype(np.uint8)
        obs["view"] = z

        if False:
            if not self.window_initialised:
                pygame.init()
                self.screen = pygame.display.set_mode((int(128), int(128)))
                pygame.display.set_caption("reconstruction")
                self.window_initialised = True
            surface = pygame.surfarray.make_surface(rec)
            surface_resized = pygame.transform.smoothscale(surface, (128,128))
            self.screen.blit(surface_resized, (0,0))
            pygame.display.flip()
            time.sleep(0.1)
        elif True:
            if not self.window_initialised:
                cv2.namedWindow("name", cv2.WINDOW_NORMAL)
                self.window_initialised = True
            cv2.imshow("name", rec[:, :, ::-1])
            cv2.waitKey(1)

        return obs
    


DEBUG_HISTOGRAMS = False

class Dict2SingleVectorWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(-1., 1., shape=(15+128,), dtype=float)
        self.maxs = dict()
        self.mins = dict()
        self.names = ["accelerometer", "gyro", "joint_position", "joint_velocity", "magnetometer", "view"]
        if DEBUG_HISTOGRAMS:
            self.histograms = {}
            self.bins = np.arange(-1.31, 1.32, 0.05)
            for name in self.names:
                self.histograms[name] = None
            self.iter = 0

    def observation(self, obs):
        obs["accelerometer"]  = obs["accelerometer"].reshape(1,-1)  / 15.
        obs["gyro"]           = obs["gyro"].reshape(1,-1)           / 4.
        obs["joint_velocity"] = obs["joint_velocity"].reshape(1,-1) / 10.
        obs["magnetometer"]   = obs["magnetometer"].reshape(1,-1)   / 1.
        obs["view"]           = obs["view"].reshape(1,-1)           / MAX_VAE
        obs["joint_position"] = np.concatenate([
                                    np.sin(obs["joint_position"].reshape(1,-1)),
                                    np.cos(obs["joint_position"].reshape(1,-1))
                                    ], axis=1)

        for k in self.names:
            v = obs[k]
            v[v < -1.] = -1.
            v[v > +1.] = +1.

            if DEBUG_HISTOGRAMS:
                if self.histograms[k] is None:
                    self.histograms[k], _ = np.histogram(v, bins=self.bins, density=True)
                else:
                    self.histograms[k] += np.histogram(v, bins=self.bins, density=True)[0]

                m = v.min()
                M = v.max()
                if k not in self.maxs:
                    self.mins[k] = m
                    self.maxs[k] = M
                else:
                    if m < self.mins[k]:
                        self.mins[k] = m
                    if M > self.maxs[k]:
                        self.maxs[k] = M

                if self.iter % 500 == 0:
                    from matplotlib import pyplot
                    for k in self.names:
                        pyplot.stairs(self.histograms[k], self.bins, label=k)
                    pyplot.legend(loc='upper right')
                    pyplot.show(block=True)
                self.iter += 1

        return np.concatenate([obs[x] for x in self.names], axis=1).squeeze()


def SurvivalVAE(xml_file=None, **kwargs):
    kwargs['render_cameras'] = True
    kwargs['render_view'] = True
    env = SurvivalEnv(xml_file=survival.default_base, **kwargs)

    wrapped_env = VAEObservationWrapper(env)
    return wrapped_env

def SurvivalVAEVector(xml_file=None, **kwargs):
    env = SurvivalVAE(xml_file=survival.default_base, **kwargs)
    wrapped_env = Dict2SingleVectorWrapper(env)
    return wrapped_env

