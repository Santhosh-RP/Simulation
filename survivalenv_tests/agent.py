from cmath import log
from subprocess import call
import sys

import numpy as np

from stable_baselines3 import DDPG, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import gymnasium
import cv2
import survivalenv

from mycallback import MyVideoSavingCallback  #MyCallback,

MODEL_TYPE = None
for arg in sys.argv:
    if arg.startswith("model="):
        split = arg.split('=')
        MODEL_TYPE = split[1].upper()
if MODEL_TYPE == None:
    print("Please, specify the baseline [model=ppo|td3|ddpg]")
    sys.exit(1)

SAVE_IDENTIFIER = None
for arg in sys.argv:
    if arg.startswith("id="):
        split = arg.split('=')
        SAVE_IDENTIFIER = split[1]
if SAVE_IDENTIFIER == None:
    print("Please, specify the save identifier [id=\"XXX\"]")
    sys.exit(1)


# env = gymnasium.make('survivalenv/SurvivalEnv-v0', render_cameras=True)
env = gymnasium.make('survivalenv/SurvivalEnv-v0', render_cameras=True, render_view=True)

# print(f'action space: {env.action_space}')
# print(f'observation space: {env.observation_space}')
# env = survivalenv.survivalenv.SurvivalEnv(render_cameras=True, render_view=False)
# env = VAEObservationWrapper(env)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# net_arch = [256, dict(vf=[256, 256], pi=[256, 256])]
net_arch = dict(pi=[128,128,128], vf=[128,128,128])
policy_kwargs = dict(net_arch=net_arch)


if MODEL_TYPE == "DDPG":
    model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1, buffer_size=1000000, tensorboard_log="runs")
elif MODEL_TYPE == "TD3":
    model = TD3("MultiInputPolicy", env, action_noise=action_noise, verbose=1, buffer_size=1000000, tensorboard_log="runs")
elif MODEL_TYPE == "PPO":
    # Originally, clip_range was 0.15. I changed it to 0.05 on 2024-04-03
    model = PPO("MultiInputPolicy", env, verbose=1, clip_range=0.05, tensorboard_log="runs", policy_kwargs=policy_kwargs)
    #A model = PPO("MultiInputPolicy", env, verbose=1, clip_range=0.15, clip_range_vf=0.85, tensorboard_log="runs")
    #B model = PPO("MultiInputPolicy", env, verbose=1, clip_range=0.15, clip_range_vf=0.85, tensorboard_log="runs", policy_kwargs=policy_kwargs)
else:
    print("Unhandled MODEL_TYPE")
    sys.exit(1)

total_timesteps=20*100_000
log_interval=1
# callback = MyCallback(env)
callback = MyVideoSavingCallback(env)

DO_TRAINING = True

if DO_TRAINING:
    # model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    model.learn(total_timesteps=total_timesteps, log_interval=log_interval, callback=callback)
    model.save(MODEL_TYPE)

env = model.get_env()
del model

if MODEL_TYPE == "DDPG":
    model = DDPG.load(f"{MODEL_TYPE}")
elif MODEL_TYPE == "TD3":
    model = TD3.load(f"{MODEL_TYPE}")
elif MODEL_TYPE == "PPO":
    model = PPO.load(f"{MODEL_TYPE}")



obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    left_image = obs["left"]
    right_image = obs["right"]
    
    try:
        cv2.imshow("vision", np.concatenate((left_image, right_image), axis=1))
    except Exception as e:
        print("error, e")
        pass

    if True:
        k = cv2.waitKey(1)
        if k%255 == 27:
            obs = env.reset()

    # env.render()

