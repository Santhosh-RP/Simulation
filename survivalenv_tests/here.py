import os
import sys
import cv2
import math
import numpy as np

import survivalenv
import gymnasium as gym

import time
import glob
import random
import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from vae import VAE

import cma

PUBLISH = True

ENV_NAME = 'survivalenv'
EPISODES_PER_GENERATION = 15
GENERATIONS = 10000
POPULATION_SIZE = 20
SIGMA=1.
SAVE_PATH = "./"


ZDIM = 150
MAX_VAE = 10

RSSM_Z_VALUES = []
RSSM_RD_VALUES = []
RSSM_A_VALUES = []
RSSM_batch_size=50
RSSM_max_dataset_size=10000
RSSM_sequence_length=2000
RSSM_num_of_epochs = 5000
RSSM_learning_rate = 0.001
RSSM_warmup_minibatches=10

TEST_STUFF = False
if TEST_STUFF:
    RSSM_batch_size=5
    RSSM_sequence_length=50
    RSSM_warmup_minibatches=1
    POPULATION_SIZE=3


znet = VAE(zDim=ZDIM).to(device)
znet.load_state_dict(torch.load('vae_t4_b_.pth'))


class RSSM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim, num_layers=1):
        super(RSSM, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.h = None


    def forward(self, x, online=False):
        if online is True:
            x, self.h = self.rnn(x, self.h)
        else:
            x, self.h = self.rnn(x)
        x = self.fc(x)
        return x

# Instantiation of the RSSM. The observation size is 162, the (reward+done) would be 2.
# The action however is not predicted, therefore the input is 162+2+2 but the output is
# smaller (without the action) i.e. 162+2.
rssm = RSSM(164+2, 512, 164, num_layers=2).to(device)


class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_shape, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 32)
        self.lout = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.elu(self.l1(x.float()))
        x = F.elu(self.l2(x))
        x = F.elu(self.l3(x))
        return torch.tanh(self.lout(x))

    def get_params(self):
        p = np.empty((0,))
        for n in self.parameters():
            p = np.append(p, n.flatten().cpu().detach().numpy())
        return p

    def set_params(self, x):
        start = 0
        for p in self.parameters():
            e = start + np.prod(p.shape)
            p.data = torch.FloatTensor(x[start:e]).reshape(p.shape)
            start = e

def evaluate(ann, env, seed, render=False, wait_after_render=False):
    global RSSM_Z_VALUES
    global RSSM_RD_VALUES
    global RSSM_A_VALUES
    RSSM_Z_VALUES = []
    RSSM_RD_VALUES = []
    RSSM_A_VALUES = []
    obs = env.reset()
    obs = preprocess_observation(obs[0])
    total_reward = 0
    while True:
        if render is True:
            env.render()
        # Output of the neural net
        net_output = ann(obs)
        # the action is the value clipped returned by the nn
        action = net_output.data.cpu()

        obs, reward, done, _, _ = env.step(action.numpy())
        obs = preprocess_observation(obs)
        rd = torch.tensor(np.array([reward, done]))
        RSSM_Z_VALUES.append(obs)
        RSSM_RD_VALUES.append(rd)
        RSSM_A_VALUES.append(action)
        # print(f'{obs.shape=} {rd.shape=} {action.shape=}')
        if len(RSSM_A_VALUES)>=RSSM_sequence_length:
            break
        total_reward += reward
        if done:
            break
    if render and wait_after_render:
        for i in range(2):
            env.render()
            time.sleep(1)
    
    RSSM_Z_VALUES =  np.concatenate(RSSM_Z_VALUES,  0).reshape((-1, RSSM_Z_VALUES[0].shape[0]))
    RSSM_RD_VALUES = np.concatenate(RSSM_RD_VALUES, 0).reshape((-1, RSSM_RD_VALUES[0].shape[0]))
    RSSM_A_VALUES =  np.concatenate(RSSM_A_VALUES,  0).reshape((-1, RSSM_A_VALUES[0].shape[0]))
    return total_reward


def fitness(candidate, env, seed, render=False):
    ann.set_params(candidate)
    return -evaluate(ann, env, seed, render)

def preprocess_observation(obs):
    image = cv2.resize(obs["view"], (128, 128))
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)/255
    image.unsqueeze_(0)
    image = image.to(device)
    z_gpu = znet.get_z(image.to(device))
    z = z_gpu.detach().cpu()
    z_gpu[z_gpu < -MAX_VAE] = -MAX_VAE
    z_gpu[z_gpu > +MAX_VAE] = +MAX_VAE
    outimg = (znet.decoder(z_gpu).squeeze().movedim(0,2).detach().cpu().numpy()*255).astype(np.uint8)

    # out, mu, logVAR = znet(image, mle=True)
    # z = mu.detach().cpu()
    # outimg = np.transpose(out[0].detach().cpu().numpy(), [1,2,0])*255
    # outimg = outimg.astype(np.uint8)

    outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)
    # cv2.imshow("window", outimg)
    # cv2.waitKey(1)
    return torch.cat([torch.tensor(obs[o].copy()).view(-1) for o in sorted(obs.keys()) if o != 'view'] + [z.view(-1)], 0)




def train_with_cma(generations, writer_name):
    es = cma.CMAEvolutionStrategy(len(ann.get_params())*[0], SIGMA, {'popsize': POPULATION_SIZE})
    # es = cma.CMAEvolutionStrategy(len(ann.get_params())*[0], SIGMA, {'popsize': POPULATION_SIZE, 'seed': 123})
    best = 0

    def reps(generation):
        if generation < 20:
            return 1
        elif generation < 50:
            return 2
        elif generation < 100:
            return 4
        elif generation < 200:
            return 5
        elif generation < 300:
            return 6
        elif generation < 400:
            return 7
        elif generation < 500:
            return 8
        else:
             return EPISODES_PER_GENERATION

    def scale(generation):
        return 1

    optimizer = torch.optim.Adam(rssm.parameters(), lr=RSSM_learning_rate)
    loss = nn.MSELoss()

    all_values_accum = None
    RSSM_epoch_number = 0

    fill = torch.zeros((1,1,162+2+2))
    fill[0][0][162+1] = 1
    for generation in range(generations):
        global RSSM_Z_VALUES
        global RSSM_RD_VALUES
        global RSSM_A_VALUES

        all_values = []

        seeds = [random.getrandbits(32) for _ in range(reps(generation))]
        # Create population of candidates and evaluate them
        candidates, fitnesses , Maxfitnesses = es.ask(sigma_fac=scale(generation)), [],[]
        for candidate in candidates:
            reward = 0
            for seed in seeds:
                # Evaluate the agent using stable-baselines predict function
                reward += fitness(candidate, env, seed, render=False) 
            average_candidate_reward = reward / reps(generation)
            fitnesses.append(average_candidate_reward)
            Maxfitnesses.append(-average_candidate_reward)
            this_one = torch.cat((torch.tensor(RSSM_Z_VALUES), torch.tensor(RSSM_RD_VALUES), torch.tensor(RSSM_A_VALUES)), 1).unsqueeze(0)
            if this_one.shape[1]<RSSM_sequence_length:
                fill_amount = RSSM_sequence_length-this_one.shape[1]
                this_fill = fill.repeat(1, fill_amount, 1)
                this_one = torch.cat((this_one, this_fill), 1)
            all_values.append(this_one)


        all_values = torch.cat(all_values, 0).type(torch.float32)
        if all_values_accum is None:
            all_values_accum = all_values
        else:
            all_values_accum = torch.cat((all_values_accum, all_values), 0)

        print(f'RSSM dataset size{len(all_values_accum)=}')
        if all_values_accum.shape[0] > RSSM_max_dataset_size:
            perm = torch.randperm(all_values_accum.size(0))
            idx = perm[:RSSM_max_dataset_size]
            all_values_accum = all_values_accum[idx]

        torch.save(all_values_accum, 'rssm_dataset.pth')
        if all_values_accum.shape[0]>=RSSM_batch_size*RSSM_warmup_minibatches:
            # We don't want to start training too early because it may overfit and not be able to get out of a local minima?
            perm = torch.randperm(all_values_accum.size(0))
            avg_loss = 0
            grads = {}
            for batch in range(int(math.floor(all_values_accum.shape[0]/RSSM_batch_size))):
                idx = perm[batch*RSSM_batch_size:(batch+1)*RSSM_batch_size]
                batch_data = all_values_accum[idx]
                X = batch_data[:, 0:-1, :].to(device)
                TY = batch_data[:, 1:,   :-2].to(device) # -2 because the action is of size 2 (two wheels)
                y = rssm(X)
                L = loss(y, TY)
                optimizer.zero_grad()
                L.backward()

                for name, param in rssm.named_parameters():
                    if not name in grads.keys():
                        grads[name+"_grad"] = 0
                    grads[name+"_grad"] += torch.sum(param.abs())

                avg_loss += L
                optimizer.step()
            avg_loss /= (batch+1)/RSSM_batch_size
      
            print(f"\rRSSM Epoch: {RSSM_epoch_number}/{RSSM_num_of_epochs} \tL={avg_loss}")
            RSSM_epoch_number += 1
            wandb.log(dict({"epoch": RSSM_epoch_number,
                        "train_loss": avg_loss,
                        "dataset_size": len(all_values_accum)}, **grads))
            torch.save(rssm.state_dict(), f'rssm_{str(RSSM_epoch_number).zfill(6)}.pth')

        
        # CMA-ES update
        es.tell(candidates, fitnesses)

        # Display some training infos
        mean_fitness = np.mean(sorted(fitnesses)[:int(0.1 * len(candidates))])
        print("Iteration {:<3} Mean top 10% reward: {:.2f}".format(generation, -mean_fitness))
        cur_best = max(Maxfitnesses)
        best_index = np.argmax(Maxfitnesses)
        # print("current  value {}...".format(cur_best))
        writer.add_scalar('performance/mean top 10 reward', -mean_fitness, generation)
        writer.add_scalar('params/reps', reps(generation), generation)
        writer.add_scalar('params/sigma', SIGMA*scale(generation), generation)

        best_params = candidates[best_index]
        render_the_test = os.path.exists("render")
        seeds = [random.getrandbits(32) for _ in range(EPISODES_PER_GENERATION)]
        test_rew = 0
        ann.set_params(best_params)
        for seed in seeds:
            test_rew += evaluate(ann, env, seed, render=render_the_test)
        test_rew /= EPISODES_PER_GENERATION
        writer.add_scalar('performance/test reward', test_rew, generation)
        # Save model if it's best
        if not best or test_rew >= best:
            best = test_rew
            print("Saving new best with value {}...".format(best))
            torch.save(ann.state_dict(), writer_name+'_BEST.pth')
        # Saving model every 
        if (generation+1)%5 == 0:
            try:
                torch.save(ann.state_dict(), os.path.join(SAVE_PATH, writer_name+"_gen_" + str(generation+1).zfill(8) + ".pth"))
            except:
                print("Error in saving model")

    print('best reward : {}'.format(best))


if __name__ == '__main__':

    # if len(sys.argv)>2 and sys.argv[1] == '-test':
    env = gym.make('survivalenv/SurvivalEnv-v0', render_cameras=True) #, render_view=True
    # env = survival.SurvivalEnv(render_cameras=True, render_view=True)
    # else:
        # env = survival.SurvivalEnv(render_cameras=False, render_view=False)
    n_joints = env.action_space.shape[0]
    raw_obs = env.reset()[0]
    obs = preprocess_observation(raw_obs)

    print(f'Joints {n_joints}')
    ann = NeuralNetwork(obs.shape[0], n_joints)
    print(f'NN {obs.shape[0]} {n_joints}')


    task_id = "rssm_lol3"
    if PUBLISH:
        import wandb
        wandb.init(project="RSSM", notes=f"r{task_id}", name=f"r{task_id}")
        wandb.config = {
            "task_id": task_id,
            "batch_size": RSSM_batch_size,
            "learning_rate": RSSM_learning_rate,
            "num_epochs": RSSM_num_of_epochs,
            "zdim": ZDIM
        }



    if len(sys.argv)>2 and sys.argv[1] == '-test':
        env.render_mode = "human"
        ann.load_state_dict(torch.load(sys.argv[2]))
        accumulated_reward = 0
        done = 0
        while True:
            reward = evaluate(ann, env, seed=random.getrandbits(32), render=True, wait_after_render=True)
            accumulated_reward += reward
            done += 1
            print(f'Reward: {reward}    (avg:{accumulated_reward/done})')
    else:
        # while not os.path.exists("start"):
            # time.sleep(1)

        if not os.path.isdir(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        np.random.seed(123)
        now = datetime.datetime.now()
        date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
        writer_name = f'cmaC3_{ENV_NAME}_pop{POPULATION_SIZE}_k{EPISODES_PER_GENERATION}_sigma{SIGMA}_{date_time}'
        writer = SummaryWriter(log_dir='runs/'+writer_name)

        try:
            train_with_cma(GENERATIONS, writer_name)
        finally:
            if PUBLISH:
                wandb.finish()


