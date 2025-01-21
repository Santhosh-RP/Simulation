import copy
from typing import Optional, Union

import time
import os
from pickletools import uint8
import sys
import numpy as np

from pathlib import Path

from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
import gymnasium.envs.mujoco as mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

import numpy as np
import random


import torch

np.random.seed(13)

from ..helper import Helper
from pathlib import Path


SAVE = False

WIDTH = 160
HEIGHT = 160
RENDER_FPS = 50
MAX_STEPS = 15000
FRAME_SKIP = 10
ROBOT_SPAWN_MAX_DISTANCE = 3
FOOD_SPAWN_MAX_DISTANCE = 10
FOOD_SPAWN_MIN_DISTANCE = 0.2
FOOD_DISTANCE_THRESHOLD = 0.6
FOOD_ITEMS = 10 # Changed to 11 for the Helper Food
NUMBER_OF_JOINTS = 2

INIT_HEALTH = 1000
MAX_HEALTH = 2000


MIN_CURRICULUM_LEARNING_WAIT = 500
MAX_CURRICULUM_LEARNING_WAIT = 15000
INC_CURRICULUM_LEARNING_WAIT = 1

MIN_CURRICULUM_LEARNING_DIST = 0.25
MAX_CURRICULUM_LEARNING_DIST = 1.25
INC_CURRICULUM_LEARNING_DIST = (MAX_CURRICULUM_LEARNING_DIST-MIN_CURRICULUM_LEARNING_DIST)/1000


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

DEBUG = 'debug' in sys.argv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base = Path(__file__).parents[1]
default_base = str(base / "assets" / "final_file.xml")

class SurvivalEnv(MujocoEnv, utils.EzPickle):
    metadata = { "render_modes": [ "human", "rgb_array", "depth_array", ], "render_fps": RENDER_FPS, }

    def __init__(self, xml_file=default_base, **kwargs):
        utils.EzPickle.__init__(**locals())
        if SAVE is True: 
            self.rdata = []
        self.episodes = 0
        self.height = HEIGHT
        self.width = WIDTH
        self._number_of_joints = NUMBER_OF_JOINTS

        try:
            self.render_cameras = kwargs["render_cameras"]
        except:
            self.render_cameras = False
        try:
            self.render_view = kwargs["render_view"]
        except:
            self.render_view = False

        # For testing transform poses, sets agent and helper positions
        self.initial_pos_set = False

        self.right_img, self.left_img, self.top_img = None, None, None

        self.observation_space = spaces.Dict({
            # "joint_position": spaces.Box(-1., 1., shape=(NUMBER_OF_JOINTS,), dtype=float),
            "joint_velocity": spaces.Box(-1., 1., shape=(NUMBER_OF_JOINTS,), dtype=np.float32),
            "gyro":           spaces.Box(-1., 1., shape=(3,),   dtype=np.float32),
            "accelerometer":  spaces.Box(-1., 1., shape=(3,),   dtype=np.float32),
            "magnetometer":   spaces.Box(-1., 1., shape=(3,),   dtype=np.float32),
            "health":         spaces.Box(-1., 1., shape=(1,),   dtype=np.float32),
        })
        if self.render_cameras is True:
            self.observation_space["left_image"] = spaces.Box(0, 255, shape=(HEIGHT, WIDTH, 3), dtype=np.float32)
            self.observation_space["right_image"] = spaces.Box(0, 255, shape=(HEIGHT, WIDTH, 3), dtype=np.float32)
        
        self.ignore_contact_names = ['floor', 'gviewL1', 'gviewL', 'gviewR1', 'gviewR']
        self.ignore_contact_ids = []

        # self.images = { "right_view": np.zeros((HEIGHT,WIDTH,3)), "left_view": np.zeros((HEIGHT,WIDTH,3))}
        self.images = { "left_image": np.zeros((HEIGHT,WIDTH,3), dtype=np.float32),
                        "right_image": np.zeros((HEIGHT,WIDTH*3), dtype=np.float32)}

        MujocoEnv.__init__(self, xml_file, FRAME_SKIP, observation_space=self.observation_space, width=self.width, height=self.height)
        self.mujoco_renderer = MujocoRenderer(self.model, self.data)
        try:
            self.render_mode = kwargs["render_mode"]
        except:
            self.render_mode = "none"

        # Creation of Helper object
        self.helper = Helper("food_free_11", MIN_CURRICULUM_LEARNING_DIST)
        self.curriculum_wait = MIN_CURRICULUM_LEARNING_WAIT

        self.reset_model()

    def get_steps(self):
        return int(self._steps)

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low[:-2], high=high[:-2], dtype=np.float32)
        # self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    @property
    def done(self):
        return self._done

    def respawn_food(self, food_idx, avoid_locations=None, threshold=0.0):
        def get_random_position():
            WEIRD = False
            if WEIRD:
                distance = random.random()+FOOD_SPAWN_MAX_DISTANCE/4
                angle = (random.random()-0.5)*2.
                angle = angle*0.7           # This spawns the food in a weird way that points forward
                angle *=np.pi
            else:
                distance = random.random()*(FOOD_SPAWN_MAX_DISTANCE-FOOD_SPAWN_MIN_DISTANCE)+FOOD_SPAWN_MIN_DISTANCE
                angle = (random.random()-0.5)*2.*np.pi

            height = np.random.normal()/4+1.5
            height = np.clip(height, 1.0, 2.0)

            p = np.array([ -np.cos(angle)*distance, +np.sin(angle)*distance, random.random()+1.3])
            return p
        if avoid_locations is None:
            p = get_random_position()
        else:
            all_locations_distant = False
            while all_locations_distant is False:
                p = get_random_position()
                all_locations_distant = True
                for location in avoid_locations:
                    dist = np.linalg.norm(p[:2]-location[:2])
                    if dist < FOOD_DISTANCE_THRESHOLD:
                        all_locations_distant = False
                        break

        body_name = 'food_body_'+str(food_idx+1)
        geom_name = 'food_geom_'+str(food_idx+1)
        joint_name = 'food_free_'+str(food_idx+1)
        for axis in range(3):
            self.data.joint(joint_name).qpos[axis] = p[axis]
            self.data.body(body_name).xpos[axis] = p[axis]
            self.data.geom(geom_name).xpos[axis] = p[axis]
            self.data.body(body_name).xipos[axis] = p[axis]
            self.model.geom(geom_name).pos[axis] = p[axis]
        return p

    def respawn_robot(self):
        distance_p = random.random()*ROBOT_SPAWN_MAX_DISTANCE
        angle_p = random.random()*2.*np.pi
        height = np.random.normal()/4+1.2
        height = np.clip(height, 0.5, 1.9)
        p = [-np.cos(angle_p)*distance_p, +np.sin(angle_p)*distance_p, height]
        orientation = random.random()*2.*np.pi
        self.data.joint("root").qpos[:] = p + [np.sin(orientation/2), 0, 0, np.cos(orientation/2)]
        return p


    def handle_food(self):
        for contact in self.data.contact:
            if contact.geom1 in self.ignore_contact_ids:
                continue
            elif contact.geom2 in self.ignore_contact_ids:
                continue
            else:
                g1 = self.data.geom(contact.geom1)
                if g1.name in self.ignore_contact_names:
                    self.ignore_contact_ids.append(contact.geom1)
                    continue
                else:
                    g2 = self.data.geom(contact.geom2)
                    if g2.name in self.ignore_contact_names:
                        self.ignore_contact_ids.append(contact.geom2)
                        continue

                beak, food = sorted([g1.name, g2.name])
                if beak != "beak":
                    continue
                if food.startswith("food"):
                    food = int(food.split('_')[-1])
                    self.respawn_food(food-1)
                    if food == 11:
                        self.helper.set_success()
                    return True
        return False

    # Placing agent & helper into test positions
    def set_test_pos(self, agent_pos, helper_pos):
        # Setting x and y positions of agents
        self.data.joint("root").qpos[:2] = agent_pos
        self.data.joint("Hroot").qpos[:2] = helper_pos

        # # Setting angle of direction agents are facing
        # # qpos[3]=0 makes angle left
        self.data.joint("root").qpos[3] = np.sin((np.pi * 0.75) /2)
        self.data.joint("Hroot").qpos[3] = np.sin((np.pi/4)/2)

        # # qpos[6]=0 makes face right
        self.data.joint("root").qpos[6] = np.cos((np.pi * 0.75)/2)
        self.data.joint("Hroot").qpos[6] = np.cos((np.pi/4)/2)


    # Gathering data to test movement avoidance methods repulse and attract
    # Includes placing agent and food in static positions
    def movement_avoidance_test(self):
        # Contains the x and y coordiantes of each food
        self.data.joint('food_free_1').qpos[:2] = [0, 4]
        self.data.joint('food_free_2').qpos[:2] = [1.5, 4]
        self.data.joint('food_free_3').qpos[:2] = [0, 8]
        self.data.joint('food_free_4').qpos[:2] = [-1.5, 8]
        self.data.joint('food_free_5').qpos[:2] = [-7, 4]
        self.data.joint('food_free_6').qpos[:2] = [-7, 6]
        self.data.joint('food_free_7').qpos[:2] = [-7, 8]
        self.data.joint('food_free_8').qpos[:2] = [-7, 10]
        self.data.joint('food_free_9').qpos[:2] = [-7, 12]
        self.data.joint('food_free_10').qpos[:2] = [-7, 14]
        self.data.joint('root').qpos[:2] = [0, 12]
        self.data.joint('root').qpos[3:] = [0.38268343, 0, 0, 0.92387953]
        self.data.joint('Hroot').qpos[:2] = [0, 0]
        self.data.joint('Hroot').qpos[3:] = [0.70710678, 0, 0, 0.70710678]

    def movement_avoidance_test2(self, difficulty_level):
        # Base positions for all levels
        self.data.joint('root').qpos[:2] = [0, 8]
        self.data.joint('root').qpos[3:] = [0.38268343, 0, 0, 0.92387953]
        self.data.joint('Hroot').qpos[:2] = [0, 0]
        self.data.joint('Hroot').qpos[3:] = [0.70710678, 0, 0, 0.70710678]
        self.data.joint('food_free_1').qpos[:2] = [0, 4]
        self.data.joint('food_free_2').qpos[:2] = [1.5, 4]
        self.data.joint('food_free_3').qpos[:2] = [-1, 3]
        self.data.joint('food_free_4').qpos[:2] = [2, 3]
        self.data.joint('food_free_5').qpos[:2] = [-2, 2]
        self.data.joint('food_free_6').qpos[:2] = [3, 2]
        self.data.joint('food_free_7').qpos[:2] = [-3, 1]
        self.data.joint('food_free_8').qpos[:2] = [4, 1]
        self.data.joint('food_free_9').qpos[:2] = [-4, 0]  # Dead end requiring backtracking
        self.data.joint('food_free_10').qpos[:2] = [5, 0]

    def movement_avoidance_test3(self):
        # Contains the x and y coordiantes of each food
        self.data.joint('food_free_1').qpos[:2] = [0, 4]
        self.data.joint('food_free_2').qpos[:2] = [1.5, 4]
        self.data.joint('food_free_3').qpos[:2] = [-1.5, 4]
        self.data.joint('food_free_4').qpos[:2] = [-3, 4]
        self.data.joint('food_free_5').qpos[:2] = [-4.5, 4]
        self.data.joint('food_free_6').qpos[:2] = [-7, 6]
        self.data.joint('food_free_7').qpos[:2] = [-7, 8]
        self.data.joint('food_free_8').qpos[:2] = [-7, 10]
        self.data.joint('food_free_9').qpos[:2] = [-7, 12]
        self.data.joint('food_free_10').qpos[:2] = [-7, 14]
        self.data.joint('root').qpos[:2] = [0, 8]
        self.data.joint('root').qpos[3:] = [0.38268343, 0, 0, 0.92387953]
        self.data.joint('Hroot').qpos[:2] = [0, 0]
        self.data.joint('Hroot').qpos[3:] = [0.70710678, 0, 0, 0.70710678]

    def collision_test_start(self):
        self.data.joint('root').qpos[:2] = [2, 0]
        self.data.joint('root').qpos[3:] = [0.38268343, 0, 0, 0.92387953]
        self.data.joint('Hroot').qpos[:2] = [-2, 0]
        self.data.joint('Hroot').qpos[3:] = [0.92387953, 0, 0, 0.38268343]
        self.data.joint('food_free_1').qpos[:2] = [-7, 0]
        self.data.joint('food_free_2').qpos[:2] = [-7, 2]
        self.data.joint('food_free_3').qpos[:2] = [-7, 4]
        self.data.joint('food_free_4').qpos[:2] = [-7, 6]
        self.data.joint('food_free_5').qpos[:2] = [-7, 8]
        self.data.joint('food_free_6').qpos[:2] = [-7, 10]
        self.data.joint('food_free_7').qpos[:2] = [-7, 12]
        self.data.joint('food_free_8').qpos[:2] = [-7, 14]
        self.data.joint('food_free_9').qpos[:2] = [-7, 16]
        self.data.joint('food_free_10').qpos[:2] = [-7, 18]
        self.data.joint('food_free_11').qpos[:2] = [-7, 20]

    def collision_test_actions(self):
        helper_action = np.array([1, 1])
        learner_action = np.array([1, 1])
        return helper_action, learner_action

    def helper_waiting_test_start(self):
        self.data.joint('food_free_1').qpos[:2] = [-7, 0]
        self.data.joint('food_free_2').qpos[:2] = [-7, 2]
        self.data.joint('food_free_3').qpos[:2] = [-7, 4]
        self.data.joint('food_free_4').qpos[:2] = [-7, 6]
        self.data.joint('food_free_5').qpos[:2] = [-7, 8]
        self.data.joint('food_free_6').qpos[:2] = [-7, 10]
        self.data.joint('food_free_7').qpos[:2] = [-7, 12]
        self.data.joint('food_free_8').qpos[:2] = [-7, 14]
        self.data.joint('food_free_9').qpos[:2] = [-7, 16]
        self.data.joint('food_free_10').qpos[:2] = [-7, 18]
        self.data.joint('food_free_11').qpos[:2] = [-7, 20]


    def step(self, action):
        self.last_action = copy.deepcopy(action)
        # Uncommenting this makes the simulation real-time-ish
        # while time.time()-self._timestamp<1./RENDER_FPS:
            # time.sleep(0.002)
    
        self._timestamp = time.time()

        # We get the action that the helper takes
        helper_action = np.array(self.helper.get_action(self.data, [0, 10], self.curriculum_wait))
        
        combined_action = np.concatenate((action*30, helper_action))
        self.do_simulation(combined_action,  self.frame_skip)




        got_food = self.handle_food()
        if got_food:
            self._health += INIT_HEALTH
            if self._health > MAX_HEALTH:
                self._health = MAX_HEALTH
            reward = float(INIT_HEALTH)/float(MAX_STEPS)
        else:
            reward = 0.

        self._steps += 1
        self._health -= 1
        if self._steps >= MAX_STEPS or self._health < 0:
            self._done = True
        self.terminated = self._done

        # Reward calculation
        current_xpos = np.array(self.data.body("torso").xpos)
        if self.previous_xpos is None:
            self.previous_xpos = np.array(current_xpos)
        xpos_inc = current_xpos - self.previous_xpos
        energy = np.sum(np.abs(np.sum(action)/100))
        # FORWARD_REWARD = False
        # if FORWARD_REWARD:
        #     if xpos_inc[0] > 0:
        #         reward = xpos_inc[0]/(energy+1)
        #     else:
        #         reward = xpos_inc[0]*(energy+1)
        # else:
        #     reward = xpos_inc[0]
        self.previous_xpos = np.array(current_xpos)


        # observation
        if self.render_cameras:
            for camera in ["left_image", "right_image"]:
                self.camera_name = camera
                self.render_mode = "rgb_array"
                self.images[self.camera_name] = self.render().astype(np.float32)

        observation = self._get_obs()

        self.info = {
            "done": self._done,
            "reward": np.float32(reward),
            "energy": np.float32(energy),
            "delta": np.float32(xpos_inc[0])
        }

        # render
        if self.render_view:
            self.render_mode = "human"
            self.camera_name = None
            self.render()

        self.reward = np.float32(reward)


        if SAVE is True:
            self.rdata.append({'observation': observation, 'action': action, 'reward': reward, 'done': self._done, 'terminated': self.terminated, 'info': self.info})
        if self.terminated or self._done:
            if SAVE is True:
                spath = f'/dl/episode_{str(self.episodes+201).zfill(6)}.npz'
                fd = open(spath, 'wb')
                np.savez_compressed(fd, data=self.rdata)
                fd.close()
                self.rdata = []
            self.episodes += 1

        return observation, reward, self._done, self.terminated, self.info

    def _get_obs(self):
        def normalise(array, maximum):
            return np.clip(array/maximum, -1, 1)

        gyro = 3
        accelerometer = 3
        magnetometer = 3
        joint_pos_offset = 0
        joint_vel_offset = joint_pos_offset + NUMBER_OF_JOINTS
        gyro_offset = joint_vel_offset + NUMBER_OF_JOINTS
        accelerometer_offset = gyro_offset + gyro
        magnetometer_offset = accelerometer_offset + accelerometer
        self.last_obs = {
            # "joint_position": normalise(np.array(self.data.sensordata[joint_pos_offset:joint_vel_offset]), np.pi/2),
            "joint_velocity": normalise(np.array(self.data.sensordata[joint_vel_offset:gyro_offset]), 20).astype(np.float32),
            "gyro":           normalise(np.array(self.data.sensordata[gyro_offset:accelerometer_offset]), 8).astype(np.float32),
            "accelerometer":  normalise(np.array(self.data.sensordata[accelerometer_offset:magnetometer_offset]), 100).astype(np.float32),
            "magnetometer":   normalise(np.array(self.data.sensordata[magnetometer_offset:]), 1.).astype(np.float32),
            "health":         normalise(np.array([self._health]), MAX_HEALTH).astype(np.float32),
            }
        self.last_obs_wo_images = dict(self.last_obs)
        if self.render_cameras is True:
            self.last_obs["left_image"] = self.images["left_image"]
            self.last_obs["right_image"] = self.images["right_image"]

        return self.last_obs


    def respawn_all_food(self):
        avoid_locations = []
        for i in range(FOOD_ITEMS):
            avoid_locations.append(self.respawn_food(i, avoid_locations, FOOD_DISTANCE_THRESHOLD))


    def reset_model(self):
        self.previous_xpos = None
        self._steps = 0
        self._health = INIT_HEALTH
        self._done = False

        self.respawn_robot()
        self.respawn_all_food()

        # self.movement_avoidance_test()
        # self.movement_avoidance_test2()
        # self.movement_avoidance_test3()
        # self.collision_test_start()
        # self.helper.reset()

        self._steps = 0
        self._done = False
        self._timestamp = time.time()
        self.previous_xpos = None
        observation = self._get_obs()
        return observation


    def render(self):
        return self.mujoco_renderer.render(self.render_mode, self.camera_id, self.camera_name)


    def close(self):
        self.mujoco_renderer.close()


    def reset(self, **kwargs):
        super().reset(**kwargs)

        observation, _, done, terminated, info = self.step(np.zeros(NUMBER_OF_JOINTS))

        while done is True or terminated is True:
            # In case the agent dies in the first step
            observation, _, done, terminated, info = self.step(np.zeros(NUMBER_OF_JOINTS))

        if SAVE is True: 
            self.rdata.append({'observation': observation, 'action': None,   'reward': None,   'done': done,       'terminated': terminated,      'info': info})

        self.curriculum_wait = min(
            self.curriculum_wait + INC_CURRICULUM_LEARNING_WAIT,
            MAX_CURRICULUM_LEARNING_WAIT)

        self.helper.curriculum_learning_dist = min(
            self.helper.curriculum_learning_dist * (1 + INC_CURRICULUM_LEARNING_DIST),
            MAX_CURRICULUM_LEARNING_DIST)

        return observation, info


