import gymnasium as gym
from gymnasium.wrappers import TransformReward
from stable_baselines3 import DQN
import json,os
import utils,envs
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from stable_baselines3.common.evaluation import evaluate_policy
import sys
import time
import logging
import argparse
import itertools
from six import StringIO
from random import sample, randint

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from IPython import display
import matplotlib
import matplotlib.pyplot as plt


class wrapper(gym.Wrapper):
   def __init__(self, env):
      super().__init__(env)
      self.env = env
   
   def step(self, action): 
      observation, reward, terminated, truncated, info = self.env.step(action)
      return observation, self.reward(reward,observation), terminated, truncated, info
   

   def reward(self, reward,state):
      return reward - 2*(abs(state[0]) + abs(state[1]) + abs(state[2]) )



def test():
   pass


if __name__ == "__main__":
   global args
   env_name = 'game2048'

   # args = utils.get_args('DQN',env_name,wrapper)
   args = utils.get_args('DQN',env_name,envs.make)

   # utils.train(args)

   # %%%  
   # model = DQN.load(args['save_path'])
   # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
   # print(f'mean_reward: "{mean_reward}, std_reward: {std_reward}')
   # %%%  
   # utils.test_custom_human(args)

#加载模型
   # %%%%%% 
   device = 'cuda'
   model = DQN.load(args['save_path'])
   env = args['model_args']['env']

   state,info = env.reset()
   img = plt.imshow(env.render(mode='rgb_array'))
   while True:
      plt.axis("off")
      img.set_data(env.render(mode='rgb_array'))
      display.display(plt.gcf())
      display.clear_output(wait=True)
      action,_state = model.predict(state)
      # take action
      s_, r, done, _,info = env.step(action)
      time.sleep(0.1)
      if done:
         break
      state = s_
   env.close()
   plt.close()




   
   


# %%
