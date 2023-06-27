import gymnasium as gym
from gymnasium.wrappers import TransformReward
from stable_baselines3 import DQN
import json,os
import utils

from stable_baselines3.common.evaluation import evaluate_policy



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
   args = utils.get_args('DQN','CartPole-v1',wrapper = wrapper)
   # args = utils.get_args('DQN','CartPole-v1')

   # utils.train(args)
   utils.test_human(args)
   
   # model = DQN.load(args['save_path'])
   # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
   # print(f'mean_reward: "{mean_reward}, std_reward: {std_reward}')






   
   

