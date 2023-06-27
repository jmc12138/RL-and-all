import gymnasium as gym
from gymnasium.wrappers import TransformReward
from stable_baselines3 import DQN
import json,os
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np


import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback,CheckpointCallback,CallbackList



def ctn_train(args,model_path):
   env = args['model_args']['env']
   model = DQN(**args['model_args'])
   model.set_parameters(model_path)


   checkpoint_callback = CheckpointCallback(
   save_freq=args['save_freq'],
   save_path=args['save_path'],
   name_prefix=args['save_name'],
   save_replay_buffer=False,
   save_vecnormalize=True,
   verbose=2
)
   eval_callback = EvalCallback(env, best_model_save_path=args['save_path'],
                             eval_freq=args['save_freq'],
                             deterministic=True, render=False)
   
   model.learn(**args['learn_args'],callback = CallbackList([checkpoint_callback, eval_callback]))
   model.save(args['last_model_path'])


def train(args):
   env = args['model_args']['env']
   model = DQN(**args['model_args'])
   # model.set_parameters(model_path)

   print(args['save_freq'])
   print(args['save_path'])
   print(args['save_name'])

   checkpoint_callback = CheckpointCallback(
   save_freq=args['save_freq'],
   save_path=args['save_path'],
   name_prefix=args['save_name'],
   save_replay_buffer=False,
   save_vecnormalize=True,
   verbose=2
)
   eval_callback = EvalCallback(env, best_model_save_path=args['save_path'],
                             eval_freq=args['save_freq'],
                             deterministic=True, render=False)
   
   model.learn(**args['learn_args'],callback = CallbackList([checkpoint_callback, eval_callback]))
   model.save(args['last_model_path'])



def test_custom_human(args):

   model = DQN.load(args['last_model_path'])
   env = args['model_args']['env']

      #   env = VecFrameStack(env, n_stack=4)
   
   print("开始测试！")
   rewards = []  # 记录所有回合的奖励
   steps = []
   for i_ep in range(args['test_eps']):
      ep_reward = 0  # 记录一回合内的奖励
      ep_step = 0
      state,info = env.reset()  # 重置环境，返回初始状态
      env.render()
      for _ in range(args['test_max_steps']):
         ep_step+=1
         action,_state = model.predict(state)  # 选择动作
         next_state, reward, terminated, truncated, info = env.step(action)  # 更新环境，返回transition
         env.render()

         state = next_state  # 更新下一个状态
         ep_reward += reward  # 累加奖励
         if terminated:
               break
      steps.append(ep_step)
      rewards.append(ep_reward)
      print(f"回合：{i_ep+1}/{args['test_eps']}，奖励：{ep_reward:.2f}")
   print("完成测试")

def test_human(args,make = gym.make):

   model = DQN.load(args['last_model_path'])
   args['make_args'].update({'render_mode':'human'})
   env = make(**args['make_args'])

      #   env = VecFrameStack(env, n_stack=4)
   
   print("开始测试！")
   rewards = []  # 记录所有回合的奖励
   steps = []
   for i_ep in range(args['test_eps']):
      ep_reward = 0  # 记录一回合内的奖励
      ep_step = 0
      a = env.reset()
      state,info = env.reset()  # 重置环境，返回初始状态
      for _ in range(args['test_max_steps']):
         ep_step+=1
         action,_state = model.predict(state)  # 选择动作
         next_state, reward, terminated, truncated, info = env.step(action)  # 更新环境，返回transition

         state = next_state  # 更新下一个状态
         ep_reward += reward  # 累加奖励
         if terminated:
               break
      steps.append(ep_step)
      rewards.append(ep_reward)
      print(f"回合：{i_ep+1}/{args['test_eps']}，奖励：{ep_reward:.2f}")
   print("完成测试")


def test_atari_human(args):

   if args['test_model_path'] != '':
      model = DQN.load(args['test_model_path'])
   else:
      model = DQN.load(args['last_model_path'])
      
   env = args['model_args']['env']
   ones = np.ones(4)
   
   print("开始测试！")

   for i_ep in range(args['test_eps']):
      ep_reward = np.zeros(4)   # 记录一回合内的奖励
      ep_step = np.zeros(4)
      state = env.reset()  # 重置环境，返回初始状态
      for _ in range(args['test_max_steps']):

         action,_state = model.predict(state)  # 选择动作
         next_state, reward, dones, info = env.step(action)  # 更新环境，返回transition
         env.render("human")
         state = next_state  # 更新下一个状态
         ep_reward += reward*(ones-dones)  # 累加奖励
         ep_step += ones*(ones-dones)

   

      print(f"回合：{i_ep}/{args['test_eps']}，奖励：{ep_reward},step:{ep_step}")
   print("完成测试")