
## 杂项函数，不知道该怎么分类

import gym
import keyboard
import numpy as np
import time,os,json
import yaml
try:
    from .constant import *
except:
    from constant import *

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def play_with_keyboard(env_name):
    global action_in_play_with_keyboard
    action_in_play_with_keyboard = 0
    def actions(x,n):
        global action_in_play_with_keyboard
        for i in range(n):
            if x.event_type == "down" and x.name == str(i):
                action_in_play_with_keyboard = i            
            if x.event_type == "up" and x.name == str(i):
                action_in_play_with_keyboard = 0

    env = gym.make(env_name,render_mode = 'human')
    env.metadata['render_fps'] = 60    

    print('观测空间={}'.format(env.observation_space))
    print('动作空间={}'.format(env.action_space))

    keyboard.hook(lambda x:actions(x,env.action_space.n))
    env.reset()
    env.render()
    total_reward = 0
    while True:

        next_state,reward,terminated ,truncated ,info  = env.step(action_in_play_with_keyboard)
        total_reward += reward
        print(f'action:{action_in_play_with_keyboard} , reward:{total_reward}')
        if terminated:
            print('游戏结束')
            break

        time.sleep(0.1)




def get_args(method,env_name,make = gym.make,wrapper = lambda x:x):

    args_path = os.path.join(yaml_path,f"{method}.yml" )
    with open(args_path, 'r') as f:
      args = yaml.safe_load(f)[env_name]
    with open(main_args_path, 'r') as f:
      main_args = yaml.safe_load(f)[env_name]

    env_name = env_name.replace('/','_')
    new_args = {'save_path':os.path.join(model_path,f"{method}_{env_name}")}


    args.update(main_args)
    args.update(new_args)


    env = wrapper(make(**args['make_args']))

        # env = VecFrameStack(env, n_stack=args['n_stack'])

    
    model_args = {
        "tensorboard_log": log_path,
        "env":env
    }
    args["model_args"].update(model_args)


    return args



def get_atari_args(method,env_name):

    args_path = os.path.join(yaml_path,f"{method}.yml" )


    with open(args_path, 'r') as f:
      args = yaml.safe_load(f)[env_name]

    with open(main_args_path, 'r') as f:
      main_args = yaml.safe_load(f)[env_name]


    env_name = env_name.replace('/','_')
    new_args = {'save_path':os.path.join(model_path,f"{method}_{env_name}"),'last_model_path':os.path.join(os.path.join(model_path,f"{method}_{env_name}"),'last_model')}


    args.update(main_args)
    args.update(new_args)


    env = make_atari_env(**args['make_args'])
    # Frame-stacking with 4 frames
    env = VecFrameStack(env, n_stack=args['n_stack'])

        # env = VecFrameStack(env, n_stack=args['n_stack'])

    
    model_args = {
        "tensorboard_log": log_path,
        "env":env
    }
    args["model_args"].update(model_args)


    return args



def get_info(args):

    env = args["model_args"]['env']
    # env =  make_atari_env(args['env_name'])
    # env = gym.make(args['env_name'],obs_type = 'grayscale')
    action_space_size = env.action_space.n

    # 获取观测空间大小
    observation_space_size = env.observation_space.shape

    print("操作空间大小:", action_space_size)
    print("观测空间大小:", observation_space_size)



if __name__ == "__main__":
    
    # print(get_args('DQN','CartPole-v1'))

    args = get_atari_args('DQN','ALE/Boxing-v5')
    # print(args)
    print('-------------------------')
    get_info(args)    
    # args = get_args('DQN','CartPole-v1')
    # print(args)
    # get_info(args)

