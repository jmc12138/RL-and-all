import gymnasium as gym
from gymnasium.wrappers import TransformReward
from stable_baselines3 import DQN
import json,os
import utils
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

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
   env_name = 'ALE/Boxing-v5'
   args = utils.get_atari_args('DQN',env_name)

   opt = {
      'model_path':os.path.join(args['save_path'],'best_model'),
      'test_model_path': os.path.join(args['save_path'],'best_model') 
      
   }



   # %%% 训练
   # utils.train(args)
   utils.ctn_train(args,opt['model_path'])



   # %%%  测试

   args.update({'test_model_path':opt['test_model_path']})
   utils.test_atari_human(args)

   # %%%  
   # model = DQN.load(args['save_path'])
   # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
   # print(f'mean_reward: "{mean_reward}, std_reward: {std_reward}')




   
   

