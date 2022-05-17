import pybulletgym
import gym
import json
import os
from stable_baselines3 import DDPG, PPO, DDPG, SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from MyReplayBuffer import MyReplayBuffer
from typing import Callable


env_name = 'PeriodicHumanoidPyBulletEnv-v0'
num_cpu = 4
env = make_vec_env(env_name, n_envs=num_cpu)
# env = gym.make(env_name)


lr = 0.00001
total_timesteps = 5000000
old_log_path = './3.21/33M/'
log_path = './5.16/5M/'
load_old_replay_buffer = False
save_replay_buffer = False
eval_freq = 1
batch_size = 100
replay_buffer_class = MyReplayBuffer
# replay_buffer_kwargs = {'n_envs' : num_cpu, 'counter_interval' : 500, 'buffer_size' : 1500}
replay_buffer_kwargs = {'counter_interval' : 200000}
tb_log = os.path.join(log_path, 'tb_log')
eval_log_path = os.path.join(log_path, 'eval')

if not os.path.exists(log_path):
    os.makedirs(log_path)


training_algo = DDPG
model = training_algo('MlpPolicy', env, verbose = 1, train_freq = 1, learning_rate = lr, tensorboard_log = tb_log, replay_buffer_class = replay_buffer_class, replay_buffer_kwargs = replay_buffer_kwargs)
# model = DDPG.load(os.path.join(old_log_path, 'DDPG'), env, verbose = 1, learning_rate = lr, tensorboard_log = tb_log, batch_size = batch_size, replay_buffer_class = replay_buffer_class, replay_buffer_kwargs = replay_buffer_kwargs)
model.set_parameters(os.path.join(old_log_path, 'DDPG'))
if load_old_replay_buffer:
    model.load_replay_buffer(os.path.join(old_log_path, 'rb'))
model.learn(total_timesteps, eval_freq = eval_freq, eval_log_path = eval_log_path)
model.save(os.path.join(log_path + training_algo.__name__))
if save_replay_buffer:
    model.save_replay_buffer(log_path + 'rb')
env.close()

config = {
    'env_name' : env_name,
    'lr' : lr,
    'total_timesteps' : total_timesteps,
    'batch_size' : batch_size,
    'load_old_replay_buffer' : load_old_replay_buffer,
    'save_replay_buffer' : save_replay_buffer,
    'replay_buffer_class' : 'None' if replay_buffer_class is None else replay_buffer_class.__name__,
    'replay_buffer_kwargs' : replay_buffer_kwargs
}
with open(os.path.join(log_path, 'config.json'), 'w') as f:
    json.dump(json.dumps(config), f)
