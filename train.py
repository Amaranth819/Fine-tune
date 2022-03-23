import pybulletgym
import gym
import json
import os
from stable_baselines3 import DDPG
from stable_baselines3.her import HerReplayBuffer
from MyReplayBuffer import MyReplayBuffer

env_name = 'HumanoidPyBulletEnv-v0'
lr = 0.0000002
total_timesteps = 10000000
old_log_path = './3.21/23M/'
log_path = './3.21/33M/'
load_old_replay_buffer = False
save_replay_buffer = False
eval_freq = 1
batch_size = 100
replay_buffer_class = MyReplayBuffer
tb_log = os.path.join(log_path, 'tb_log')
eval_log_path = os.path.join(log_path, 'eval')

if not os.path.exists(log_path):
    os.mkdir(log_path)

env = gym.make(env_name)
# model = DDPG('MlpPolicy', env, verbose = 1, learning_rate = lr, tensorboard_log = tb_log)
model = DDPG.load(os.path.join(old_log_path, 'DDPG'), env, verbose = 1, learning_rate = lr, tensorboard_log = tb_log, batch_size = batch_size, replay_buffer_class = replay_buffer_class)
if load_old_replay_buffer:
    model.load_replay_buffer(os.path.join(old_log_path, 'rb'))
model.learn(total_timesteps, eval_freq = eval_freq, eval_log_path = eval_log_path)
model.save(os.path.join(log_path + 'DDPG'))
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
    'replay_buffer_class' : 'None' if replay_buffer_class is None else replay_buffer_class.__name__
}
with open(os.path.join(log_path, 'config.json'), 'w') as f:
    json.dump(json.dumps(config), f)
