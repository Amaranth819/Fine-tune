import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import time
import numpy as np
import torch
import pybullet as pb
from stable_baselines3 import DDPG

env = gym.make('HumanoidPyBulletEnv-v0')

'''
    Env settings
    Action space: torque (computed by power * coefficient * clip(a, (-1,1)))

    Observation space (clipped):
    1. difference between the initial and current base height
    2. the angle formed by the origin, the current robot position and the destination
    3. velocity (in terms of the body frame, also simplified by yaw angle)
    4. roll and pitch angle of robot
    5. feet contact: whether the feet contact with the ground

    Reward function:
    1. alive bonus
    2. distance to destination / dt
    3. abs(joint speed).mean() * electricity_cost (-0.1) + stall_torque_cost (-0.1) * a^2.mean()
    4. self.joints_at_limit_cost * self.robot.joints_at_limit
    5. feet_collision_cost (=0)
'''


# env = Monitor(gym.make('HumanoidPyBulletEnv-v0'), './video', force=True)
env.render() # call this before env.reset, if you want a window showing the environment
obs = env.reset()  # should return a state vector if everything worked

model = DDPG.load('E:/Syracuse/PaperWork/fine-tune/experiments/3.16/22M-v2/22M-v2/DDPG')
rewards = []

for i in range(10):
    obs = env.reset()
    done = False
    reward = 0

    images = []


    while not done:
        action, _ = model.predict(obs, deterministic = True)
        # action = env.action_space.sample()
        obs, r, done, _ = env.step(action)
        reward += r
        time.sleep(0.002)

    rewards.append(reward)

print(rewards)
print(np.average(rewards))