import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
import random
from gym import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer


class MyReplayBuffer(ReplayBuffer):
    def __init__(self, 
            buffer_size: int, 
            observation_space: spaces.Space, 
            action_space: spaces.Space, 
            device: Union[th.device, str] = "cpu", 
            n_envs: int = 1, 
            optimize_memory_usage: bool = False, 
            handle_timeout_termination: bool = True,
            counter_interval = 100000):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)

        # self.recent_sample_indices = []
        # self.not_recent_sample_indices = []
        
        self.history_buffer_num = 0
        self.counter_interval = counter_interval
        self.recent_sample_ratio = 0.5

    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        # if len(self.recent_sample_indices) >= self.counter_interval:
        #    tmp = self.recent_sample_indices.pop(0)
        #    self.not_recent_sample_indices.append(tmp)
        # if len(self.not_recent_sample_indices) >= self.buffer_size - self.counter_interval:
        #    self.not_recent_sample_indices.pop(0)
        # self.recent_sample_indices.append(self.pos)

        # if self.recent_sample_ratio < 0.5:
        #    self.recent_sample_ratio += (0.5 - 0.0) / 2000000
        
        self.history_buffer_num += 1
        super().add(obs, next_obs, action, reward, done, infos)

    
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # if not self.optimize_memory_usage:
        #     # return super().sample(batch_size=batch_size, env=env)
            
        #     recent_bs = int(batch_size * self.recent_sample_ratio)
        #     non_recent_bs = batch_size - recent_bs
            
        #     if len(self.recent_sample_indices) < self.counter_interval:
        #         return super().sample(batch_size = batch_size, env = env)
        #     else:
        #         recent_inds = random.choices(self.recent_sample_indices, k = recent_bs)
                    
        #         # upper_bound = self.buffer_size if self.full else self.pos
        #         not_recent_inds = random.choices(self.not_recent_sample_indices, k = non_recent_bs)
                
        #         batch_inds = np.append(recent_inds, not_recent_inds)
        #         return self._get_samples(batch_inds, env=env)


        # if self.full:
        #     batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        # else:
        #     batch_inds = np.random.randint(0, self.pos, size=batch_size)

        if self.history_buffer_num >= self.counter_interval:
            batch_inds = np.random.randint(self.pos - self.counter_interval, self.pos, size = batch_size)
        else:
            batch_inds = np.random.randint(0, self.pos, size = batch_size)


        return self._get_samples(batch_inds, env=env)
