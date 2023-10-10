
import os
os.environ["PATH"] += ":."

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import requests
import subprocess

import stable_baselines3
from stable_baselines3 import PPO
import gym
from gym import spaces
import numpy as np
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete,Space 

import random
import os
from stable_baselines3.common.vec_env import VecFrameStack
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from util.base18 import toBase18, fromBase18,fromBase128,price_to_valid_tick
from util.constants import GOD_ACCOUNT
from util.globaltokens import weth_usdc_pool
import numpy as np
from gym import Env, spaces
import subprocess  # Assuming you're using this for some shell commands

class DiscreteSimpleEnvBox(Env):
    def __init__(self, price_lower_low, price_lower_high, price_upper_low, price_upper_high, agent_budget_usd, pool):
        super(DiscreteSimpleEnvBox, self).__init__()

        self.pool = pool
        self.global_state = self.pool.get_global_state()

        self.action_space = spaces.Box(
            low=np.array([price_lower_low, price_upper_low], dtype=np.float32),
            high=np.array([price_lower_high, price_upper_high], dtype=np.float32),
            dtype=np.float32
        )

        self.reward = 0
        self.cumulative_reward = 0
        self.done = False

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        self.agent_budget_usd = agent_budget_usd

    def reset(self):
        self.global_state = self.pool.get_global_state()
        self.state = np.array([
            float(self.global_state['curr_price']),
            float(self.global_state['liquidity_raw']),
            float(self.global_state['feeGrowthGlobal0X128']),
            float(self.global_state['feeGrowthGlobal1X128'])
        ], dtype=np.float32)
        self.done = False
        self.reward = 0
        self.cumulative_reward = 0
        return self.state

    def step(self, action):
        self._take_action(action)
        self.state = self.get_state()
        self.reward = self._calculate_reward(action)
        
        self.cumulative_reward += self.reward
        self.done = self._is_done()
        return self.state, self.reward, self.done, {}

    def _take_action(self, action):
        tick_lower = action[0]
        tick_upper = action[1]
        amount = self.agent_budget_usd
        tx_receipt=self.pool.add_liquidity(GOD_ACCOUNT, tick_lower, tick_upper, amount, b'')

    def get_state(self):
        self.global_state = self.pool.get_global_state()
        self.state = np.array([
            float(self.global_state['curr_price']),
            float(self.global_state['liquidity_raw']),
            float(self.global_state['feeGrowthGlobal0X128']),
            float(self.global_state['feeGrowthGlobal1X128'])
        ], dtype=np.float32)
        return self.state

    def _calculate_reward(self, action):
        tick_lower = price_to_valid_tick(action[0], 60)
        tick_upper = price_to_valid_tick(action[1], 60)
        amount = self.agent_budget_usd
        _, _, liquidity = self.pool.budget_to_liquidity(action[0], action[1], amount)
        _, fee_income = self.pool.collect_fee(GOD_ACCOUNT, tick_lower, tick_upper)
        self.pool.remove_liquidity(GOD_ACCOUNT, tick_lower, tick_upper, liquidity)
        self.reward = fee_income
        print(f'Reward: {self.reward}')
        return self.reward

    def _is_done(self):
        threshold = 100 
        return self.reward >= threshold
