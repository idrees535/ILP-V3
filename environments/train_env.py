import numpy as np
import pandas as pd
import random
# import matplotlib.pyplot as plt
import gymnasium as gym
# import tensorflow as tf
# from tensorflow.keras.callbacks import TensorBoard
# import tensorflow_probability as tfp
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
from util.utility_functions import *
process = start_hardhat_node()
from util.constants import GOD_ACCOUNT, WALLET_LP, WALLET_SWAPPER, RL_AGENT_ACCOUNT, BASE_PATH,TIMESTAMP
from util.pool_configs import *
from models.SimEngine import SimEngine

# import mlflow
# import mlflow.tensorflow
# mlflow.tensorflow.autolog()

class DiscreteSimpleEnv(gym.Env):
    def __init__(self, agent_budget_usd=10000,alpha = 0.5, exploration_std_dev = 0.01, beta=0.1,penalty_param_magnitude=-1,use_running_statistics=False,action_transform='linear'):
        super(DiscreteSimpleEnv, self).__init__()

        self.pool=None
        self.global_state=None
        self.curr_price=None
        self.action_lower_bound=None
        self.action_upper_bound=None
        self.state=None
        self.engine=None
        self.action_transform=action_transform
        self.train_data_log=[]
        
        self.action_space = gym.spaces.Dict({
            'price_relative_lower': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'price_relative_upper': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        
        self.reward=0
        self.cumulative_reward = 0
        self.done=False
        self.episode=0
        self.step_count=0

        self.observation_space = gym.spaces.Dict({
            'scaled_curr_price': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'scaled_liquidity': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'scaled_feeGrowthGlobal0x128': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'scaled_feeGrowthGlobal1x128': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),   
        })
        self.agent_budget_usd = agent_budget_usd
        self.initial_budget_usd = agent_budget_usd

        # Initialize rewrad normalization running statistics
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_count = 0

        self.exploration_std_dev = exploration_std_dev
        self.penalty=0
        self.penalty_param=0
        self.penalty_param_magnitude=penalty_param_magnitude

        # Initialize running statistics for state normalization
        self.use_running_statistics=use_running_statistics
        self.curr_price_mean = 0
        self.curr_price_std = 1
        self.liquidity_mean = 0
        self.liquidity_std = 1
        self.fee_growth_diff_0 = 0
        self.fee_growth_diff_1 = 0
        self.fee_growth_0_mean = 0
        self.fee_growth_1_mean = 0
        self.fee_growth_0_std = 1
        self.fee_growth_1_std = 1
        self.previous_fee_growth_0 = 0
        self.previous_fee_growth_1 = 0

        #Obs space scaling param
        self.alpha = alpha
        #Rewrad scaling param
        self.beta=beta
        
    def reset(self):
        self.pool=random.choice([weth_usdc_pool,eth_dai_pool,btc_usdt_pool,btc_weth_pool])
        #self.pool = btc_weth_pool
        
        print(f'Pool selcted for this episode: {self.pool.pool_id}')
        # sim_strategy = SimStrategy()
        # sim_state = SimState(ss=sim_strategy,pool=self.pool)

        output_dir = "model_output"
        # netlist_log_func = netlist_createLogData

        #from engine.SimEngine import SimEngine
        self.engine = SimEngine(self.pool)

        self.global_state=self.pool.get_global_state()
        self.curr_price=self.global_state['curr_price']
        self.action_lower_bound=self.curr_price*0.1
        self.action_upper_bound=self.curr_price*2
        self.state = self.get_obs_space()
        
        self.done=False
        self.reward=0
        self.cumulative_reward = 0
        self.episode+=1
        self.step_count=0
    
        # Used for evaluation only
        self.cumulative_reward_rl_agent = 0
        self.cumulative_reward_baseline_agent = 0

        self.agent_budget_usd = self.initial_budget_usd
         
        # reset running statistics for reward normalization
        '''
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_count = 0
        '''
        # reset running statistics for state normalization
        '''
        self.curr_price_mean = 0
        self.curr_price_std = 1
        self.liquidity_mean = 0
        self.liquidity_std = 1
        self.fee_growth_diff_0 = 0
        self.fee_growth_diff_1 = 0
        self.fee_growth_0_mean = 0
        self.fee_growth_1_mean = 0
        self.fee_growth_0_std = 1
        self.fee_growth_1_std = 1
        self.previous_fee_growth_0 = 0
        self.previous_fee_growth_1 = 0
        '''  
        return self.state

    def step(self, raw_action):
              
        # Execute agent's action using pool's interface of add/remove liquidity
        mint_tx_receipt,action=self._take_action(raw_action)
        
        # run uniswap abm env of n_steps
        print()
        print('_______________________________Environment Step________________________________')
        # self.engine.reset()
        self.engine.run()
        print()
        
        self.state=self.get_obs_space()

        scaled_reward,raw_reward,fee_income,impermanent_loss = self._calculate_reward(action,mint_tx_receipt)
        self.reward=scaled_reward
        self.cumulative_reward += self.reward

        self.step_count+=1
        
        print(f"\nepisode: {self.episode}, step_count: {self.step_count}, scaled_reward: {self.reward}, raw_reward: {raw_reward} cumulative_reward: {self.cumulative_reward}")
        print(f"\nraw_pool_state: {self.pool.get_global_state()}")
        print(f"\nscaled_pool_state: {self.state}")
        print()

        self.train_data_log.append((self.episode, self.step_count, action, self.pool.get_global_state(), raw_action, self.state, raw_reward, self.reward, self.cumulative_reward, fee_income, impermanent_loss))

        self.done = self._is_done()
        return self.state, self.reward, self.done, {}

    def get_obs_space(self):
        self.global_state = self.pool.get_global_state()

        # Scaling for curr_price and liquidity
        curr_price = float(self.global_state['curr_price'])
        liquidity = float(self.global_state['liquidity_raw'])
        fee_growth_0 = float(self.global_state['feeGrowthGlobal0X128'])
        fee_growth_1 = float(self.global_state['feeGrowthGlobal1X128'])

        self.curr_price_mean = self.alpha * curr_price + (1 - self.alpha) * self.curr_price_mean
        self.curr_price_std = np.sqrt(self.alpha * (curr_price - self.curr_price_mean)**2 + (1 - self.alpha) * self.curr_price_std**2)

        self.liquidity_mean = self.alpha * liquidity + (1 - self.alpha) * self.liquidity_mean
        self.liquidity_std = np.sqrt(self.alpha * (liquidity - self.liquidity_mean)**2 + (1 - self.alpha) * self.liquidity_std**2)

        # Scaling for fee growth differences 
        self.fee_growth_diff_0 = fee_growth_0 - self.previous_fee_growth_0
        self.fee_growth_diff_1 = fee_growth_1 - self.previous_fee_growth_1

        self.fee_growth_0_mean = self.alpha * self.fee_growth_diff_0 + (1 - self.alpha) * self.fee_growth_0_mean
        self.fee_growth_0_std = np.sqrt(self.alpha * (self.fee_growth_diff_0 - self.fee_growth_0_mean)**2 + (1 - self.alpha) * self.fee_growth_0_std**2)

        self.fee_growth_1_mean = self.alpha * self.fee_growth_diff_1 + (1 - self.alpha) * self.fee_growth_1_mean
        self.fee_growth_1_std = np.sqrt(self.alpha * (self.fee_growth_diff_1 - self.fee_growth_1_mean)**2 + (1 - self.alpha) * self.fee_growth_1_std**2)

        if self.use_running_statistics==True:
            #Use running stats
            obs = {'scaled_curr_price': (curr_price - self.curr_price_mean) / (self.curr_price_std + 1e-10),'scaled_liquidity': (liquidity - self.liquidity_mean) / (self.liquidity_std + 1e-10),}
            obs['scaled_feeGrowthGlobal0x128'] = (self.fee_growth_diff_0 - self.fee_growth_0_mean) / (self.fee_growth_0_std + 1e-10)
            obs['scaled_feeGrowthGlobal1x128'] = (self.fee_growth_diff_1 - self.fee_growth_1_mean) / (self.fee_growth_1_std + 1e-10)

        else:
            # Scale obs space using global stats
            obs = {'scaled_curr_price': curr_price/5000,'scaled_liquidity': liquidity/1e24,'scaled_feeGrowthGlobal0x128': fee_growth_0/1e34,'scaled_feeGrowthGlobal1x128': fee_growth_1/1e34}

        self.previous_fee_growth_0 = fee_growth_0
        self.previous_fee_growth_1 = fee_growth_1

        return obs

    def _take_action(self, action):
        self.penalty=0

        raw_a, raw_b = action[0, 0].numpy(), action[0, 1].numpy()

        # Add exploration noise
        a_0 = raw_a + np.random.normal(0, self.exploration_std_dev)
        a_1 = raw_b + np.random.normal(0, self.exploration_std_dev)
        
        if self.action_transform=='linear':
            a_0 = np.clip(a_0, 0, 1)
            a_1 = np.clip(a_1, 0, 1)
            price_lower = self.action_lower_bound + a_0 * (self.action_upper_bound - self.action_lower_bound)/2
            price_upper = (self.action_upper_bound - self.action_lower_bound)/2 + a_1 * (self.action_upper_bound - self.action_lower_bound)/2
        
            # Enabling agent to place range orders too (Only feasible when using multiple positions)
            #price_lower = self.action_lower_bound + a_0 * (self.action_upper_bound - self.action_lower_bound)
            #price_upper = self.action_lower_bound + a_1 * (self.action_upper_bound - self.action_lower_bound)
            
        elif self.action_transform=='exp':
            #Exponential transforom
            exp_a_0 = np.exp(a_0)
            exp_a_1 = np.exp(a_1)

            # Normalize the exponential values to the range [0, 1]
            norm_exp_a_0 = exp_a_0 / (exp_a_0 + exp_a_1)
            norm_exp_a_1 = exp_a_1 / (exp_a_0 + exp_a_1)

            # Calculate the range between action_lower_bound and action_upper_bound
            range_bound = self.action_upper_bound - self.action_lower_bound

            # Calculate price_lower and price_upper using the normalized exponential values
            price_lower = self.action_lower_bound + norm_exp_a_0 * range_bound
            price_upper = self.action_lower_bound + norm_exp_a_1 * range_bound

        # Check if price_lower or price_upper are NaN
        if np.isnan(price_lower) or np.isnan(price_upper):
            price_lower = np.random.uniform(0, 1)
            price_upper = np.random.uniform(0, 1)
            print("Warning: price_lower or price_upper was NaN. Assigned random values.")
            
        # Ensure price_lower is less than price_upper - Add penalty
        if price_lower>price_upper:
            price_lower = min(price_lower, price_upper)
            price_upper = max(price_lower, price_upper)
            self.penalty=self.penalty_param_magnitude

        # ensure actions are not too close - Add penalty
        min_diff_percentage = 0.05  # 5% difference
        price_diff = price_upper - price_lower
        
        if price_diff < min_diff_percentage * price_lower:
            self.penalty+=self.penalty_param_magnitude
            price_upper = price_lower + min_diff_percentage * price_lower
        
        action_dict = {
            'price_lower': price_lower,
            'price_upper': price_upper
        }
        tick_lower=price_to_valid_tick(action_dict['price_lower'])
        tick_upper=price_to_valid_tick(action_dict['price_upper'])
        amount=self.agent_budget_usd
        
        print('\nRL Agent Action')
        print(f"____RL Agent WALLET {self.pool.get_wallet_balances(GOD_ACCOUNT.address)} ")
        print(f"____Liquidity amount: {amount} toBase18: {toBase18(amount)} ")
        print(f"raw_action: {action}, scaled_action: {action_dict} \ns")

        #print(f"\nAmount for liquidity: {amount}")
        mint_tx_receipt=self.pool.add_liquidity(GOD_ACCOUNT, tick_lower, tick_upper, amount, b'')

        return mint_tx_receipt,action_dict
        
    def _calculate_reward(self,action,mint_tx_receipt):
       
        tick_lower=price_to_valid_tick(action['price_lower'],60)
        tick_upper=price_to_valid_tick(action['price_upper'],60)
        liquidity=mint_tx_receipt.events['Mint']['amount']

        # Collecting fee earned by position
        print('Collect fee')
        collect_tx_receipt,fee_income = self.pool.collect_fee(GOD_ACCOUNT, tick_lower, tick_upper,poke=True)
    
        print("\nBurn Position and Collect Tokens")
        # Remove position and collect tokens
        burn_tx_receipt=self.pool.remove_liquidity_with_liquidty(GOD_ACCOUNT, tick_lower, tick_upper, liquidity)
        collect_tx_receipt,curr_budget_usd = self.pool.collect_fee(GOD_ACCOUNT, tick_lower, tick_upper,poke=False)

        # Can use online scaling approach as used for reward for this
        rel_portofolio_value = 1 - curr_budget_usd/self.agent_budget_usd
        
        # Instead of using full budget for next step use previous step's reomved liquidity amount as budget in next step
        #self.agent_budget_usd = curr_budget_usd
        
        # Calculate IL
        amount0_initial = mint_tx_receipt.events['Mint']['amount0']
        amount1_initial = mint_tx_receipt.events['Mint']['amount1']
        
        amount0_final = burn_tx_receipt.events['Burn']['amount0']
        amount1_final = burn_tx_receipt.events['Burn']['amount1']
        self.global_state = self.pool.get_global_state()
        pool_price = float(self.global_state['curr_price'])

        value_initial = (amount0_initial * pool_price + amount1_initial) / 1e18
        value_final = (amount0_final * pool_price + amount1_final) / 1e18

        impermanent_loss = value_initial - value_final

        if fee_income==0:
            self.penalty_param+= 0.05
            self.penalty += self.penalty_param_magnitude*(1+self.penalty_param)

        print(f'fee_earned:{fee_income}, impermannet_loss: {impermanent_loss}, penalty: {self.penalty}, initial_agent_portofolio_value: {value_initial}, final_agent_portofolio_value: {value_final}, reward_mean: {self.reward_mean}, rewrad_std_dev: {self.reward_std}, reward_count: {self.reward_count}')
        print()
        
        raw_reward = fee_income - impermanent_loss + self.penalty
        
        if self.penalty==0:
            self.reward_count += 1
            #new_mean = self.reward_mean + (raw_reward - self.reward_mean) / self.reward_count
            #new_std = ((self.reward_std ** 2 + (raw_reward - self.reward_mean) * (raw_reward - new_mean)) / self.reward_count) ** 0.5
            new_mean = self.beta * raw_reward + (1 - self.beta) * self.reward_mean
            new_std = np.sqrt(self.beta * ((raw_reward - new_mean) ** 2) + (1 - self.beta) * (self.reward_std ** 2))
            self.reward_mean = new_mean
            self.reward_std = new_std

        #scaled_reward = (raw_reward - self.reward_mean) / (self.reward_std + 1e-10)
        scaled_reward = raw_reward*10

        #Reset penlaty for next step
        self.penalty=0

        return scaled_reward,raw_reward,fee_income, impermanent_loss

    def _is_done(self):
        
        max_reward_threshold = 100000
        min_reward_threshold= -100000
        max_budget_threshold = 1.5*self.initial_budget_usd
        min_budget_threshold = 0.5*self.initial_budget_usd

        if self.cumulative_reward >= max_reward_threshold or self.cumulative_reward<=min_reward_threshold or self.agent_budget_usd>max_budget_threshold or self.agent_budget_usd<min_budget_threshold:
            return True
        else:
            return False
        
    def revert_back_to_snapshot(self): 
            self.pool.revert_to_snapshot(self.snapshot_id)