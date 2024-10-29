import logging
import os
import sys
import pandas as pd
from util.constants import BASE_PATH
from util.utility_functions import fromBase18

# Add parent directory to sys.path to handle imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.uniswap_lp_agent import UniswapV3LiquidityProviderAgent
from agents.uniswap_swapper_agent import UniswapV3SwapperAgent
from util.agent_policies import retail_lp_policy,noise_trader_policy


class SimEngine:
    def __init__(self, pool):
        self.pool = pool
        self.lp_agent = UniswapV3LiquidityProviderAgent(initial_token0=1e10, initial_token1=1e10, pool=self.pool)
        self.swapper_agent = UniswapV3SwapperAgent(initial_token0=1e10, initial_token1=1e10, pool=self.pool)
        self.df = pd.read_csv(f"{BASE_PATH}/events_data/WBTC-ETH_all_events.csv")
        self.df = self.df.iloc[:10]
        #print(f"token0: {token0}    token1:  {token1}")
    def run(self):
        i=1
        for index, row in self.df.iterrows():
            if row['type'] == 'mint':
                tick_lower = pd.to_numeric(row['tickLower'], errors='coerce')  # Access value in the current row
                tick_upper = pd.to_numeric(row['tickUpper'], errors='coerce')
                liquidity_amount = pd.to_numeric(row['amount'], errors='coerce')
                token0_amount = pd.to_numeric(row['amount0'], errors='coerce')*10**10
                token1_amount = pd.to_numeric(row['amount1'], errors='coerce')
                block_time = row['evt_block_time']
                action = 'add_liquidity'
                self.lp_agent.takeStep(action, tick_lower, tick_upper, liquidity_amount, block_time)

            if row['type'] == 'burn':
                tick_lower = pd.to_numeric(row['tickLower'], errors='coerce')  # Access value in the current row
                tick_upper = pd.to_numeric(row['tickUpper'], errors='coerce')
                liquidity_amount = pd.to_numeric(row['amount'], errors='coerce')
                token0_amount = pd.to_numeric(row['amount0'], errors='coerce')*10**10
                token1_amount = pd.to_numeric(row['amount1'], errors='coerce')
                block_time = row['evt_block_time']
                action = 'remove_liquidity'
                self.lp_agent.takeStep(action, tick_lower, tick_upper, liquidity_amount, block_time)

            if row['type'] == 'swap' and pd.to_numeric(row['amount0'], errors='coerce', downcast='float') < 0 :
                token0_amount = pd.to_numeric(row['amount0'], errors='coerce')*10**10
                token1_amount = pd.to_numeric(row['amount1'], errors='coerce')
                block_time = row['evt_block_time']
                swap_action = 'swap_token1_for_token0'
                self.swapper_agent.takeStep(swap_action,token0_amount,block_time)
            
            if row['type'] == 'swap' and pd.to_numeric(row['amount1'], errors='coerce', downcast='float') < 0 :
                token0_amount = pd.to_numeric(row['amount0'], errors='coerce')*10**10
                token1_amount = pd.to_numeric(row['amount1'], errors='coerce')
                block_time = row['evt_block_time']
                swap_action = 'swap_token0_for_token1'
                self.swapper_agent.takeStep(swap_action,token1_amount,block_time)
            print(f"---------------------------------CSV ROW :{i}")
            i +=1 
        



