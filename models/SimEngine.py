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
    def __init__(self, pool, batch_size = 7):
        self.pool = pool
        self.lp_agent = UniswapV3LiquidityProviderAgent(initial_token0=1e10, initial_token1=1e10, pool=self.pool)
        self.swapper_agent = UniswapV3SwapperAgent(initial_token0=1e10, initial_token1=1e10, pool=self.pool)
        self.df = pd.read_csv(f"{BASE_PATH}/events_data/WBTC-ETH_all_events.csv")
        # self.df = self.df.iloc[:10]
        #print(f"token0: {token0}    token1:  {token1}")
        self.batch_size = batch_size
        self.current_index = 0
        self.i = 0
    def run(self):
        # Determine the end index for the current batch
        end_index = min(self.current_index + self.batch_size, len(self.df))
        
        # Process the next batch of actions
        for index, row in self.df.iloc[self.current_index:end_index].iterrows():
            self.i+=1
            print(f"---------------------------------CSV ROW :{self.i}")
            if row['type'] == 'mint':
                tick_lower = int(pd.to_numeric(row['tickLower'], errors='coerce'))  # Access value in the current row
                tick_upper = int(pd.to_numeric(row['tickUpper'], errors='coerce'))
                liquidity_amount = int(pd.to_numeric(row['amount'], errors='coerce'))
                token0_amount = int(pd.to_numeric(row['amount0'], errors='coerce'))
                token1_amount = int(pd.to_numeric(row['amount1'], errors='coerce'))
                block_time = row['evt_block_time']
                action = 'add_liquidity'
                self.lp_agent.takeStep(action, tick_lower, tick_upper, token0_amount, token1_amount, block_time)

            # elif row['type'] == 'burn':
            #     tick_lower = int(pd.to_numeric(row['tickLower'], errors='coerce'))  # Access value in the current row
            #     tick_upper = int(pd.to_numeric(row['tickUpper'], errors='coerce'))
            #     liquidity_amount = int(pd.to_numeric(row['amount'], errors='coerce'))
            #     token0_amount = int(pd.to_numeric(row['amount0'], errors='coerce'))
            #     token1_amount = int(pd.to_numeric(row['amount1'], errors='coerce'))
            #     block_time = row['evt_block_time']
            #     action = 'remove_liquidity'
            #     self.lp_agent.takeStep(action, tick_lower, tick_upper, liquidity_amount, block_time)

            elif row['type'] == 'swap' and int(pd.to_numeric(row['amount0'], errors='coerce', downcast='float')) < 0 :
                token0_amount = int(pd.to_numeric(row['amount0'], errors='coerce'))
                token1_amount = int(pd.to_numeric(row['amount1'], errors='coerce'))
                sqrt_price = int(pd.to_numeric(row['sqrtPriceX96'], errors='coerce'))
                block_time = row['evt_block_time']
                swap_action = 'swap_token1_for_token0'
                self.swapper_agent.takeStep(swap_action,token0_amount,block_time,sqrt_price)
            
            elif row['type'] == 'swap' and int(pd.to_numeric(row['amount1'], errors='coerce', downcast='float')) < 0 :
                token0_amount = int(pd.to_numeric(row['amount0'], errors='coerce'))
                token1_amount = int(pd.to_numeric(row['amount1'], errors='coerce'))
                sqrt_price = int(pd.to_numeric(row['sqrtPriceX96'], errors='coerce'))
                block_time = row['evt_block_time']
                swap_action = 'swap_token0_for_token1'
                self.swapper_agent.takeStep(swap_action,token1_amount,block_time,sqrt_price)
            else:
                print(f"\nDID NOTHING  --> {row['type']}\n")
            
        
        # Update the current index for the next run
        self.current_index = end_index
        # Check if all actions have been processed
        if self.current_index >= len(self.df):
            print("All actions have been processed.")
        



