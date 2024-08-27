# Imports
from enforce_typing import enforce_types
from typing import List, Set
from agents.UniswapV3LiquidityProviderAgent import UniswapV3LiquidityProviderAgent
from agents.UniswapV3SwapperAgent import UniswapV3SwapperAgent
from engine.SimStrategyBase import SimStrategyBase
from engine.SimStateBase import SimStateBase
from engine.KPIsBase import KPIsBase
from util.constants import *
from util.constants import S_PER_HOUR,S_PER_MONTH, S_PER_YEAR,S_PER_DAY
from util.base18 import toBase18, fromBase18,fromBase128,price_to_valid_tick,price_to_raw_tick,price_to_sqrtp,sqrtp_to_price,tick_to_sqrtp,liquidity0,liquidity1,eth
import brownie
import random
import numpy as np
import pandas as pd
from model_scripts.agents_policies import retail_lp_policy,noise_trader_policy


# SimStrategy Class
class SimStrategy(SimStrategyBase):
    def __init__(self):
        self.time_step = S_PER_DAY
        self.stop_condition = S_PER_MONTH
        
        self.log_interval = S_PER_DAY
        self.max_ticks = 3

        #self.pool = weth_usdc_pool
# SimState Class
class SimState(SimStateBase):
    def __init__(self, ss: SimStrategy, pool):
        super().__init__(ss)

        self.pool = pool # Give pool as an arguemnet to simstate class

        # Liquidity provider agents
        self.retail_lp = UniswapV3LiquidityProviderAgent("retail_lp", 1e18,1e18,retail_lp_policy,self.pool)
        #self.inst_lp = UniswapV3LiquidityProviderAgent("inst_lp", 1000000000000.0,110000000000000.0,inst_LP_policy)
        #self.rl_lp = UniswapV3LiquidityProviderAgent("rl_lp", 1000000000000.0,110000000000000.0,rl_LP_policy)
        
        # Trader agents
        self.noise_trader = UniswapV3SwapperAgent("noise_trader",1e18,1e18, noise_trader_policy,self.pool)
        #self.whale_trader = UniswapV3SwapperAgent("whale_trader",5000000000000000.0,5500000000000000.0, whale_trader_policy)
      

        self.agents["retail_lp"] = self.retail_lp
        self.agents["noise_trader"] = self.noise_trader
        #self.agents["inst_lp"] = self.inst_lp
        #self.agents["whale_trader"] = self.whale_trader
        #self.agents["rl_lp"] = self.rl_lp
        

        # Initialize KPIs object
        self.kpis = KPIs(time_step=ss.time_step)

    def reset(self):
        self.tick = 0
# KPIs Class
class KPIs(KPIsBase):
    pass  

# netlist_createLogData Function
def netlist_createLogData(state: SimState):
    
    # Static variables
    try:
        s = []
        dataheader = []
        datarow = []
        
        # Log agents state
        retail_lp = state.agents["retail_lp"]
        noise_trader = state.agents["noise_trader"]

        # Log LP's funds
        s.append(f"LP_token1: {retail_lp.token1()}, LP_token0: {retail_lp.token0()}")
        dataheader.extend(["LP_token1", "LP_token0"])
        datarow.extend([retail_lp.token1(), retail_lp.token0()])
        

        # Log Trader's funds
        s.append(f"Trader_token1: {noise_trader.token1()}, Trader_token0: {noise_trader.token0()}")
        dataheader.extend(["Trader_token1", "Trader_token0"])
        datarow.extend([noise_trader.token1(), noise_trader.token0()])
        
    except Exception as e:
        print("An error occurred while logging data:", e)

    return s, dataheader, datarow