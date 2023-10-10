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
from util.globaltokens import weth_usdc_pool
import numpy as np
import pandas as pd


#from agents.rl_ilp_agent import DiscreteSimpleEnvBox
#import stable_baselines3
#from stable_baselines3 import PPO

# SimStrategy Class
class SimStrategy(SimStrategyBase):
    def __init__(self):
        self.time_step = S_PER_DAY
        self.stop_condition = S_PER_MONTH
        
        self.log_interval = S_PER_DAY
        self.max_ticks = 5

        self.pool = weth_usdc_pool
        '''
        price_lower_low = 1000
        price_lower_high = 2000
        price_upper_low = 2000
        price_upper_high = 3000
        agent_budget_usd = 1000
        pool = None  # Your pool object here

        env_box = DiscreteSimpleEnvBox(price_lower_low, price_lower_high, price_upper_low, price_upper_high, agent_budget_usd,weth_usdc_pool)
        self.model = PPO('MlpPolicy', env_box, verbose=1)
        '''
# SimState Class
class SimState(SimStateBase):
    def __init__(self, ss: SimStrategy):
        super().__init__(ss)

        # Liquidity provider agents
        self.retail_lp = UniswapV3LiquidityProviderAgent("retail_lp", 1000000000000.0,110000000000000.0,retail_LP_policy)
        #self.inst_lp = UniswapV3LiquidityProviderAgent("inst_lp", 1000000000000.0,110000000000000.0,inst_LP_policy)
        #self.rl_lp = UniswapV3LiquidityProviderAgent("rl_lp", 1000000000000.0,110000000000000.0,rl_LP_policy)
        
        # Trader agents
        self.noise_trader = UniswapV3SwapperAgent("noise_trader",5000000000000000.0,5500000000000000.0, noise_trader_policy)
        #self.whale_trader = UniswapV3SwapperAgent("whale_trader",5000000000000000.0,5500000000000000.0, whale_trader_policy)
      
        #self.agents = [self.lp, self.trader]
        self.agents["retail_lp"] = self.retail_lp
        self.agents["noise_trader"] = self.noise_trader
        #self.agents["inst_lp"] = self.inst_lp
        #self.agents["whale_trader"] = self.whale_trader
        #self.agents["rl_lp"] = self.rl_lp
        

          # Initialize KPIs object
        self.kpis = KPIs(time_step=ss.time_step)
# KPIs Class
class KPIs(KPIsBase):
    pass  # For now, we'll just use the base class

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
        s.append(f"LP_USDC: {retail_lp.USDC()}, LP_WETH: {retail_lp.WETH()}")
        dataheader.extend(["LP_USDC", "LP_WETH"])
        datarow.extend([retail_lp.USDC(), retail_lp.WETH()])
        

        # Log Trader's funds
        #s.append(f"Trader_USDC: {noise_trader.USDC()}, Trader_WETH: {noise_trader.WETH()}")
        #dataheader.extend(["Trader_USDC", "Trader_WETH"])
        #datarow.extend([noise_trader.USDC(), noise_trader.WETH()])

        # Log global pool state
        global_state = weth_usdc_pool.get_global_state()  # Replace with the actual method call
        for key, value in global_state.items():
            s.append(f"{key}: {value}")
            dataheader.append(key)
            datarow.append(value)
        
        # Log pool state
        # Initialize storage for tick and position data
        if not hasattr(netlist_createLogData, "tick_data_frames"):
            netlist_createLogData.tick_data_frames = []
        if not hasattr(netlist_createLogData, "position_data_frames"):
            netlist_createLogData.position_data_frames = []

        # Retrieve tick and position data as dictionaries
        tick_data_dict = weth_usdc_pool.get_pool_state_for_all_ticks(lower_price_interested=1000, upper_price_interested=2500)
        position_data_dict = weth_usdc_pool.get_pool_state_for_all_positions()

        # Convert dictionaries to DataFrames
        tick_df = pd.DataFrame.from_dict(tick_data_dict, orient='index')
        position_df = pd.DataFrame.from_dict(position_data_dict, orient='index')

        # Append DataFrames to lists
        netlist_createLogData.tick_data_frames.append(tick_df)
        netlist_createLogData.position_data_frames.append(position_df)

        # Write to CSV and reset
        tick_final_df = pd.concat(netlist_createLogData.tick_data_frames, keys=range(len(netlist_createLogData.tick_data_frames)))
        position_final_df = pd.concat(netlist_createLogData.position_data_frames, keys=range(len(netlist_createLogData.position_data_frames)))

        tick_final_df.to_csv("outdir_csv/tick_data.csv")
        position_final_df.to_csv("outdir_csv/position_data.csv")

        netlist_createLogData.tick_data_frames = []
        netlist_createLogData.position_data_frames = []
    except Exception as e:
        print("An error occurred:", e)

    return s, dataheader, datarow

def retail_LP_policy(state,agent):
    # for retail LP policy, ticks should be closer to current market price, positions should be added and reomved with price movement
    # State will carry the hyperparameters of policy (frequency, volatility,risk etc)
    # retail has less capital to invest
    # More wide position range choices

    actions = ['add_liquidity', 'remove_liquidity']
    
    # Choose a action (As price moves LP decides to add/ remove liquidty, more price movements more rebalancing)
    action = random.choice(actions)
    if action =='add_liquidity':
        current_price = sqrtp_to_price(weth_usdc_pool.pool.slot0()[0])
        price_lower = current_price * random.uniform(0.5, 0.9)  
        price_upper = current_price * random.uniform(1.1, 1.5)  
        amount_usd = random.uniform(10, 5000)
        return action, price_lower,price_upper,amount_usd 
    
    elif action =='remove_liquidity':
        lp_positions = weth_usdc_pool.get_lp_all_positions(agent._wallet.address)
        if lp_positions:
            position_to_remove = random.choice(lp_positions)
            tick_lower = position_to_remove['tick_lower']
            tick_upper = position_to_remove['tick_upper']
            amount = position_to_remove['liquidity']
            return action, tick_lower, tick_upper, amount
        else:
            #print("This LP doesn't contain any positions.")
            return None   

def inst_LP_policy(state):
    # More capital to invest
    # has concentrated positions
    # Doesn't rebalnces frequently
    actions = ['add_liquidity', 'remove_liquidity', 'hold']
    action_prob = [0.2, 0.1, 0.7]
    action = np.random.choice(actions, p=action_prob)

    if action == 'hold':
        return action, None, None, None
    
    elif action=='add_liquidity':
        current_price = sqrtp_to_price(weth_usdc_pool.pool.slot0()[0])
        price_lower = current_price * random.uniform(0.95, 0.98)
        price_upper = current_price * random.uniform(1.02, 1.05)
        amount_usd = random.uniform(5000, 10000)
        return action, price_lower, price_upper, amount_usd
    
    elif action=='remove_liquidity':
        #Add logic to select position to reomve for this LP (e.g if position is inactive then romve it)
        lp_positions = weth_usdc_pool.get_lp_all_positions(agent._wallet.address)
        if lp_positions:
            position_to_remove = random.choice(lp_positions)
            tick_lower = position_to_remove['tick_lower']
            tick_upper = position_to_remove['tick_upper']
            amount = position_to_remove['liquidity']
            return action, tick_lower, tick_upper, amount
        else:
            print("This LP doesn't contain any positions.")
            return None   
        
def stoikov_LP_policy(state):
    pass    

def grid_LP_policy(state):
    pass

def rl_LP_policy(state):
    # Here integrate RL agent which will perform some action based on it's policy
    
    action = "add_liquidity"
    print("Implement RL policy here")   
    return action,None,None,None

def noise_trader_policy(state):
    actions = ['swap_token0_for_token1', 'swap_token1_for_token0']

    # Performs random swaps (No strategy)
    # Amount should be less than trader's token 0 balance and should be a function of liquidty in pool (If there is less slippage more volume will be tarded)
    action = random.choice(actions)
    
    # Generate a random amount
    if action == 'swap_token0_for_token1':
        amount = random.uniform(0.05, 0.1) 
    else:
        amount=random.uniform(100,500)

    return action, amount

def whale_trader_policy(state):
    # Swap amount a function of liquidity depth Elastisity to execution price data from ganutlet's analysis
    current_price = sqrtp_to_price(weth_usdc_pool.pool.slot0()[0])
    if current_price < 1450 and current_price > 2500:
        action = 'swap_token1_for_token0'
        amount = random.uniform(1000, 5000)
    else:
        action='swap_token0_to_token1'
        amount=random.uniform(0.1, 0.5)

    return action, amount

# Agnets and pool reinitialization (netlist reinitilaze agents everytime simulation is run which is not correct beacuse in RL we want to same agents to interact with pool in each step of RL agent)
# Model trader behaviours based on gauntlet's analysis of trader elasticity
# Include feeCollected by agent in agent's state (balances), how can IL be claculated?
# Create Log data function (Make log data more comrehensive, by adding events at each step)
# Get derived state from raw state for infernce tasks (volume from swaps, fees_cumulative, volatility from price, etc.)
# Initialize new pool when done with testing because Tick data and position data is incomplete, as we are getting position data and tick from only those ticks which were stored in json file, but pool has money more initialized ticks and positions than those stored in json file