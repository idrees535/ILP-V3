import numpy as np
#from util.globaltokens import weth_usdc_pool
from util.base18 import toBase18, fromBase18,fromBase128,price_to_valid_tick,price_to_raw_tick,price_to_sqrtp,sqrtp_to_price,tick_to_sqrtp,liquidity0,liquidity1,eth
import random

def retail_lp_policy(state):
    # for retail LP policy, ticks should be closer to current market price, positions should be added and reomved with price movement
    # State will carry the hyperparameters of policy (frequency, volatility,risk etc)
    # retail has less capital to invest
    # More wide position range choices

    actions = ['add_liquidity', 'remove_liquidity']
    
    # Choose a action (As price moves LP decides to add/ remove liquidty, more price movements more rebalancing)
    action = random.choice(actions)
    if action =='add_liquidity':
        current_price = sqrtp_to_price(state.pool.pool.slot0()[0])
        tick_lower = price_to_valid_tick(current_price * random.uniform(0.5, 0.9))  
        tick_upper = price_to_valid_tick(current_price * random.uniform(1.1, 1.5))  
        amount_usd = random.uniform(1000, 10000)

        return action, tick_lower,tick_upper,amount_usd 
    
    elif action =='remove_liquidity':
        lp_positions = state.pool.get_lp_all_positions(state._wallet.address)
        
        if lp_positions:
            position_to_remove = random.choice(lp_positions)
            
            tick_lower = position_to_remove['tick_lower']
            tick_upper = position_to_remove['tick_upper']
            amount = position_to_remove['liquidity']
            return action, tick_lower, tick_upper, amount
        else:
            #print("This LP doesn't contain any positions.")
            return None
        

def inst_lp_policy(state):
    # More capital to invest
    # has concentrated positions
    # Doesn't rebalnces frequently
    actions = ['add_liquidity', 'remove_liquidity', 'hold']
    action_prob = [0.2, 0.1, 0.7]
    action = np.random.choice(actions, p=action_prob)

    if action == 'hold':
        return action, None, None, None
    
    elif action=='add_liquidity':
        current_price = sqrtp_to_price(state.pool.pool.slot0()[0])
        price_lower = current_price * random.uniform(0.95, 0.98)
        price_upper = current_price * random.uniform(1.02, 1.05)
        amount_usd = random.uniform(5000, 50000)
        return action, price_lower, price_upper, amount_usd
    
    elif action=='remove_liquidity':
        #Add logic to select position to reomve for this LP (e.g if position is inactive then romve it)
        lp_positions = state.pool.get_lp_all_positions(state._wallet.address)
        if lp_positions:
            position_to_remove = random.choice(lp_positions)
            tick_lower = position_to_remove['tick_lower']
            tick_upper = position_to_remove['tick_upper']
            amount = position_to_remove['liquidity']
            return action, tick_lower, tick_upper, amount
        else:
            #print("This LP doesn't contain any positions.")
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
        amount = random.uniform(1, 5) 
    else:
        amount=random.uniform(2000,10000)

    return action, amount

def whale_trader_policy(state):
    # Swap amount a function of liquidity depth Elastisity to execution price data from ganutlet's analysis
    current_price = sqrtp_to_price(state.pool.pool.slot0()[0])
    if current_price < 1450 and current_price > 2500:
        action = 'swap_token1_for_token0'
        amount = random.uniform(1000, 5000)
    else:
        action='swap_token0_to_token1'
        amount=random.uniform(0.1, 0.5)

    return action, amount


def noise_trader_policy_1(state, params):
    drift, volatility, slippage_tolerance = params['drift'], params['volatility'], params['slippage_tolerance']
    signal = np.random.normal(drift, volatility)
    expected_slippage = state['expected_slippage']
    
    if expected_slippage > slippage_tolerance:
        return 'no_action'
    
    action = 'buy' if signal > 0 else 'sell'
    return action

def whale_trader_policy_1(state, params):
    market_depth = state['market_depth']
    limit_price = params['limit_price']
    market_price = state['market_price']
    
    if market_price > limit_price:
        return 'no_action'
    
    if market_depth > params['threshold']:
        return 'large_buy'
    else:
        return 'twap_buy'

def arb_trader_policy_1(state, params):
    price_diff = state['price_pool1'] - state['price_pool2']
    expected_slippage = state['expected_slippage']
    execution_cost = state['execution_cost']
    
    net_profit = price_diff - expected_slippage - execution_cost
    
    if net_profit > params['profit_threshold']:
        return 'buy_from_pool1_sell_in_pool2'
    else:
        return 'no_action'
    
def update_slippage_tolerance(state, params):
    recent_slippages = state['recent_slippages']  # List or other data structure
    avg_recent_slippage = sum(recent_slippages) / len(recent_slippages)
    elasticity_coefficient = params['elasticity_coefficient']  # Hyperparameter
    
    new_tolerance = avg_recent_slippage * elasticity_coefficient
    return new_tolerance

def retail_lp_policy_1(state, params):
    if state['price'] > params['high_threshold']:
        return 'remove_liquidity'
    elif state['price'] < params['low_threshold']:
        return 'add_liquidity'
    
def stoikov_lp_policy_1(state, params):
    # Implement Stoikov's model and return action
    pass

def static_lp_policy(state, params):
    # +- 10%, 20%,30% around current price and passively readjust position when out of the money
    
    return 'fixed_range'

def rl_lp_policy_1(state, params,agent):
    obs=weth_usdc_pool.get_global_state()
    # ML or optimization logic here
    model=agent.load_model()
    action, _states = model.predict(obs, deterministic=True)
    return action

