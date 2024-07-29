import numpy as np
#from util.globaltokens import weth_usdc_pool
from util.base18 import toBase18, fromBase18,fromBase128,price_to_valid_tick,price_to_raw_tick,price_to_sqrtp,sqrtp_to_price,tick_to_sqrtp,liquidity0,liquidity1,eth
import random

import random

q96 = 2**96

def calc_amount0(liq, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    return int(liq * q96 * (pb - pa) / pa / pb)

def calc_amount1(liq, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    return int(liq * (pb - pa) / q96)

def noise_trader_policy(state):
    actions = ['swap_token0_for_token1', 'swap_token1_for_token0']
    
    action = random.choice(actions)
    
    # Determine slippage tolerance between 1% and 14%
    slippage_tolerance = random.uniform(0.01, 0.14)
    
    global_state = state.get_global_state()
    pool_price = global_state['curr_price']
    liquidity = global_state['liquidity'] 
    
    # Calculate the upper and lower price bounds based on slippage tolerance
    if action == 'swap_token0_for_token1':
        price_impact_upper_bound = pool_price * (1 + slippage_tolerance)
        token0_amount = calc_amount0(liquidity, pool_price, price_impact_upper_bound)
        swap_amount = token0_amount
    else:
        price_impact_lower_bound = pool_price * (1 - slippage_tolerance)
        token1_amount = calc_amount1(liquidity, price_impact_lower_bound, pool_price)
        swap_amount = token1_amount
    
    return action, swap_amount



def retail_lp_policy(state):
    # for retail LP policy, ticks should be closer to current market price, positions should be added and reomved with price movement
    # retail has less capital to invest

    actions = ['add_liquidity', 'remove_liquidity']
    
    # Choose a action (As price moves LP decides to add/ remove liquidty, more price movements more rebalancing)
    action = random.choice(actions)
    if action =='add_liquidity':
        current_price = sqrtp_to_price(state.pool.pool.slot0()[0])
        tick_lower = price_to_valid_tick(current_price * random.uniform(0.5, 0.9))  
        tick_upper = price_to_valid_tick(current_price * random.uniform(1.1, 1.5))  
        amount_token1 = random.uniform(5000, 50000)

        return action, tick_lower,tick_upper,amount_token1 
    
    elif action =='remove_liquidity':
        lp_positions = state.pool.get_lp_all_positions(state._wallet.address)
        
        if lp_positions:
            position_to_remove = random.choice(lp_positions)
            
            tick_lower = position_to_remove['tick_lower']
            tick_upper = position_to_remove['tick_upper']
            amount = position_to_remove['liquidity']
            return action, tick_lower, tick_upper, amount
        else:
            #print("This LP doesn't have any positions.")
            return None
        

def noise_trader_policy_1(state):
    # Define actions
    actions = ['swap_token0_for_token1', 'swap_token1_for_token0']

    global_state = state.get_global_state()
    tick_state = state.get_tick_state(global_state['tick'])
    lp_positions = state.get_lp_all_positions(state._wallet.address)

    # Extract relevant information from the Uniswap model
    liquidity_depth = global_state['liquidity_raw']
    tick_liquidity_net = tick_state['liquidityNet_raw']
    pool_price = global_state['curr_price']
    recent_price_trend = random.choice(['up','down'])

    # Define threshold for price impact
    price_impact_threshold = 0.01  # 1% price impact threshold

    # Choose an action based on the state
    if liquidity_depth > 1000:
        # Check recent price trend and price impact
        if recent_price_trend == 'up':
            action = 'swap_token1_for_token0'
        elif recent_price_trend == 'down':
            action = 'swap_token0_for_token1'
        else:
            action = random.choice(actions)
    else:
        # Low liquidity, perform a random action
        action = random.choice(actions)

    # Generate a random amount
    max_amount = liquidity_depth / 10  # Example: Trade up to 10% of liquidity
    amount = random.uniform(1, max_amount)

    if amount<state.pool.token0.balanceOf(state._wallet_address) and action=='swap_token0_for_token1':
        amount=state.pool.token0.balanceOf(state._wallet_address)
        
    if amount<state.pool.token1.balanceOf(state._wallet_address) and action=='swap_token1_for_token0':
        amount=state.pool.token1.balanceOf(state._wallet_address)

    return action, amount


def informed_trader_policy(state):
    # Swap amount a function of liquidity depth Elastisity to execution price data from ganutlet's analysis
    current_price = sqrtp_to_price(state.pool.pool.slot0()[0])
    if current_price < 1450 and current_price > 2500:
        action = 'swap_token1_for_token0'
        amount = random.uniform(1000, 5000)
    else:
        action='swap_token0_to_token1'
        amount=random.uniform(0.1, 0.5)

    return action, amount


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


def arb_trader_policy(state, params):
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


def rl_lp_policy_1(state, params,agent):
    obs=state.pool.get_global_state()
    model=agent.load_model()
    action, _states = model.predict(obs, deterministic=True)
    return action

