import numpy as np
#from util.globaltokens import weth_usdc_pool
from util.base18 import toBase18, fromBase18,fromBase128,price_to_valid_tick,price_to_raw_tick,price_to_sqrtp,sqrtp_to_price,tick_to_sqrtp,liquidity0,liquidity1,eth
import random
import pprint


q96 = 2**96
MAX_SAFE_INTEGER = (1 << 53) - 1

def calc_amount0(liq, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    # Ensure the result does not exceed MAX_SAFE_INTEGER
    amount0 = int(liq * q96 * (pb - pa) / pa / pb)
    return min(amount0, MAX_SAFE_INTEGER)

def calc_amount1(liq, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    # Ensure the result does not exceed MAX_SAFE_INTEGER
    amount1 = int(liq * (pb - pa) / q96)
    return min(amount1, MAX_SAFE_INTEGER)

def noise_trader_policy(state):
    actions = ['swap_token0_for_token1', 'swap_token1_for_token0']
    
    action = random.choice(actions)
    
    # Determine slippage tolerance between 1% and 14%
    slippage_tolerance = random.uniform(0.01, 0.05)
    
    global_state = state.pool.get_global_state()
    pool_price = global_state['curr_price']
    sqrt_price = price_to_sqrtp(pool_price)
    liquidity = global_state['liquidity_raw'] 
    
    # Calculate the upper and lower price bounds based on slippage tolerance
    if action == 'swap_token0_for_token1':
        price_impact_upper_bound = price_to_sqrtp(pool_price * (1 + slippage_tolerance))
        
        token0_amount = calc_amount0(liquidity, sqrt_price, price_impact_upper_bound)
        # Cap token0_amount to avoid exceeding MAX_SAFE_INTEGER
        token0_amount = min(token0_amount, MAX_SAFE_INTEGER)
        swap_amount = fromBase18(token0_amount)
    else:
        price_impact_lower_bound = price_to_sqrtp(pool_price * (1 - slippage_tolerance))
        
        token1_amount = calc_amount1(liquidity, price_impact_lower_bound, pool_price)
        # Cap token1_amount to avoid exceeding MAX_SAFE_INTEGER
        token1_amount = min(token1_amount, MAX_SAFE_INTEGER)
        swap_amount = fromBase18(token1_amount)
    
    # Cap the final swap amount and add randomness to simulate trader behavior
    swap_amount = min(swap_amount, MAX_SAFE_INTEGER)
    swap_amount =  swap_amount * random.uniform(0.0001, 0.0005)  #random.uniform(0,1)*pool_price
 
    return action, swap_amount


def retail_lp_policy(state):
    actions = ['add_liquidity', 'remove_liquidity']
    
    # Choose an action (retail LPs add/remove liquidity with price movements)
    action = random.choice(actions)
    
    if action == 'add_liquidity':
        print("\nADD LIQUIDITY\n")
        global_state = state.pool.get_global_state()
        liquidity = global_state['liquidity_raw']
        current_price = sqrtp_to_price(state.pool.pool.slot0()[0])
        
        # Calculate price bounds
        price_lower = current_price * random.uniform(0.5, 0.9)
        tick_lower = price_to_valid_tick(price_lower)
        
        price_upper = current_price * random.uniform(1.1, 1.5)
        tick_upper = price_to_valid_tick(price_upper)
        
        # Calculate liquidity for token0 and token1
        liq_token0 = calc_amount0(liquidity, price_to_sqrtp(price_lower), price_to_sqrtp(price_upper))
        liq_token1 = calc_amount1(liquidity, price_to_sqrtp(price_lower), price_to_sqrtp(price_upper))
        
        # Calculate total liquidity and cap it to avoid exceeding MAX_SAFE_INTEGER
        total_liq = liq_token0 * current_price + liq_token1
        total_liq = min(total_liq, MAX_SAFE_INTEGER)
        
        liquidity_percentage = random.uniform(0.0001, 0.0005)  # Retail LPs allocate a small percentage of liquidity
        liq_amount_token1 = fromBase18(liquidity_percentage * total_liq) #random.uniform(0, 1)*current_price 

        return action, tick_lower, tick_upper, liq_amount_token1
    
    elif action == 'remove_liquidity':
        print("\nREMOVE LIQUIDITY\n")
        lp_positions = state.pool.get_lp_all_positions(state._wallet.address)

        if lp_positions:
            position_to_remove = random.choice(lp_positions)
            print(f"Position going to remove : {position_to_remove}")
            tick_lower = position_to_remove['tick_lower']
            tick_upper = position_to_remove['tick_upper']
            amount = position_to_remove['liquidity']
            
            return action, tick_lower, tick_upper, amount
        else:
            # No positions to remove
            return None

def update_slippage_tolerance(state, params):
    recent_slippages = state['recent_slippages']  # List or other data structure
    avg_recent_slippage = sum(recent_slippages) / len(recent_slippages)
    elasticity_coefficient = params['elasticity_coefficient']  # Hyperparameter
    
    # Calculate new slippage tolerance and cap it to MAX_SAFE_INTEGER
    new_tolerance = avg_recent_slippage * elasticity_coefficient
    new_tolerance = min(new_tolerance, MAX_SAFE_INTEGER)
    
    return new_tolerance
