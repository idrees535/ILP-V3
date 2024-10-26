import numpy as np
#from util.globaltokens import weth_usdc_pool
from util.utility_functions import toBase18, fromBase18,fromBase128,price_to_valid_tick,price_to_raw_tick,price_to_sqrtp,sqrtp_to_price,tick_to_sqrtp,liquidity0,liquidity1,eth
import random
from .constants import WALLET_LP


q96 = 2**96
MAX_SAFE_INTEGER = (1 << 53) - 1

def calc_amount0(liq, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    amount0 = int(liq * q96 * (pb - pa) / pa / pb)
    return amount0

def calc_amount1(liq, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    amount1 = int(liq * (pb - pa) / q96)
    return amount1


def noise_trader_policy(state):
    actions = ['swap_token0_for_token1', 'swap_token1_for_token0']
    action = random.choice(actions)
    
    # Determine slippage tolerance between 1% and 14%
    slippage_tolerance = random.uniform(0.01, 0.05)
    
    global_state = state.pool.get_global_state()
    pool_price = global_state['curr_price']
    sqrt_price = price_to_sqrtp(pool_price)
    liquidity = global_state['liquidity_raw'] 
    price_impact_upper_bound = price_to_sqrtp(pool_price * (1 + slippage_tolerance))
    price_impact_lower_bound = price_to_sqrtp(pool_price * (1 - slippage_tolerance))
    print (f"\n```````````````````````````````````````Current pool price  : {pool_price}")
    print (f"``````````````````````````````````````Current pool loquidity : {fromBase18(liquidity)}")
    if action == 'swap_token0_for_token1':
        # token0_amount = calc_amount0(liquidity, sqrt_price, price_impact_upper_bound)
        token0_amount=liquidity0(liquidity,price_impact_upper_bound,price_impact_lower_bound)
        token0_amount = fromBase18(token0_amount)
        swap_amount = min(token0_amount ,100000)
        swap_amount = swap_amount * random.uniform(0.00009,0.0009)
    else:
        # token1_amount = calc_amount1(liquidity, price_impact_lower_bound, pool_price)
        token1_amount=liquidity1(liquidity,price_impact_upper_bound,price_impact_lower_bound)
        token1_amount = fromBase18(token1_amount)
        swap_amount = min(token1_amount,100000)
        swap_amount = swap_amount* random.uniform(0.08, 1.4)
        
    print (f"SWAP AMOUNT : {swap_amount}")
    return action, swap_amount


def retail_lp_policy(state):
    actions = ['add_liquidity', 'remove_liquidity']
    action = random.choice(actions)
    
    if action == 'add_liquidity':
        print("\nADD LIQUIDITY")
        global_state = state.pool.get_global_state()
        liquidity = global_state['liquidity_raw']
        pool_price = global_state['curr_price']
        print (f"\n```````````````````````````````````````Current pool price  : {pool_price}")
        print (f"``````````````````````````````````````Current pool loquidity : {fromBase18(liquidity)}")    
        # Calculate price bounds
        price_lower = pool_price * random.uniform(0.5, 0.9)
        tick_lower = price_to_valid_tick(price_lower)
        
        price_upper = pool_price * random.uniform(1.1, 1.5)
        tick_upper = price_to_valid_tick(price_upper)
        
        # Calculate liquidity for token0 and token1
        liq_token0 = calc_amount0(liquidity, price_to_sqrtp(price_lower), price_to_sqrtp(price_upper))
        liq_token1 = calc_amount1(liquidity, price_to_sqrtp(price_lower), price_to_sqrtp(price_upper))
        
        # Calculate total liquidity and cap it to avoid exceeding MAX_SAFE_INTEGER
        total_liq = liq_token0 * pool_price + liq_token1
        liquidity_percentage = random.uniform(0.01, 0.5)  # Retail LPs allocate a small percentage of liquidity
        liq_amount_token1 = fromBase18(liquidity_percentage * total_liq) #random.uniform(0, 1)*current_price 
        liq_amount_token1 = min (liq_amount_token1, 1000)
        return action, tick_lower, tick_upper, liq_amount_token1
    
    elif action == 'remove_liquidity':
        print("\nREMOVE LIQUIDITY")
        lp_positions = state.pool.get_lp_all_positions(WALLET_LP.address)

        if lp_positions:
            position_to_remove = random.choice(lp_positions)
            print(f"Position going to remove : {position_to_remove}")
            tick_lower = position_to_remove['tick_lower']
            tick_upper = position_to_remove['tick_upper']
            amount = position_to_remove['liquidity']
            
            return action, tick_lower, tick_upper, amount
        else:
            print(" No positions to remove ")
            return None

def update_slippage_tolerance(state, params):
    recent_slippages = state['recent_slippages']  # List or other data structure
    avg_recent_slippage = sum(recent_slippages) / len(recent_slippages)
    elasticity_coefficient = params['elasticity_coefficient']  # Hyperparameter
    
    # Calculate new slippage tolerance and cap it to MAX_SAFE_INTEGER
    new_tolerance = avg_recent_slippage * elasticity_coefficient
    new_tolerance = min(new_tolerance, MAX_SAFE_INTEGER)
    
    return new_tolerance
