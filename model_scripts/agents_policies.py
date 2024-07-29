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
    
    global_state = state.ppool.get_global_state()
    pool_price = global_state['curr_price']
    sqrt_price=price_to_sqrtp(pool_price)
    liquidity = global_state['liquidity_raw'] 
    
    # Calculate the upper and lower price bounds based on slippage tolerance
    if action == 'swap_token0_for_token1':
        price_impact_upper_bound = price_to_sqrtp(pool_price * (1 + slippage_tolerance))

        token0_amount = calc_amount0(liquidity, sqrt_price, price_impact_upper_bound)
        swap_amount = token0_amount
    else:
        price_impact_lower_bound = price_to_sqrtp(pool_price * (1 - slippage_tolerance))
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
        global_state = state.pool.get_global_state()
        liquidity = global_state['liquidity_raw'] 
        current_price = sqrtp_to_price(state.pool.pool.slot0()[0])
        price_lower=current_price * random.uniform(0.5, 0.9)
        tick_lower = price_to_valid_tick(price_lower)  
        price_upper=current_price * random.uniform(1.1, 1.5)
        tick_upper = price_to_valid_tick(price_upper)

        liq_token0=calc_amount0(liquidity,price_to_sqrtp(price_lower),price_to_sqrtp(price_upper)) 
        liq_token1=calc_amount1(liquidity,price_to_sqrtp(price_lower),price_to_sqrtp(price_upper))
        total_liq=liq_token0*current_price+liq_token1
        liquidity_percentage = random.uniform(0.01, 0.05)  # Retail LPs may allocate 1% to 5% of total liquidity
        liq_amount_token1 = liquidity_percentage * total_liq 
        #amount_token1 = random.uniform(5000, 50000)

        return action, tick_lower,tick_upper,liq_amount_token1 
    
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

