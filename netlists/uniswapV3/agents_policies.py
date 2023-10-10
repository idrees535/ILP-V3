import numpy as np
from util.globaltokens import weth_usdc_pool

def noise_trader_policy(state, params):
    drift, volatility, slippage_tolerance = params['drift'], params['volatility'], params['slippage_tolerance']
    signal = np.random.normal(drift, volatility)
    expected_slippage = state['expected_slippage']
    
    if expected_slippage > slippage_tolerance:
        return 'no_action'
    
    action = 'buy' if signal > 0 else 'sell'
    return action

def whale_trader_policy(state, params):
    market_depth = state['market_depth']
    limit_price = params['limit_price']
    market_price = state['market_price']
    
    if market_price > limit_price:
        return 'no_action'
    
    if market_depth > params['threshold']:
        return 'large_buy'
    else:
        return 'twap_buy'

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

def retail_lp_policy(state, params):
    if state['price'] > params['high_threshold']:
        return 'remove_liquidity'
    elif state['price'] < params['low_threshold']:
        return 'add_liquidity'
    
def stoikov_lp_policy(state, params):
    # Implement Stoikov's model and return action
    pass

def static_lp_policy(state, params):
    # +- 10%, 20%,30% around current price and passively readjust position when out of the money
    
    return 'fixed_range'

def rl_lp_policy(state, params,agent):
    obs=weth_usdc_pool.get_global_state()
    # ML or optimization logic here
    model=agent.load_model()
    action, _states = model.predict(obs, deterministic=True)
    return action

