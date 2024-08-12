# %% [markdown]
# # Backtest Strategy

# %%
from datetime import datetime, timedelta
from datetime import datetime, timedelta
import requests
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

def save_data_to_df(response_json):
    data = []
    for position in response_json.get('LP_positions', []):
        burn_data = position.get('burn', {})
        info_data = position.get('info', {})
        mint_data = position.get('mint', {})
        swap_data = position.get('swap', {})

        data.append({
            'start': info_data.get('start'),
            'end': info_data.get('end'),
            'curr_price': burn_data.get('burn_price') / 1e10,
            'lower_price': info_data.get('lower_price'),
            'upper_price': info_data.get('upper_price'),
            'X_start': info_data.get('X_start'),
            'Y_start': info_data.get('Y_start'),
            'liquidity': mint_data.get('liquidity'),
            'X_left': mint_data.get('X_left')/1e8,
            'X_mint': mint_data.get('X_mint')/1e8,
            'Y_left': mint_data.get('Y_left')/1e18,
            'Y_mint': mint_data.get('Y_mint')/1e18,
            'X_fee': burn_data.get('X_fee')/1e8,
            'X_reserve': burn_data.get('X_reserve')/1e8,
            'Y_fee': burn_data.get('Y_fee')/1e18,
            'Y_reserve': burn_data.get('Y_reserve')/1e18,
            'APR': info_data.get('APR'),
            'Impermanent_loss': info_data.get('Impermanent_loss'),
            'PnL': info_data.get('PnL'),
            'Yield': info_data.get('Yield')
        })

    final_result = response_json.get('final_result', {})
    final_result_data = {
        'final_PnL': final_result.get('PnL'),
        'final_fee_value': final_result.get('fee_value')/1e18,
        'final_fee_yield': final_result.get('fee_yield'),
        'final_impermanent_loss': final_result.get('impermanent_loss'),
        'final_portfolio_value_end': final_result.get('portfolio_value_end')/1e18,
        'final_portfolio_value_start': final_result.get('portfolio_value_start')/1e18
    }

    data_df = pd.DataFrame(data)
    final_result_df = pd.DataFrame([final_result_data])

    return data_df, final_result_df

def plot_prices_over_time(data_df):
    plt.figure(figsize=(10, 6))
    plt.plot(data_df['start'], data_df['lower_price'], label='Lower Price')
    plt.plot(data_df['start'], data_df['upper_price'], label='Upper Price')
    plt.plot(data_df['start'], data_df['curr_price'], label='Current Price')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Price Ranges Over Time')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_apr_over_time(data_df):
    plt.figure(figsize=(10, 6))
    plt.plot(data_df['start'], data_df['APR'])
    plt.xlabel('Date')
    plt.ylabel('APR')
    plt.title('APR Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_il_over_time(data_df):
    plt.figure(figsize=(10, 6))
    plt.plot(data_df['start'], data_df['Impermanent_loss'])
    plt.xlabel('Date')
    plt.ylabel('Impermanent Loss')
    plt.title('Impermanent Loss Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_pnl_over_time(data_df):
    plt.figure(figsize=(10, 6))
    plt.plot(data_df['start'], data_df['PnL'])
    plt.xlabel('Date')
    plt.ylabel('PnL')
    plt.title('PnL Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_yield_over_time(data_df):
    plt.figure(figsize=(10, 6))
    plt.plot(data_df['start'], data_df['Yield'])
    plt.xlabel('Date')
    plt.ylabel('Yield')
    plt.title('Yield Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def price_to_raw_tick(price):
    return math.floor(math.log(price) / math.log(1.0001))

def price_to_valid_tick(price, tick_spacing=60):
    raw_tick = math.floor(math.log(price, 1.0001))
    remainder = raw_tick % tick_spacing
    if remainder != 0:
        # Round to the nearest valid tick, considering tick spacing.
        raw_tick += tick_spacing - remainder if remainder >= tick_spacing // 2 else -remainder
    return raw_tick

def datetime_to_unix_timestamp(date_str, format='%Y-%m-%d %H:%M:%S'):
    dt = datetime.strptime(date_str, format)
    return int(time.mktime(dt.timetuple()))

def backtest_ilp(start_date, end_date, X_reserve, Y_reserve, pool_id, ddpg_agent_path, ppo_agent_path, rebalancing_frequency, agent):
    current_date = datetime.strptime(start_date, '%d-%m-%y')
    end_date = datetime.strptime(end_date, '%d-%m-%y')

    all_positions = []

    while current_date <= end_date:
        curr_date_str = current_date.strftime('%Y-%m-%d')
        # Step 3: Predict new positions
        ddpg_action, ppo_action = get_inference(ddpg_agent_path, ppo_agent_path, pool_id, curr_date_str)
        
        # Step 4: Rebalance portfolio
        start_interval = current_date
        end_interval = current_date + timedelta(days=rebalancing_frequency)
        start_date_str = start_interval.strftime('%Y-%m-%d %H:%M:%S')
        end_date_str = end_interval.strftime('%Y-%m-%d %H:%M:%S')

        start_timestamp = datetime_to_unix_timestamp(start_date_str)
        end_timestamp = datetime_to_unix_timestamp(end_date_str)

        if agent == "ddpg":
            price_lower = ddpg_action['price_lower']
            price_upper = ddpg_action['price_upper']
            
        else:
            price_lower = ppo_action['price_lower']
            price_upper = ppo_action['price_upper']
            
        print(f'price_lower: {price_lower} & price_upper: {price_upper}')
        # Collect all positions in a list
        all_positions.append({
            "start": start_timestamp,
            "end": end_timestamp,
            "lower_tick": price_to_valid_tick(price_lower,1),
            "upper_tick": price_to_valid_tick(price_upper,1),
        })


        # Move to the next rebalancing date
        current_date += timedelta(days=rebalancing_frequency)

    # Step 5: Send all positions to the simulator API in a single request
    response = simulate_position(X_reserve, Y_reserve, all_positions)
    response_json = response.json()

    if 'LP_positions' not in response_json:
        print(f"Voyager API response:{response_json}")
        return pd.DataFrame(), pd.DataFrame()

    # Process the response to save data to a DataFrame
    data_df, results_df = save_data_to_df(response_json)

    return data_df, results_df

def get_inference(ddpg_agent_path='model_storage/ddpg/ddpg_1', ppo_agent_path='model_storage/ppo/lstm_actor_critic_batch_norm', pool_id="0xcbcdf9626bc03e24f779434178a73a0b4bad62ed", date_str='2024-05-05'):
    url = "https://ilp.tempestfinance.xyz/api/v1/inference/"
    data = {
        "pool_state": {
            "current_profit": 500,
            "price_out_of_range": False,
            "time_since_last_adjustment": 40000,
            "pool_volatility": 0.2
        },
        "user_preferences": {
            "risk_tolerance": {
                "profit_taking": 50,
                "stop_loss": -500
            },
            "investment_horizon": 7,
            "liquidity_preference": {
                "adjust_on_price_out_of_range": True
            },
            "risk_aversion_threshold": 0.1,
            "user_status": "new_user"
        },
        "pool_id": pool_id,
        "ddpg_agent_path": ddpg_agent_path,
        "ppo_agent_path": ppo_agent_path,
    }
    
    # Make the POST request
    response = requests.post(url, json=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        try:
            response_json = response.json()
            ddpg_action = response_json.get('ddpg_action', {})
            ppo_action = response_json.get('ppo_action', {})
            #print(f'ppo-action: {ppo_action} & ddpg_action: {ddpg_action}')
            return ddpg_action, ppo_action
        except ValueError:
            print("Failed to parse JSON response.")
            return None, None
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return None, None

def simulate_position(X_reserve, Y_reserve, positions):
    print(positions)
    vector = {
        "datatype": "raw",
        "fee_tier": 1000,
        "pool": "0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa",
        "token0": X_reserve,
        "token1": Y_reserve,
        "range_type": "tick",
        "positions": positions
    }
    url = "https://voyager-simulation.tempestfinance.xyz/MVP"
    response = requests.post(url, json=vector)
    print(response.text)

    return response

start_date = '11-05-24'
end_date = '11-06-24'
ddpg_agent_path = 'model_storage/ddpg/ddpg_1'
ppo_agent_path = 'model_storage/ppo/lstm_actor_critic_batch_norm'
pool_id = "0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa"
agent = "ppo"

budget_eth=10 # Initail total ETH reserves for liquidity position
btc_eth_price=18 #btc/ETH price
X_reserve = 1
Y_reserve = 1
rebalancing_frequency = 7


data_df, results_df = backtest_ilp(start_date, end_date, X_reserve, Y_reserve, pool_id, ddpg_agent_path, ppo_agent_path, rebalancing_frequency, agent)


# %%
print(results_df)
print(data_df)


# Plotting the results
plot_prices_over_time(data_df)
plot_apr_over_time(data_df)
plot_il_over_time(data_df)
plot_pnl_over_time(data_df)
plot_yield_over_time(data_df)
