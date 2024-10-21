from datetime import datetime, timedelta
# import datetime
import requests
import pandas as pd
import os 
import sys 
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from util.utility_functions import price_to_valid_tick
from util.constants import *
from scripts.predict_action import predict_action


def backtest_ilp(start_date, end_date, token0, token1, pool_id, ddpg_agent_path, ppo_agent_path, rebalancing_frequency, agent):
    current_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    all_positions = []

    while current_date <= end_date:
        curr_date_str = current_date.strftime('%Y-%m-%d')
        # Step 3: Predict new positions
        ddpg_action,ddpg_action_dict,ddpg_action_ticks,ppo_action, ppo_action_dict,ppo_action_ticks = get_inference(ddpg_agent_path, ppo_agent_path, pool_id, curr_date_str)
        print(f"DDPG Action:     {ddpg_action}")
        
        # Step 4: Rebalance portfolio
        start_interval = current_date
        end_interval = current_date + timedelta(days=rebalancing_frequency)
        start_date_str = start_interval.strftime('%Y-%m-%d %H:%M:%S')
        end_date_str = end_interval.strftime('%Y-%m-%d %H:%M:%S')

        if agent == "ddpg":
            action_lower = ddpg_action_dict["price_lower"]
            action_upper = ddpg_action_dict["price_upper"]
            print(f"\n_______________________________ANGENT :  DDPG")
            print(f"\nDDPG ACTION: {ddpg_action}")
            print(f"\n{ddpg_action_dict}")
            print(f"\n{ddpg_action_ticks}\n")
            
        else:
            action_lower = ppo_action_dict["price_lower"]
            action_upper = ppo_action_dict["price_upper"]
            print(f"\n_______________________________ANGENT :  PPO")
            print(f"\nPPO ACTION: {ppo_action}")
            print(f"\n{ppo_action_dict}")
            print(f"\n{ppo_action_ticks}\n")

        # print(f"Price lower  :  {action_lower}    :  {type(action_lower)}")
        # print(f"Price upper  :  {action_upper}    :  {type(action_upper)}")
        
        # Collect all positions in a list
        all_positions.append({
            "start": convert_to_unix_timestamp(start_date_str),
            "end": convert_to_unix_timestamp(end_date_str),
            "lower_tick": (action_lower),
            "upper_tick": (action_upper),
        })

        # Move to the next rebalancing date
        current_date += timedelta(days=rebalancing_frequency)

    # Step 5: Send all positions to the simulator API in a single request
    response = simulate_position(token0, token1, all_positions)
    response_json = response.json()

    if 'LP_positions' not in response_json:
        print(f"Error: 'LP_positions' not found in response or response is None: {response_json}")
        return pd.DataFrame(), pd.DataFrame()

    # Process the response to save data to a DataFrame
    data_df, results_df = save_data_to_df(response_json)

    return data_df, results_df

# Function to convert date string to Unix timestamp
def convert_to_unix_timestamp(date_str):
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    return int(dt.timestamp())

def get_inference(ddpg_agent_path='model_storage/ddpg/ddpg_1', ppo_agent_path='model_storage/ppo/lstm_actor_critic_batch_norm', pool_id="0xcbcdf9626bc03e24f779434178a73a0b4bad62ed", date_str='2024-05-05'):
    # url = "http://127.0.0.1:8000/predict_action/"
    # data = {
    #     "pool_id": pool_id,
    #     "ddpg_agent_path": ddpg_agent_path,
    #     "ppo_agent_path": ppo_agent_path,
    #     "date_str": date_str
    # }
    ddpg_action,ddpg_action_dict,ddpg_action_ticks,ppo_action, ppo_action_dict,ppo_action_ticks = predict_action(pool_id,ddpg_agent_path,ppo_agent_path,date_str)
    # response = requests.post(url, json=data)
    # response_json = response.json()
    # ddpg_action = response_json['ddpg_action']
    # ppo_action = response_json['ppo_action']
    # print(f"\n\nPPO ACTION: {ppo_action}")
    # print(f"\n{ppo_action_dict}")
    # print(f"\n{ppo_action_ticks}\n")

    return ddpg_action,ddpg_action_dict,ddpg_action_ticks,ppo_action, ppo_action_dict,ppo_action_ticks

def simulate_position(token0, token1, positions):
    vector = {
        "datatype": "raw",
        "fee_tier": 1000,
        "token0": token0,
        "token1": token1,
        "range_type": "price",
        "positions": positions
    }
    print(vector)
    url = "http://localhost:5050/MVP"
    response = requests.post(url, json=vector)
    print(response.text)

    return response

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

# Example usage
start_date = '2024-03-01'
end_date = '2024-04-01'
agent_name = "ddpg_1"
ddpg_agent_path = f'model_storage/ddpg/{agent_name}'
ppo_agent_path = 'model_storage/ppo/lstm_actor_critic_batch_norm'
pool_id = "0xcbcdf9626bc03e24f779434178a73a0b4bad62ed" # BTC/ETH pool
agent = "ddpg"

budget_eth=10 # Initail total ETH reserves for liquidity position
btc_eth_price=18 #btc/ETH price
token0 = (budget_eth/2)/btc_eth_price 
token1 = budget_eth/2
rebalancing_frequency = 7

data_df, results_df = backtest_ilp(start_date, end_date, token0, token1, pool_id, ddpg_agent_path, ppo_agent_path, rebalancing_frequency, agent)

results_df.to_csv(f"model_output/backtest/results_{agent_name}.csv")
data_df.to_csv(f"model_output/backtest/data_df_{agent_name}.csv")