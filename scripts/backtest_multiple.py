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
from util.hardhat_control import start_hardhat_node,stop_hardhat_node
from scripts.predict_action import PredictAction
import json

start_hardhat_node()
def backtest_ilp(start_date, end_date, token0, token1, pool_id, agent_path, rebalancing_frequency, agent):
    current_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    # Initialize the PredictAction class
    predictor = PredictAction(agent_path, agent)
    all_positions = []

    while current_date <= end_date:
        curr_date_str = current_date.strftime('%Y-%m-%d')
        # Step 3: Predict new positions
        action, action_dict, action_ticks = predictor.predict_action(pool_id,curr_date_str)
        
        # Step 4: Rebalance portfolio
        end_interval = current_date + timedelta(days=rebalancing_frequency)
        start_date_str = current_date.strftime('%Y-%m-%d %H:%M:%S')
        end_date_str = end_interval.strftime('%Y-%m-%d %H:%M:%S')

        if agent == "ddpg":
            action_lower = action_dict["price_lower"]
            action_upper = action_dict["price_upper"]
            print(f"\n_______________________________DDPG AGENT ACTIONS")
            print(f"\n{action}")
            print(f"\n{action_dict}")
            print(f"\n{action_ticks}\n")
            
        else:
            action_lower = action_dict["price_lower"]
            action_upper = action_dict["price_upper"]
            print(f"\n_______________________________PPO AGENT ACTIONS")
            print(f"\n{action}")
            print(f"\n{action_dict}")
            print(f"\n{action_ticks}\n")
        
        # Collect all positions in a list
        all_positions.append({
            "start": convert_to_unix_timestamp(start_date_str),
            "end": convert_to_unix_timestamp(end_date_str),
            "lower_price": (action_lower),
            "upper_price": (action_upper),
        })
        # Move to the next rebalancing date
        current_date = end_interval

    # Step 5: Send all positions to the simulator API in a single request
    response_list = simulate_position(token0, token1, all_positions)
    # if 'LP_positions' not in response_json:
    #     print(f"Error: 'LP_positions' not found in response or response is None: {response_json}")
    #     return pd.DataFrame(), pd.DataFrame()
    data_df, result_df = convert_responses_to_dataframe_and_aggregate_final(response_list)

    return data_df, result_df

# Function to convert date string to Unix timestamp
def convert_to_unix_timestamp(date_str):
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    return int(dt.timestamp())

# Function to split the positions while maintaining the original structure
def split_positions_for_api(data):
    separated_objects = []

    for position in data['positions']:
        # Create a new object with a single-element positions list
        new_object = {**data, "positions": [position]}
        separated_objects.append(new_object)
    return separated_objects

# Function to convert responses to a DataFrame
def convert_responses_to_dataframe_and_aggregate_final(responses):
    data = []
    final_result_aggregate = {
        "total_PnL": 0,
        "total_fee_value": 0,
        "total_fee_yield": 0,
        "total_impermanent_loss": 0,
        "total_portfolio_value_end": 0,
        "total_portfolio_value_start": 0,
        "count": 0
    }
    
    for response in responses:
        if response["status"] == "success":
            lp_info = response["LP_positions"][0]["info"]
            burn = response["LP_positions"][0]["burn"]
            mint = response["LP_positions"][0]["mint"]
            swap = response["LP_positions"][0]["swap"]
            final_result = response["final_result"]

            # Add relevant data to a dictionary
            row = {
                "APR": lp_info["APR"],
                "Impermanent_loss_info": lp_info["Impermanent_loss"],
                "PnL_info": lp_info["PnL"],
                "X_start": lp_info["X_start"],
                "Y_start": lp_info["Y_start"],
                "lower_price_info": lp_info["lower_price"],
                "upper_price_info": lp_info["upper_price"],
                "burn_X_collect": burn["X_collect"],
                "burn_Y_collect": burn["Y_collect"],
                "mint_X_left": mint["X_left"],
                "mint_Y_left": mint["Y_left"],
                "swap_X_after_swap": swap["X_after_swap"],
                "swap_Y_after_swap": swap["Y_after_swap"]
            }
            data.append(row)

            # Aggregate final results
            final_result_aggregate["total_PnL"] += final_result["PnL"]
            final_result_aggregate["total_fee_value"] = final_result["fee_value"]
            final_result_aggregate["total_fee_yield"] += final_result["fee_yield"]
            final_result_aggregate["total_impermanent_loss"] += final_result["impermanent_loss"]
            if final_result_aggregate["count"] == 0:
                final_result_aggregate["portfolio_value_start"] = final_result["portfolio_value_start"]
            final_result_aggregate["total_portfolio_value_end"] = final_result["portfolio_value_end"]
            final_result_aggregate["count"] += 1

    # Convert to DataFrame
    data_df = pd.DataFrame(data)

    # Calculate averages for aggregated fields
    aggregated_final_result = {
        "average_PnL": final_result_aggregate["total_PnL"],
        "average_fee_value": final_result_aggregate["total_fee_value"],
        "average_fee_yield": final_result_aggregate["total_fee_yield"],
        "average_impermanent_loss": final_result_aggregate["total_impermanent_loss"],
        "average_portfolio_value_start": final_result_aggregate["total_portfolio_value_start"],
        "average_portfolio_value_end": final_result_aggregate["total_portfolio_value_end"]
    }
    print(f"\n\n {json.dumps(aggregated_final_result,indent=4)}")
    result_df = pd.DataFrame([aggregated_final_result])
    return data_df, result_df

def simulate_position(token0, token1, positions):
    vector = {
        "datatype": "raw",
        "pool": f"{pool_id}",
        "fee_tier": 1000,
        "token0": token0,
        "token1": token1,
        "range_type": "price",
        "positions": positions
    }
    url = "http://localhost:5050/MVP"

    multiple_vectors = split_positions_for_api(vector)
    all_responses =[]
    for vec in multiple_vectors:
        print(json.dumps(vec,indent=4))
        response = requests.post(url, json=vec)
        print(f"___________Voyager Response:\n{response.text}")
        response_json = response.json()
        all_responses.append(response_json)

    # print(f"\n\n {all_responses}")
    return all_responses

# def save_data_to_df(response_json):
#     data = []
#     for position in response_json.get('LP_positions', []):
#         burn_data = position.get('burn', {})
#         info_data = position.get('info', {})
#         mint_data = position.get('mint', {})
#         swap_data = position.get('swap', {})

#         data.append({
#             'start': info_data.get('start'),
#             'end': info_data.get('end'),
#             'curr_price': burn_data.get('burn_price') / 1e10,
#             'lower_price': info_data.get('lower_price'),
#             'upper_price': info_data.get('upper_price'),
#             'X_start': info_data.get('X_start'),
#             'Y_start': info_data.get('Y_start'),
#             'liquidity': mint_data.get('liquidity'),
#             'X_left': mint_data.get('X_left')/1e8,
#             'X_mint': mint_data.get('X_mint')/1e8,
#             'Y_left': mint_data.get('Y_left')/1e18,
#             'Y_mint': mint_data.get('Y_mint')/1e18,
#             'X_fee': burn_data.get('X_fee')/1e8,
#             'X_reserve': burn_data.get('X_reserve')/1e8,
#             'Y_fee': burn_data.get('Y_fee')/1e18,
#             'Y_reserve': burn_data.get('Y_reserve')/1e18,
#             'APR': info_data.get('APR'),
#             'Impermanent_loss': info_data.get('Impermanent_loss'),
#             'PnL': info_data.get('PnL'),
#             'Yield': info_data.get('Yield')
#         })

#     final_result = response_json.get('final_result', {})
#     final_result_data = {
#         'final_PnL': final_result.get('PnL'),
#         'final_fee_value': final_result.get('fee_value')/1e18,
#         'final_fee_yield': final_result.get('fee_yield'),
#         'final_impermanent_loss': final_result.get('impermanent_loss'),
#         'final_portfolio_value_end': final_result.get('portfolio_value_end')/1e18,
#         'final_portfolio_value_start': final_result.get('portfolio_value_start')/1e18
#     }

#     data_df = pd.DataFrame(data)
#     result_df = pd.DataFrame([final_result_data])

#     return data_df, result_df

# Example usage
start_date = '2024-02-01'
end_date = '2024-02-14'
agent = "ppo"
agent_name = "ppo_20241026_215047"
agent_path = f'model_storage/{agent}/{agent_name}'
pool_id = "0x4e68ccd3e89f51c3074ca5072bbac773960dfa36" 

budget = 10000 # Initail total budget 
price=2284  
token0 = ((budget/2)/price)*1e18
token1 = (budget/2)*1e6
rebalancing_frequency = 7

data_df, results_df = backtest_ilp(start_date, end_date, token0, token1, pool_id, agent_path, rebalancing_frequency, agent)

results_df.to_csv(f"model_output/backtest/results_{agent_name}.csv")
data_df.to_csv(f"model_output/backtest/data_df_{agent_name}.csv")
stop_hardhat_node()