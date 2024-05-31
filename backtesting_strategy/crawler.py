import requests
import json
from datetime import datetime
import pandas as pd
import sys
import os
from time import time
from tqdm import tqdm

def fetch_swap_events(pool_address, first=1000, output_file=None):
    all_swaps = []
    last_id = ""
    batch_number = 0
    total_records_saved = 0

    while True:
        # Define the GraphQL query with variables for pagination
        query = """
        {
          swaps(
            first: %d
            where: {pool: "%s", id_gt: "%s"}
            orderBy: id
            orderDirection: asc
          ) {
            id
            transaction {
              id
              blockNumber
            }
            recipient
            sender
            sqrtPriceX96
            tick
            amount0
            amount1
            logIndex
            timestamp
          }
        }
        """ % (first, pool_address, last_id)

        # Define the endpoint URL
        url = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'

        # Define the headers
        headers = {
            'Content-Type': 'application/json'
        }

        try:
            # Make the request
            response = requests.post(url, headers=headers, data=json.dumps({'query': query}))

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                swaps = data['data']['swaps']

                if not swaps:
                    break  # Exit the loop if no more swaps are returned

                # Reformat the swaps
                reformatted_swaps = reformat_swaps(swaps)

                # Convert to DataFrame and append to CSV
                df = pd.DataFrame(reformatted_swaps, dtype=str)
                if batch_number == 0:
                    df.to_csv(output_file, index=False, mode='w')  # Write header for the first batch
                else:
                    df.to_csv(output_file, index=False, header=False,
                              mode='a')  # Append without header for subsequent batches

                batch_number += 1
                total_records_saved += len(reformatted_swaps)
                last_id = swaps[-1]['id']  # Update last_id to the id of the last fetched swap

                # Print the number of records saved every 10,000 records
                if total_records_saved % 1000 == 0:
                    sys.stdout.write(f"\r\t{total_records_saved} records have been saved.")
                    sys.stdout.flush()
            else:
                print(f"Query failed to run with status code {response.status_code}")
                print(response.text)
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    # Ensure to print the final count
    sys.stdout.write(f"\r\t{total_records_saved} records have been saved.\n")
    sys.stdout.flush()


def reformat_swaps(swaps):
    reformatted = []
    for swap in swaps:
        reformatted.append({
            'evt_block_number': swap['transaction']['blockNumber'],
            'evt_block_time': datetime.fromtimestamp(int(swap['timestamp'])).strftime('%Y-%m-%d %H:%M:%S'),
            'evt_tx_hash': swap['transaction']['id'],
            'evt_index': swap['logIndex'],
            'recipient': swap['recipient'],
            'sender': swap['sender'],
            'sqrt_price_x96': swap['sqrtPriceX96'],
            'tick': swap['tick'],
            'amount0': swap['amount0'],
            'amount1': swap['amount1']
        })
    return reformatted


def fetch_mint_events(pool_address, first=1000, output_file=None):
    all_mints = []
    last_id = ""
    batch_number = 0
    total_records_saved = 0

    while True:
        # Define the GraphQL query with variables for pagination
        query = """
        {
          mints(
            first: %d
            where: {pool: "%s", timestamp_gte: "%s"}
            orderBy: timestamp
            orderDirection: asc
          ) {
            id
            transaction {
              id
              blockNumber
            }
            owner
            origin
            amount
            amount0
            amount1
            timestamp
            tickLower
            tickUpper
            logIndex
          }
        }
        """ % (first, pool_address, last_id)

        # Define the endpoint URL
        url = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'

        # Define the headers
        headers = {
            'Content-Type': 'application/json'
        }

        try:
            # Make the request
            response = requests.post(url, headers=headers, data=json.dumps({'query': query}))

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                mints = data['data']['mints']

                if not mints:
                    break  # Exit the loop if no more mints are returned

                # Reformat the mints
                reformatted_mints = reformat_mints(mints)

                # Convert to DataFrame and append to CSV
                df = pd.DataFrame(reformatted_mints, dtype=str)
                if batch_number == 0:
                    df.to_csv(output_file, index=False, mode='w')  # Write header for the first batch
                else:
                    df.to_csv(output_file, index=False, header=False,
                              mode='a')  # Append without header for subsequent batches

                batch_number += 1
                total_records_saved += len(reformatted_mints)
                last_id = mints[-1]['id']  # Update last_id to the id of the last fetched mint

                # Print the number of records saved every 10,000 records
                if total_records_saved % 1000 == 0:
                    sys.stdout.write(f"\r\t{total_records_saved} records have been saved.")
                    sys.stdout.flush()
            else:
                print(f"Query failed to run with status code {response.status_code}")
                print(response.text)
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    # Ensure to print the final count
    sys.stdout.write(f"\r\t{total_records_saved} records have been saved.\n")
    sys.stdout.flush()


def reformat_mints(mints):
    reformatted = []
    for mint in mints:
        reformatted.append({
            'evt_block_number': mint['transaction']['blockNumber'],
            'evt_block_time': datetime.fromtimestamp(int(mint['timestamp'])).strftime('%Y-%m-%d %H:%M:%S'),
            'evt_tx_hash': mint['transaction']['id'],
            'owner': mint['owner'],
            'origin': mint['origin'],
            'amount': mint['amount'],
            'amount0': mint['amount0'],
            'amount1': mint['amount1'],
            'tick_lower': mint['tickLower'],
            'tick_upper': mint['tickUpper'],
            "evt_index": mint['logIndex'],
        })
    return reformatted


def fetch_burn_events(pool_address, first=1000, output_file= None):
    all_burns = []
    last_id = ""
    batch_number = 0
    total_records_saved = 0

    while True:
        # Define the GraphQL query with variables for pagination
        query = """
        {
          burns(
            first: %d
            where: {pool: "%s", id_gt: "%s"}
            orderBy: id
            orderDirection: asc
          ) {
            id
            transaction {
              id
              blockNumber
            }
            owner
            origin
            amount
            amount0
            amount1
            timestamp
            tickLower
            tickUpper
            logIndex
          }
        }
        """ % (first, pool_address, last_id)

        # Define the endpoint URL
        url = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'

        # Define the headers
        headers = {
            'Content-Type': 'application/json'
        }

        try:
            # Make the request
            response = requests.post(url, headers=headers, data=json.dumps({'query': query}))

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()
                burns = data['data']['burns']

                if not burns:
                    break  # Exit the loop if no more burns are returned

                # Reformat the burns
                reformatted_burns = reformat_burns(burns)

                # Convert to DataFrame and append to CSV
                df = pd.DataFrame(reformatted_burns, dtype=str)
                if batch_number == 0:
                    df.to_csv(output_file, index=False, mode='w')  # Write header for the first batch
                else:
                    df.to_csv(output_file, index=False, header=False,
                              mode='a')  # Append without header for subsequent batches

                batch_number += 1
                total_records_saved += len(reformatted_burns)
                last_id = burns[-1]['id']  # Update last_id to the id of the last fetched burn

                # Print the number of records saved every 10,000 records
                if total_records_saved % 1000 == 0:
                    sys.stdout.write(f"\r\t{total_records_saved} records have been saved.")
                    sys.stdout.flush()
            else:
                print(f"Query failed to run with status code {response.status_code}")
                print(response.text)
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    # Ensure to print the final count
    sys.stdout.write(f"\r\t{total_records_saved} records have been saved.\n")
    sys.stdout.flush()


def reformat_burns(burns):
    reformatted = []
    for burn in burns:
        reformatted.append({
            'evt_block_number': burn['transaction']['blockNumber'],
            'evt_block_time': datetime.fromtimestamp(int(burn['timestamp'])).strftime('%Y-%m-%d %H:%M:%S'),
            'evt_tx_hash': burn['transaction']['id'],
            'owner': burn['owner'],
            'origin': burn['origin'],
            'amount': burn['amount'],
            'amount0': burn['amount0'],
            'amount1': burn['amount1'],
            'tick_lower': burn['tickLower'],
            'tick_upper': burn['tickUpper'],
            "evt_index": burn['logIndex'],
        })
    return reformatted


def crawl_events(pools_data_path, output_dir, limit_pool = 1000):
    pools_data = json.load(open(pools_data_path, "r"))
    for pool in pools_data[:limit_pool]:
        print(f"Fetching events for pool {pool['token0']}-{pool['token1']}-{pool['fee']}...")
        pool_address = pool["address"]
        pool_dir_name = f'{pool["token0"]}_{pool["token1"]}_{pool["fee"]}'
        pool_path = os.path.join(output_dir, pool_dir_name)
        if not os.path.exists(pool_path):
            os.makedirs(pool_path)
        print("\tFetching swap events...")
        swaps_output_file = os.path.join(pool_path, 'swap_events.csv')
        fetch_swap_events(pool_address, output_file=swaps_output_file)
        print("\tFetching mint events...")
        mints_output_file = os.path.join(pool_path, 'mint_events.csv')
        fetch_mint_events(pool_address, output_file=mints_output_file)
        print("\tFetching burn events...")
        burns_output_file = os.path.join(pool_path, 'burn_events.csv')
        fetch_burn_events(pool_address, output_file=burns_output_file)


def merge_events(pools_data_path, output_dir):
    pools_data = json.load(open(pools_data_path, "r"))
    for pool in tqdm(pools_data):
        pool_dir_name = f'{pool["token0"]}_{pool["token1"]}_{pool["fee"]}'
        pool_path = os.path.join(output_dir, pool_dir_name)
        decimal0 = pool['decimals0']
        decimal1 = pool['decimals1']
        df_swaps = pd.read_csv(os.path.join(pool_path, 'swap_events.csv'))
        df_mints = pd.read_csv(os.path.join(pool_path, 'mint_events.csv'))
        df_burns = pd.read_csv(os.path.join(pool_path, 'burn_events.csv'))

        # Add 'type' column to each DataFrame
        df_swaps['type'] = 'swap'
        df_mints['type'] = 'mint'
        df_burns['type'] = 'burn'

        # Drop the 'origin' column from the mint and burn DataFrames
        df_mints.drop(columns=['origin', 'owner'], inplace=True)
        df_burns.drop(columns=['origin', 'owner'], inplace=True)
        rename_columns = {
            'tick_lower': 'tickLower',
            'tick_upper': 'tickUpper',
            'sqrt_price_x96': 'sqrtPriceX96'
        }
        df_mints.rename(columns=rename_columns, inplace=True)
        df_burns.rename(columns=rename_columns, inplace=True)
        df_swaps.rename(columns=rename_columns, inplace=True)

        df_all_events = pd.concat([df_swaps, df_mints, df_burns], ignore_index=True)
        df_all_events['amount0'] = df_all_events['amount0'].apply(lambda x: int(float(x) * 10 ** decimal0))
        df_all_events['amount1'] = df_all_events['amount1'].apply(lambda x: int(float(x) * 10 ** decimal1))
        df_all_events.sort_values(['evt_block_number', 'evt_index'], inplace=True)
        df_all_events.to_csv(os.path.join(pool_path, 'all_events.csv'), index=False)


def get_pool_state(address, block_number):
        query = """
        {
          pool(
            id: "%s",
            block: { number: %d }
          ) {
            sqrtPrice
            tick
            liquidity
            feeGrowthGlobal0X128
            feeGrowthGlobal1X128
          }
        }
        """ % (address, block_number)

        url = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, json={'query': query})
        if response.status_code == 200:
            data = response.json()
            return data['data']['pool']
        else:
            print(f"Query failed for block {block_number} with status code {response.status_code}")
            print(response.text)
            return None


def crawl_pool_state(pools_data_path, output_dir):
    pools_data = json.load(open(pools_data_path, "r"))
    for pool in pools_data:
        print(f"Fetching pool state for pool {pool['token0']}-{pool['token1']}-{pool['fee']}...")
        pool_dir_name = f'{pool["token0"]}_{pool["token1"]}_{pool["fee"]}'
        pool_path = os.path.join(output_dir, pool_dir_name)
        events_file = os.path.join(pool_path, 'all_events.csv')
        # Load the CSV file into a DataFrame, selecting only the required columns and parsing dates
        df_blocks = pd.read_csv(events_file, usecols=['evt_block_number', 'evt_block_time'],
                                parse_dates=['evt_block_time']).drop_duplicates()

        # Add a new column for the date only
        df_blocks['date'] = df_blocks['evt_block_time'].dt.date

        # Group by date and get the first block of each day
        first_blocks_each_day = df_blocks.sort_values(by='evt_block_time').groupby('date').first().reset_index()

        # List to hold pool states
        pool_states = []

        # Iterate over each block number in the DataFrame and get the pool state
        total_blocks = len(first_blocks_each_day)
        for index, row in first_blocks_each_day.iterrows():
            block_number = row['evt_block_number']
            pool_state = get_pool_state(pool['address'], block_number)
            if pool_state:
                pool_state['block_number'] = block_number
                pool_state['block_time'] = row['evt_block_time']
                pool_states.append(pool_state)
            # Print progress
            progress = (index + 1) / total_blocks * 100
            sys.stdout.write(f"\r\tProgress: {progress:.2f}% ({index + 1}/{total_blocks} blocks processed)")
            sys.stdout.flush()

        # Convert the list of pool states to a DataFrame
        df_pool_states = pd.DataFrame(pool_states)

        # Display the resulting DataFrame
        output_path = os.path.join(pool_path, 'pool_states.csv')
        df_pool_states.to_csv(output_path, index=False)
        print("\nFinished.")
        print("=====================================")


if __name__ == "__main__":
    pools_data_path = "./top_pools.json"
    output_dir = "./data/crawl/"
    crawl_events(pools_data_path, output_dir)
    merge_events(pools_data_path, output_dir)
    # crawl_pool_state(pools_data_path, output_dir)
    #
    # pools_data = json.load(open(pools_data_path, "r"))
    # # for pool in tqdm(pools_data):
    # pool = pools_data[0]
    # pool_dir_name = f'{pool["token0"]}_{pool["token1"]}_{pool["fee"]}'
    # pool_path = os.path.join(output_dir, pool_dir_name)
    # pool_address = pool["address"]
    # fetch_burn_events(pool_address, output_file=os.path.join(pool_path, 'burn_events_.csv'))
    #
    # decimal0 = pool['decimals0']
    # decimal1 = pool['decimals1']
    # df_swaps = pd.read_csv(os.path.join(pool_path, 'swap_events.csv'))
    # df_mints = pd.read_csv(os.path.join(pool_path, 'mint_events.csv'))
    # df_burns = pd.read_csv(os.path.join(pool_path, 'burn_events.csv'))
    # df_swaps['type'] = 'swap'
    # df_mints['type'] = 'mint'
    # df_burns['type'] = 'burn'
    #
    # # Drop the 'origin' column from the mint and burn DataFrames
    # df_mints.drop(columns=['origin', 'owner'], inplace=True)
    # df_burns.drop(columns=['origin', 'owner'], inplace=True)
    # rename_columns = {
    #     'tick_lower': 'tickLower',
    #     'tick_upper': 'tickUpper',
    #     'sqrt_price_x96': 'sqrtPriceX96'
    # }
    # df_mints.rename(columns=rename_columns, inplace=True)
    # df_burns.rename(columns=rename_columns, inplace=True)
    # df_swaps.rename(columns=rename_columns, inplace=True)
    # tx = '0xf5e3b7b55ed7323dd2d9031abe551517ae3f635e445b4854e5dfe6ae25b1d1e8'
    # df_all_events = pd.concat([df_swaps, df_mints, df_burns], ignore_index=True)
    # # row = df_all_events[df_all_events['evt_tx_hash'] == tx].iloc[0]
    # df_all_events['amount0_'] = df_all_events['amount0'].apply(lambda x: str(float(x) * 10 ** decimal0))
    # df_all_events['amount1_'] = df_all_events['amount1'].apply(lambda x: str(float(x) * 10 ** decimal1))
    # row = df_all_events[df_all_events['evt_tx_hash'] == tx].iloc[0]

