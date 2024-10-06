import requests
import pandas as pd
import datetime
from datetime import datetime, timedelta,timezone
def sync_pool_data(pool_id= "0x4e68ccd3e89f51c3074ca5072bbac773960dfa36", UNISWAP_V3_SUBGRAPH_URL = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'):
    # Fetching liquidity positions
    liquidity_query = """
    {
      positions(where: { pool: "%s" ,liquidity_gt: "0"}) {
        liquidity
        tickLower {
          price0
          tickIdx
        }
        tickUpper {
          price0
          tickIdx
        }
      }
    }
    """ % pool_id
    liquidity_data = requests.post(UNISWAP_V3_SUBGRAPH_URL, json={'query': liquidity_query}).json()

    tick_query="""
     {
     ticks(where: {liquidityGross_gt: "0", liquidityNet_gt: "0", pool: "%s"}) {
     
      tickIdx
      liquidityGross
      liquidityNet
      
    }
  }
  """% pool_id
    tick_data = requests.post(UNISWAP_V3_SUBGRAPH_URL, json={'query': tick_query}).json()
    # Fetching pool price, volume, fees, and reserves
    global_query = """
    {
      pools(where: { id: "%s" }) {
        token0Price
        token1Price
        volumeUSD
        feesUSD
        totalValueLockedToken0
        totalValueLockedToken1
      }
    }
    """ % pool_id
    global_data = requests.post(UNISWAP_V3_SUBGRAPH_URL, json={'query': global_query}).json()

    # Fetching swaps and trades
    swaps_query = """
    {
      swaps(where: { pool: "%s" }, first: 10, orderBy: timestamp, orderDirection: desc) {
        amount0
        amount1
        amountUSD
        timestamp
      }
    }
    """ % pool_id
    swaps_data = requests.post(UNISWAP_V3_SUBGRAPH_URL, json={'query': swaps_query}).json()

        # Constructing the state representation with float data type
    state = {
        'positions': [(float(position['tickLower']['tickIdx']), float(position['tickUpper']['tickIdx']), float(position['liquidity'])) for position in liquidity_data['data']['positions']],
        'pool_price': float(global_data['data']['pools'][0]['token1Price']),
        'ticks': [(float(tick['tickIdx']), float(tick['liquidityGross']), float(tick['liquidityNet'])) for tick in tick_data['data']['ticks']]
    }

    return state

import requests
def fetch_inference_pool_data(pool_id):
    UNISWAP_V3_SUBGRAPH_URL = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'
    
    # GraphQL query
    query = """
    {
      pool(id: \"%s\") {
        feeGrowthGlobal0X128
        feeGrowthGlobal1X128
        liquidity
        token1Price
      }
    }
    """ % pool_id

    response = requests.post(UNISWAP_V3_SUBGRAPH_URL,json={'query': query}).json()

    # Extracting data from response
    data = response['data']['pool']

    # Constructing the state representation with float data type
    state = {
        'feeGrowthGlobal0X128': float(data['feeGrowthGlobal0X128']),
        'feeGrowthGlobal1X128': float(data['feeGrowthGlobal1X128']),
        'liquidity': float(data['liquidity']),
        'token1Price': float(data['token1Price'])
    }
    return state

from datetime import timezone
def fetch_inference_pool_data_1(pool_id='0xcbcdf9626bc03e24f779434178a73a0b4bad62ed', date_str='2024-05-03'):
    UNISWAP_V3_SUBGRAPH_URL = "https://api.goldsky.com/api/public/project_clvon2puehf5a01zb9axv0oa8/subgraphs/uniswap-v3-mainnet/1.0.0/gn"
    
    # Convert date to timestamp
    date = datetime.strptime(date_str, '%Y-%m-%d')
    timestamp = int(date.timestamp())
    timestamp = int(date.replace(tzinfo=timezone.utc).timestamp())
    # GraphQL query for poolDayData
    query = """
    {
      poolDayDatas(where: {pool: \"%s\", date: %d}) {
        date
        volumeToken0
        volumeToken1
        liquidity
        token1Price
      }
    }
    """ % (pool_id, timestamp)

    response = requests.post(UNISWAP_V3_SUBGRAPH_URL, json={'query': query}).json()

    # Extracting data from response
    if 'data' in response and 'poolDayDatas' in response['data'] and len(response['data']['poolDayDatas']) > 0:
        data = response['data']['poolDayDatas'][0]

        # Constructing the state representation with float data type
        state = {
            #'date': datetime.utcfromtimestamp(data['date']).strftime('%Y-%m-%d'),
            'feeGrowthGlobal0X128': float(data['volumeToken0']),
            'feeGrowthGlobal1X128': float(data['volumeToken1']),
            'liquidity': float(data['liquidity']),
            'token1Price': float(data['token1Price'])
        }
        return state
    else:
        print(f"No data found for pool {pool_id} on {date_str}")
        return None
