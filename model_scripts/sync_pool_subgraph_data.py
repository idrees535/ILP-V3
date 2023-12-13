import requests
import pandas as pd
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