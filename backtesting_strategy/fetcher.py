import pandas as pd
import requests
import pickle
from itertools import compress
import time

##############################################################
# Get Swaps from Uniswap v3's subgraph, and liquidity at each swap from Flipside Crypto
##############################################################

def fetchUniswapv3(query: str,network='mainnet') -> dict:
    if network == 'mainnet':
        univ3_graph_url = 'https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3'
      
    params = {'query': query}
        
    response = requests.post(univ3_graph_url, json=params)
    return response.json()

# def getSwapData(poolAddress,fileName,downloadData,network='mainnet',rangeBlocks=[]):       
#     """
#     Internal function to query full history of swap data from Uniswap v3's subgraph.
#     Use GetPoolData.get_pool_data_flipside which preprocesses the data in order to conduct simualtions with the Active Strategy Framework.
#     """
#     request_swap = [] 
    
#     if downloadData:

#         for blockNumber in rangeBlocks:
#           queryBuilt = queryBuilderUniswapv3Swap(poolAddress,blockNumber)
#           response = fetchUniswapv3(queryBuilt,network)["data"]["swaps"]
          
#           request_swap.extend(response)
                
#           with open('./data/'+fileName+'_swap.pkl', 'wb+') as output:
#             pickle.dump(request_swap, output, pickle.HIGHEST_PROTOCOL)
#     else:
#         with open('./data/'+fileName+'_swap.pkl', 'rb') as input:
#             request_swap = pickle.load(input)
           
#     return request_swap

def getSwapsData(poolAddress,fromTimestamp,toTimestamp,fileName,downloadData,network='mainnet'):       
    """
    Internal function to query full history of swap data from Uniswap v3's subgraph.
    Use GetPoolData.get_pool_data_flipside which preprocesses the data in order to conduct simualtions with the Active Strategy Framework.
    """
    swaps = [] 
    fetch = True

    if downloadData:
      while fetch:
        if not len(swaps):
          _fromTimestamp = fromTimestamp
        else:
          _fromTimestamp = swaps[-1]["timestamp"]

        queryBuilt = queryBuilderUniswapv3Swap(poolAddress,_fromTimestamp,toTimestamp)
        response = fetchUniswapv3(queryBuilt,network)["data"]["swaps"]

        swaps.extend(response)

        if len(response) < 100:
          fetch = False
    else:
      with open('./data/'+fileName+'_swap.pkl', 'rb') as input:
        swaps = pickle.load(input)
    
    if downloadData:
      with open('./data/'+fileName+'_swap.pkl', 'wb+') as output:
        pickle.dump(swaps, output, pickle.HIGHEST_PROTOCOL)

    return swaps

# 0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640
def queryBuilderUniswapv3Swap(poolAddress,fromT,toT):
  return '''
  {
    swaps(
        where:{ 
        pool:"'''+poolAddress+'''",
        timestamp_gt:'''+str(fromT)+''',
        timestamp_lt:'''+str(toT)+'''
      },
      first:100,
      orderBy: timestamp,
      orderDirection: asc
    )
    {
      amount0
      amount1
      amountUSD
      tick
      timestamp
      token0{
        symbol
        decimals
      }
      token1{
        symbol
        decimals
      }
    }
  }
  '''
   