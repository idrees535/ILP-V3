# import brownie
import sys
import os

# Add parent directory to sys.path to handle imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .constants import  GOD_ACCOUNT
# from .utility_functions import *
from models.uniswap_model import UniV3Model

import random

# Pool configurations as a list of dictionaries
pool_configs = [
    {
        "token0": "WETH",
        "token1": "USDC",
        "supply_token0": 1e40,
        "supply_token1": 1e40,
        "decimals_token0": 18,
        "decimals_token1": 18,
        "fee_tier": 3000,
        "initial_pool_price": 2000,
        "initial_liquidity_amount_token1": 10000000
    },
    {
        "token0": "ETH",
        "token1": "DAI",
        "supply_token0": 1e40,
        "supply_token1": 1e40,
        "decimals_token0": 18,
        "decimals_token1": 18,
        "fee_tier": 3000,
        "initial_pool_price": 1000,
        "initial_liquidity_amount_token1": 10000000
    },
    {
        "token0": "BTC",
        "token1": "USDT",
        "supply_token0": 1e40,
        "supply_token1": 1e40,
        "decimals_token0": 18,
        "decimals_token1": 18,
        "fee_tier": 3000,
        "initial_pool_price": 60000,
        "initial_liquidity_amount_token1": 1000000000
    },
    {
        "token0": "BTC",
        "token1": "WETH",
        "supply_token0": 1e40,
        "supply_token1": 1e40,
        "decimals_token0": 18,
        "decimals_token1": 18,
        "fee_tier": 3000,
        "initial_pool_price": 20,  # 20 WETH
        "initial_liquidity_amount_token1": 10000
    }
]

# Select a random pool configuration
selected_pool_config = random.choice(pool_configs)

# Parameters common to all pools
deployer = GOD_ACCOUNT
sync_pool = True

# Create a single instance of UniV3Model based on the randomly selected configuration
selected_pool = UniV3Model(
    token0=selected_pool_config["token0"],
    token1=selected_pool_config["token1"],
    decimals_token0=selected_pool_config["decimals_token0"],
    decimals_token1=selected_pool_config["decimals_token1"],
    supply_token0=selected_pool_config["supply_token0"],
    supply_token1=selected_pool_config["supply_token1"],
    fee_tier=selected_pool_config["fee_tier"],
    initial_pool_price=selected_pool_config["initial_pool_price"],
    deployer=deployer,
    sync_pool=sync_pool,
    initial_liquidity_amount_token1=selected_pool_config["initial_liquidity_amount_token1"]
)

# Now `selected_pool` contains the UniV3Model instance created with the randomly selected pool configuration
