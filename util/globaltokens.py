import brownie
from enforce_typing import enforce_types


from .constants import  GOD_ACCOUNT
from util.base18 import toBase18
from util.tx import txdict
from model_scripts.UniswapV3_Model_v2 import UniV3Model
from model_scripts.sync_pool_subgraph_data import sync_pool_data


_TOKEN1 = None
_TOKEN0 = None

token0 = "WETH"
token1 = "USDC"
supply_token0 = 1e18
supply_token1 = 1e18
decimals_token0 = 18
decimals_token1 = 18
fee_tier = 3000
initial_pool_price = 2000
deployer = GOD_ACCOUNT
sync_pool=True
initial_liquidity_amount_token1=10000000
#state=sync_pool_data(pool_id= "0x4e68ccd3e89f51c3074ca5072bbac773960dfa36")
weth_usdc_pool = UniV3Model(token0, token1,decimals_token0,decimals_token1,supply_token0,supply_token1,fee_tier,initial_pool_price,deployer,sync_pool, initial_liquidity_amount_token1)

token0 = "ETH"
token1 = "DAI"
supply_token0 = 1e18
supply_token1 = 1e18
decimals_token0 = 18
decimals_token1 = 18
fee_tier = 3000
initial_pool_price = 1000
deployer = GOD_ACCOUNT
sync_pool=True
initial_liquidity_amount_token1=10000000
eth_dai_pool = UniV3Model(token0, token1,decimals_token0,decimals_token1,supply_token0,supply_token1,fee_tier,initial_pool_price,deployer,sync_pool, initial_liquidity_amount_token1)

token0 = "BTC"
token1 = "USDT"
supply_token0 = 1e18
supply_token1 = 1e18
decimals_token0 = 18
decimals_token1 = 18
fee_tier = 3000
initial_pool_price = 60000
deployer = GOD_ACCOUNT
sync_pool=True
initial_liquidity_amount_token1=10000000000

btc_usdt_pool = UniV3Model(token0, token1,decimals_token0,decimals_token1,supply_token0,supply_token1,fee_tier,initial_pool_price,deployer,sync_pool, initial_liquidity_amount_token1)


token0 = "BTC"
token1 = "WETH"
supply_token0 = 1e18
supply_token1 = 1e18
decimals_token0 = 18
decimals_token1 = 18
fee_tier = 3000
initial_pool_price = 20 # 20 WETH
deployer = GOD_ACCOUNT
sync_pool=True
initial_liquidity_amount_token1=10000000000

btc_weth_pool = UniV3Model(token0, token1,decimals_token0,decimals_token1,supply_token0,supply_token1,fee_tier,initial_pool_price,deployer,sync_pool, initial_liquidity_amount_token1)
#Instantiante pool class
#weth_usdc_pool = UniV3Model(True,10000000000,2000)

@enforce_types
def Token1():
    global _TOKEN1  # pylint: disable=global-statementcocoe
    token=_TOKEN1=weth_usdc_pool.token1
    return token

@enforce_types
def Token0():
    global _TOKEN0  # pylint: disable=global-statement
    token=_TOKEN0=weth_usdc_pool.token0
    return token

@enforce_types
def Token1_address() -> str:
    return Token1().address

@enforce_types
def Token0_address() -> str:
    return Token0().address

@enforce_types
def fundToken1FromAbove(dst_address: str, amount_base: int):
    tx_receipt=Token1().transfer(dst_address, amount_base, txdict(GOD_ACCOUNT))
    print(f'funded account with token1: {tx_receipt.events}')

@enforce_types
def fundToken0FromAbove(dst_address: str, amount_base: int):
    tx_receipt=Token0().transfer(dst_address, amount_base, txdict(GOD_ACCOUNT))
    print(f'funded account with token0: {tx_receipt.events}')