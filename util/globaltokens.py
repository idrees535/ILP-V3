import brownie
from enforce_typing import enforce_types


from .constants import  GOD_ACCOUNT
from util.base18 import toBase18
from util.tx import txdict
from model_scripts.UniswapV3_Model_v2 import UniV3Model


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
initial_liquidity_amount=10000
weth_usdc_pool = UniV3Model(token0, token1,decimals_token0,decimals_token1,supply_token0,supply_token1,fee_tier,initial_pool_price,deployer,sync_pool, initial_liquidity_amount)

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
    #print(tx_receipt.events)

@enforce_types
def fundToken0FromAbove(dst_address: str, amount_base: int):
    tx_receipt=Token0().transfer(dst_address, amount_base, txdict(GOD_ACCOUNT))
    #print(tx_receipt.events)