import brownie
from enforce_typing import enforce_types


from .constants import  GOD_ACCOUNT
from util.base18 import toBase18
from util.tx import txdict
from scripts.UniswapV3_Model import UniV3Model


_USDC_TOKEN = None
_WETH_TOKEN = None

#Instantiante pool class
weth_usdc_pool = UniV3Model(True,10000000000,2000)

@enforce_types
def USDCToken():
    global _USDC_TOKEN  # pylint: disable=global-statementcocoe
    token=_USDC_TOKEN=weth_usdc_pool.usdc
    return token

@enforce_types
def WETHToken():
    global _WETH_TOKEN  # pylint: disable=global-statement
    token=_WETH_TOKEN=weth_usdc_pool.weth
    return token

@enforce_types
def USDC_address() -> str:
    return USDCToken().address

@enforce_types
def WETH_address() -> str:
    return WETHToken().address

@enforce_types
def fundUSDCFromAbove(dst_address: str, amount_base: int):
    tx_receipt=USDCToken().transfer(dst_address, amount_base, txdict(GOD_ACCOUNT))
    #print(tx_receipt.events)

@enforce_types
def fundWETHFromAbove(dst_address: str, amount_base: int):
    tx_receipt=WETHToken().transfer(dst_address, amount_base, txdict(GOD_ACCOUNT))
    #print(tx_receipt.events)