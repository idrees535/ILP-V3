
from enforce_typing import enforce_types
import os
import sys
from util.constants import GOD_ACCOUNT,WALLET_LP,WALLET_SWAPPER
import brownie
# Add parent directory to sys.path to handle imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.utility_functions import *

@enforce_types
class UniswapV3SwapperAgent():
    def __init__(self, token0, token1 ,policy_func,pool):
        #super().__init__( token0, token1)
        
        self.pool=pool
        self.policy=policy_func
        self._token0=token0
        self._token1=token1
        #wallet_swapper =brownie.network.accounts[0]
        self.pool.fundToken0FromAbove(WALLET_SWAPPER.address, toBase18(token0))
        self.pool.fundToken1FromAbove(WALLET_SWAPPER.address, toBase18(token1))
        # transferETH(GOD_ACCOUNT,WALLET_SWAPPER.address,toBase18(10000))

    def takeStep(self):
        action,amount = self.policy(self)

        print(f"____SWAPPER WALLET {self.pool.get_wallet_balances(WALLET_SWAPPER.address)} ")
        print(f"____Swap amount:   {amount} toBase18: {toBase18(amount)} \n")

        if action == 'swap_token0_for_token1':
            tx_receipt=self.pool.swap_token0_for_token1(WALLET_SWAPPER.address, toBase18(amount), data=b'')
            #print(tx_receipt.events)
            #log_event_to_csv(tx_receipt)
        elif action == 'swap_token1_for_token0':
            tx_receipt=self.pool.swap_token1_for_token0(WALLET_SWAPPER.address, toBase18(amount), data=b'')
            #print(tx_receipt.events)
            #log_event_to_csv(tx_receipt)


 