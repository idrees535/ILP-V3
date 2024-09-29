
from enforce_typing import enforce_types
from engine import AgentBase
from util.constants import GOD_ACCOUNT
from util.tx import txdict
import brownie
from util.base18 import toBase18,log_event_to_csv
from util.globaltokens import weth_usdc_pool
from util.tx import _fees, transferETH

@enforce_types
class UniswapV3SwapperAgent():
    def __init__(self, name, token0, token1 ,policy_func,pool):
        super().__init__(name, token0, token1)
        
        self.pool=pool
        self.policy=policy_func
        self._token0=token0
        self._token1=token1
        self.pool.fundToken0FromAbove(self._wallet.address, toBase18(token0))
        self.pool.fundToken1FromAbove(self._wallet.address, toBase18(token1))
        transferETH(GOD_ACCOUNT,self._wallet.address,1)

    def takeStep(self):
        action,amount = self.policy(self)

        if action == 'swap_token0_for_token1':
            tx_receipt=self.pool.swap_token0_for_token1(self._wallet.address, toBase18(amount), data=b'')
            #print(tx_receipt.events)
            #log_event_to_csv(tx_receipt)
            
        
        elif action == 'swap_token1_for_token0':
            tx_receipt=self.pool.swap_token1_for_token0(self._wallet.address, toBase18(amount), data=b'')
            #print(tx_receipt.events)
            #log_event_to_csv(tx_receipt)

    