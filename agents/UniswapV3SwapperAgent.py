#Kuch hua
# Nahi hua
from enforce_typing import enforce_types
from engine import AgentBase
from util.constants import GOD_ACCOUNT
from util.tx import txdict
import brownie
from util.base18 import toBase18,log_event_to_csv
from util.globaltokens import weth_usdc_pool

@enforce_types
class UniswapV3SwapperAgent(AgentBase.AgentBaseEvmBoth):
    def __init__(self, name, weth, usdc ,policy_func):
        super().__init__(name, weth, usdc)
        
        self.pool=weth_usdc_pool
        self.policy=policy_func
        self.weth=weth
        self.usdc=usdc

    def takeStep(self, state):
        action,amount = self.policy(state)

        if action == 'swap_token0_for_token1':
            tx_receipt=self.pool.swap_token0_for_token1(self._wallet.address, toBase18(amount), data=b'')
            log_event_to_csv(tx_receipt)
            
        
        elif action == 'swap_token1_for_token0':
            tx_receipt=self.pool.swap_token1_for_token0(self._wallet.address, toBase18(amount), data=b'')
            log_event_to_csv(tx_receipt)

    