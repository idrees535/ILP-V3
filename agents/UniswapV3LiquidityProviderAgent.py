import brownie
from enforce_typing import enforce_types
from engine import AgentBase
from util.globaltokens import weth_usdc_pool
import random
import pandas as pd
from util.base18 import log_event_to_csv

@enforce_types
class UniswapV3LiquidityProviderAgent(AgentBase.AgentBaseEvmBoth):
    def __init__(self, name: str,token0,token1,policy_func):
        super().__init__(name,token0,token1)

        self.pool=weth_usdc_pool
        self.policy=policy_func
        self._token0=token0
        self._token1=token1
        self.agent_name=name
    
    def takeStep(self, state):
        try:
            liquidity_action, tick_lower, tick_upper, amount = self.policy(state, self)
        except TypeError:
            print(f"Policy returned None, no action will be taken by {self.agent_name} wallet: {self._wallet.address} at this timestep.")
            return None

        if liquidity_action == "add_liquidity":
            tx_receipt= self.pool.add_liquidity(self._wallet.address, tick_lower, tick_upper, amount, b'')
            #log_event_to_csv(tx_receipt)

        elif liquidity_action == "remove_liquidity":
            tx_receipt = self.pool.remove_liquidity_with_liquidty(self._wallet.address, tick_lower, tick_upper, amount)
            log_event_to_csv(tx_receipt)
            tx_receipt,_ = self.pool.collect_fee(self._wallet.address, tick_lower, tick_upper)
            log_event_to_csv(tx_receipt)

        elif liquidity_action == "hold":
            print("Do Nothing (HODOOR)")
