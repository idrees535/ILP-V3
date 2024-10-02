import brownie
from enforce_typing import enforce_types
from engine import AgentBase
from util.globaltokens import GOD_ACCOUNT
import random
import pandas as pd
from util.base18 import log_event_to_csv
from util.base18 import toBase18, fromBase18
from util.tx import _fees, transferETH
from model_scripts.UniswapV3_Model_v2 import UniV3Model


@enforce_types
class UniswapV3LiquidityProviderAgent(AgentBase.AgentBaseEvmBoth):
    def __init__(self, name: str,token0,token1,policy_func,pool):
        super().__init__(name,token0,token1)

        self.pool=pool
        self.policy=policy_func
        self._token0=token0
        self._token1=token1
        self.agent_name=name
        #self._wallet =brownie.network.accounts[0]
        self.pool.fundToken0FromAbove(self._wallet.address, toBase18(token0))
        self.pool.fundToken1FromAbove(self._wallet.address, toBase18(token1))
        transferETH(GOD_ACCOUNT,self._wallet.address,100* 10**18)
        
    def takeStep(self, state):
        try:
            liquidity_action, tick_lower, tick_upper, amount = self.policy(self)
        except TypeError:
            print(f"Policy returned None, no action will be taken by {self.agent_name}")
            return None

        if liquidity_action == "add_liquidity":
            tx_receipt= self.pool.add_liquidity(self._wallet.address, tick_lower, tick_upper, amount, b'')
            #print(tx_receipt.events)
            #log_event_to_csv(tx_receipt)
            print(f"____________________{UniV3Model().get_wallet_balances(self._wallet.address)} ")

        elif liquidity_action == "remove_liquidity":
            # collect_tx_receipt,_ = self.pool.collect_fee(self._wallet.address, tick_lower, tick_upper)
            #log_event_to_csv(tx_receipt)
            burn_tx_receipt = self.pool.remove_liquidity_with_liquidty(self._wallet.address, tick_lower, tick_upper, amount)
            #print(burn_tx_receipt.events)
            #log_event_to_csv(tx_receipt)
            print(f"____________________ {UniV3Model().get_wallet_balances(self._wallet.address)}")
            

        elif liquidity_action == "hold":
            print("Do Nothing (HODOOR)")
