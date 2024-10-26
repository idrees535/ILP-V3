# import brownie
import random
import pandas as pd
import os
import sys
# Add parent directory to sys.path to handle imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.utility_functions import *
from util.constants import GOD_ACCOUNT,WALLET_LP,WALLET_SWAPPER



class UniswapV3LiquidityProviderAgent():
    def __init__(self,token0,token1,policy_func,pool):

        self.pool=pool
        self.policy=policy_func
        self.pool.fundToken0FromAbove(WALLET_LP.address, toBase18(token0))
        self.pool.fundToken1FromAbove(WALLET_LP.address, toBase18(token1))
        #transferETH(GOD_ACCOUNT,WALLET_LP.address,toBase18(10000))
        
    def takeStep(self):
        try:
            liquidity_action, tick_lower, tick_upper, amount = self.policy(self)
        except TypeError:
            print(f"Policy returned None, no action will be taken check your policy functions")
            return None
        
        print(f"____LIQUIDITY_PROVIDER WALLET {self.pool.get_wallet_balances(WALLET_LP.address)} ")
        print(f"____Liquidity amount: {amount} \n")

        if liquidity_action == "add_liquidity":
            tx_receipt= self.pool.add_liquidity_with_liquidity(WALLET_LP.address, tick_lower, tick_upper, amount, b'')
            #print(tx_receipt.events)
            #log_event_to_csv(tx_receipt)
        elif liquidity_action == "remove_liquidity":
            #collect_tx_receipt,_ = self.pool.collect_fee(wallet_lp.address, tick_lower, tick_upper)
            #log_event_to_csv(tx_receipt)
            burn_tx_receipt = self.pool.remove_liquidity_with_liquidty(WALLET_LP.address, tick_lower, tick_upper, amount)
            #print(burn_tx_receipt.events)
            #log_event_to_csv(tx_receipt)
        elif liquidity_action == "hold":
            print("Do Nothing (HODOOR)")
        
