import os
import sys
# Add parent directory to sys.path to handle imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.utility_functions import *
from util.constants import GOD_ACCOUNT,WALLET_LP,WALLET_SWAPPER


class UniswapV3LiquidityProviderAgent():
    def __init__(self, initial_token0, initial_token1, pool):

        self.pool=pool
        # self.policy=policy_func
        self.pool.fundToken0FromAbove(WALLET_LP.address, toBase18(initial_token0))
        self.pool.fundToken1FromAbove(WALLET_LP.address, toBase18(initial_token1))
        #transferETH(GOD_ACCOUNT,WALLET_LP.address,toBase18(10000))
        
    def takeStep(self,action, tick_lower, tick_upper, token0_amount, token1_amount, block_time):
        
        print(f"____LIQUIDITY_PROVIDER WALLET {self.pool.get_wallet_balances(WALLET_LP.address)} ")
        print(f"____Liquidity amount0: {token0_amount}  amount1:  {token1_amount}     Block_time: {block_time} \n")

        if action == "add_liquidity":
            tx_receipt= self.pool.add_liquidity_with_amounts(WALLET_LP.address, tick_lower, tick_upper, token0_amount, token1_amount,  b'')
            #print(tx_receipt.events)
            #log_event_to_csv(tx_receipt)
        elif action == "remove_liquidity":
            #collect_tx_receipt,_ = self.pool.collect_fee(wallet_lp.address, tick_lower, tick_upper)
            #log_event_to_csv(tx_receipt)
            burn_tx_receipt = self.pool.remove_liquidity_with_liquidty(WALLET_LP.address, tick_lower, tick_upper, token1_amount)
            #print(burn_tx_receipt.events)
            #log_event_to_csv(tx_receipt)
        else :
            print("Do Nothing (HODOOR)")