import sys
sys.path.append('/mnt/c/Users/hijaz tr/Desktop/cadCADProject1/tokenspice')

import os
os.environ["PATH"] += ":."

from util.constants import BROWNIE_PROJECTUniV3, GOD_ACCOUNT
from util.constants import BROWNIE_PROJECTUniV3, GOD_ACCOUNT
from util.base18 import toBase18, fromBase18,fromBase128,price_to_valid_tick,price_to_raw_tick,price_to_sqrtp,sqrtp_to_price,tick_to_sqrtp,liquidity0,liquidity1,eth
import brownie
from web3 import Web3
import json
import math
import random
from brownie.exceptions import VirtualMachineError

class UniV3Model():
    def __init__(self, token0='token0', token1='token1', token0_decimals=18, token1_decimals=18, supply_token0=1e18, supply_token1=1e18, fee_tier=3000, initial_pool_price=1,deployer=GOD_ACCOUNT,sync_pool_with_liq=True,sync_pool_with_ticks=False,sync_pool_with_positions=False,sync_pool_with_events=False, state=None, initial_liquidity_amount=1000000):
        self.deployer = deployer
        self.token0_name = token0
        self.token1_name = token1
        self.token0_symbol = token0
        self.token1_symbol = token1
        self.token0_decimals = token0_decimals
        self.token1_decimals = token1_decimals
        self.supply_token0 = supply_token0
        self.supply_token1 = supply_token1
        self.fee_tier = fee_tier
        self.initial_pool_price = initial_pool_price
        self.sync_pool_with_liq=sync_pool_with_liq
        self.sync_pool_with_ticks=sync_pool_with_ticks
        self.sync_pool_with_positions=sync_pool_with_positions
        self.sync_pool_with_events=sync_pool_with_events
        self.initial_liquidity_amount=initial_liquidity_amount
        self.pool_id = f"{token0}_{token1}_{fee_tier}"
        

        w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
        self.base_fee = w3.eth.getBlock('latest')['baseFeePerGas']
        
        self.deploy_load_tokens()
        self.deploy_load_pool()

    def load_addresses(self):
        try:
            with open("model_storage/token_pool_addresses.json", "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_addresses(self, addresses):
        with open("model_storage/token_pool_addresses.json", "w") as f:
            json.dump(addresses, f)

    def deploy_load_tokens(self):
        SimpleToken = BROWNIE_PROJECTUniV3.Simpletoken
        addresses = self.load_addresses()
        pool_addresses = addresses.get(self.pool_id, {})

        # This function deploys a token and saves its address in the JSON file
        def deploy_and_save_token(name, symbol, decimals, supply, key):
            token = SimpleToken.deploy(name, symbol, decimals, toBase18(supply),  {'from': self.deployer, 'gas_price': self.base_fee + 1})
            print(f"New {symbol} token deployed at {token.address}")
            pool_addresses[key] = token.address
            addresses[self.pool_id] = pool_addresses
            self.save_addresses(addresses)
            return token

        # Load or deploy token1
        if "token1_address" in pool_addresses:
            self.token1 = SimpleToken.at(pool_addresses["token1_address"])
        else:
            self.token1 = deploy_and_save_token(self.token1_name, self.token1_symbol, self.token1_decimals, self.supply_token1, "token1_address")

        # Load or deploy token0
        if "token0_address" in pool_addresses:
            self.token0 = SimpleToken.at(pool_addresses["token0_address"])
        else:
            self.token0 = deploy_and_save_token(self.token0_name, self.token0_symbol, self.token0_decimals, self.supply_token0, "token0_address")
            # Ensure token0 address is less than token1 address
            while int(self.token0.address, 16) >= int(self.token1.address, 16):
                self.token0 = deploy_and_save_token(self.token0_name, self.token0_symbol, self.token0_decimals, self.supply_token0, "token0_address")


    def deploy_load_pool(self):
        UniswapV3Factory = BROWNIE_PROJECTUniV3.UniswapV3Factory
        UniswapV3Pool = BROWNIE_PROJECTUniV3.UniswapV3Pool
        addresses = self.load_addresses()
        pool_addresses = addresses.get(self.pool_id, {})

        if "pool_address" in pool_addresses:
            self.pool = UniswapV3Pool.at(pool_addresses["pool_address"])
            print(f"Existing pool:{self.pool_id} having pool address: {self.pool} loaded")
        else:
            self.factory = UniswapV3Factory.deploy( {'from': self.deployer, 'gas_price': self.base_fee + 1})
            pool_creation_txn = self.factory.createPool(self.token0.address, self.token1.address, self.fee_tier,  {'from': self.deployer, 'gas_price': self.base_fee + 1})
            self.pool_address = pool_creation_txn.events['PoolCreated']['pool']
            print(pool_creation_txn.events)
            self.pool = UniswapV3Pool.at(self.pool_address)

            sqrtPriceX96 = price_to_sqrtp(self.initial_pool_price)
            tx_receipt=self.pool.initialize(sqrtPriceX96,  {'from': self.deployer, 'gas_price': self.base_fee + 100000})
            print(tx_receipt.events)

            pool_addresses["pool_address"] = self.pool_address
            addresses[self.pool_id] = pool_addresses
            self.save_addresses(addresses)
            
            self.sync_pool_state()


    def ensure_token_order(self):
        # Check if token0's address is greater than token1's address
        if int(self.token0.address, 16) > int(self.token1.address, 16):
            SimpleToken = BROWNIE_PROJECTUniV3.Simpletoken

            # Continue deploying token0 until its address is less than token1's address
            while True:
                new_token0 = SimpleToken.deploy(self.token0_name, self.token0_symbol, self.token0_decimals, self.supply_token0, {'from': self.deployer, 'gas_price': self.base_fee + 1})
                if int(new_token0.address, 16) < int(self.token1.address, 16):
                    break

            # Update the model's token0 reference to point to the new token0 contract
            self.token0 = new_token0
            print(f"New {self.token0_symbol} token deployed at {self.token0.address} to ensure desired token order in the pool")

    def sync_pool_state(self):
        # Can add any other logic to sync pool with real pool
        if self.sync_pool_with_liq:
            tick_lower = price_to_valid_tick(self.initial_pool_price-self.initial_pool_price*0.5)
            tick_upper =price_to_valid_tick(self.initial_pool_price+self.initial_pool_price*0.5)
            self.add_liquidity(self.deployer, tick_lower, tick_upper, self.initial_liquidity_amount, b'')
            print(f'Initial liq amount {self.initial_liquidity_amount} added in pool')

        elif self.sync_pool_with_positions:
            for tick_idx, liquidity_gross, liquidity_net in self.state['ticks']:
                # Determine the tickLower and tickUpper based on tick_idx.
                # This is a simplified example; in a real scenario, you'd need to be more precise.
                tick_lower = int(tick_idx - 10)
                tick_upper = int(tick_idx + 10)

                # Calculate the amount of liquidity to add. 
                # This is a simplified example; you'd need to calculate this based on the pool's requirements.
                amount_liquidity = int((liquidity_gross + liquidity_net) / 2)

                # Call the contract method to add liquidity.
                # This is a placeholder; replace it with an actual contract interaction.
                self.pool.add_liquidity(GOD_ACCOUNT, tick_lower, tick_upper, amount_liquidity, '')

                print(f"Liquidity of {amount_liquidity} added between ticks {tick_lower} and {tick_upper}")

        elif self.sync_pool_with_ticks:
                for tick_idx, liquidity_gross, liquidity_net in self.state['ticks']:
                    # Determine the tickLower and tickUpper based on tick_idx.
                    # This is a simplified example; in a real scenario, you'd need to be more precise.
                    tick_lower = int(tick_idx - 10)
                    tick_upper = int(tick_idx + 10)

                    # Calculate the amount of liquidity to add. 
                    # This is a simplified example; you'd need to calculate this based on the pool's requirements.
                    amount_liquidity = int((liquidity_gross + liquidity_net) / 2)

                    # Call the contract method to add liquidity.
                    # This is a placeholder; replace it with an actual contract interaction.
                    self.pool.add_liquidity(GOD_ACCOUNT, tick_lower, tick_upper, amount_liquidity, '')

                    print(f"Liquidity of {amount_liquidity} added between ticks {tick_lower} and {tick_upper}")
        elif self.sync_pool_with_events:
            print("Computionally expensive process")
        else:
            print("No pool sync applied")


    def add_liquidity(self, liquidity_provider, tick_lower, tick_upper, usd_budget, data):
        tx_params = {'from': str(liquidity_provider), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
        tx_params1 = {'from': str(GOD_ACCOUNT), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
        tx_receipt=None
        try:
            pool_actions = self.pool
            liquidity=self.budget_to_liquidity(tick_lower,tick_upper,usd_budget)
            #print(liquidity)

            tx_receipt = pool_actions.mint(liquidity_provider, tick_lower, tick_upper, liquidity, data, tx_params)

            # Implement callback
            amount0 = tx_receipt.events['Mint']['amount0']
            amount1 = tx_receipt.events['Mint']['amount1']
            #print(tx_receipt.events['Mint']['amount'])
            print(tx_receipt.events)
            if amount0 > 0:
                tx_receipt_token0_transfer = self.token0.transfer(self.pool.address, amount0, tx_params)
            if amount1 > 0:
                tx_receipt_token1_transfer=self.token1.transfer(self.pool.address, amount1, tx_params)
                #print(f'token1 amount:{amount1}transfered to contract:{tx_receipt_token1_transfer}')

        except VirtualMachineError as e:
            print("Failed to add liquidty", e.revert_msg)

        #Transfer tokens token0 and token1 from GOD_ACCOUNT to agent's wallet(For safety instaed of this add acheck statement in policy which checks that agent's abalnce should be greater than amound he is adding in liquidty)
        #self.token0.transfer(liquidity_provider, amount0, tx_params1)
        #self.token1.transfer(liquidity_provider, amount1, tx_params1)

        # Store position in json file
        liquidity_provider_str = str(liquidity_provider)
        
        try:
            with open("model_storage/liq_positions.json", "r") as f:
                all_positions = json.load(f)
        except FileNotFoundError:
            all_positions = {}
        
        # Initialize if this pool_id is not in the list
        if self.pool_id not in all_positions:
            all_positions[self.pool_id] = {}
        
        # Initialize if this liquidity provider is not in the list
        if liquidity_provider_str not in all_positions[self.pool_id]:
            all_positions[self.pool_id][liquidity_provider_str] = []
        
        existing_position = None
        for position in all_positions[self.pool_id][liquidity_provider_str]:
            if position['tick_lower'] == tick_lower and position['tick_upper'] == tick_upper:
                existing_position = position
                break
    
        if existing_position:
            existing_position['liquidity'] += liquidity 
            existing_position['amount_usd'] += usd_budget # Add new liquidity to existing position
        else:
        # Add new position to list
            all_positions[self.pool_id][liquidity_provider_str].append({
                'tick_lower': tick_lower,
                'tick_upper': tick_upper,
                'liquidity': liquidity,
                'amount_usd':usd_budget
            })
        
        # Store updated positions
        with open("model_storage/liq_positions.json", "w") as f:
            json.dump(all_positions, f)
        
        return tx_receipt
    
    def add_liquidity_with_liquidity(self, liquidity_provider, tick_lower, tick_upper, liquidity, data):
        tx_params = {'from': str(liquidity_provider), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
        tx_params1 = {'from': str(GOD_ACCOUNT), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
        tx_receipt=None
        try:
            pool_actions = self.pool
            liquidity=liquidity

            tx_receipt = pool_actions.mint(liquidity_provider, tick_lower, tick_upper, liquidity, data, tx_params)

            # Implement callback
            amount0 = tx_receipt.events['Mint']['amount0']
            amount1 = tx_receipt.events['Mint']['amount1']
            print(tx_receipt.events)
            if amount0 > 0:
                tx_receipt_token0_transfer = self.token0.transfer(self.pool.address, amount0, tx_params)
            if amount1 > 0:
                tx_receipt_token1_transfer=self.token1.transfer(self.pool.address, amount1, tx_params)
                #print(f'token1 amount:{amount1}transfered to contract:{tx_receipt_token1_transfer}')


        except VirtualMachineError as e:
            print("Failed to add liquidty", e.revert_msg)

        #Transfer tokens token0 and token1 from GOD_ACCOUNT to agent's wallet(For safety instaed of this add acheck statement in policy which checks that agent's abalnce should be greater than amound he is adding in liquidty)
        #self.token0.transfer(liquidity_provider, amount0, tx_params1)
        #self.token1.transfer(liquidity_provider, amount1, tx_params1)

        # Store position in json file
        liquidity_provider_str = str(liquidity_provider)
        
        try:
            with open("model_storage/liq_positions.json", "r") as f:
                all_positions = json.load(f)
        except FileNotFoundError:
            all_positions = {}
        
        # Initialize if this pool_id is not in the list
        if self.pool_id not in all_positions:
            all_positions[self.pool_id] = {}
        
        # Initialize if this liquidity provider is not in the list
        if liquidity_provider_str not in all_positions[self.pool_id]:
            all_positions[self.pool_id][liquidity_provider_str] = []
        
        existing_position = None
        for position in all_positions[self.pool_id][liquidity_provider_str]:
            if position['tick_lower'] == tick_lower and position['tick_upper'] == tick_upper:
                existing_position = position
                break
    
        if existing_position:
            existing_position['liquidity'] += liquidity 
            existing_position['amount_usd'] += liquidity # Add new liquidity to existing position
        else:
        # Add new position to list
            all_positions[self.pool_id][liquidity_provider_str].append({
                'tick_lower': tick_lower,
                'tick_upper': tick_upper,
                'liquidity': liquidity,
                'amount_usd':liquidity
            })
        
        # Store updated positions
        with open("model_storage/liq_positions.json", "w") as f:
            json.dump(all_positions, f)
        
        return tx_receipt
    
    
    def remove_liquidity(self, liquidity_provider, tick_lower, tick_upper, amount_usd):
        liquidity_provider_str = str(liquidity_provider)
        tx_receipt = None
        
        # Convert budget to liquidity amount
        liquidity = self.budget_to_liquidity(tick_lower, tick_upper, amount_usd)

        try:
            tx_params = {'from': str(liquidity_provider), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
            tx_receipt = self.pool.burn(tick_lower, tick_upper, liquidity, tx_params)
            print(tx_receipt.events)
            self.collect_fee(liquidity_provider_str,tick_lower,tick_upper)
        except VirtualMachineError as e:
            print("Failed to remove liquidity", e.revert_msg)
            return tx_receipt  # Exit early if smart contract interaction fails

        try:
            with open("model_storage/liq_positions.json", "r") as f:
                all_positions = json.load(f)
        except FileNotFoundError:
            all_positions = {}
            
        if self.pool_id not in all_positions or \
        liquidity_provider_str not in all_positions[self.pool_id]:
            print("Position not found.")
            return tx_receipt  # Exit early if no positions are found

        existing_position = None
        for position in all_positions[self.pool_id][liquidity_provider_str]:
            if position['tick_lower'] == tick_lower and position['tick_upper'] == tick_upper:
                existing_position = position
                break

        if not existing_position:
            print("Position not found.")
            return tx_receipt  # Exit early if the specific position is not found

        if existing_position['liquidity'] > liquidity:
            existing_position['liquidity'] -= liquidity
            existing_position['amount_usd'] -= amount_usd  # Deduct removed liquidity
        else:
            all_positions[self.pool_id][liquidity_provider_str].remove(existing_position)  # Remove position if liquidity becomes zero
        
        # Update the JSON file
        with open("model_storage/liq_positions.json", "w") as f:
            json.dump(all_positions, f)

        return tx_receipt
    
    def remove_liquidity_with_liquidty(self, liquidity_provider, tick_lower, tick_upper, liquidity):
        liquidity_provider_str = str(liquidity_provider)
        tx_receipt = None
        
        # Convert budget to liquidity amount

        try:
            tx_params = {'from': str(liquidity_provider), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
            tx_receipt = self.pool.burn(tick_lower, tick_upper, liquidity, tx_params)
            print(tx_receipt.events)
            self.collect_fee(liquidity_provider_str,tick_lower,tick_upper)
        except VirtualMachineError as e:
            print("Failed to remove liquidity", e.revert_msg)
            return tx_receipt  # Exit early if smart contract interaction fails

        try:
            with open("model_storage/liq_positions.json", "r") as f:
                all_positions = json.load(f)
        except FileNotFoundError:
            all_positions = {}
            
        if self.pool_id not in all_positions or \
        liquidity_provider_str not in all_positions[self.pool_id]:
            print("Position not found.")
            return tx_receipt  # Exit early if no positions are found

        existing_position = None
        for position in all_positions[self.pool_id][liquidity_provider_str]:
            if position['tick_lower'] == tick_lower and position['tick_upper'] == tick_upper:
                existing_position = position
                break

        if not existing_position:
            print("Position not found.")
            return tx_receipt  # Exit early if the specific position is not found

        if existing_position['liquidity'] > liquidity:
            existing_position['liquidity'] -= liquidity
            existing_position['amount_usd'] -= liquidity  # Deduct removed liquidity
        else:
            all_positions[self.pool_id][liquidity_provider_str].remove(existing_position)  # Remove position if liquidity becomes zero
        
        # Update the JSON file
        with open("model_storage/liq_positions.json", "w") as f:
            json.dump(all_positions, f)

        return tx_receipt
            

    def swap_token0_for_token1(self, recipient, amount_specified, data):
        tx_params = {'from': str(recipient), 'gas_price': self.base_fee + 1000000, 'gas_limit':  5000000, 'allow_revert': True}
        #tx_params1={'from': str(GOD_ACCOUNT), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
        sqrt_price_limit_x96=4295128740+1

        pool_actions = self.pool
        zero_for_one = True
        tx_receipt=None
        
        try:
            tx_receipt= pool_actions.swap(recipient, zero_for_one, amount_specified,sqrt_price_limit_x96, data,tx_params)
            
            print(tx_receipt.events)
            amount0 = tx_receipt.events['Swap']['amount0']

            #Transfer tokens from GOD_ACCOUNT to agent's wallet
            #self.token0.transfer(recipient, amount0, tx_params1)

            # Transfer token0 to pool (callback)
            tx_receipt_token0_transfer = self.token0.transfer(self.pool.address, amount0, tx_params)
            
        
        except VirtualMachineError as e:
            print("Swap token 0 to Token 1 Transaction failed:", e.revert_msg)
            slot0_data = self.pool.slot0()
            print(f'contract_token1_balance - approx_token1_amount: {self.token1.balanceOf(self.pool)-amount_specified*sqrtp_to_price(slot0_data[0])}, approx_token1_amount: {amount_specified*sqrtp_to_price(slot0_data[0])}), contract_token1_balance: {self.token1.balanceOf(self.pool)}, amount_swap_token0: {amount_specified}, contract_token0 _balance - amount_swap_token0: {self.token0.balanceOf(self.pool)-amount_specified}')

        return tx_receipt
     
    def swap_token1_for_token0(self, recipient, amount_specified, data):
        tx_params = {'from': str(recipient), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
        tx_params1={'from': str(GOD_ACCOUNT), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
        sqrt_price_limit_x96=1461446703485210103287273052203988822378723970342-1

        pool_actions = self.pool   
        zero_for_one = False
        tx_receipt=None 

        try:
            tx_receipt = pool_actions.swap(recipient, zero_for_one, amount_specified, sqrt_price_limit_x96, data,tx_params)
            print(tx_receipt.events)
        
            amount1 = tx_receipt.events['Swap']['amount1']

            #Transfer tokens token0 and token1 from GOD_ACCOUNT to agent's wallet (This should be removed latter as our agent will have balace in their accounts while initialized and they should not be allowed to make a transaction greater than their balance which will result in failure of transaction)
            #self.token1.transfer(recipient, amount1, tx_params1)

            # Trasfer token1 to pool (callabck)
            tx_receipt_token1_transfer = self.token1.transfer(self.pool.address, amount1, tx_params)
            #print(f'token1 amount:{amount1} transfered to contract:{tx_receipt_token1_transfer}')
            
        
        except VirtualMachineError as e:
            print("Swap token 1 to Token 0 Transaction failed:", e.revert_msg)
            slot0_data = self.pool.slot0()
            print(f'contract_token0_balance - approx_token0_amount: {self.token0.balanceOf(self.pool)-amount_specified/sqrtp_to_price(slot0_data[0])}, approx_token0_amount: {amount_specified/sqrtp_to_price(slot0_data[0])}, contract_token0_balance: {self.token0.balanceOf(self.pool)}, contract_token1_balance - amount_swap_token1: {self.token1.balanceOf(self.pool)-amount_specified}')
        return tx_receipt

    def collect_fee(self,recipient,tick_lower,tick_upper):
        tx_params = {'from': str(recipient), 'gas_price': self.base_fee + 1, 'gas_limit': 500000000, 'allow_revert': True}
        # Poke to update variables
        try:
            tx_receipt = self.pool.burn(tick_lower, tick_upper, 0, tx_params)
        except VirtualMachineError as e:
            print("Poke:", e.revert_msg)
        
        position_key = Web3.solidityKeccak(['address', 'int24', 'int24'], [str(recipient), tick_lower, tick_upper]).hex()

        position_info = self.pool.positions(position_key)
        
        amount0Owed = position_info[3]
        amount1Owed = position_info[4]

        print(f'amount0Owed: {position_info[3]}, ,amount1Owed: {position_info[4]}')

        tx_receipt=None
        fee_collected_usd=0
        try:
            tx_receipt=self.pool.collect(recipient,tick_lower,tick_upper,amount0Owed, amount1Owed,tx_params)
            print(tx_receipt.events)

            amount0Collected=tx_receipt.events['Collect']['amount0']
            amount1Collected=tx_receipt.events['Collect']['amount1']

            slot0_data = self.pool.slot0()
            fee_collected_usd = fromBase18(amount0Collected*sqrtp_to_price(slot0_data[0]) + amount1Collected)
        except VirtualMachineError as e:
            print("Fee collection failed:", e.revert_msg)
            print(f"contract_token0_balance - amount0Owed: {self.token0.balanceOf(self.pool)-amount0Owed} ,contract_token1_balance - amount1Owed: {self.token1.balanceOf(self.pool)-amount1Owed}, position_tick_lower: {tick_lower}, position_tick_upper: {tick_upper}")

        
        #print(f"Fee collected usd: {fee_collected_usd}")
        return tx_receipt,fee_collected_usd
    
   
    # Get All positions of all LPs in the pool
    def get_all_liquidity_positions(self):
        try:
            with open("model_storage/liq_positions.json", "r") as f:
                all_positions = json.load(f)
        except FileNotFoundError:
            print("No positions found.")
            return {}
        except json.JSONDecodeError:
            print("Error decoding JSON. File might be malformed.")
            return {}

        if self.pool_id in all_positions:
            return all_positions[self.pool_id]
        else:
            print(f"No positions found for pool {self.pool_id}.")
            return {}

    # Get all positions of an LP in the pool
    def get_lp_all_positions(self, liquidity_provider):
        liquidity_provider_str = str(liquidity_provider)
        all_positions = self.get_all_liquidity_positions()

        if not all_positions:
            print("Pool has no LP positions.")
            return None

        if liquidity_provider_str in all_positions:
            return all_positions[liquidity_provider_str]
        else:
            print(f"No positions found for this liquidity provider {liquidity_provider_str} in pool.")
            return None

    def get_position_state(self,tick_lower,tick_upper,agent):
        position_key = Web3.solidityKeccak(['address', 'int24', 'int24'], [str(agent), tick_lower, tick_upper]).hex()
        pool_state = self.pool
        position_info = self.pool.positions(position_key)
        
        return {
        "position_key":f"{str(agent)}_{tick_lower}_{tick_upper}",
        "liquidity_provider": str(agent),
        "tick_lower":tick_lower,
        "tick_upper":tick_upper,
        "_liquidity_raw": position_info[0],
        #"_liquidity_converted": fromBase18(position_info[0]),
        "feeGrowthInside0LastX128": position_info[1],
        #"feeGrowthInside0Last": fromBase128(position_info[1]),
        "feeGrowthInside1LastX128": position_info[2],
        #"feeGrowthInside1Last": fromBase128(position_info[2]),
        "tokensOwed0_raw": position_info[3],
        #"tokensOwed0_converted": fromBase18(position_info[3]),
        "tokensOwed1_raw": position_info[4],
        #"tokensOwed1_converted": fromBase18(position_info[4])
    }


    def get_tick_state(self,tick):
        pool_state = self.pool
        word_position = tick >> 8

        tick_info = pool_state.ticks(tick)
        tick_bitmap = pool_state.tickBitmap(word_position)

        return {
        'tick':tick,
        "liquidityGross_raw": tick_info[0],
        #"liquidityGross_converted": fromBase18(tick_info[0]),
        "liquidityNet_raw": tick_info[1],
        #"liquidityNet_converted": fromBase18(tick_info[1]),
        "feeGrowthOutside0X128": tick_info[2],
        #"feeGrowthOutside0": fromBase128(tick_info[2]),
        "feeGrowthOutside1X128": tick_info[3],
        #"feeGrowthOutside1": fromBase128(tick_info[3]),
        "tickCumulativeOutside": tick_info[4],
        #"secondsPerLiquidityOutsideX128": tick_info[5],
        #"secondsPerLiquidityOutside": fromBase128(tick_info[5]),
        #"secondsOutside": tick_info[6],
        #"initialized": tick_info[7],
        "tickBitmap": tick_bitmap
    }
    
    def get_global_state(self):
        pool_state = self.pool
        slot0_data = pool_state.slot0()
        observation_index = slot0_data[2]

        feeGrowthGlobal0X128 = pool_state.feeGrowthGlobal0X128()
        feeGrowthGlobal1X128 = pool_state.feeGrowthGlobal1X128()
        protocol_fees = pool_state.protocolFees()
        
        liquidity = pool_state.liquidity()

        observation_info = pool_state.observations(observation_index)
        
        return {
        "curr_sqrtPriceX96": slot0_data[0],
        "curr_price": sqrtp_to_price(slot0_data[0]),
        "tick": slot0_data[1],
        #"locking_status": slot0_data[6],
        "feeGrowthGlobal0X128": feeGrowthGlobal0X128,
        #"feeGrowthGlobal0": fromBase128(feeGrowthGlobal0X128),
        "feeGrowthGlobal1X128": feeGrowthGlobal1X128,
        #"feeGrowthGlobal1": fromBase128(feeGrowthGlobal1X128),
        "liquidity_raw": liquidity,
        #"liquidity_converted": fromBase18(liquidity),
        "blockTimestamp": observation_info[0],
        "tickCumulative": observation_info[1],
        "secondsPerLiquidityCumulativeX128": observation_info[2],
        #"secondsPerLiquidityCumulative": fromBase128(observation_info[2]),
        #"initialized": observation_info[3]
    }
           
    def get_pool_state_for_all_ticks(self, lower_price_interested, upper_price_interested):
        tick_states = {} 
        try:
            with open("model_storage/liq_positions.json", "r") as f:
                file_content = f.read().strip()  # Read and remove any leading/trailing whitespace
                if not file_content:  # Check if file is empty
                    print("File is empty. Returning.")
                    return tick_states  # Return empty dict
                all_positions = json.loads(file_content)
        except FileNotFoundError:
            print("No positions found.")
            return tick_states  # Return empty dict
        except json.JSONDecodeError:
            print("Error decoding JSON. File might be malformed.")
            return tick_states  # Return empty dict

        # Convert interested prices to ticks
        lower_tick_interested = price_to_valid_tick(lower_price_interested, tick_spacing=60)
        upper_tick_interested = price_to_valid_tick(upper_price_interested, tick_spacing=60)

        unique_ticks = set()

        # Filter only the positions related to this specific pool.
        if self.pool_id not in all_positions:
            print("No positions for this pool.")
            return tick_states

        for liquidity_provider, positions in all_positions[self.pool_id].items():
            for position in positions:
                tick_lower = position['tick_lower']
                tick_upper = position['tick_upper']

                # Check if the tick_lower or tick_upper falls within the interested range
                if lower_tick_interested <= tick_lower <= upper_tick_interested or \
                lower_tick_interested <= tick_upper <= upper_tick_interested:
                    unique_ticks.add(tick_lower)
                    unique_ticks.add(tick_upper)

        # Fetch pool states for unique ticks within the range
        for tick in unique_ticks:
            tick_states[tick] = self.get_tick_state(tick)  # Fetch and store each tick state

        return tick_states

    
    def get_pool_state_for_all_positions(self):
        position_states = {}
        # Load all positions from the JSON file
        try:
            with open("model_storage/liq_positions.json", "r") as f:
                all_positions = json.load(f)
        except FileNotFoundError:
            print("No positions found.")
            return

        # Check if this pool_id exists in all_positions
        if self.pool_id not in all_positions:
            print(f"No positions found for pool {self.pool_id}.")
            return

        # Fetch positions for this specific pool
        for liquidity_provider_str, positions in all_positions[self.pool_id].items():
            for position in positions:
                tick_lower = position['tick_lower']
                tick_upper = position['tick_upper']
                liquidity = position['liquidity']
                position_key = f"{liquidity_provider_str}_{tick_lower}_{tick_upper}"
                position_states[position_key] = self.get_position_state(tick_lower, tick_upper,liquidity_provider_str)

        return position_states

        

    def get_wallet_balances(self, recipient):
        recipient_address = recipient.address  # Assuming recipient is a brownie account object
        balances = {
        recipient_address: {
            'ETH': fromBase18(recipient.balance()),
            'token0': fromBase18(self.token0.balanceOf(recipient_address)),
            'token1': fromBase18(self.token1.balanceOf(recipient_address))
        }
    }
        return balances    
        

    def set_pool_allowance(self, recipient, amount0,amount1):
        '''
        w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
        base_fee = w3.eth.getBlock('latest')['baseFeePerGas']
        '''
        tx_params = {'from': str(recipient),'gas_price': self.base_fee + 1}
        
        tx_receipt_token0 = self.token0.approve(self.pool.address, amount0, tx_params)
           
        print(tx_receipt_token0.events)
            
        
        
        tx_receipt_token1 = self.token1.approve(self.pool.address, amount1, tx_params)
            
        print(tx_receipt_token1.events)
    
   
    def budget_to_liquidity(self,tick_lower,tick_upper,usd_budget):
            
        # Calculating liquidity (Not needed here: Can shift this function to helper_functions)
        def get_liquidity_for_amounts(sqrt_ratio_x96, sqrt_ratio_a_x96, sqrt_ratio_b_x96, amount0, amount1):
            if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
                sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96
            
            if sqrt_ratio_x96 <= sqrt_ratio_a_x96:
                return liquidity0(amount0, sqrt_ratio_a_x96, sqrt_ratio_b_x96)
            elif sqrt_ratio_x96 < sqrt_ratio_b_x96:
                liquidity0_value = liquidity0(amount0, sqrt_ratio_x96, sqrt_ratio_b_x96)
                liquidity1_value = liquidity1(amount1, sqrt_ratio_a_x96, sqrt_ratio_x96)
                return min(liquidity0_value, liquidity1_value)
            else:
                return liquidity1(amount1, sqrt_ratio_a_x96, sqrt_ratio_b_x96)

        slot0_data = self.pool.slot0()
        sqrtp_cur =slot0_data[0]

        usdp_cur = sqrtp_to_price(sqrtp_cur)
        amount_token0 =  ((0.5 * usd_budget)/usdp_cur) * eth
        amount_token1 = 0.5 * usd_budget * eth

        sqrtp_low = tick_to_sqrtp(tick_lower)
        sqrtp_upp = tick_to_sqrtp(tick_upper)
        
        
        liquidity=get_liquidity_for_amounts(sqrt_ratio_x96=sqrtp_cur, sqrt_ratio_a_x96=sqrtp_low, sqrt_ratio_b_x96=sqrtp_upp, amount0=amount_token0, amount1=amount_token1)
        return liquidity

'''
Usage
token0 = "ETH"
token1 = "DAI"
supply_token0 = 1e18
supply_token1 = 1e18
decimals_token0 = 18
decimals_token1 = 18
fee_tier = 3000
initial_pool_price = 2000
deployer = GOD_ACCOUNT
sync_pool=True
initial_liquidity_amount=10000
env = UniV3Model(token0, token1,decimals_token0,decimals_token1,supply_token0,supply_token1,fee_tier,initial_pool_price,deployer,sync_pool, initial_liquidity_amount)
'''
