import sys
sys.path.append('/mnt/c/Users/hijaz tr/Desktop/cadCADProject1/tokenspice')

from util.constants import BROWNIE_PROJECTUniV3, GOD_ACCOUNT
from util.base18 import toBase18, fromBase18,fromBase128,price_to_valid_tick,price_to_raw_tick,price_to_sqrtp,sqrtp_to_price,tick_to_sqrtp,liquidity0,liquidity1,eth
import brownie
from web3 import Web3
import json
import math
import random
from brownie.exceptions import VirtualMachineError

class UniV3Model:

    def __init__(self,init_with_liquidity=False, initial_liquidity_amount=0,initial_price=1):
        self.deployer = GOD_ACCOUNT
        self.init_with_liquidity = init_with_liquidity
        self.initial_liquidity_amount = initial_liquidity_amount
        self.initial_price=initial_price
        w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
        self.base_fee = w3.eth.getBlock('latest')['baseFeePerGas']
        addresses = self.load_addresses()
        self.is_pool_initialized = addresses.get("is_pool_initialized", False)
        self.ensure_pool_and_tokens()

    def ensure_pool_and_tokens(self):
        if not hasattr(self, 'weth'):
            self.deploy_tokens()
        if not hasattr(self, 'pool'):
            self.deploy_pool()
        if not self.is_pool_initialized:
            self.initialize_pool(price_to_sqrtp(self.initial_price))  # sqrtPriceX96

    def load_addresses(self):
            try:
                with open("addresses.json", "r") as f:
                    content = f.read()
                    if not content:
                        raise ValueError("File is empty")
                    return json.loads(content)
            except (FileNotFoundError, ValueError):
                # Create the file with default values if it doesn't exist or is empty
                addresses = {
                    "weth_address": None,
                    "usdc_address": None,
                    "pool_address": None,
                    "is_pool_initialized": False
                }
                with open("addresses.json", "w") as f:
                    json.dump(addresses, f)
                return addresses

    def save_addresses(self, addresses):
        addresses["is_pool_initialized"] = self.is_pool_initialized
        with open("addresses.json", "w") as f:
            json.dump(addresses, f)


    def deploy_tokens(self):
        Simpletoken = BROWNIE_PROJECTUniV3.Simpletoken
        initial_total_supply = toBase18(1e18)
        addresses = self.load_addresses()

        if addresses["weth_address"]:
            self.weth = Simpletoken.at(addresses["weth_address"])
            print("Loaded existing WETH")
            #print(tx_receipt_weth.events)
        else:
            tx_receipt_weth = self.weth = Simpletoken.deploy("Wrapped Ether", "WETH", 18, initial_total_supply, {'from': self.deployer, 'gas_price': self.base_fee + 1})
            addresses["weth_address"] = self.weth.address
            self.save_addresses(addresses)
            print(tx_receipt_weth.events)
            print(f"Deployed new WETH at address: {self.weth.address}")

        if addresses["usdc_address"]:
            self.usdc = Simpletoken.at(addresses["usdc_address"])
            print("Loaded existing USDC")
            #print(tx_receipt_usdc.events)
        else:
            tx_receipt_usdc = self.usdc = Simpletoken.deploy("USD Coin", "USDC", 18, initial_total_supply, {'from': self.deployer, 'gas_price': self.base_fee + 1})
            addresses["usdc_address"] = self.usdc.address
            self.save_addresses(addresses)
            print(tx_receipt_usdc.events)
            print(f"Deployed new USDC at address: {self.usdc.address}")
            

    def deploy_pool(self):
        UniswapV3Factory = BROWNIE_PROJECTUniV3.UniswapV3Factory
        UniswapV3Pool = BROWNIE_PROJECTUniV3.UniswapV3Pool
        addresses = self.load_addresses()

        if addresses["pool_address"]:
            self.pool = UniswapV3Pool.at(addresses["pool_address"])
            print("Loaded existing pool")
            #print(pool_creation_txn.events)

        else:     
            self.factory = UniswapV3Factory.deploy({'from': self.deployer, 'gas_price': self.base_fee + 1})
            pool_creation_txn = self.factory.createPool(self.weth.address, self.usdc.address, 3000, {'from': self.deployer, 'gas_price': self.base_fee + 1})
            self.pool_address = pool_creation_txn.events['PoolCreated']['pool']
            self.pool = UniswapV3Pool.at(self.pool_address)
            addresses["pool_address"] = pool_creation_txn.events['PoolCreated']['pool']
            self.pool = UniswapV3Pool.at(addresses["pool_address"])
            self.save_addresses(addresses)
            print(pool_creation_txn.events)
            print(f"Factory address: {self.factory.address}")

            
    def initialize_pool(self, sqrtPriceX96):
        if not hasattr(self, 'pool'):
            raise Exception("Pool has not been deployed or loaded")
        if not self.is_pool_initialized:
            tx_receipt = self.pool.initialize(sqrtPriceX96, {'from': self.deployer, 'gas_price': self.base_fee + 1, 'gas_limit': 5000000,'allow_revert': True})
            print("New pool iniitialized")
            print(tx_receipt.events)
            
            self.is_pool_initialized = True 
            self.is_pool_initialized = True
            self.save_addresses(self.load_addresses())

            if self.init_with_liquidity:
                self.initialize_pool_with_liq()
                
        else:
            print("Pool is already initialized")
            #print(tx_receipt.events)
            
    def initialize_pool_with_liq(self):
        # Can add any other logic to sync pool with real pool
        initial_price_low = self.initial_price-self.initial_price*0.5
        initial_price_high =self.initial_price+self.initial_price*0.5
        tx_receipt=self.add_liquidity(self.deployer, initial_price_low, initial_price_high, self.initial_liquidity_amount, b'')
        print(f'Initial liq added in pool : {tx_receipt.events}')

    
    def add_liquidity(self, liquidity_provider, price_low, price_upp, usd_budget, data):
        tx_params = {'from': str(liquidity_provider), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
        tx_params1 = {'from': str(GOD_ACCOUNT), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
        tx_receipt=None
        try:
            pool_actions = self.pool
            tick_lower,tick_upper,liquidity=self.budget_to_liquidity(price_low,price_upp,usd_budget)

            tx_receipt = pool_actions.mint(liquidity_provider, tick_lower, tick_upper, liquidity, data, tx_params)

        except VirtualMachineError as e:
            print("Failed to add liquidty", e.revert_msg)

        if 'Mint' in tx_receipt.events:
            amount0 = tx_receipt.events['Mint']['amount0']
            amount1 = tx_receipt.events['Mint']['amount1']
            print(tx_receipt.events)

            #Transfer tokens weth and usdc from GOD_ACCOUNT to agent's wallet(For safety instaed of this add acheck statement in policy which checks that agent's abalnce should be greater than amound he is adding in liquidty)
            #self.weth.transfer(liquidity_provider, amount0, tx_params1)
            #self.usdc.transfer(liquidity_provider, amount1, tx_params1)

            # Emulate the callback
            if amount0 > 0:
                tx_receipt_weth_transfer = self.weth.transfer(self.pool.address, amount0, tx_params)
                #print(tx_receipt_weth_transfer.events)

            if amount1 > 0:
                tx_receipt_usdc_transfer=self.usdc.transfer(self.pool.address, amount1, tx_params)
                #print(tx_receipt_usdc_transfer.events)
            
        else:
            print("Mint event not found in transaction receipt.")

        # Store position in json file
        liquidity_provider_str = str(liquidity_provider)
        try:
            with open("all_positions.json", "r") as f:
                file_contents = f.read()
                if not file_contents.strip():  # Check if the file is empty or just whitespace
                    all_positions = {}  # Initialize as empty dict
                else:
                    all_positions = json.loads(file_contents)
        except FileNotFoundError:
            all_positions = {}
        
        # Initialize if this liquidity provider is not in the list
        if liquidity_provider_str not in all_positions:
            all_positions[liquidity_provider_str] = []
        
        # Add new position to list
        all_positions[liquidity_provider_str].append({
            'tick_lower': tick_lower,
            'tick_upper': tick_upper,
            'liquidity': liquidity
        })
        
        # Store updated positions
        with open("all_positions.json", "w") as f:
            json.dump(all_positions, f)
        
        return tx_receipt
    
    def remove_liquidity(self, liquidity_provider,tick_lower,tick_upper,liquidity):
        liquidity_provider_str = str(liquidity_provider)
        tx_receipt=None
        
        # Remove liquidity via smart contract burn function
        try:
            tx_params = {'from': str(liquidity_provider), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
            tx_receipt = self.pool.burn(tick_lower, tick_upper, liquidity, tx_params)
            print(tx_receipt.events)
            
            # Remove this liquidity position from local storage
            all_positions = self.get_all_liquidity_positions()
            try:
                all_positions[liquidity_provider_str].remove({'tick_lower': tick_lower, 'tick_upper': tick_upper, 'liquidity': liquidity})
            except ValueError:
                    print("Position not found in the list.")
            # Update the file
            with open("all_positions.json", "w") as f:
                json.dump(all_positions, f)
        
        except VirtualMachineError as e:
            print("Failed to remove liquidty", e.revert_msg)
        return tx_receipt
        

    def swap_token0_for_token1(self, recipient, amount_specified, data):
        tx_params = {'from': str(recipient), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
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
            #self.weth.transfer(recipient, amount0, tx_params1)

            # Transfer weth to pool (callback)
            tx_receipt_weth_transfer = self.weth.transfer(self.pool.address, amount0, tx_params)
            
        
        except VirtualMachineError as e:
            print("Swap token 0 to Token 1 Transaction failed:", e.revert_msg)
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

            #Transfer tokens weth and usdc from GOD_ACCOUNT to agent's wallet (This should be removed latter as our agent will have balace in their accounts while initialized and they should not be allowed to make a transaction greater than their balance which will result in failure of transaction)
            #self.usdc.transfer(recipient, amount1, tx_params1)

            # Trasfer token1 to pool (callabck)
            tx_receipt_usdc_transfer = self.usdc.transfer(self.pool.address, amount1, tx_params)
            
        
        except VirtualMachineError as e:
            print("Swap token 1 to Token 0 Transaction failed:", e.revert_msg)
        return tx_receipt

    def collect_fee(self,recipient,tick_lower,tick_upper):
        tx_params = {'from': str(recipient), 'gas_price': self.base_fee + 1, 'gas_limit': 5000000, 'allow_revert': True}
        
        position_key = Web3.solidityKeccak(['address', 'int24', 'int24'], [str(recipient), tick_lower, tick_upper]).hex()

        position_info = self.pool.positions(position_key)
        amount0Owed = position_info[3]
        amount1Owed = position_info[4]
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
        
        #print(f"Fee collected usd: {fee_collected_usd}")
        return tx_receipt,fee_collected_usd
    
    # For now I am considering all positions of a liquiidty provider as distinct positions, and each position is removed fully instead of partially so there is no nedd of these two functions for now
    def increase_liquidty():
        pass
    def decrease_lquidity():
        pass


    #Get All positions of all Lps in pool
    def get_all_liquidity_positions(self):
        all_positions = {}  # Initialize as an empty dictionary
        try:
            with open("all_positions.json", "r") as f:
                file_content = f.read().strip()  # Read and remove any leading/trailing whitespace
                if not file_content:  # Check if file is empty
                    print("File is empty. Returning empty positions.")
                    return all_positions  # Return empty dict
                all_positions = json.loads(file_content)
        except FileNotFoundError:
            print("No positions found in pool.")
            return all_positions  # Return empty dict
        except json.JSONDecodeError:
            print("Error decoding JSON. File might be malformed.")
            return all_positions  # Return empty dict

        return all_positions

    
    # Get all poistions of an LP in Pool
    def get_lp_all_positions(self,liquidity_provider):
        liquidity_provider_str = str(liquidity_provider)
        all_positions=self.get_all_liquidity_positions()
        
        if all_positions is None:
            print("Pool has no LP position")
            return None
        if liquidity_provider_str not in all_positions or not all_positions[liquidity_provider_str]:
            #print("No positions found for this liquidity provider in pool")
            return None
        lp_positions = all_positions[liquidity_provider_str]
        return lp_positions
    
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
            with open("all_positions.json", "r") as f:
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
        lower_tick_interested = price_to_valid_tick(lower_price_interested,tick_spacing=60)
        upper_tick_interested = price_to_valid_tick(upper_price_interested,tick_spacing=60)

        unique_ticks = set()

        for liquidity_provider, positions in all_positions.items():
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
            #print(f"Fetching pool state for tick: {tick}")
            tick_states[tick] = self.get_tick_state(tick)  # Fetch and store each tick state

        return tick_states

    
    def get_pool_state_for_all_positions(self):
        position_states = {} 
        # Load all positions from the JSON file
        try:
            with open("all_positions.json", "r") as f:
                all_positions = json.load(f)
        except FileNotFoundError:
            print("No positions found.")
            return

        for liquidity_provider, positions in all_positions.items():
            #print(f"Fetching pool states for liquidity provider: {liquidity_provider}")
            for position in positions:
                tick_lower = position['tick_lower']
                tick_upper = position['tick_upper']
                position_key = f"{liquidity_provider}_{tick_lower}_{tick_upper}"
                position_states[position_key] = self.get_position_state(tick_lower, tick_upper, liquidity_provider)

        return position_states
        

    def get_wallet_balances(self, recipient):
        recipient_address = recipient.address  # Assuming recipient is a brownie account object
        balances = {
        recipient_address: {
            'ETH': fromBase18(recipient.balance()),
            'WETH': fromBase18(self.weth.balanceOf(recipient_address)),
            'USDC': fromBase18(self.usdc.balanceOf(recipient_address))
        }
    }
        return balances    
        

    def set_pool_allowance(self, recipient, amount0,amount1):
        '''
        w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
        base_fee = w3.eth.getBlock('latest')['baseFeePerGas']
        '''
        tx_params = {'from': str(recipient),'gas_price': self.base_fee + 1}
        
        tx_receipt_weth = self.weth.approve(self.pool.address, amount0, tx_params)
           
        print(tx_receipt_weth.events)
            
        
        
        tx_receipt_usdc = self.usdc.approve(self.pool.address, amount1, tx_params)
            
        print(tx_receipt_usdc.events)
    
   
    def budget_to_liquidity(self,price_low,price_upp,usd_budget):
            
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
        amount_weth =  ((0.5 * usd_budget)/usdp_cur) * eth
        amount_usdc = 0.5 * usd_budget * eth

        tick_lower=price_to_valid_tick(price_low)
        tick_upper=price_to_valid_tick(price_upp)

        sqrtp_low = tick_to_sqrtp(tick_lower)
        sqrtp_upp = tick_to_sqrtp(tick_upper)
        
        
        liquidity=get_liquidity_for_amounts(sqrt_ratio_x96=sqrtp_cur, sqrt_ratio_a_x96=sqrtp_low, sqrt_ratio_b_x96=sqrtp_upp, amount0=amount_weth, amount1=amount_usdc)

        return tick_lower,tick_upper,liquidity


def main():
    env = UniV3Model(True,1000000,2000)
    #env.check_balances(GOD_ACCOUNT)
    #env.set_pool_allowance(GOD_ACCOUNT,toBase18(1e18),toBase18(1e18))
    #env.set_pool_allowance(GOD_ACCOUNT,toBase18(100000000))
        
    #env.check_balances(GOD_ACCOUNT)
    # Price(2203.087634560301,1796.553389941556),ticks(74940,76980),sqrtprice(3358146572399269645810728435712,3718737045571708900035560210432)
    

    #w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
    #base_fee = w3.eth.getBlock('latest')['baseFeePerGas']
    #tx_params = {'from': str(GOD_ACCOUNT),'gas_price': base_fee + 1}
    
    agent=brownie.network.accounts[0]
    

    #env.get_global_state()

    #env.add_liquidity(agent,1950,2050,10000,b'')
    #env.get_global_state()

    #env.remove_liquidity_given_ticks(tick_lower=73800,tick_upper=78240,amount_usd=100)

    #env.remove_liquidity(agent)
    #env.get_global_state()
    #env.weth.transfer(agent,toBase18(10000),tx_params)
    #env.usdc.transfer(agent, toBase18(100000000000), tx_params)
    #env.weth.transfer(agent, toBase18(1000000), tx_params)

    #env.add_liquidity(agent,1950,2050,1000000000,b'')
    #env.swap_token0_for_token1(agent, toBase18(5), data=b'')
    

    #env.swap_token1_for_token0(agent, toBase18(1000), data=b'')
    #env.swap_token1_for_token0(agent, toBase18(5), data=b'')
    
    
    #env.get_pool_state_for_all_positions()
    #env.get_pool_state_for_all_ticks(1000,3000)
    
    #env.check_balances(GOD_ACCOUNT)
if __name__ == "__main__":
    main()



# Why the fuck address is being returned in a numeric form by position function
# Tick bitmap??
# Reconsider adding liquiidty with budget specified as in policy we will be ensuring that LP has enough budget to add liquiidty, which doesn't mean that he has enough amounts of token0 and token1 (Had to implement a swap before if he has enough budget but not enough amount of token0 or token1)

# Future Work:
# Handle Mutiple pools using generic token and deployemnt and get agent's to interact with multiple pools
# Sync pool states with uniswap pool
# For incraesing/decreasing and xisting position the storage part should update the values of a position (Optional: As we can maintain each poition disticntly, instead of merging them(unless there is some problem in fee calculations which I have to identify if there is any))