import math
import csv
import os 
import random
import subprocess
import time
import signal
import logging
import brownie
# from brownie.network import chain
logging.getLogger("brownie").setLevel(logging.ERROR)

min_tick = -887272
max_tick = 887272
q96 = 2**96
eth = 10**18

def toBase18(amt: float) -> int:
    return int(amt * 1e18)

def fromBase18(amt_base: int) -> float:
    return amt_base / 1e18

def fromBase128(value):
    return value / (2 ** 128)

def toBase128(value):
    return value *(2 **128)

def price_to_raw_tick(price):
    return math.floor(math.log(price) / math.log(1.0001))

def price_to_valid_tick(price, tick_spacing=60):
    raw_tick = math.floor(math.log(price, 1.0001))
    remainder = raw_tick % tick_spacing
    if remainder != 0:
        # Round to the nearest valid tick, considering tick spacing.
        raw_tick += tick_spacing - remainder if remainder >= tick_spacing // 2 else -remainder
    return raw_tick

def tick_to_price(tick):
    price = (1.0001 ** tick)
    return price

def price_to_sqrtp(p):
    return int(math.sqrt(p) * q96)

def sqrtp_to_price(sqrtp):
    return (sqrtp / q96) ** 2

def tick_to_sqrtp(t):
    return int((1.0001 ** (t / 2)) * q96)

def liquidity0(amount, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    return (amount * (pa * pb) / q96) / (pb - pa)

def liquidity1(amount, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    return amount * q96 / (pb - pa)

def log_event_to_csv(tx_receipt):
    # Maximum set of possible fields across all event types
    if tx_receipt is None:
        print("Transaction was not successful, skipping event logging.")
        return
    max_fields = ['sender', 'owner', 'tickLower', 'tickUpper', 'amount', 'amount0', 'amount1', 
                  'from', 'to', 'value', 'sqrtPriceX96', 'liquidity', 'tick', 'recipient']
    
    csv_file_path = "model_output/events_log.csv"
    
    # Check if file exists to write headers
    try:
        with open(csv_file_path, 'r') as f:
            pass
    except FileNotFoundError:
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Event Type'] + max_fields)
    
    # Append new rows
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        
        for event_type, event_list in tx_receipt.events.items():
            for event in event_list:
                row = [event_type]
                for field in max_fields:
                   row.append(str(event[field]) if field in event else '')
                writer.writerow(row)

# def txdict(from_account) -> dict:
#     """Return a tx dict that includes priority_fee and max_fee for EIP1559"""
#     priority_fee, max_fee = _fees()
#     return {
#         "from": from_account,
#         "priority_fee": priority_fee,
#         "max_fee": max_fee,
#     }


def txdict(from_account) -> dict:
    """Return a tx dict with a valid gas price."""
    gas_price = 875000000  # 875 gwei, adjust as needed for your local network
    return {
        "from": from_account,
        "gas_price": gas_price,
    }

# def transferETH(from_account, to_account, amount):
#     """
#     Transfer ETH accounting for priority_fee and max_fee, for EIP1559.
#     Returns a TransactionReceipt instance.
#     """
#     priority_fee, max_fee = _fees()
#     #return from_account.transfer(to_account, amount, priority_fee=priority_fee, max_fee=max_fee)
#     return from_account.transfer(to_account, amount)

def transferETH(from_account, to_account, amount):
    """
    Transfer ETH accounting for priority_fee and max_fee for EIP-1559 transactions.
    Returns a TransactionReceipt instance.
    """
    gas_price = 875000000  # 875 gwei, adjust as needed

    # Use the priority_fee and max_fee for the transaction
    return from_account.transfer(to_account, amount, gas_price=gas_price)

def _fees() -> tuple:
    assert brownie.network.is_connected()
    #priority_fee = chain.priority_fee #875000000
    #max_fee = chain.base_fee + 2 * chain.priority_fee #875000000

    priority_fee = 8750000
    max_fee = 1000000000

    return (priority_fee, max_fee)

# Function to calculate liquidity for token0
def calculate_liquidity_token0(token0_amount, P_lower, P_upper):
    sqrt_P_lower = math.sqrt(P_lower)
    sqrt_P_upper = math.sqrt(P_upper)
    L = (token0_amount * sqrt_P_upper * sqrt_P_lower) / (sqrt_P_upper - sqrt_P_lower)
    return L

# Function to calculate liquidity for token1
def calculate_liquidity_token1(token1_amount, P_lower, P_upper):
    sqrt_P_lower = math.sqrt(P_lower)
    sqrt_P_upper = math.sqrt(P_upper)
    L = token1_amount / (sqrt_P_upper - sqrt_P_lower)
    return L

# Function to calculate swap amount for token0
def calculate_swap_token0(L, P_lower, P_upper):
    sqrt_P_lower = math.sqrt(P_lower)
    sqrt_P_upper = math.sqrt(P_upper)
    swap_amount_token0 = L * (sqrt_P_upper - sqrt_P_lower) / (sqrt_P_upper * sqrt_P_lower)
    return swap_amount_token0

# Function to calculate swap amount for token1
def calculate_swap_token1(L, P_lower, P_upper):
    sqrt_P_lower = math.sqrt(P_lower)
    sqrt_P_upper = math.sqrt(P_upper)
    swap_amount_token1 = L * (sqrt_P_upper - sqrt_P_lower)
    return swap_amount_token1

def generate_random_token0_amount(liquidity, pool_price):
    """
    Generate a random token0 amount for swapping based on liquidity and price sensitivity,
    with additional scaling to reduce swap amounts for high prices and liquidity.
    
    Args:
    liquidity (float): The current total liquidity in the pool.
    pool_price (float): The current pool price of token0 in terms of token1.
    
    Returns:
    float: A capped and scaled token0 amount for swapping.
    """
    # Define a smaller range for the percentage of liquidity to swap (0.001% to 0.1%)
    percentage_of_liquidity = random.uniform(0.00001, 0.001)  # Smaller percentage range to reduce amount
    
    # Calculate base token0 amount as a fraction of the liquidity
    token0_amount = percentage_of_liquidity * min(liquidity, 10000) * pool_price  # Adjust based on price
    
    # Apply dynamic scaling based on price and liquidity to limit extreme values
    scaling_factor = 1 / (pool_price ** 0.5)  # Inverse square root of price to reduce large values
    
    # Calculate the final token0 amount
    token0_amount *= scaling_factor
    
    # Introduce a hard cap to prevent the swap amount from being too large
    max_swap_cap = 1000  # Adjust this cap based on pool behavior
    
    return min(token0_amount, max_swap_cap)

def generate_random_token1_amount(liquidity, pool_price):
    """
    Generate a random token1 amount for swapping based on liquidity and price sensitivity,
    with additional scaling to reduce swap amounts for high prices and liquidity.
    
    Args:
    liquidity (float): The current total liquidity in the pool.
    pool_price (float): The current pool price of token0 in terms of token1.
    
    Returns:
    float: A capped and scaled token1 amount for swapping.
    """
    # Define a smaller range for the percentage of liquidity to swap (0.001% to 0.1%)
    percentage_of_liquidity = random.uniform(0.00001, 0.001)  # Smaller percentage range to reduce swap amount
    
    # Calculate base token1 amount as a fraction of the liquidity, adjusted by the pool price
    token1_amount = percentage_of_liquidity * min(liquidity, 10000) / pool_price  # Adjust for price in terms of token1
    
    # Apply dynamic scaling based on price and liquidity to limit extreme values
    scaling_factor = 1 / (pool_price ** 0.5)  # Inverse square root of price to reduce large values
    
    # Calculate the final token1 amount
    token1_amount *= scaling_factor
    
    # Introduce a hard cap to prevent the swap amount from being too large
    max_swap_cap = 1000  # Adjust this cap based on pool behavior
    
    return min(token1_amount, max_swap_cap)

def generate_random_liquidity_addition_q96(current_price, current_liquidity_q96, max_liquidity_cap_q96):

    # Define a small random percentage of the current Q96 liquidity to add (between 0.001% and 0.05%)
    percentage_of_liquidity_to_add = random.uniform(0.00001, 0.0005)  # Smaller range to account for large Q96 liquidity
    
    # Calculate the liquidity addition in Q96 format
    liquidity_addition_q96 = int(percentage_of_liquidity_to_add * current_liquidity_q96)
    
    # Scale liquidity addition based on the current price (use square root scaling)
    price_scaling_factor = current_price ** 0.5  # Square root of price for scaling
    
    # Apply scaling to liquidity addition
    liquidity_addition_q96 = int(liquidity_addition_q96 * price_scaling_factor)
    
    # Introduce a hard cap to prevent too much liquidity addition
    return min(liquidity_addition_q96, max_liquidity_cap_q96)
