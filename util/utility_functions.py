import math
import csv
import os 
import subprocess
import time
import signal
from util.constants import GOD_ACCOUNT, WALLET_LP, WALLET_SWAPPER, RL_AGENT_ACCOUNT, BASE_PATH,TIMESTAMP,HARDHAT_PROJECT_PATH
# from collections import OrderedDict
import brownie
# from brownie.network import chain

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

# Function to start the Hardhat node
def start_hardhat_node():
    print("Starting Hardhat node...")
    # Start Hardhat node in the background
    process = subprocess.Popen(
        ["npx", "hardhat", "node"], 
        cwd=HARDHAT_PROJECT_PATH,
        preexec_fn=os.setsid  # This makes it possible to stop the process later
    )
    time.sleep(60)  # Give some time for the node to start
    return process

# Function to stop the Hardhat node
def stop_hardhat_node():
    print("Stopping Hardhat node...")
    # Kill any process using port 8545
    subprocess.run(["sudo", "fuser", "-k", "8545/tcp"])
    print("Hardhat node stopped.")