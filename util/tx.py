"""tx utilities"""
import brownie
from brownie.network import chain


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
