import brownie
from brownie import accounts
from model_scripts.UniswapV3_Model_v2 import UniV3Model


# Ensure you're connected to a network (optional)
# brownie.network.connect('development')  # or your desired network, e.g., 'ganache'

# # # Print the list of available accounts and their addresses
# accounts = brownie.network.accounts

_wallet = accounts.at("0xa31014fDF60494ad2AD4Dba69D525D09E27f87C6", force=True)

# Loop through and print each account address
for i, account in enumerate(accounts):
    # print(f"Account {i}: {account} : {accounts[i].balance()/10**18}")
    print (f"Account {i}: {UniV3Model().get_wallet_balances(accounts[i])}")

# # Accessing the GOD_ACCOUNT and RL_AGENT_ACCOUNT if available
# if len(accounts) > 9:
GOD_ACCOUNT = accounts[9]


#print(f"GOD_ACCOUNT {GOD_ACCOUNT}: {type(GOD_ACCOUNT)} :   {GOD_ACCOUNT.balance()/10**18} Ether")

#print(f"_wallet : {_wallet} : {type(_wallet)}  :  {_wallet.balance()/10**18}")

#print (f"{UniV3Model().get_wallet_balances(_wallet)}")

