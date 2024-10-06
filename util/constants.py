import pathlib
import datetime
import brownie
# from brownie._config import CONFIG  

BASE_PATH = pathlib.Path().resolve().parent.as_posix()
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

BROWNIE_PROJECTUniV3 = brownie.project.load("../v3_core/", name="UniV3Project")
#print(BROWNIE_PROJECTUniV3.available_contracts)

if brownie.network.show_active() != "development":
    brownie.network.connect("development")

GOD_ACCOUNT = brownie.network.accounts[9]
RL_AGENT_ACCOUNT = brownie.network.accounts[8]
WALLET_LP = brownie.network.accounts[0]
WALLET_SWAPPER = brownie.network.accounts[1]

# evm stuff
GASLIMIT_DEFAULT = 5000000
BURN_ADDRESS = "0x000000000000000000000000000000000000dEaD"

# OPF_ACCOUNT = brownie.network.accounts[8]
# OPF_ADDRESS = OPF_ACCOUNT.address

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
