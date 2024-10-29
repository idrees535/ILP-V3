import os 
import sys
import pathlib
import datetime
import brownie
import logging
# Suppress all logging across the entire application
logging.disable(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("web3").setLevel(logging.ERROR)
logging.getLogger("eth_utils").setLevel(logging.WARNING)
logging.getLogger("brownie").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
logging.getLogger("web3.middleware.geth_poa").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

BASE_PATH = pathlib.Path().resolve().parent.as_posix()
# BASE_PATH = '/mnt/c/Users/MuhammadSaqib/Documents/ILP-Agent-Framework'
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
os.chdir(BASE_PATH)
if "." not in os.environ["PATH"]:
    os.environ["PATH"] += ":."

if brownie.network.show_active() != "development":
    brownie.network.connect("development")
BROWNIE_PROJECTUniV3 = None
BROWNIE_PROJECTUniV3 = brownie.project.load(f"{BASE_PATH}/v3_core/", name="UniV3Project")

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
