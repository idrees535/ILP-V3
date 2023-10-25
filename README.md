
# Intelligent Liquidity Provisioning Framework - Reinforcement Learning Approach to Optimize Liquidity Provisioning on Uniswap V3 Pools

# 🏗 Initial Setup

## Prerequisites

- Linux/MacOS
- Python 3.8.5+
- solc 0.7.6+ [[Instructions](https://docs.soliditylang.org/en/v0.8.9/installing-solidity.html)]
- ganache. To install: `npm install ganache --global`
- nvm 16.13.2, _not_ nvm 17. To install: `nvm install 16.13.2; nvm use 16.13.2`

## Install TokenSPICE

Open a new terminal and:
```console
#clone repo
git clone https://github.com/idrees535/Intelligent-Liquidity-Provisioning-Framework-V1

#create a virtual environment
python3 -m venv venv

#activate env
source venv/bin/activate

#install dependencies
pip install -r requirements.txt

#install brownie packages 
./brownie-install.sh
```

## Run Ganache

```console
source venv/bin/activate
#add pwd to bash path
export PATH=$PATH:.

#run ganache
tsp ganache
```
This will start a Ganache chain, and populate 9 accounts.

## Compile the contracts

```console
tsp compile
```
## RL Agent

1. model_notebooks/rl_lp_agent_ipynb contains RL agent environemnt and DDPG defined with it's training and evaluation scripts and run experiments. (No need to run reset env cell for first run or any subsequent run until you want to refresh deployed pools/tokens)
2. util/globaltokens.py file loads brownie compiled project from util/constants.py and deploys pools using model_scripts/UniswapV3_Model_V2.py class, which are being imported in model_notebooks/rl_lp_agent_ipynb to tarin RL agent
3. model_outdir_csv directory contains csv data of ABM, RL Agnt training and evaluation
4. model_storage directory contains  tensorboard RL agent training logs, saved actor critic models, liq_positions.json (contains local storage of all liquiidty position agent wise and pool wise), token_pool_addresses.json (contains deployed token and pool addresses in local storage)
5. For more details about setup and configuration of Tokenspice Agent based Simulator refer to tokenspice official Github Repo: https://github.com/tokenspice/tokenspice
6. model_scripts/agent_policies.py defines the policies of Uniswap agents (trader and liquidity provider)
7. model_scripts/plot.py contains visualization functions of training and evaluation
8. Instead of using Tokenspice CLI command (tsp run) to run agent based simulation in model_notebook/rl_lp_agent.ipynb notebook we use Folllowing script to initialize and run abm in agent environemnt:

### Initialize ABM with specific Pool

```console
from netlists.uniswapV3.netlist import SimStrategy,SimState,netlist_createLogData
from util.globaltokens import weth_usdc_pool,eth_dai_pool,btc_usdt_pool

sim_strategy = SimStrategy()
sim_state = SimState(ss=sim_strategy,pool=weth_usdc_pool)

output_dir = "model_outdir_csv"
netlist_log_func = netlist_createLogData

from engine.SimEngine import SimEngine
engine = SimEngine(sim_state, output_dir, netlist_log_func)

retail_lp_agent=sim_state.agents['retail_lp']._wallet.address
print(f'retail_lp_agent: {retail_lp_agent}')

noise_trader=sim_state.agents['noise_trader']._wallet.address
print(f'noise_trader_agent: {noise_trader}')
```

### Run ABM

```console
engine.reset()
engine.run()
```
