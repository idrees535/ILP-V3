# ILP Agent Framework V3
ILP agent framewrok let's liquidty providers/liquidty managers  to build/devlop, train and deploy their liquidity managememnt agents with their custom insticts and prefernces. An ILP Agent consists if three differnt type of agents working together to efficiently manage liquidty On Uniswap V3. Following are three differnt components of this framewrok:

1. **Strategy Agent**
This agent mamnges the budget of vault/portofolio. Basically it allocates budget across differnt lqiuidty pools, and manges the distribution of budget based on historical performmance data of the lqiuidty pools, while incorporating the prefernces of user (Retail LP/ LM). User prefernce include total budget, investment horizon, risk factor, stop loss etc. Based on this stratgy agent allocates the total budget in differnt pools and speciifices the amount of budget allocated to each pool, and in case of multiple position based stratgy it alos speciifies the amount of budget allocated to each position in pool. The output of strategy agent for some input i.e {10000USD,30 Dasys, 0.6 Risk factor} willl be position 1: ETH/USD Pool, 3000USD, position 2 ETH/USD Pool, 1000USD, position 3: BTC/WETH Pool, 5000USD
This strategu agent also monitors these liquidty position on epoch basis i.e daily and then based on the progress and goals of the overall startegy adjusts these positions, it can remove positions, rebalance positions or add new posistions for ecrtain pools
2. **Predictor Agent**
This agent based on the output of stratgy agent {predicts the tick_lower,tick_upper} of each add position action suggested by stratgy agent. This agent is basically the current RL ILP agnet in V2 framework which given the state space predicts a liquidty position in given pool

3. **Executor Agent**
This executor agent executes the actions of stratgy agent and predictor agent. This is where giza agents platform which provides a secure and verifiable infrastructure of deploytemnt of these agents in production

## Implementtaion

### Improvements in Current RL Agent
1. Integarte voyager simulator
2. Improve policy functions
3. Improve obs space, action space and reward function

### Impelent Stratgy Agent
1. A gen AI model which is fine tuned on liquidty provisioning, portofolio managemnt and risk managemnt data, which devlops optimal budget allocation startgeies across pools
2. Function calling/tool calling capabilities to analyze the histroical data
3. Functions yto traanslate the user prefernces/risk appetite in stratgy actions
4. Monitor current LP positions and quantiatvely measure their performance across benchmarks and baselines
5. Based on monitoring analysis dynamicallya djust stratgies

### Integrate these agents with giza agents
1. Automate contarct calls execute stratgy actions

### Prerequisites

- Linux/MacOS environment
- Python 3.9.19 (Please ensure this strictly, otherwise you will get into dependency conflict issues)
- NVM, Node.js, ganache and solc
- Ganache CLI v6.12.2 (ganache-core: 2.13.2)

Install NVM:
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
```
Note: Ensure you have curl installed. If not, install it with sudo apt install curl.
Replace v0.39.1 with the latest release available on the nvm GitHub page.
Restart your terminal or source your profile to ensure nvm commands are available:

```bash
source ~/.bashrc
```
Install Node.js using nvm:
```bash
nvm install 16.3.0
nvm use 16.3.0
```

Install Ganache CLI using npm:
```bash
npm install -g ganache-cli@7.9.1
```

Install solc-js via npm: 
```bash
npm install -g solc
```
### Installing Dependencies

1. Clone the repository:

```bash
git clone https://github.com/yourgithubusername/Intelligent-Liquidity-Provisioning-Framework-V2.git
```

2. Navigate into the cloned directory:

```bash
cd Intelligent-Liquidity-Provisioning-Framework-V2
```

3. Create and activate a virtual environment:

```bash
python3 -m venv ilp_venv
source ilp_venv/bin/activate
```

4. Install required Python packages:

```bash
pip install -r requirements_td.txt
```

5. Ensure you have the `solc`, `ganache`, and `nvm` installed and upto date

6. Open a new terminal and start ganache cli in this, please ensure that latest version of ganache is installed in your systems
```bash
ganache-cli --gasLimit 10000000000 --gasPrice 1 --hardfork istanbul
```

### Django App Deployment

1. In a new terminal Navigate to the Django project directory:

```bash
cd django_app
```

2. Apply migrations to initialize the database:

```bash
python manage.py migrate
```

3. Run the Django development server:

```bash
python manage.py runserver 0.0.0.0:8000
```

The server will start, and the API endpoints will be accessible locally.

