## V3 Improvements
1. Integarte voyager simulator
2. Improve policy functions
3. Improve obs space, action space and reward function


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

## Usage

This framework offers a set of API endpoints to initialize the environment, train and evaluate models using both DDPG and PPO algorithms, and perform inference based on the Uniswap V3 model and user-defined parameters. Below are example `curl` commands to interact with these endpoints.

### Initialize Script

To set the base path and reset environment variables for the framework, use the following command:

```bash
curl -X POST http://127.0.0.1:8000/initialize_script/ \
-H "Content-Type: application/json" \
-d '{
    "base_path": "<path_to_your_cloned_repository>",
    "reset_env_var": true
}'
```

Replace `<path_to_your_cloned_repository>` with the actual path where the repository is located on your system.

### Train DDPG Agent

To train the DDPG agent with custom parameters:

```bash
curl -X POST http://127.0.0.1:8000/train_ddpg/ \
-H "Content-Type: application/json" \
-d '{
    "max_steps": 2,
    "n_episodes": 2,
    "model_name": "model_storage/ddpg/ddpg_fazool",
    "alpha": 0.001,
    "beta": 0.001,
    "tau": 0.8,
    "batch_size": 50,
    "training": true,
    "agent_budget_usd": 10000,
    "use_running_statistics": false
}'
```

### Evaluate DDPG Agent

To evaluate the DDPG agent:

```bash
curl -X POST http://127.0.0.1:8000/evaluate_ddpg/ \
-H "Content-Type: application/json" \
-d '{
    "eval_steps": 2,
    "eval_episodes": 2,
    "model_name": "model_storage/ddpg/ddpg_fazool",
    "percentage_range": 0.6,
    "agent_budget_usd": 10000,
    "use_running_statistics": false
}'
```

### Train PPO Agent

To train the PPO agent with custom parameters:

```bash
curl -X POST http://127.0.0.1:8000/train_ppo/ \
-H "Content-Type: application/json" \
-d '{
    "max_steps": 2,
    "n_episodes": 2,
    "model_name": "model_storage/ppo/ppo2_fazool22",
    "buffer_size": 5,
    "n_epochs": 20,
    "gamma": 0.5,
    "alpha": 0.001,
    "gae_lambda": 0.75,
    "policy_clip": 0.6,
    "max_grad_norm": 0.6,
    "agent_budget_usd": 10000,
    "use_running_statistics": false,
    "action_transform": "linear"
}'
```

### Evaluate PPO Agent

To evaluate the PPO agent:

```bash
curl -X POST http://127.0.0.1:8000/evaluate_ppo/ \
-H "Content-Type: application/json" \
-d '{
    "eval_steps": 2,
    "eval_episodes": 2,
    "model_name": "model_storage/ppo/ppo2_fazool",
    "percentage_range": 0.5,
    "agent_budget_usd": 10000,
    "use_running_statistics": false,
    "action_transform": "linear"
}'
```

### Perform Inference

To perform inference using the model, providing details about the pool state and user preferences:

```bash
curl -X POST http://127.0.0.1:8000/inference/ \
-H "Content-Type: application/json" \
-d '{
    "pool_state": {
        "current_profit": 500,
        "price_out_of_range": false,
        "time_since_last_adjustment": 40000,
        "pool_volatility": 0.2
    },
    "user_preferences": {
        "risk_tolerance": {"profit_taking": 50, "stop_loss": -500},
        "investment_horizon": 7,
        "liquidity_preference": {"adjust_on_price_out_of_range": true},
        "risk_aversion_threshold": 0.1,
        "user_status": "new_user"
    },
    "pool_id": "0x4e68ccd3e89f51c3074ca5072bbac773960dfa36",
    "ddpg_agent_path": "model_storage/ddpg/ddpg_1",

    "ppo_agent_path": "model_storage/ppo/lstm_actor_critic_batch_norm"
}'
```

These commands serve as a starting point for interacting with the framework. Customize the JSON payload as needed to fit your specific requirements for training, evaluation, and inference.

---
