from util.globaltokens import weth_usdc_pool,eth_dai_pool,btc_usdt_pool,btc_weth_pool
from enforce_typing import enforce_types
from django_app.django_app.rl_ilp_script import train_ddpg_agent,reset_env
# base_path = '/mnt/c/Users/MuhammadSaqib/Documents/ILP-Agent-Framework/'
#reset_env()
train_ddpg_agent(max_steps=10, n_episodes=1)

