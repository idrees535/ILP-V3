from util.globaltokens import weth_usdc_pool,eth_dai_pool,btc_usdt_pool,btc_weth_pool
from django_app.django_app.rl_ilp_script import *
# base_path = '/mnt/c/Users/MuhammadSaqib/Documents/ILP-Agent-Framework/'
env_setup(base_path, reset_env_var=True)
train_ddpg_agent(max_steps=10, n_episodes=1)

