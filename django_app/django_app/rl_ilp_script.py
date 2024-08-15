# %% [markdown]
# # Set Paths

# %%
#export PATH=$PATH:.
#base_path="/home/azureuser/Intelligent-Liquidity-Provisioning-Framework"
import sys
import os
import pathlib

base_path = pathlib.Path().resolve().parent.as_posix()
reset_env_var = False
sys.path.append(base_path)
os.chdir(base_path)
os.environ["PATH"] += ":."

def env_setup(base_path, reset_env_var):
    base_path=base_path
    sys.path.append(base_path)
    os.chdir(base_path)
    os.environ["PATH"] += ":."
    reset_env_var=reset_env_var

# %% [markdown]
# # Reset Env

# %%
def reset_env():
    import shutil
    import os
    import json

    # Define the paths
    folder_path = os.path.join(base_path, "v3_core/build/deployments")
    json_file1_path = os.path.join(base_path, "model_storage/token_pool_addresses.json")
    json_file2_path = os.path.join(base_path, "model_storage/liq_positions.json")

    # 1. Delete the folder and its contents
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    # 2. Delete contents of the first JSON file
    with open(json_file1_path, 'w') as file:
        file.write("{}")

    # 3. Delete contents of the second JSON file and add {}
    with open(json_file2_path, 'w') as file:
        file.write("{}")
        
if reset_env_var==True:
    reset_env()

# %% [markdown]
# # Env Setup

# %%
from netlists.uniswapV3.netlist import SimStrategy,SimState,netlist_createLogData
from engine.SimEngine import SimEngine
from util.globaltokens import weth_usdc_pool,eth_dai_pool,btc_usdt_pool,btc_weth_pool
import brownie
from util.constants import GOD_ACCOUNT,RL_AGENT_ACCOUNT
from util.base18 import toBase18, fromBase18,fromBase128,price_to_valid_tick
from model_scripts.plot import train_rewards_plot,eval_rewards_plot,train_raw_actions_plot,train_scaled_actions_plot,train_combined_metrics_plot,train_separate_episode_action_plot
from model_scripts.sync_pool_subgraph_data import fetch_inference_pool_data,fetch_inference_pool_data_1

#Imports
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import mlflow
import mlflow.tensorflow
mlflow.tensorflow.autolog()

# %% [markdown]
# # Training env

# %%
#env=DiscreteSimpleEnv(agent_budget_usd=10000,use_running_statistics=False)
#n_actions = sum(action_space.shape[0] for action_space in env.action_space.values())
#input_dims = sum(np.prod(env.observation_space.spaces[key].shape) for key in env.observation_space.spaces.keys())


# %% [markdown]
## RL Agents

# %% [markdown]
# ## DDPG Agent

# %%
    
# Training Loop
def train_ddpg_agent(max_steps=100, n_episodes=10, model_name='model_storage/ddpg/ddpg_2',alpha=0.001, beta=0.001, tau=0.8,batch_size=50, training=True,agent_budget_usd=10000,use_running_statistics=False,action_transform='linear'):
    env=DiscreteSimpleEnv(agent_budget_usd=agent_budget_usd,use_running_statistics=use_running_statistics,action_transform=action_transform)
    n_actions = sum(action_space.shape[0] for action_space in env.action_space.values())
    input_dims = sum(np.prod(env.observation_space.spaces[key].shape) for key in env.observation_space.spaces.keys())
    ddpg_agent = DDPG(alpha=alpha, beta=beta, input_dims=input_dims, tau=tau, env=env, n_actions=n_actions, batch_size=batch_size, training=training)
    
    for i in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            action = ddpg_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ddpg_agent.remember(state, action, reward, next_state, done)
            ddpg_agent.learn()
            state = next_state
            episode_reward += reward
            if done:
                break
        print(f"Episode {i+1}: Reward = {episode_reward}")
        #ddpg_agent.memory.clear()
    
    # Create dummy data for model input shape
    dummy_state = np.random.random((1, input_dims))
    dummy_action = np.random.random((1, n_actions))

    # Run dummy data through models to build them
    ddpg_agent.actor(dummy_state)
    ddpg_agent.critic(dummy_state, dummy_action)

    
    model_base_path = os.path.join(base_path,model_name)
    ddpg_actor_model_path = os.path.join(model_base_path, 'actor')
    ddpg_critic_model_path = os.path.join(model_base_path, 'critic')

    # Saved Trained weights
    ddpg_agent.actor.save_weights(ddpg_actor_model_path)
    ddpg_agent.critic.save_weights(ddpg_critic_model_path)

    # Save trained model
    ddpg_agent.actor.save(ddpg_actor_model_path)
    ddpg_agent.critic.save(ddpg_critic_model_path)

    ddpg_train_data_log=env.train_data_log
    output_file=ddpg_training_vis(ddpg_train_data_log,model_name)

    return ddpg_train_data_log,ddpg_actor_model_path,ddpg_critic_model_path

def ddpg_training_vis(ddpg_train_data_log,model_name):
    df_data = []

    for entry in ddpg_train_data_log:
        episode, step_count, scaled_action, raw_state, tensor_data, scaled_state, raw_reward, scaled_reward, cumulative_reward, fee_earned, impermanent_loss = entry
        
        # Extract raw_action values from tensor_data
        raw_action_0 = float(tensor_data[0][0].numpy())
        raw_action_1 = float(tensor_data[0][1].numpy())

        scaled_action_0 = scaled_action['price_lower']
        scaled_action_1 = scaled_action['price_upper']
        
        # Combine all data into a single dictionary
        data = {
            'episode': episode,
            'step_count': step_count,
            'scaled_action_0':scaled_action_0,
            'scaled_action_1':scaled_action_1,
            'raw_reward': raw_reward,
            'scaled_reward': scaled_reward,
            'fee_earned': fee_earned,
            'impermanent_loss': impermanent_loss,
            'cumulative_reward': cumulative_reward,
            'raw_action_0': raw_action_0,
            'raw_action_1': raw_action_1,
        }
        
        # Add raw_action, global_state, and state data
        #data.update(scaled_action)
        data.update(raw_state)
        data.update(scaled_state)
        
        df_data.append(data)

    ddpg_train_data_df = pd.DataFrame(df_data)
    base_model_name = os.path.basename(model_name)
    output_dir = os.path.join('model_outdir_csv', 'ddpg', base_model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'train_logs.csv')
            
    ddpg_train_data_df.to_csv(output_file, index=False)

    #train_rewards_plot(ddpg_train_data_df)
    #train_raw_actions_plot(ddpg_train_data_df)
    #train_scaled_actions_plot(ddpg_train_data_df)
    #train_combined_metrics_plot(ddpg_train_data_df)
    #train_separate_episode_action_plot(ddpg_train_data_df)
    
    train_rewards_plot(ddpg_train_data_df, output_dir)
    train_raw_actions_plot(ddpg_train_data_df, output_dir)
    train_scaled_actions_plot(ddpg_train_data_df, output_dir)
    train_combined_metrics_plot(ddpg_train_data_df, output_dir)
    train_separate_episode_action_plot(ddpg_train_data_df, output_dir)

    return output_file

def eval_ddpg_agent(eval_steps=100,eval_episodes=2,model_name='model_storage/ddpg/200_100_step_running_stats_lstm_bn_global_obs_norm',percentage_range=0.5,agent_budget_usd=10000,use_running_statistics=False):

    eval_env = DiscreteSimpleEnvEval(agent_budget_usd=agent_budget_usd,percentage_range=percentage_range, seed=42,use_running_statistics=use_running_statistics)
    n_actions = sum(action_space.shape[0] for action_space in eval_env.action_space.values())
    input_dims = sum(np.prod(eval_env.observation_space.spaces[key].shape) for key in eval_env.observation_space.spaces.keys())
    ddpg_eval_agent = DDGPEval(env=eval_env, n_actions=n_actions, input_dims=input_dims, training=False)
    model_base_path = os.path.join(base_path, model_name)

    ddpg_actor_model_path = os.path.join(model_base_path, 'actor')
    ddpg_critic_model_path = os.path.join(model_base_path, 'critic')

    ddpg_eval_agent.actor.load_weights(ddpg_actor_model_path)
    ddpg_eval_agent.critic.load_weights(ddpg_critic_model_path)

    for episode in range(eval_episodes):
        state = eval_env.reset()
        episode_reward = 0
        
        for step in range(eval_steps):
            action = ddpg_eval_agent.choose_action(state)
            next_state, reward, done, _ = eval_env.step(action)
            
            episode_reward += reward
            state = next_state
            if done:
                break
        print(f"Episode {episode+1}/{eval_episodes}, Reward: {episode_reward}")

    ppo_eval_data_log=eval_env.eval_data_log
    ddpg_eval_vis(ppo_eval_data_log,model_name)

    return ppo_eval_data_log

def ddpg_eval_vis(ppo_eval_data_log,model_name):
    df_eval_data = []

    for entry in ppo_eval_data_log:
        (episode, step_count, global_state, raw_action_rl_agent, action_rl_agent, 
        raw_action_baseline_agent, action_baseline_agent, state, 
        raw_reward_rl_agent, raw_reward_baseline_agent,scaled_reward_rl_agent,scaled_reward_baseline_agent, cumulative_reward_rl_agent, 
        cumulative_reward_baseline_agent, fee_income_rl_agent, impermanent_loss_rl_agent, 
        fee_income_baseline_agent, impermanent_loss_baseline_agent) = entry
        
        # Extract raw_action values for RL agent
        raw_action_rl_agent_0 = float(raw_action_rl_agent[0][0].numpy())
        raw_action_rl_agent_1 = float(raw_action_rl_agent[0][1].numpy())

        scaled_action_rl_agent_0 = action_rl_agent['price_lower']
        scaled_action_rl_agent_1 = action_rl_agent['price_upper']
        
        # Extract raw_action values for baseline agent
        raw_action_baseline_agent_0 = raw_action_baseline_agent['price_lower']
        raw_action_baseline_agent_1 = raw_action_baseline_agent['price_upper']
        
        # Combine all data into a single dictionary
        data = {
            'episode': episode,
            'step_count': step_count,
            'raw_reward_rl_agent': raw_reward_rl_agent,
            'scaled_reward_rl_agent':scaled_reward_rl_agent,
            'cumulative_reward_rl_agent': cumulative_reward_rl_agent,
            'scaled_action_rl_agent_0':scaled_action_rl_agent_0,
            'scaled_action_rl_agent_1':scaled_action_rl_agent_1,
            'fee_income_rl_agent': fee_income_rl_agent,
            'impermanent_loss_rl_agent': impermanent_loss_rl_agent,
            'raw_reward_baseline_agent': raw_reward_baseline_agent,
            'scaled_reward_baseline_agent':scaled_reward_baseline_agent,
            'cumulative_reward_baseline_agent': cumulative_reward_baseline_agent,
            'raw_action_baseline_agent_0': raw_action_baseline_agent_0,
            'raw_action_baseline_agent_1': raw_action_baseline_agent_1,
            'fee_income_baseline_agent': fee_income_baseline_agent,
            'impermanent_loss_baseline_agent': impermanent_loss_baseline_agent,
            'raw_action_rl_agent_0': raw_action_rl_agent_0,
            'raw_action_rl_agent_1': raw_action_rl_agent_1,

        }
        
        # Add action, global_state, and state data
        
        data.update(global_state)
        data.update(state)
        
        df_eval_data.append(data)

    ddpg_eval_data_df = pd.DataFrame(df_eval_data)
    base_model_name = os.path.basename(model_name)
    output_dir = os.path.join('model_outdir_csv', 'ddpg', base_model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'eval_logs.csv')
            
    ddpg_eval_data_df.to_csv(output_file, index=False)

    eval_rewards_plot(ddpg_eval_data_df,output_dir)

# %% [markdown]
# ## PPO Agent (Stochastic Policy)

# %%

def train_ppo_agent(max_steps=100, n_episodes=10, model_name='model_storage/ppo/ppo2', buffer_size=50,n_epochs=10, gamma=0.5, alpha=0.01, gae_lambda=0.75, policy_clip=0.8, max_grad_norm=10,agent_budget_usd=10000,use_running_statistics=False,action_transform='linear'):
    
    env=DiscreteSimpleEnv(agent_budget_usd=agent_budget_usd,use_running_statistics=use_running_statistics, action_transform=action_transform)
    n_actions = sum(action_space.shape[0] for action_space in env.action_space.values())
    input_dims = sum(np.prod(env.observation_space.spaces[key].shape) for key in env.observation_space.spaces.keys())
    ppo_agent = PPO(env, n_actions, observation_dims=input_dims,buffer_size=buffer_size,n_epochs=n_epochs, gamma=gamma, alpha=alpha, gae_lambda=gae_lambda, policy_clip=policy_clip, max_grad_norm=max_grad_norm)
    for i in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            action,log_prob = ppo_agent.choose_action(state)
        

            next_state, reward, done, _ = env.step(action)

            ppo_agent.remember(state, action, reward, next_state, done,log_prob)

            state = next_state
            episode_reward += reward

            if ppo_agent.rollout_buffer.is_full():
                ppo_agent.learn()
            if done:
                break
        print(f"Episode {i+1}: Reward = {episode_reward}")

    ppo_model_base_path = os.path.join(base_path,model_name)
    ppo_actor_model_path = os.path.join(ppo_model_base_path, 'actor')
    ppo_critic_model_path = os.path.join(ppo_model_base_path, 'critic')
    
    # Save trained weights
    ppo_agent.actor.save_weights(ppo_actor_model_path)
    ppo_agent.critic.save_weights(ppo_critic_model_path)

    # Save trained model
    ppo_agent.actor.save(ppo_actor_model_path)
    ppo_agent.critic.save(ppo_critic_model_path)

    ppo_train_data_log=env.train_data_log
    ppo_training_vis(ppo_train_data_log,model_name)

    return ppo_train_data_log,ppo_actor_model_path,ppo_critic_model_path

def ppo_training_vis(ppo_train_data_log,model_name):
    df_data = []

    for entry in ppo_train_data_log:
        episode, step_count, scaled_action, raw_state, tensor_data, scaled_state, raw_reward, scaled_reward, cumulative_reward, fee_earned, impermanent_loss = entry
        
        # Extract raw_action values from tensor_data
        raw_action_0 = float(tensor_data[0][0].numpy())
        raw_action_1 = float(tensor_data[0][1].numpy())

        scaled_action_0 = scaled_action['price_lower']
        scaled_action_1 = scaled_action['price_upper']
        
        # Combine all data into a single dictionary
        data = {
            'episode': episode,
            'step_count': step_count,
            'scaled_action_0':scaled_action_0,
            'scaled_action_1':scaled_action_1,
            'raw_reward': raw_reward,
            'scaled_reward': scaled_reward,
            'fee_earned': fee_earned,
            'impermanent_loss': impermanent_loss,
            'cumulative_reward': cumulative_reward,
            'raw_action_0': raw_action_0,
            'raw_action_1': raw_action_1,
        }
        
        # Add raw_action, global_state, and state data
        #data.update(scaled_action)
        data.update(raw_state)
        data.update(scaled_state)
        
        df_data.append(data)

    ppo_train_data_df = pd.DataFrame(df_data)
    base_model_name = os.path.basename(model_name)
    output_dir = os.path.join('model_outdir_csv', 'ppo', base_model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'train_logs.csv')
            
    ppo_train_data_df.to_csv(output_file, index=False)

    train_rewards_plot(ppo_train_data_df,output_dir)
    train_raw_actions_plot(ppo_train_data_df,output_dir)
    train_scaled_actions_plot(ppo_train_data_df,output_dir)
    train_combined_metrics_plot(ppo_train_data_df,output_dir)
    train_separate_episode_action_plot(ppo_train_data_df,output_dir)

def eval_ppo_agent(eval_steps=100, eval_episodes=2, model_name='model_storage/ppo/lstm_actor_critic_batch_norm',percentage_range=0.6,agent_budget_usd=10000, use_running_statistics=False, action_transform="linear"):

    eval_env = DiscreteSimpleEnvEval(agent_budget_usd=agent_budget_usd,percentage_range=percentage_range, seed=42, penalty_param_magnitude=0, use_running_statistics=use_running_statistics, action_transform=action_transform)
    n_actions = sum(action_space.shape[0] for action_space in eval_env.action_space.values())
    input_dims = sum(np.prod(eval_env.observation_space.spaces[key].shape) for key in eval_env.observation_space.spaces.keys())
    ppo_eval_agent = PPOEval(eval_env, n_actions, observation_dims=input_dims, buffer_size=5,training=False)

    model_base_path = os.path.join(base_path,model_name)

    ppo_actor_model_path = os.path.join(model_base_path, 'actor')
    ppo_critic_model_path = os.path.join(model_base_path, 'critic')

    ppo_eval_agent.actor.load_weights(ppo_actor_model_path)
    ppo_eval_agent.critic.load_weights(ppo_critic_model_path)

    for episode in range(eval_episodes):
        state = eval_env.reset()
        episode_reward = 0
        
        for step in range(eval_steps):
            action,_ = ppo_eval_agent.choose_action(state)
            next_state, reward, done, _ = eval_env.step(action)
            
            episode_reward += reward
            
            state = next_state
            if done:
                break
        print(f"Episode {episode+1}/{eval_episodes}, Reward: {episode_reward}")
    ppo_eval_data_log=eval_env.eval_data_log
    ppo_eval_vis(ppo_eval_data_log,model_name)
    return ppo_eval_data_log


def ppo_eval_vis(ppo_eval_data_log,model_name):
    df_eval_data = []    
    for entry in ppo_eval_data_log:
        (episode, step_count, global_state, raw_action_rl_agent, action_rl_agent, 
        raw_action_baseline_agent, action_baseline_agent, state, 
        raw_reward_rl_agent, raw_reward_baseline_agent,scaled_reward_rl_agent,scaled_reward_baseline_agent, cumulative_reward_rl_agent, 
        cumulative_reward_baseline_agent, fee_income_rl_agent, impermanent_loss_rl_agent, 
        fee_income_baseline_agent, impermanent_loss_baseline_agent) = entry
        
        # Extract raw_action values for RL agent
        raw_action_rl_agent_0 = float(raw_action_rl_agent[0][0].numpy())
        raw_action_rl_agent_1 = float(raw_action_rl_agent[0][1].numpy())

        scaled_action_rl_agent_0 = action_rl_agent['price_lower']
        scaled_action_rl_agent_1 = action_rl_agent['price_upper']
        
        # Extract raw_action values for baseline agent
        raw_action_baseline_agent_0 = raw_action_baseline_agent['price_lower']
        raw_action_baseline_agent_1 = raw_action_baseline_agent['price_upper']
        
        # Combine all data into a single dictionary
        data = {
            'episode': episode,
            'step_count': step_count,
            'raw_reward_rl_agent': raw_reward_rl_agent,
            'scaled_reward_rl_agent':scaled_reward_rl_agent,
            'cumulative_reward_rl_agent': cumulative_reward_rl_agent,
            'scaled_action_rl_agent_0':scaled_action_rl_agent_0,
            'scaled_action_rl_agent_1':scaled_action_rl_agent_1,
            'fee_income_rl_agent': fee_income_rl_agent,
            'impermanent_loss_rl_agent': impermanent_loss_rl_agent,
            'raw_reward_baseline_agent': raw_reward_baseline_agent,
            'scaled_reward_baseline_agent':scaled_reward_baseline_agent,
            'cumulative_reward_baseline_agent': cumulative_reward_baseline_agent,
            'raw_action_baseline_agent_0': raw_action_baseline_agent_0,
            'raw_action_baseline_agent_1': raw_action_baseline_agent_1,
            'fee_income_baseline_agent': fee_income_baseline_agent,
            'impermanent_loss_baseline_agent': impermanent_loss_baseline_agent,
            'raw_action_rl_agent_0': raw_action_rl_agent_0,
            'raw_action_rl_agent_1': raw_action_rl_agent_1,

        }
        # Add action, global_state, and state data
        data.update(global_state)
        data.update(state)
        
        df_eval_data.append(data)

    ppo_eval_data_df = pd.DataFrame(df_eval_data)

    base_model_name = os.path.basename(model_name)
    output_dir = os.path.join('model_outdir_csv', 'ppo', base_model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'eval_logs.csv')
            
    ppo_eval_data_df.to_csv(output_file, index=False)

    eval_rewards_plot(ppo_eval_data_df,output_dir)


# %%
#%load_ext tensorboard
#%tensorboard --logdir ./model_storage

# %% [markdown]
# # Inference Pipeline

# %%


def postprocess_action(action, curr_price,action_transform='linear'):
    a_0, a_1 = action[0, 0].numpy(), action[0, 1].numpy()

    action_lower_bound = curr_price * 0.1
    action_upper_bound = curr_price * 2

    if action_transform=='linear':
        a_0 = np.clip(a_0, 0, 1)
        a_1 = np.clip(a_1, 0, 1)
        price_lower = action_lower_bound + a_0 * (action_upper_bound - action_lower_bound)/2
        price_upper = (action_upper_bound - action_lower_bound)/2 + a_1 * (action_upper_bound - action_lower_bound)/2

    elif action_transform=='exp':
        exp_a_0 = np.exp(a_0)
        exp_a_1 = np.exp(a_1)
        norm_exp_a_0 = exp_a_0 / (exp_a_0 + exp_a_1)
        norm_exp_a_1 = exp_a_1 / (exp_a_0 + exp_a_1)
        range_bound = action_upper_bound - action_lower_bound
        price_lower = action_lower_bound + norm_exp_a_0 * range_bound
        price_upper = action_lower_bound + norm_exp_a_1 * range_bound


    if price_lower > price_upper:
        price_lower, price_upper = price_upper, price_lower

    action_dict = {
        'price_lower': price_lower,
        'price_upper': price_upper
    }
    ticks_dict={
    'tick_lower':price_to_valid_tick(action_dict['price_lower']),
    'tick_upper':price_to_valid_tick(action_dict['price_upper'])
    }
    return action_dict,ticks_dict

def load_inference_agent(ddpg_agent_path='model_storage/ddpg/ddpg_1',ppo_agent_path='model_storage/ppo/lstm_actor_critic_batch_norm'):
    eval_data_log=[]
    eval_env = DiscreteSimpleEnvEval(agent_budget_usd=10000,percentage_range=0.30, seed=42)
    n_actions = sum(action_space.shape[0] for action_space in eval_env.action_space.values())
    input_dims = sum(np.prod(eval_env.observation_space.spaces[key].shape) for key in eval_env.observation_space.spaces.keys())

    # Load ddpg eval agent for predictions
    ddpg_eval_agent = DDGPEval(env=eval_env, n_actions=n_actions, input_dims=input_dims, training=False)
    model_base_path = os.path.join(base_path,ddpg_agent_path)
    ddpg_actor_model_path = os.path.join(model_base_path, 'actor')
    ddpg_critic_model_path = os.path.join(model_base_path, 'critic')
    ddpg_eval_agent.actor.load_weights(ddpg_actor_model_path)
    ddpg_eval_agent.critic.load_weights(ddpg_critic_model_path)

    # Load ppo eval agent for infernece
    eval_data_log=[]
    eval_env = DiscreteSimpleEnvEval(agent_budget_usd=10000,percentage_range=0.50, seed=42,penalty_param_magnitude=0,use_running_statistics=False)
    ppo_eval_agent = PPOEval(eval_env, n_actions, observation_dims=input_dims, buffer_size=5,training=False)
    model_base_path = os.path.join(base_path,ppo_agent_path)
    ppo_actor_model_path = os.path.join(model_base_path, 'actor')
    ppo_critic_model_path = os.path.join(model_base_path, 'critic')
    ppo_eval_agent.actor.load_weights(ppo_actor_model_path)
    ppo_eval_agent.critic.load_weights(ppo_critic_model_path)

    return ddpg_eval_agent,ppo_eval_agent, eval_env


def predict_action(pool_id="0x3416cf6c708da44db2624d63ea0aaef7113527c6",ddpg_agent_path='model_storage/ddpg/ddpg_1',ppo_agent_path='model_storage/ppo/lstm_actor_critic_batch_norm',date_str='2024-05-03'):
    pool_data = fetch_inference_pool_data_1(pool_id,date_str)
    print(f"State Space: {pool_data}")

    global_state = pool_data
    curr_price = global_state['token1Price']
    liquidity = global_state['liquidity']
    fee_growth_0 = global_state['feeGrowthGlobal0X128']
    fee_growth_1 = global_state['feeGrowthGlobal1X128']

    obs = {'scaled_curr_price': curr_price/5000, 'scaled_liquidity': liquidity/1e20, 
           'scaled_feeGrowthGlobal0x128': fee_growth_0/1e34, 'scaled_feeGrowthGlobal1x128': fee_growth_1/1e34}
    print(f"Obs Space: {obs}")

    ddpg_eval_agent,ppo_eval_agent,eval_env=load_inference_agent(ddpg_agent_path,ppo_agent_path)

    eval_env.reset()

    # DDPG Agent Action
    ddpg_action = ddpg_eval_agent.choose_action(obs)
    ddpg_action_dict,ddpg_action_ticks = postprocess_action(ddpg_action, curr_price,action_transform='linear')

    # PPO Agent Action
    ppo_action, _ = ppo_eval_agent.choose_action(obs)
    ppo_action_dict,ppo_action_ticks = postprocess_action(ppo_action, curr_price,action_transform='exp')

    return ddpg_action,ddpg_action_dict,ddpg_action_ticks,ppo_action, ppo_action_dict,ppo_action_ticks

def perform_inference(user_preferences,pool_state,pool_id="0x99ac8ca7087fa4a2a1fb6357269965a2014abc35",ddpg_agent_path='model_storage/ddpg/ddpg_1',ppo_agent_path='model_storage/ppo/lstm_actor_critic_batch_norm',date_str = '2024-05-05'):
    # Extracting necessary information from the pool state
    current_profit = pool_state['current_profit']
    price_out_of_range = pool_state['price_out_of_range']
    time_since_last_adjustment = pool_state['time_since_last_adjustment']
    pool_volatility = pool_state['pool_volatility']

    # User Preferences
    risk_tolerance = user_preferences['risk_tolerance']
    investment_horizon = user_preferences['investment_horizon']
    liquidity_preference = user_preferences['liquidity_preference']
    volatility_threshold=user_preferences['risk_aversion_threshold']


    # Adjust thresholds based on user preferences
    profit_taking_threshold = risk_tolerance['profit_taking']
    stop_loss_threshold = risk_tolerance['stop_loss']
    rebalance_interval = investment_horizon * 24 * 60 * 60  # Convert days to seconds

    # Predicted actions from RL agents
    ddpg_action,ddpg_action_dict,ddpg_action_ticks, ppo_action,ppo_action_dict,ppo_action_ticks=predict_action(pool_id=pool_id,ddpg_agent_path=ddpg_agent_path,ppo_agent_path=ppo_agent_path,date_str=date_str)

    # Decision Logic
    if user_preferences['user_status']=='new_user':
        return 'Add new liquidity position',ddpg_action_dict,ppo_action_dict
    if current_profit >= profit_taking_threshold:
        return 'adjust_position', ddpg_action_dict,ppo_action_dict
    elif price_out_of_range and liquidity_preference['adjust_on_price_out_of_range']:
        return 'adjust_position', ddpg_action_dict,ppo_action_dict  
    elif pool_volatility > volatility_threshold:
        return 'exit_position', None,None  # Exit position in case of high volatility
    elif time_since_last_adjustment >= rebalance_interval:
        return 'rebalance_position', None,None  
    elif current_profit <= stop_loss_threshold:
        return 'exit_position', None,None  # Exit the position to stop further losses
    else:
        return 'maintain_position', None,None  # Maintain the current position
