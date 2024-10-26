import sys
import os 
import pandas as pd 
import numpy as np
from tqdm import tqdm 

# Add parent directory to sys.path to handle imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.hardhat_control import start_hardhat_node,stop_hardhat_node
start_hardhat_node()
from environments.train_env import DiscreteSimpleEnv
from environments.eval_env import DiscreteSimpleEnvEval
from agents.ddpg_agent import DDPG,DDGPEval
from util.plot import *
from agents.ppo_agent import PPO,PPOEval
from util.sync_pool_subgraph_data import *
from util.utility_functions import *
from util.constants import BROWNIE_PROJECTUniV3, GOD_ACCOUNT, WALLET_LP, WALLET_SWAPPER, RL_AGENT_ACCOUNT, BASE_PATH,TIMESTAMP


def train_ppo_agent(max_steps=100, n_episodes=10, model_name=f'model_storage/ppo/ppo_{TIMESTAMP}', buffer_size=50,n_epochs=10, gamma=0.5, alpha=0.01, gae_lambda=0.75, policy_clip=0.8, max_grad_norm=10,agent_budget_usd=10000,use_running_statistics=False,action_transform='linear'):
    env=DiscreteSimpleEnv(agent_budget_usd=agent_budget_usd,use_running_statistics=use_running_statistics, action_transform=action_transform)
    n_actions = sum(action_space.shape[0] for action_space in env.action_space.values())
    input_dims = sum(np.prod(env.observation_space.spaces[key].shape) for key in env.observation_space.spaces.keys())
    ppo_agent = PPO(env, n_actions, observation_dims=input_dims,buffer_size=buffer_size,n_epochs=n_epochs, gamma=gamma, alpha=alpha, gae_lambda=gae_lambda, policy_clip=policy_clip, max_grad_norm=max_grad_norm)
    
    for i in range(n_episodes):
        start_hardhat_node()
        state = env.reset()
        episode_reward = 0
        
        for  _ in tqdm(range(max_steps), desc= f'EPISODE {i+1}/{len(range(n_episodes))} Progress'):
            # sys.stdout = open(os.devnull, 'w') # Redirect standard output to null (suppress output)
            action,log_prob = ppo_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ppo_agent.remember(state, action, reward, next_state, done,log_prob)
            state = next_state
            episode_reward += reward
            if ppo_agent.rollout_buffer.is_full():
                ppo_agent.learn()
            # sys.stdout = sys.__stdout__ # Restore normal standard output
            if done:
                break
        print(f"Episode {i+1}: Reward = {episode_reward}")
        stop_hardhat_node()

    ppo_model_base_path = os.path.join(BASE_PATH,model_name)
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
    output_dir = os.path.join(BASE_PATH,'model_output', 'ppo', base_model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'ppo_train_logs.csv')
            
    ppo_train_data_df.to_csv(output_file, index=False)
    print(f"Output Directory: {output_dir}")

    train_rewards_plot(ppo_train_data_df,output_dir)
    train_raw_actions_plot(ppo_train_data_df,output_dir)
    train_scaled_actions_plot(ppo_train_data_df,output_dir)
    train_combined_metrics_plot(ppo_train_data_df,output_dir)
    train_separate_episode_action_plot(ppo_train_data_df,output_dir)
    
    return output_file

train_ppo_agent(max_steps=1000, n_episodes=20, buffer_size=10,n_epochs=5, gamma=0.5, alpha=0.001, gae_lambda=0.75, policy_clip=0.6, max_grad_norm=0.6)