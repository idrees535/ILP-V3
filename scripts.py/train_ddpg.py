import sys
import os 
import pandas as pd 
import numpy as np
from tqdm import tqdm 
import importlib

# Add parent directory to sys.path to handle imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.train_env import DiscreteSimpleEnv
from agents.ddpg_agent import DDPG,DDGPEval
from util.plot import *
from util.sync_pool_subgraph_data import *
from util.utility_functions import *
from util.constants import BROWNIE_PROJECTUniV3, GOD_ACCOUNT, WALLET_LP, WALLET_SWAPPER, RL_AGENT_ACCOUNT, BASE_PATH,TIMESTAMP

def train_ddpg_agent(max_steps=100, n_episodes=10, model_name=f'model_storage/ddpg/ddpg_{TIMESTAMP}',alpha=0.001, beta=0.001, tau=0.8,batch_size=50, training=True,agent_budget_usd=10000,use_running_statistics=False,action_transform='linear'):
    start_hardhat_node()
    env=DiscreteSimpleEnv(agent_budget_usd=agent_budget_usd,use_running_statistics=use_running_statistics,action_transform=action_transform)
    n_actions = sum(action_space.shape[0] for action_space in env.action_space.values())
    input_dims = sum(np.prod(env.observation_space.spaces[key].shape) for key in env.observation_space.spaces.keys())
    ddpg_agent = DDPG(alpha=alpha, beta=beta, input_dims=input_dims, tau=tau, env=env, n_actions=n_actions, batch_size=batch_size, training=training)
     
    for i in range(n_episodes):
        start_hardhat_node()
        import util.pool_configs
        importlib.reload(util.pool_configs)
        state = env.reset()
        episode_reward = 0
        
        for _ in tqdm(range(max_steps), desc= f'EPISODE {i+1}/{len(range(n_episodes))} Progress'):
            sys.stdout = open(os.devnull, 'w') # Redirect standard output to null (suppress output)
            action = ddpg_agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ddpg_agent.remember(state, action, reward, next_state, done)
            ddpg_agent.learn()
            state = next_state
            episode_reward += reward
            sys.stdout = sys.__stdout__ # Restore normal standard output
            if done:
                break
        print(f"Episode {i+1}: Reward = {episode_reward}")
        stop_hardhat_node()
        #ddpg_agent.memory.clear()
    
    # Create dummy data for model input shape
    dummy_state = np.random.random((1, input_dims))
    dummy_action = np.random.random((1, n_actions))

    # Run dummy data through models to build them
    ddpg_agent.actor(dummy_state)
    ddpg_agent.critic(dummy_state, dummy_action)

    print(f"Base path: {BASE_PATH}")
    ddpg_actor_model_path = os.path.join(BASE_PATH,model_name, 'actor')
    ddpg_critic_model_path = os.path.join(BASE_PATH,model_name, 'critic')

    # Saved Trained weights
    ddpg_agent.actor.save_weights(ddpg_actor_model_path)
    ddpg_agent.critic.save_weights(ddpg_critic_model_path)

    # Save trained model
    ddpg_agent.actor.save(ddpg_actor_model_path)
    ddpg_agent.critic.save(ddpg_critic_model_path)

    ddpg_train_data_log=env.train_data_log
    ddpg_training_vis(ddpg_train_data_log,model_name)

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
    output_dir = os.path.join(BASE_PATH,'model_output', 'ddpg', base_model_name)
    print(f"Output Directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'train_logs.csv')
         
    ddpg_train_data_df.to_csv(output_file, index=False)
    #print(ddpg_train_data_df)
    
    train_rewards_plot(ddpg_train_data_df, output_dir)
    train_raw_actions_plot(ddpg_train_data_df, output_dir)
    train_scaled_actions_plot(ddpg_train_data_df, output_dir)
    train_combined_metrics_plot(ddpg_train_data_df, output_dir)
    train_separate_episode_action_plot(ddpg_train_data_df, output_dir)

    return output_file

train_ddpg_agent(max_steps=500, n_episodes=50, alpha=0.0001, beta=0.0001)