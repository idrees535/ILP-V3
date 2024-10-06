import sys
import os 
import pandas as pd 
import numpy as np
from tqdm import tqdm 


# Add parent directory to sys.path to handle imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.train_env import DiscreteSimpleEnv
from environments.eval_env import DiscreteSimpleEnvEval
from agents.ddpg_agent import DDPG,DDGPEval
from util.plot import *
from agents.ppo_agent import PPO,PPOEval
from util.sync_pool_subgraph_data import *
from util.utility_functions import *
from util.constants import *

def train_ddpg_agent(max_steps=100, n_episodes=10, model_name=f'model_storage/ddpg/ddpg_{TIMESTAMP}',alpha=0.001, beta=0.001, tau=0.8,batch_size=50, training=True,agent_budget_usd=10000,use_running_statistics=False,action_transform='linear'):
    env=DiscreteSimpleEnv(agent_budget_usd=agent_budget_usd,use_running_statistics=use_running_statistics,action_transform=action_transform)
    n_actions = sum(action_space.shape[0] for action_space in env.action_space.values())
    input_dims = sum(np.prod(env.observation_space.spaces[key].shape) for key in env.observation_space.spaces.keys())
    ddpg_agent = DDPG(alpha=alpha, beta=beta, input_dims=input_dims, tau=tau, env=env, n_actions=n_actions, batch_size=batch_size, training=training)
    
    for i in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        for _ in tqdm(range(max_steps), desc= f'EPISODE {i+1}/{len(range(n_episodes))} Progress'):
            #print(f"\n__________________GOD_ACCOUNT balance: {GOD_ACCOUNT.balance()/10**18} Eth __________________\n")
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

    print(f"Base path for output : {BASE_PATH}")
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
    output_dir = os.path.join(BASE_PATH,'model_outdir_csv', 'ddpg', base_model_name)
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'train_logs.csv')
            
    ddpg_train_data_df.to_csv(output_file, index=False)

    #train_rewards_plot(ddpg_train_data_df)
    #train_raw_actions_plot(ddpg_train_data_df)
    #train_scaled_actions_plot(ddpg_train_data_df)
    #train_combined_metrics_plot(ddpg_train_data_df)
    #train_separate_episode_action_plot(ddpg_train_data_df)
    
    # train_rewards_plot(ddpg_train_data_df, output_dir)
    # train_raw_actions_plot(ddpg_train_data_df, output_dir)
    # train_scaled_actions_plot(ddpg_train_data_df, output_dir)
    # train_combined_metrics_plot(ddpg_train_data_df, output_dir)
    # train_separate_episode_action_plot(ddpg_train_data_df, output_dir)

    return output_file

def eval_ddpg_agent(eval_steps=100,eval_episodes=2,model_name='model_storage/ddpg/200_100_step_running_stats_lstm_bn_global_obs_norm',percentage_range=0.5,agent_budget_usd=10000,use_running_statistics=False):

    eval_env = DiscreteSimpleEnvEval(agent_budget_usd=agent_budget_usd,percentage_range=percentage_range, seed=42,use_running_statistics=use_running_statistics)
    n_actions = sum(action_space.shape[0] for action_space in eval_env.action_space.values())
    input_dims = sum(np.prod(eval_env.observation_space.spaces[key].shape) for key in eval_env.observation_space.spaces.keys())
    ddpg_eval_agent = DDGPEval(env=eval_env, n_actions=n_actions, input_dims=input_dims, training=False)

    ddpg_actor_model_path = os.path.join(BASE_PATH,model_name, 'actor')
    ddpg_critic_model_path = os.path.join(BASE_PATH,model_name, 'critic')

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

train_ddpg_agent(max_steps=10, n_episodes=1)