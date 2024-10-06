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

def eval_ppo_agent(eval_steps=100, eval_episodes=2, model_name='model_storage/ppo/lstm_actor_critic_batch_norm',percentage_range=0.6,agent_budget_usd=10000, use_running_statistics=False, action_transform="linear"):
    eval_env = DiscreteSimpleEnvEval(agent_budget_usd=agent_budget_usd,percentage_range=percentage_range, seed=42, penalty_param_magnitude=0, use_running_statistics=use_running_statistics, action_transform=action_transform)
    n_actions = sum(action_space.shape[0] for action_space in eval_env.action_space.values())
    input_dims = sum(np.prod(eval_env.observation_space.spaces[key].shape) for key in eval_env.observation_space.spaces.keys())
    ppo_eval_agent = PPOEval(eval_env, n_actions, observation_dims=input_dims, buffer_size=5,training=False)

    model_base_path = os.path.join(BASE_PATH,model_name)

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