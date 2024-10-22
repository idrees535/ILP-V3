import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

# Add parent directory to sys.path to handle imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environments.eval_env import DiscreteSimpleEnvEval
from agents.ddpg_agent import DDGPEval
from agents.ppo_agent import PPOEval
from util.sync_pool_subgraph_data import *
from util.utility_functions import *
from util.constants import *

class PredictAction:
    def __init__(self, ddpg_agent_path='model_storage/ddpg/ddpg_1', ppo_agent_path='model_storage/ppo/lstm_actor_critic_batch_norm'):
        # Initialize the agents and the environment
        self.ddpg_eval_agent, self.ppo_eval_agent, self.eval_env = self.load_inference_agents(ddpg_agent_path, ppo_agent_path)

    def load_inference_agents(self, ddpg_agent_path, ppo_agent_path):
        # Initialize environment
        eval_env = DiscreteSimpleEnvEval(agent_budget_usd=10000, percentage_range=0.30, seed=42)
        n_actions = sum(action_space.shape[0] for action_space in eval_env.action_space.values())
        input_dims = sum(np.prod(eval_env.observation_space.spaces[key].shape) for key in eval_env.observation_space.spaces.keys())

        # Load DDPG agent
        ddpg_eval_agent = DDGPEval(env=eval_env, n_actions=n_actions, input_dims=input_dims, training=False)
        ddpg_actor_model_path = os.path.join(BASE_PATH, ddpg_agent_path, 'actor')
        ddpg_critic_model_path = os.path.join(BASE_PATH, ddpg_agent_path, 'critic')
        ddpg_eval_agent.actor.load_weights(ddpg_actor_model_path)
        ddpg_eval_agent.critic.load_weights(ddpg_critic_model_path)

        # Load PPO agent
        ppo_eval_agent = PPOEval(eval_env, n_actions, observation_dims=input_dims, buffer_size=5, training=False)
        ppo_actor_model_path = os.path.join(BASE_PATH, ppo_agent_path, 'actor')
        ppo_critic_model_path = os.path.join(BASE_PATH, ppo_agent_path, 'critic')
        ppo_eval_agent.actor.load_weights(ppo_actor_model_path)
        ppo_eval_agent.critic.load_weights(ppo_critic_model_path)

        return ddpg_eval_agent, ppo_eval_agent, eval_env

    def predict_action(self, pool_id="0xcbcdf9626bc03e24f779434178a73a0b4bad62ed", date_str='2024-01-01'):
        # Fetch pool data
        global_state = fetch_inference_pool_data_1(pool_id, date_str)
        print(f"\n\nState Space: {global_state}\n\n")

        # Extract state
        curr_price = global_state['token1Price']
        liquidity = global_state['liquidity']
        fee_growth_0 = global_state['feeGrowthGlobal0X128']
        fee_growth_1 = global_state['feeGrowthGlobal1X128']

        obs = {
            'scaled_curr_price': curr_price / 5000,
            'scaled_liquidity': liquidity / 1e20,
            'scaled_feeGrowthGlobal0x128': fee_growth_0 / 1e34,
            'scaled_feeGrowthGlobal1x128': fee_growth_1 / 1e34
        }
        print(f"Obs Space: {obs}")

        # Reset environment
        # self.eval_env.reset()

        # DDPG Agent Action
        ddpg_action = self.ddpg_eval_agent.choose_action(obs)
        ddpg_action_dict, ddpg_action_ticks = self.postprocess_action(ddpg_action, curr_price, action_transform='linear')

        # PPO Agent Action
        ppo_action, _ = self.ppo_eval_agent.choose_action(obs)
        print(f"\n\nPPO ACTION:   {ppo_action} \n\n")
        ppo_action_dict, ppo_action_ticks = self.postprocess_action(ppo_action, curr_price, action_transform='exp')

        return ddpg_action, ddpg_action_dict, ddpg_action_ticks, ppo_action, ppo_action_dict, ppo_action_ticks

    def postprocess_action(self, action, curr_price, action_transform='linear'):
        action_numpy = action.numpy()  # Convert the action tensor to NumPy array
        a_0, a_1 = action_numpy[0, 0], action_numpy[0, 1]

        action_lower_bound = curr_price * 0.1
        action_upper_bound = curr_price * 2

        if action_transform == 'linear':
            a_0 = np.clip(a_0, 0, 1)
            a_1 = np.clip(a_1, 0, 1)
            price_lower = action_lower_bound + a_0 * (action_upper_bound - action_lower_bound) / 2
            price_upper = (action_upper_bound - action_lower_bound) / 2 + a_1 * (action_upper_bound - action_lower_bound) / 2

        elif action_transform == 'exp':
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
        ticks_dict = {
            'tick_lower': price_to_valid_tick(action_dict['price_lower']),
            'tick_upper': price_to_valid_tick(action_dict['price_upper'])
        }
        return action_dict, ticks_dict

    def perform_inference(self, user_preferences, pool_state, pool_id="0x99ac8ca7087fa4a2a1fb6357269965a2014abc35", date_str='2024-05-05'):
        # Extract necessary information from pool state
        current_profit = pool_state['current_profit']
        price_out_of_range = pool_state['price_out_of_range']
        time_since_last_adjustment = pool_state['time_since_last_adjustment']
        pool_volatility = pool_state['pool_volatility']

        # User Preferences
        risk_tolerance = user_preferences['risk_tolerance']
        investment_horizon = user_preferences['investment_horizon']
        liquidity_preference = user_preferences['liquidity_preference']
        volatility_threshold = user_preferences['risk_aversion_threshold']

        # Adjust thresholds based on user preferences
        profit_taking_threshold = risk_tolerance['profit_taking']
        stop_loss_threshold = risk_tolerance['stop_loss']
        rebalance_interval = investment_horizon * 24 * 60 * 60  # Convert days to seconds

        # Predicted actions from RL agents
        ddpg_action, ddpg_action_dict, ddpg_action_ticks, ppo_action, ppo_action_dict, ppo_action_ticks = self.predict_action(
            pool_id=pool_id, date_str=date_str)

        # Decision Logic
        if user_preferences['user_status'] == 'new_user':
            return 'Add new liquidity position', ddpg_action_dict, ppo_action_dict
        if current_profit >= profit_taking_threshold:
            return 'adjust_position', ddpg_action_dict, ppo_action_dict
        elif price_out_of_range and liquidity_preference['adjust_on_price_out_of_range']:
            return 'adjust_position', ddpg_action_dict, ppo_action_dict
        elif pool_volatility > volatility_threshold:
            return 'exit_position', None, None  # Exit position in case of high volatility
        elif time_since_last_adjustment >= rebalance_interval:
            return 'rebalance_position', None, None
        elif current_profit <= stop_loss_threshold:
            return 'exit_position', None, None  # Exit the position to stop further losses
        else:
            return 'maintain_position', None, None  # Maintain the current position
