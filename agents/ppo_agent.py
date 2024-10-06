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

class RolloutBuffer:
    def __init__(self, buffer_size, observation_dims, n_actions):
        self.states = np.zeros((buffer_size, observation_dims), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_actions), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=bool)
        self.log_probs = np.zeros((buffer_size, n_actions), dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.next_values = np.zeros(buffer_size, dtype=np.float32)
        self.ptr, self.max_size = 0, buffer_size
        self.buffer_size=buffer_size
        self.observation_dims=observation_dims
        self.n_actions=n_actions

        self.reset()

    def store_transition(self, state, action, reward, done, log_prob, value, next_value):
        index = self.ptr % self.max_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.log_probs[index] = log_prob
        self.values[index] = value
        self.next_values[index] = next_value
        self.ptr += 1

    def sample(self):
        self.ptr = 0
        return (self.states, self.actions, self.rewards, self.dones, 
                self.log_probs, self.values, self.next_values)
    def reset(self):
        self.states = np.zeros((self.buffer_size, self.observation_dims), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_actions), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=bool)
        self.log_probs = np.zeros((self.buffer_size, self.n_actions), dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.next_values = np.zeros(self.buffer_size, dtype=np.float32)
        self.ptr = 0

    def is_full(self):
        return self.ptr >= self.buffer_size
    
class PPO_Actor(tf.keras.Model):
    def __init__(self, n_actions):
        super(PPO_Actor, self).__init__()

        self.bn_input = tf.keras.layers.BatchNormalization()
        self.fc1 = tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.lstm1 = tf.keras.layers.LSTM(64, return_sequences=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.lstm2 = tf.keras.layers.LSTM(32, return_sequences=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.fc3 = tf.keras.layers.Dense(25, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn3 = tf.keras.layers.BatchNormalization()
        # Output the mean of the actions
        self.mu = tf.keras.layers.Dense(n_actions, activation='sigmoid')
        # Output the standard deviation of the actions (log std for numerical stability)
        self.sigma = tf.keras.layers.Dense(n_actions, activation='sigmoid')

    def call(self, state):
        state = self.bn_input(state)
        x = self.fc1(state)
        x = tf.expand_dims(x, axis=1)
        x = self.lstm1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = tf.expand_dims(x, axis=1)
        x = self.lstm2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        
        return mu, sigma
    
class PPO_Critic(tf.keras.Model):
    def __init__(self):
        super(PPO_Critic, self).__init__()

        self.bn_state = tf.keras.layers.BatchNormalization()
        self.fc1 = tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.lstm1 = tf.keras.layers.LSTM(64, return_sequences=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.lstm2 = tf.keras.layers.LSTM(32, return_sequences=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.fc3 = tf.keras.layers.Dense(25, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.q = tf.keras.layers.Dense(1, activation=None)

    def call(self, state):
        x = self.bn_state(state) 
        x = self.fc1(x)
        x = tf.expand_dims(x, axis=1)
        x = self.lstm1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = tf.expand_dims(x, axis=1)
        x = self.lstm2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        q = self.q(x)

        return q

class PPO:
    def __init__(self, env, n_actions,observation_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, buffer_size=64, max_grad_norm=0.5, n_epochs=1,training=True):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.buffer_size=buffer_size
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=alpha)
        self.observation_dims=observation_dims
        self.max_grad_norm = max_grad_norm
        self.n_epochs=n_epochs
        
        self.actor = PPO_Actor(n_actions)
        self.critic = PPO_Critic()
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))

        self.rollout_buffer = RolloutBuffer(self.buffer_size, observation_dims, n_actions)

        self.env=env
        self.training=training
        

         # For tensorboard logging
        self.log_dir = os.path.join(base_path,'model_storage/tensorboard_ppo_logs')
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.tensorboard_counter=0
 
    def choose_action(self, state):

        state_dict = state
        state_dict_float = {
            key: float(value) for key, value in state_dict.items()
        }

        state_array = np.array(list(state_dict_float.values()), dtype=np.float32)
        state_array = state_array.reshape(1, -1)
        state_tensor = tf.convert_to_tensor(state_array, dtype=tf.float32)

        mu, sigma = self.actor(state_tensor,training=self.training)
        
        action_prob = tfp.distributions.Normal(mu, sigma)
        action = action_prob.sample()
        
        #Action clipping
        #action = tf.clip_by_value(action, 0, 1)
        log_prob = action_prob.log_prob(action)
        print(f"mu: {mu}, sigma: {sigma}, action_prob: {action_prob}, action: {action}, log_prob: {log_prob}")
        return action,log_prob

    def remember(self, state, action, reward, next_state, done, log_prob):
        
        flat_state = self.flatten_state(state)
        flat_action = self.flatten_action(action)
        flat_next_state = self.flatten_state(next_state)
        value = self.critic(tf.convert_to_tensor([flat_state], dtype=tf.float32))
        next_value = self.critic(tf.convert_to_tensor([flat_next_state], dtype=tf.float32))
        self.rollout_buffer.store_transition(flat_state, flat_action, reward, done, log_prob, value, next_value)
    
    def learn(self):
        states, actions, rewards, dones, old_log_probs, values, next_values = self.rollout_buffer.sample()
        returns, advantages = self.compute_gae(rewards, values, next_values, dones, self.gamma, self.gae_lambda)
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        # Update policy and value networks for a number of epochs
        for _ in range(self.n_epochs):
            with tf.GradientTape() as tape:
                total_loss = self.ppo_loss(states, actions, old_log_probs, advantages, returns, self.policy_clip)
            gradients = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
            # Gradient clipping
            gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.critic.trainable_variables))
                
        self.rollout_buffer.reset()
       
    def compute_gae(self, rewards, values, next_values, dones, gamma=0.99, lam=0.95):
        gae = 0
        returns = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            returns[t] = gae + values[t]
        advantages = returns - values
        return returns, (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    def ppo_loss(self, states, actions, old_log_probs, advantages, returns, clip_param=0.2):
        mu, sigma = self.actor(states)
        values = tf.squeeze(self.critic(states))

        # Calculate new log probabilities using the updated policy
        new_policy = tfp.distributions.Normal(mu, sigma)
        new_log_probs = new_policy.log_prob(actions)

        # Policy loss
        ratios = tf.exp(new_log_probs - old_log_probs)
        #ratios = tf.reduce_mean(ratios, axis=1)
        advantages = tf.expand_dims(advantages, -1)
        surr1 = ratios * advantages
        surr2 = tf.clip_by_value(ratios, 1 - clip_param, 1 + clip_param) * advantages
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        # Value loss
        value_loss = tf.reduce_mean(tf.square(returns - values))
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        with self.train_summary_writer.as_default():
            tf.summary.scalar('total_loss', total_loss.numpy(), step=self.tensorboard_counter)
            tf.summary.scalar('policy_loss', total_loss.numpy(), step=self.tensorboard_counter)
            tf.summary.scalar('value_loss', total_loss.numpy(), step=self.tensorboard_counter)

        print(f"total_loss:{total_loss}, policy_loss:{policy_loss}, value_loss:{value_loss}, advantages:{advantages}, returns:{returns}")
        self.tensorboard_counter+=1
        return total_loss
          
    def flatten_state(self,state_dict):
        scaled_curr_price = float(state_dict['scaled_curr_price'])
        scaled_liquidity = float(state_dict['scaled_liquidity'])
        scaled_fee_growth_0 = float(state_dict['scaled_feeGrowthGlobal0x128'])
        scaled_fee_growth_1 = float(state_dict['scaled_feeGrowthGlobal1x128'])
        
        return np.array([scaled_curr_price, scaled_liquidity, scaled_fee_growth_0, scaled_fee_growth_1])

    def unflatten_state(self,state_array):
        return {
            'scaled_curr_price': state_array[0],
            'scaled_liquidity': state_array[1],
            'scaled_feeGrowthGlobal0x128': state_array[2],
            'scaled_feeGrowthGlobal1x128': state_array[3]
        }

    def flatten_action(self,action):
        return tf.reshape(action, [-1])

    def unflatten_action(self,action):
        return tf.reshape(action, [1, -1])

    def map_indices_to_action_values(self, action_indices):
        action_dict = {
            'price_relative_lower': action_indices[0],
            'price_relative_upper': action_indices[1]
        }
        return action_dict

class PPOEval(PPO):
    def choose_action(self, state):
        # Disable exploration noise
        action = super().choose_action(state)
        return action