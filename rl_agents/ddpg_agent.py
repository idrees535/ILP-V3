

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

import sys
import os
import pathlib




class ReplayBuffer:
    def __init__(self, max_size, input_dims, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_dims))
        self.new_state_memory = np.zeros((self.mem_size, input_dims))  
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
    
    def clear(self):
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *self.state_memory.shape[1:]))
        self.new_state_memory = np.zeros((self.mem_size, *self.new_state_memory.shape[1:]))
        self.action_memory = np.zeros((self.mem_size, *self.action_memory.shape[1:]))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        
class DDPG_Actor(tf.keras.Model):
    def __init__(self, n_actions):
        super(DDPG_Actor, self).__init__()

        self.bn_input = tf.keras.layers.BatchNormalization()
        self.fc1 = tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.lstm1 = tf.keras.layers.LSTM(64, return_sequences=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.lstm2 = tf.keras.layers.LSTM(32, return_sequences=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.fc3 = tf.keras.layers.Dense(25, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.output_layer = tf.keras.layers.Dense(n_actions, activation='sigmoid')  # Two output units for 'price_lower' and 'price_upper'

    def call(self, state):
        state = self.bn_input(state)
        x = self.fc1(state)
        x = tf.expand_dims(x, axis=1)
        x = self.lstm1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = tf.expand_dims(x, axis=1)
        x = self.lstm2(x)
        x = self.bn3(x)
        x = self.fc3(x)
        x = self.bn2(x)
        actions = self.output_layer(x)
        
        return actions
        
class DDPG_Critic(tf.keras.Model):
    def __init__(self, n_actions):
        super(DDPG_Critic, self).__init__()
        
        self.bn_state = tf.keras.layers.BatchNormalization()
        self.bn_action = tf.keras.layers.BatchNormalization()
        self.fc1 = tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.lstm1 = tf.keras.layers.LSTM(64, return_sequences=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.lstm2 = tf.keras.layers.LSTM(32, return_sequences=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.fc3 = tf.keras.layers.Dense(25, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.q = tf.keras.layers.Dense(1, activation=None)

    def call(self, state, action):
        state = self.bn_state(state)
        action = self.bn_action(action)
        x = tf.concat([state, action], axis=1) 
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
    
class DDPG:
    def __init__(self, alpha=0.001, beta=0.002, input_dims=[8], tau=0.005, env=None,gamma=0.99, n_actions=2, max_size=1000000, batch_size=64,training=True,max_grad_norm=10):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = DDPG_Actor(n_actions=n_actions)
        self.critic = DDPG_Critic(n_actions=n_actions)
        self.target_actor = DDPG_Actor(n_actions=n_actions)
        self.target_critic = DDPG_Critic(n_actions=n_actions)

        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

        self.env = env
        self.training=training
        self.max_grad_norm=max_grad_norm

        # For tensorboard logging
        self.log_dir = os.path.join(base_path,'model_storage/tensorboard_ddpg_logs')
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        flat_state = self.flatten_state(state)
        flat_action = self.flatten_action(action)
        flat_new_state = self.flatten_state(new_state)
        self.memory.store_transition(flat_state, flat_action, reward, flat_new_state, done)
        
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
    
    def choose_action(self, state):
        state_dict = state
        state_dict_float = {
            key: float(value) for key, value in state_dict.items()
        }

        state_array = np.array(list(state_dict_float.values()), dtype=np.float32)
        state_array = state_array.reshape(1, -1)
        state_tensor = tf.convert_to_tensor(state_array, dtype=tf.float32)
        raw_actions_tensor = self.actor(state_tensor,training=False)
        
        return raw_actions_tensor
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        #print(f"{state},{action},{reward},{new_state}")
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_,training=False)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions,training=False), 1)
            critic_value = tf.squeeze(self.critic(states, actions,training=True), 1)
            target = rewards + self.gamma*critic_value_*(1-done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_network_gradient, _ = tf.clip_by_global_norm(critic_network_gradient, self.max_grad_norm)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states,training=True)
            actor_loss = -self.critic(states, new_policy_actions,training=True)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_network_gradient, _ = tf.clip_by_global_norm(actor_network_gradient, self.max_grad_norm)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
        
        print(f"Actor_Loss: {actor_loss.numpy()}, Critic_Loss: {critic_loss.numpy()}")
       
        with self.train_summary_writer.as_default():
            tf.summary.scalar('critic_loss', critic_loss.numpy(), step=self.memory.mem_cntr)
            tf.summary.scalar('actor_loss', actor_loss.numpy(), step=self.memory.mem_cntr)

        self.update_network_parameters()

class DDGPEval(DDPG):
    def choose_action(self, state):
        # Disable exploration noise
        action = super().choose_action(state)
        return action
