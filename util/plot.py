import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
#print(sys.path)
#sys.path.append('/mnt/c/Users/hijaz tr/Desktop/cadCADProject1/tokenspice')
'''

def train_rewards_plot(data):
    # Find out how many unique episodes are in the data
    unique_episodes = data['episode'].unique()
    num_episodes = len(unique_episodes)

    # Set up the plotting area
    fig, axes = plt.subplots(nrows=5, ncols=num_episodes, figsize=(5 * num_episodes, 25), sharey='row')

    # Define the plots for each episode
    for i, episode in enumerate(unique_episodes):
        episode_data = data[data['episode'] == episode]

        # Raw Rewards
        sns.lineplot(x='step_count', y='raw_reward', data=episode_data, ax=axes[0, i], label='RL Agent')
        axes[0, i].set_title(f'Episode {episode} - Raw Rewards')
        axes[0, i].legend()

        # Scaled Rewards
        sns.lineplot(x='step_count', y='scaled_reward', data=episode_data, ax=axes[1, i], label='RL Agent')
        axes[1, i].set_title(f'Episode {episode} - Scaled Rewards')
        axes[1, i].legend()

        # Cumulative Rewards
        sns.lineplot(x='step_count', y='cumulative_reward', data=episode_data, ax=axes[2, i], label='RL Agent')
        axes[2, i].set_title(f'Episode {episode} - Cumulative Rewards')
        axes[2, i].legend()

        # Fee Earned
        sns.lineplot(x='step_count', y='fee_earned', data=episode_data, ax=axes[3, i], label='RL Agent')
        axes[3, i].set_title(f'Episode {episode} - Fee Earned')
        axes[3, i].legend()

        # Impermanent Loss
        sns.lineplot(x='step_count', y='impermanent_loss', data=episode_data, ax=axes[4, i], label='RL Agent')
        axes[4, i].set_title(f'Episode {episode} - Impermanent Loss')
        axes[4, i].legend()

    plt.tight_layout()
    plt.show()

def train_raw_actions_plot(data):
    # Find out how many unique episodes are in the data
    unique_episodes = data['episode'].unique()
    num_episodes = len(unique_episodes)

    # Set up the plotting area
    fig, axes = plt.subplots(nrows=1, ncols=num_episodes, figsize=(5 * num_episodes, 5), sharey=True)

    # Define the plots for each episode
    for i, episode in enumerate(unique_episodes):
        episode_data = data[data['episode'] == episode]

        # Raw Actions
        axes[i].fill_between(episode_data['step_count'], episode_data['raw_action_0'], episode_data['raw_action_1'], alpha=0.5, label='Actions Difference')
        
        # Scaled Current Price
        sns.lineplot(x='step_count', y='scaled_curr_price', data=episode_data, ax=axes[i], color='black', label='Scaled Current Price')

        # Setting the title and legend
        axes[i].set_title(f'Episode {episode} - Raw Actions and Scaled Current Price')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


def train_scaled_actions_plot(data):
    # Find out how many unique episodes are in the data
    unique_episodes = data['episode'].unique()
    num_episodes = len(unique_episodes)


    # Set up the plotting area
    fig, axes = plt.subplots(nrows=1, ncols=num_episodes, figsize=(5 * num_episodes, 5), sharey=True)

    # Define the plots for each episode
    for i, episode in enumerate(unique_episodes):
        episode_data = data[data['episode'] == episode]

        # Scaled Actions
        axes[i].fill_between(episode_data['step_count'], episode_data['scaled_action_0'], episode_data['scaled_action_1'], alpha=0.5, label='Actions Difference')
        
        # Current Price
        sns.lineplot(x='step_count', y='curr_price', data=episode_data, ax=axes[i], color='black', label='Current Price')

        # Setting the title and legend
        axes[i].set_title(f'Episode {episode} - Scaled Actions and Current Price')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


def eval_rewards_plot(data):
    # Find out how many unique episodes are in the data
    data['cumulative_raw_reward_rl_agent'] = data.groupby('episode')['raw_reward_rl_agent'].cumsum()
    data['cumulative_raw_reward_baseline_agent'] = data.groupby('episode')['raw_reward_baseline_agent'].cumsum()
    data['cumulative_fee_rl_agent'] = data.groupby('episode')['fee_income_rl_agent'].cumsum()
    data['cumulative_fee_baseline_agent'] = data.groupby('episode')['fee_income_baseline_agent'].cumsum()
    data['cumulative_impermanent_loss_rl_agent'] = data.groupby('episode')['impermanent_loss_rl_agent'].cumsum()
    data['cumulative_impermanent_loss_baseline_agent'] = data.groupby('episode')['impermanent_loss_baseline_agent'].cumsum()
    unique_episodes = data['episode'].unique()
    num_episodes = len(unique_episodes)

    # Set up the plotting area
    fig, axes = plt.subplots(nrows=8, ncols=num_episodes, figsize=(5 * num_episodes, 25))#, sharey='row')

    # Define the plots for each episode
    for i, episode in enumerate(unique_episodes):
        episode_data = data[data['episode'] == episode]

        # Raw Rewards
        sns.lineplot(x='step_count', y='raw_reward_rl_agent', data=episode_data, ax=axes[0, i], label='RL Agent')
        sns.lineplot(x='step_count', y='raw_reward_baseline_agent', data=episode_data, ax=axes[0, i], label='Baseline Agent', linestyle='--')
        axes[0, i].set_title(f'Episode {episode} - Raw Rewards')
        axes[0, i].legend()

        # Cumulative Raw Rewards
        sns.lineplot(x='step_count', y='cumulative_raw_reward_rl_agent', data=episode_data, ax=axes[1, i], label='RL Agent')
        sns.lineplot(x='step_count', y='cumulative_raw_reward_baseline_agent', data=episode_data, ax=axes[1, i], label='Baseline Agent', linestyle='--')
        axes[1, i].set_title(f'Episode {episode} - Cumulative Raw Rewards')
        axes[1, i].legend()

        # Scaled Rewards
        sns.lineplot(x='step_count', y='scaled_reward_rl_agent', data=episode_data, ax=axes[2, i], label='RL Agent')
        sns.lineplot(x='step_count', y='scaled_reward_baseline_agent', data=episode_data, ax=axes[2, i], label='Baseline Agent', linestyle='--')
        axes[2, i].set_title(f'Episode {episode} - Scaled Rewards')
        axes[2, i].legend()

        # Cumulative scaled Rewards
        sns.lineplot(x='step_count', y='cumulative_reward_rl_agent', data=episode_data, ax=axes[3, i], label='RL Agent')
        sns.lineplot(x='step_count', y='cumulative_reward_baseline_agent', data=episode_data, ax=axes[3, i], label='Baseline Agent', linestyle='--')
        axes[3, i].set_title(f'Episode {episode} - Cumulative Scaled Rewards')
        axes[3, i].legend()

        # Fee Income
        sns.lineplot(x='step_count', y='fee_income_rl_agent', data=episode_data, ax=axes[4, i], label='RL Agent')
        sns.lineplot(x='step_count', y='fee_income_baseline_agent', data=episode_data, ax=axes[4, i], label='Baseline Agent', linestyle='--')
        axes[4, i].set_title(f'Episode {episode} - Fee Income')
        axes[4, i].legend()

        # Cumulative Fee Income
        sns.lineplot(x='step_count', y='cumulative_fee_rl_agent', data=episode_data, ax=axes[5, i], label='RL Agent')
        sns.lineplot(x='step_count', y='cumulative_fee_baseline_agent', data=episode_data, ax=axes[5, i], label='Baseline Agent', linestyle='--')
        axes[5, i].set_title(f'Episode {episode} - Cumulative Fee Income')
        axes[5, i].legend()
        
        # Impermanent Loss
        sns.lineplot(x='step_count', y='impermanent_loss_rl_agent', data=episode_data, ax=axes[6, i], label='RL Agent')
        sns.lineplot(x='step_count', y='impermanent_loss_baseline_agent', data=episode_data, ax=axes[6, i], label='Baseline Agent', linestyle='--')
        axes[6, i].set_title(f'Episode {episode} - Impermanent Loss')
        axes[6, i].legend()

        # Cumulative Impermanent Loss
        sns.lineplot(x='step_count', y='cumulative_impermanent_loss_rl_agent', data=episode_data, ax=axes[7, i], label='RL Agent')
        sns.lineplot(x='step_count', y='cumulative_impermanent_loss_baseline_agent', data=episode_data, ax=axes[7, i], label='Baseline Agent', linestyle='--')
        axes[7, i].set_title(f'Episode {episode} - Cumulative Impermanent Loss')
        axes[7, i].legend()

    plt.tight_layout()
    plt.show()

def train_combined_metrics_plot(data_df):
    
    episodes = data_df['episode'].unique()
    episode_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'cyan', 'yellow', 'grey']

    data_df['action_midpoint'] = (data_df['scaled_action_0'] + data_df['scaled_action_1']) / 2
    data_df['action_range'] = data_df['scaled_action_1'] - data_df['scaled_action_0']

    data_df['extended_step_count'] = data_df.groupby('episode').cumcount()

    for i in range(1, len(episodes)):
        data_df.loc[data_df['episode'] == episodes[i], 'extended_step_count'] += data_df[data_df['episode'] == episodes[i-1]]['extended_step_count'].max() + 1

    # Plotting with the extended step count
    fig, ax1 = plt.subplots(figsize=(15, 7))
    for episode, color in zip(episodes, episode_colors):
        episode_data = data_df[data_df['episode'] == episode]
        ax1.fill_between(episode_data['extended_step_count'], episode_data['action_midpoint'] - episode_data['action_range']/2, 
                        episode_data['action_midpoint'] + episode_data['action_range']/2, color=color, alpha=0.2)
        ax1.plot(episode_data['extended_step_count'], episode_data['curr_price'], label=f'Episode {episode} Curr Price', color=color, linestyle='--')

    ax1.set_xlabel('Extended Step Count')
    ax1.set_ylabel('Price')
    ax1.set_title('Agent Actions Over Extended Time')
    ax1.legend(loc='upper left')

    # Create a secondary y-axis for rewards
    ax2 = ax1.twinx()
    for episode, color in zip(episodes, episode_colors):
        episode_data = data_df[data_df['episode'] == episode]
        ax2.plot(episode_data['extended_step_count'], episode_data['scaled_reward'], linestyle=':', label=f'Episode {episode} Scaled Reward', color=color)
        ax2.plot(episode_data['extended_step_count'], episode_data['raw_reward'], linestyle='-.', label=f'Episode {episode} Raw Reward', color=color)
    ax2.set_ylabel('Reward')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    color_map = plt.get_cmap('tab10')
    
    # Distribution of Raw and Scaled Rewards
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        axes[0].hist(episode_data['raw_reward'], bins=50, color=color_map(i), alpha=0.7, label=f'Episode {episode}')
        axes[1].hist(episode_data['scaled_reward'], bins=50, color=color_map(i), alpha=0.7, label=f'Episode {episode}')
    axes[0].set_title('Distribution of Raw Rewards')
    axes[1].set_title('Distribution of Scaled Rewards')
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    # Fee Income and Impermanent Loss Over Time
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        ax.plot(episode_data['fee_earned'], label=f'Episode {episode} Fee Income', color=color_map(i))
        ax.plot(episode_data['impermanent_loss'], linestyle='--', label=f'Episode {episode} Impermanent Loss', color=color_map(i))
    ax.set_title('Fee Income and Impermanent Loss Over Time')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Value')
    ax.legend()
    plt.tight_layout()
    plt.show()

def train_separate_episode_action_plot(data_df):
    # Separate data for each episode
    episodes = data_df['episode'].unique()

    fig, axes = plt.subplots(nrows=len(episodes), ncols=1, figsize=(15, 5*len(episodes)))

    for i, episode in enumerate(episodes):
        episode_df = data_df[data_df['episode'] == episode]

        # Agent Actions and Rewards over Time
        ax1 = axes[i]
        ax1.plot(episode_df['step_count'], episode_df['curr_price'], label='Current Price', color='blue')
        ax1.fill_between(episode_df['step_count'], episode_df['scaled_action_0'], episode_df['scaled_action_1'], color='gray', alpha=0.3, label='Action Range (price_lower to price_upper)')
        ax1.set_xlabel('Step Count')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title(f'Agent Actions and Rewards over Time for Episode {episode}')

        # Secondary axis for rewards
        ax2 = ax1.twinx()
        ax2.plot(episode_df['step_count'], episode_df['raw_reward'], label='Reward', color='red', linestyle='--')
        ax2.set_ylabel('Reward', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    plt.show()

    # Distribution of Raw and Scaled Rewards
    fig, axes = plt.subplots(nrows=len(episodes), ncols=2, figsize=(14, 5*len(episodes)))
    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        axes[i, 0].hist(episode_data['raw_reward'], bins=50, color='blue', alpha=0.7)
        axes[i, 0].set_title(f'Episode {episode} - Distribution of Raw Rewards')
        axes[i, 1].hist(episode_data['scaled_reward'], bins=50, color='green', alpha=0.7)
        axes[i, 1].set_title(f'Episode {episode} - Distribution of Scaled Rewards')
    plt.tight_layout()
    plt.show()

    # Fee Income and Impermanent Loss Over Time
    fig, axes = plt.subplots(nrows=len(episodes), ncols=1, figsize=(15, 6*len(episodes)))
    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        axes[i].plot(episode_data['fee_earned'], label='Fee Income', color='orange')
        axes[i].plot(episode_data['impermanent_loss'], label='Impermanent Loss', color='black', linestyle='--')
        axes[i].set_title(f'Episode {episode} - Fee Income and Imperpermanent Loss Over Time')
        axes[i].set_xlabel('Steps')
        axes[i].set_ylabel('Value')
        axes[i].legend()
    plt.tight_layout()
    plt.show()
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def train_rewards_plot(data, output_dir):
    unique_episodes = data['episode'].unique()
    num_episodes = len(unique_episodes)

    fig, axes = plt.subplots(nrows=5, ncols=num_episodes, figsize=(5 * num_episodes, 25), sharey='row')
    if num_episodes == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, episode in enumerate(unique_episodes):
        episode_data = data[data['episode'] == episode]

        sns.lineplot(x='step_count', y='raw_reward', data=episode_data, ax=axes[0, i], label='RL Agent')
        axes[0, i].set_title(f'Episode {episode} - Raw Rewards')
        axes[0, i].legend()

        sns.lineplot(x='step_count', y='scaled_reward', data=episode_data, ax=axes[1, i], label='RL Agent')
        axes[1, i].set_title(f'Episode {episode} - Scaled Rewards')
        axes[1, i].legend()

        sns.lineplot(x='step_count', y='cumulative_reward', data=episode_data, ax=axes[2, i], label='RL Agent')
        axes[2, i].set_title(f'Episode {episode} - Cumulative Rewards')
        axes[2, i].legend()

        sns.lineplot(x='step_count', y='fee_earned', data=episode_data, ax=axes[3, i], label='RL Agent')
        axes[3, i].set_title(f'Episode {episode} - Fee Earned')
        axes[3, i].legend()

        sns.lineplot(x='step_count', y='impermanent_loss', data=episode_data, ax=axes[4, i], label='RL Agent')
        axes[4, i].set_title(f'Episode {episode} - Impermanent Loss')
        axes[4, i].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'rewards_plot.png')
    plt.savefig(plot_path)
    plt.close()

def train_raw_actions_plot(data, output_dir):
    unique_episodes = data['episode'].unique()
    num_episodes = len(unique_episodes)

    fig, axes = plt.subplots(nrows=1, ncols=num_episodes, figsize=(5 * num_episodes, 5), sharey=True)
    if num_episodes == 1:
        axes = [axes]
    for i, episode in enumerate(unique_episodes):
        episode_data = data[data['episode'] == episode]

        axes[i].fill_between(episode_data['step_count'], episode_data['raw_action_0'], episode_data['raw_action_1'], alpha=0.5, label='Actions Difference')
        sns.lineplot(x='step_count', y='scaled_curr_price', data=episode_data, ax=axes[i], color='black', label='Scaled Current Price')

        axes[i].set_title(f'Episode {episode} - Raw Actions and Scaled Current Price')
        axes[i].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'raw_actions_plot.png')
    plt.savefig(plot_path)
    plt.close()

def train_scaled_actions_plot(data, output_dir):
    unique_episodes = data['episode'].unique()
    num_episodes = len(unique_episodes)

    fig, axes = plt.subplots(nrows=1, ncols=num_episodes, figsize=(5 * num_episodes, 5), sharey=True)
    if num_episodes == 1:
        axes = [axes]
    for i, episode in enumerate(unique_episodes):
        episode_data = data[data['episode'] == episode]

        axes[i].fill_between(episode_data['step_count'], episode_data['scaled_action_0'], episode_data['scaled_action_1'], alpha=0.5, label='Actions Difference')
        sns.lineplot(x='step_count', y='curr_price', data=episode_data, ax=axes[i], color='black', label='Current Price')

        axes[i].set_title(f'Episode {episode} - Scaled Actions and Current Price')
        axes[i].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'scaled_actions_plot.png')
    plt.savefig(plot_path)
    plt.close()

def eval_rewards_plot(data, output_dir):
    data['cumulative_raw_reward_rl_agent'] = data.groupby('episode')['raw_reward_rl_agent'].cumsum()
    data['cumulative_raw_reward_baseline_agent'] = data.groupby('episode')['raw_reward_baseline_agent'].cumsum()
    data['cumulative_fee_rl_agent'] = data.groupby('episode')['fee_income_rl_agent'].cumsum()
    data['cumulative_fee_baseline_agent'] = data.groupby('episode')['fee_income_baseline_agent'].cumsum()
    data['cumulative_impermanent_loss_rl_agent'] = data.groupby('episode')['impermanent_loss_rl_agent'].cumsum()
    data['cumulative_impermanent_loss_baseline_agent'] = data.groupby('episode')['impermanent_loss_baseline_agent'].cumsum()
    unique_episodes = data['episode'].unique()
    num_episodes = len(unique_episodes)

    fig, axes = plt.subplots(nrows=8, ncols=num_episodes, figsize=(5 * num_episodes, 25))

    if num_episodes == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, episode in enumerate(unique_episodes):
        episode_data = data[data['episode'] == episode]

        sns.lineplot(x='step_count', y='raw_reward_rl_agent', data=episode_data, ax=axes[0, i], label='RL Agent')
        sns.lineplot(x='step_count', y='raw_reward_baseline_agent', data=episode_data, ax=axes[0, i], label='Baseline Agent', linestyle='--')
        axes[0, i].set_title(f'Episode {episode} - Raw Rewards')
        axes[0, i].legend()

        sns.lineplot(x='step_count', y='cumulative_raw_reward_rl_agent', data=episode_data, ax=axes[1, i], label='RL Agent')
        sns.lineplot(x='step_count', y='cumulative_raw_reward_baseline_agent', data=episode_data, ax=axes[1, i], label='Baseline Agent', linestyle='--')
        axes[1, i].set_title(f'Episode {episode} - Cumulative Raw Rewards')
        axes[1, i].legend()

        sns.lineplot(x='step_count', y='scaled_reward_rl_agent', data=episode_data, ax=axes[2, i], label='RL Agent')
        sns.lineplot(x='step_count', y='scaled_reward_baseline_agent', data=episode_data, ax=axes[2, i], label='Baseline Agent', linestyle='--')
        axes[2, i].set_title(f'Episode {episode} - Scaled Rewards')
        axes[2, i].legend()

        sns.lineplot(x='step_count', y='cumulative_reward_rl_agent', data=episode_data, ax=axes[3, i], label='RL Agent')
        sns.lineplot(x='step_count', y='cumulative_reward_baseline_agent', data=episode_data, ax=axes[3, i], label='Baseline Agent', linestyle='--')
        axes[3, i].set_title(f'Episode {episode} - Cumulative Scaled Rewards')
        axes[3, i].legend()

        sns.lineplot(x='step_count', y='fee_income_rl_agent', data=episode_data, ax=axes[4, i], label='RL Agent')
        sns.lineplot(x='step_count', y='fee_income_baseline_agent', data=episode_data, ax=axes[4, i], label='Baseline Agent', linestyle='--')
        axes[4, i].set_title(f'Episode {episode} - Fee Income')
        axes[4, i].legend()

        sns.lineplot(x='step_count', y='cumulative_fee_rl_agent', data=episode_data, ax=axes[5, i], label='RL Agent')
        sns.lineplot(x='step_count', y='cumulative_fee_baseline_agent', data=episode_data, ax=axes[5, i], label='Baseline Agent', linestyle='--')
        axes[5, i].set_title(f'Episode {episode} - Cumulative Fee Income')
        axes[5, i].legend()

        sns.lineplot(x='step_count', y='impermanent_loss_rl_agent', data=episode_data, ax=axes[6, i], label='RL Agent')
        sns.lineplot(x='step_count', y='impermanent_loss_baseline_agent', data=episode_data, ax=axes[6, i], label='Baseline Agent', linestyle='--')
        axes[6, i].set_title(f'Episode {episode} - Impermanent Loss')
        axes[6, i].legend()

        sns.lineplot(x='step_count', y='cumulative_impermanent_loss_rl_agent', data=episode_data, ax=axes[7, i], label='RL Agent')
        sns.lineplot(x='step_count', y='cumulative_impermanent_loss_baseline_agent', data=episode_data, ax=axes[7, i], label='Baseline Agent', linestyle='--')
        axes[7, i].set_title(f'Episode {episode} - Cumulative Impermanent Loss')
        axes[7, i].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'eval_rewards_plot.png')
    plt.savefig(plot_path)
    plt.close()

def train_combined_metrics_plot(data_df, output_dir):
    episodes = data_df['episode'].unique()
    episode_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'cyan', 'yellow', 'grey']

    data_df['action_midpoint'] = (data_df['scaled_action_0'] + data_df['scaled_action_1']) / 2
    data_df['action_range'] = data_df['scaled_action_1'] - data_df['scaled_action_0']
    data_df['extended_step_count'] = data_df.groupby('episode').cumcount()

    for i in range(1, len(episodes)):
        data_df.loc[data_df['episode'] == episodes[i], 'extended_step_count'] += data_df[data_df['episode'] == episodes[i-1]]['extended_step_count'].max() + 1

    fig, ax1 = plt.subplots(figsize=(15, 7))
    for episode, color in zip(episodes, episode_colors):
        episode_data = data_df[data_df['episode'] == episode]
        ax1.fill_between(episode_data['extended_step_count'], episode_data['action_midpoint'] - episode_data['action_range'] / 2, 
                         episode_data['action_midpoint'] + episode_data['action_range'] / 2, color=color, alpha=0.2)
        ax1.plot(episode_data['extended_step_count'], episode_data['curr_price'], label=f'Episode {episode} Curr Price', color=color, linestyle='--')

    ax1.set_xlabel('Extended Step Count')
    ax1.set_ylabel('Price')
    ax1.set_title('Agent Actions Over Extended Time')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    for episode, color in zip(episodes, episode_colors):
        episode_data = data_df[data_df['episode'] == episode]
        ax2.plot(episode_data['extended_step_count'], episode_data['scaled_reward'], linestyle=':', label=f'Episode {episode} Scaled Reward', color=color)
        ax2.plot(episode_data['extended_step_count'], episode_data['raw_reward'], linestyle='-.', label=f'Episode {episode} Raw Reward', color=color)
    ax2.set_ylabel('Reward')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'combined_metrics_plot.png')
    plt.savefig(plot_path)
    plt.close()

    color_map = plt.get_cmap('tab10')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        axes[0].hist(episode_data['raw_reward'], bins=50, color=color_map(i), alpha=0.7, label=f'Episode {episode}')
        axes[1].hist(episode_data['scaled_reward'], bins=50, color=color_map(i), alpha=0.7, label=f'Episode {episode}')
    axes[0].set_title('Distribution of Raw Rewards')
    axes[1].set_title('Distribution of Scaled Rewards')
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'rewards_distribution_plot.png')
    plt.savefig(plot_path)
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        ax.plot(episode_data['fee_earned'], label=f'Episode {episode} Fee Income', color=color_map(i))
        ax.plot(episode_data['impermanent_loss'], label=f'Episode {episode} Impermanent Loss', color=color_map(i), linestyle='--')
    ax.set_title('Fee Income and Impermanent Loss Over Time')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Value')
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'fee_income_impermanent_loss_plot.png')
    plt.savefig(plot_path)
    plt.close()

def train_separate_episode_action_plot(data_df, output_dir):
    episodes = data_df['episode'].unique()
    if len(episodes) == 1:
        return 
    fig, axes = plt.subplots(nrows=len(episodes), ncols=1, figsize=(15, 5 * len(episodes)))
    
    for i, episode in enumerate(episodes):
        episode_df = data_df[data_df['episode'] == episode]

        ax1 = axes[i]
        ax1.plot(episode_df['step_count'], episode_df['curr_price'], label='Current Price', color='blue')
        ax1.fill_between(episode_df['step_count'], episode_df['scaled_action_0'], episode_df['scaled_action_1'], color='gray', alpha=0.3, label='Action Range (price_lower to price_upper)')
        ax1.set_xlabel('Step Count')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title(f'Agent Actions and Rewards over Time for Episode {episode}')

        ax2 = ax1.twinx()
        ax2.plot(episode_df['step_count'], episode_df['raw_reward'], label='Reward', color='red', linestyle='--')
        ax2.set_ylabel('Reward', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'separate_episode_action_plot.png')
    plt.savefig(plot_path)
    plt.close()

    fig, axes = plt.subplots(nrows=len(episodes), ncols=2, figsize=(14, 5 * len(episodes)))

    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        axes[i, 0].hist(episode_data['raw_reward'], bins=50, color='blue', alpha=0.7)
        axes[i, 0].set_title(f'Episode {episode} - Distribution of Raw Rewards')
        axes[i, 1].hist(episode_data['scaled_reward'], bins=50, color='green', alpha=0.7)
        axes[i, 1].set_title(f'Episode {episode} - Distribution of Scaled Rewards')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'rewards_distribution_per_episode.png')
    plt.savefig(plot_path)
    plt.close()

    fig, axes = plt.subplots(nrows=len(episodes), ncols=1, figsize=(15, 6 * len(episodes)))

    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        axes[i].plot(episode_data['fee_earned'], label='Fee Income', color='orange')
        axes[i].plot(episode_data['impermanent_loss'], label='Impermanent Loss', color='black', linestyle='--')
        axes[i].set_title(f'Episode {episode} - Fee Income and Impermanent Loss Over Time')
        axes[i].set_xlabel('Steps')
        axes[i].set_ylabel('Value')
        axes[i].legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'fee_income_impermanent_loss_per_episode.png')
    plt.savefig(plot_path)
    plt.close()
