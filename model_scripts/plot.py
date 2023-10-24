import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




def eval_rewards_plot(data):
    
    # Load the

    # Find out how many unique episodes are in the data
    unique_episodes = data['episode'].unique()
    num_episodes = len(unique_episodes)

    # Set up the plotting area
    fig, axes = plt.subplots(nrows=3, ncols=num_episodes, figsize=(5 * num_episodes, 15), sharey='row')

    # Define the plots for each episode
    for i, episode in enumerate(unique_episodes):
        episode_data = data[data['episode'] == episode]

        # Raw Rewards
        sns.lineplot(x='step_count', y='raw_reward_rl_agent', data=episode_data, ax=axes[0, i], label='RL Agent')
        sns.lineplot(x='step_count', y='raw_reward_baseline_agent', data=episode_data, ax=axes[0, i], label='Baseline Agent', linestyle='--')
        axes[0, i].set_title(f'Episode {episode} - Raw Rewards')
        axes[0, i].legend()

        # Scaled Rewards
        sns.lineplot(x='step_count', y='scaled_reward_rl_agent', data=episode_data, ax=axes[1, i], label='RL Agent')
        sns.lineplot(x='step_count', y='scaled_reward_baseline_agent', data=episode_data, ax=axes[1, i], label='Baseline Agent', linestyle='--')
        axes[1, i].set_title(f'Episode {episode} - Scaled Rewards')
        axes[1, i].legend()

        # Cumulative Rewards
        sns.lineplot(x='step_count', y='cumulative_reward_rl_agent', data=episode_data, ax=axes[2, i], label='RL Agent')
        sns.lineplot(x='step_count', y='cumulative_reward_baseline_agent', data=episode_data, ax=axes[2, i], label='Baseline Agent', linestyle='--')
        axes[2, i].set_title(f'Episode {episode} - Cumulative Rewards')
        axes[2, i].legend()

    plt.tight_layout()
    plt.show()





def eval_plot_1(data_df):

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18))

# Plot 1: Raw Rewards
    axes[0].plot(data_df['step_count'], data_df['raw_reward_rl_agent'], label='RL Agent')
    axes[0].plot(data_df['step_count'], data_df['raw_reward_baseline_agent'], label='Baseline Agent', linestyle='dashed')
    axes[0].set_title('Raw Rewards')
    axes[0].set_xlabel('Step Count')
    axes[0].set_ylabel('Raw Reward')
    axes[0].legend()

    # Plot 2: Scaled Rewards
    axes[1].plot(data_df['step_count'], data_df['scaled_reward_rl_agent'], label='RL Agent')
    axes[1].plot(data_df['step_count'], data_df['scaled_reward_baseline_agent'], label='Baseline Agent', linestyle='dashed')
    axes[1].set_title('Scaled Rewards')
    axes[1].set_xlabel('Step Count')
    axes[1].set_ylabel('Scaled Reward')
    axes[1].legend()

    # Plot 3: Cumulative Rewards
    axes[2].plot(data_df['step_count'], data_df['cumulative_reward_rl_agent'], label='RL Agent')
    axes[2].plot(data_df['step_count'], data_df['cumulative_reward_baseline_agent'], label='Baseline Agent', linestyle='dashed')
    axes[2].set_title('Cumulative Rewards')
    axes[2].set_xlabel('Step Count')
    axes[2].set_ylabel('Cumulative Reward')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

def eval_plot_2(data_df):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the plot style
    sns.set(style="whitegrid")

    # Plot Raw Rewards
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_df, x='step_count', y='raw_reward_rl_agent', hue='episode', marker='o', label='RL Agent')
    sns.lineplot(data=data_df, x='step_count', y='raw_reward_baseline_agent', hue='episode', marker='s', label='Baseline Agent')
    plt.title('Raw Rewards')
    plt.ylabel('Raw Reward')
    plt.xlabel('Step Count')
    plt.legend()
    plt.show()

    # Plot Scaled Rewards
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_df, x='step_count', y='scaled_reward_rl_agent', hue='episode', marker='o', label='RL Agent')
    sns.lineplot(data=data_df, x='step_count', y='scaled_reward_baseline_agent', hue='episode', marker='s', label='Baseline Agent')
    plt.title('Scaled Rewards')
    plt.ylabel('Scaled Reward')
    plt.xlabel('Step Count')
    plt.legend()
    plt.show()

    # Plot Cumulative Rewards
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_df, x='step_count', y='cumulative_reward_rl_agent', hue='episode', marker='o', label='RL Agent')
    sns.lineplot(data=data_df, x='step_count', y='cumulative_reward_baseline_agent', hue='episode', marker='s', label='Baseline Agent')
    plt.title('Cumulative Rewards')
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Step Count')
    plt.legend()
    plt.show()

def plot_agent_performance_combined(data_df):
    
    episodes = data_df['episode'].unique()
    episode_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'cyan', 'yellow', 'grey']

    data_df['action_midpoint'] = (data_df['price_lower'] + data_df['price_upper']) / 2
    data_df['action_range'] = data_df['price_upper'] - data_df['price_lower']

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

# The starting point for each new episode will be the maximum step count of the previous episode
    for i in range(1, len(episodes)):
        data_df.loc[data_df['episode'] == episodes[i], 'extended_step_count'] += data_df[data_df['episode'] == episodes[i-1]]['extended_step_count'].max() + 1

    # Plotting with the extended step count
    plt.figure(figsize=(15, 7))
    for episode, color in zip(episodes, episode_colors):
        episode_data = data_df[data_df['episode'] == episode]
        plt.plot(episode_data['extended_step_count'], episode_data['action_midpoint'], label=f'Episode {episode} Midpoint', color=color)
        plt.fill_between(episode_data['extended_step_count'], episode_data['action_midpoint'] - episode_data['action_range']/2, 
                        episode_data['action_midpoint'] + episode_data['action_range']/2, color=color, alpha=0.2)
        plt.plot(episode_data['extended_step_count'], episode_data['curr_price'], label=f'Episode {episode} Curr Price', color=color, linestyle='--')

    plt.title('Agent Actions Over Extended Time')
    plt.xlabel('Extended Step Count')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculating the midpoint of the action
    data_df['action_midpoint'] = (data_df['price_lower'] + data_df['price_upper']) / 2
    data_df['action_range'] = data_df['price_upper'] - data_df['price_lower']
    

    # Plotting
    plt.figure(figsize=(15, 7))
    for episode, color in zip(episodes, episode_colors):
        episode_data = data_df[data_df['episode'] == episode]
        plt.plot(episode_data['step_count'], episode_data['action_midpoint'], label=f'Episode {episode} Midpoint', color=color)
        plt.fill_between(episode_data['step_count'], episode_data['action_midpoint'] - episode_data['action_range']/2, 
                        episode_data['action_midpoint'] + episode_data['action_range']/2, color=color, alpha=0.2)
        plt.plot(episode_data['step_count'], episode_data['curr_price'], label=f'Episode {episode} Current Price', color=color, linestyle='--')

    plt.title('Agent Actions and Current Prices Over Time')
    plt.xlabel('Step Count')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Separate data for each episode
    
    color_map = plt.get_cmap('tab10')
    
    # Agent Actions and Rewards over Time
    fig, ax1 = plt.subplots(figsize=(15, 5))
    for i, episode in enumerate(episodes):
        episode_df = data_df[data_df['episode'] == episode]
        ax1.plot(episode_df['step_count'], episode_df['curr_price'], label=f'Episode {episode} Current Price', color=color_map(i))
    ax1.set_xlabel('Step Count')
    ax1.set_ylabel('Price')
    ax1.set_title('Agent Actions and Rewards over Time')
    ax1.legend()
    plt.tight_layout()
    plt.show()
    
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

    # Other Suggested Plots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 7*3))
    
    # Reward Distribution
    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        axes[0].hist(episode_data['raw_reward'], bins=50, color=color_map(i), alpha=0.7, label=f'Episode {episode}')
        # Action Range Variability
        action_range = episode_data['price_upper'] - episode_data['price_lower']
        axes[1].plot(episode_data['step_count'], action_range, label=f'Episode {episode} Action Range', color=color_map(i))
        # Reward Rolling Average
        rolling_avg_reward = episode_data['raw_reward'].rolling(window=5).mean()
        axes[2].plot(episode_data['step_count'], rolling_avg_reward, label=f'Episode {episode} Rolling Avg Reward', color=color_map(i))
    
    axes[0].set_title('Reward Distribution')
    axes[1].set_title('Action Range Variability Over Time')
    axes[2].set_title('Reward Rolling Average Over Time')
    for ax in axes:
        ax.legend()
    
    plt.tight_layout()
    plt.show()
def plot_agent_performance_overlay(data_df):
    
    # Distinct episodes and color palette
    episodes = data_df['episode'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(episodes)))

    # Agent Actions and Color-coded Rewards over Time
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        ax.fill_between(episode_data['step_count'], episode_data['price_lower'], episode_data['price_upper'], color=colors[i], alpha=0.5, label=f'Episode {episode} Action Range')
        ax.scatter(episode_data['step_count'], episode_data['curr_price'], c=[colors[i]], s=20, label=f'Episode {episode} Current Price')
    ax.set_title('Agent Actions Over Time')
    ax.set_xlabel('Step Count')
    ax.set_ylabel('Price')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Distribution of Raw and Scaled Rewards
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        axes[0].hist(episode_data['raw_reward'], bins=50, color=colors[i], alpha=0.5, label=f'Episode {episode}')
        axes[1].hist(episode_data['scaled_reward'], bins=50, color=colors[i], alpha=0.5, label=f'Episode {episode}')
    axes[0].set_title('Distribution of Raw Rewards')
    axes[1].set_title('Distribution of Scaled Rewards')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    # Fee Income and Impermanent Loss Over Time
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        ax.plot(episode_data['fee_earned'], label=f'Episode {episode} Fee Income', color=colors[i])
        ax.plot(episode_data['impermanent_loss'], linestyle='--', label=f'Episode {episode} Impermanent Loss', color=colors[i])
    ax.set_title('Fee Income and Impermanent Loss Over Time')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Value')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Other Suggested Plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 21))
    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        # Reward Distribution
        axes[0].hist(episode_data['raw_reward'], bins=50, color=colors[i], alpha=0.5, label=f'Episode {episode}')
        # Action Range Variability
        action_range = episode_data['price_upper'] - episode_data['price_lower']
        axes[1].plot(episode_data['step_count'], action_range, label=f'Episode {episode} Action Range', color=colors[i])
        # Reward Rolling Average
        rolling_avg_reward = episode_data['raw_reward'].rolling(window=5).mean()
        axes[2].plot(episode_data['step_count'], rolling_avg_reward, label=f'Episode {episode} Rolling Average Reward', color=colors[i])
    axes[0].set_title('Reward Distribution Across Episodes')
    axes[1].set_title('Action Range Variability Over Time Across Episodes')
    axes[2].set_title('Reward Rolling Average Over Time Across Episodes')
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_agent_performance(data_df):
    # Separate data for each episode
    episodes = data_df['episode'].unique()

    fig, axes = plt.subplots(nrows=len(episodes), ncols=1, figsize=(15, 5*len(episodes)))

    for i, episode in enumerate(episodes):
        episode_df = data_df[data_df['episode'] == episode]

        # Agent Actions and Rewards over Time
        ax1 = axes[i]
        ax1.plot(episode_df['step_count'], episode_df['curr_price'], label='Current Price', color='blue')
        ax1.fill_between(episode_df['step_count'], episode_df['price_lower'], episode_df['price_upper'], color='gray', alpha=0.3, label='Action Range (price_lower to price_upper)')
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

    
    # Agent Actions and Color-coded Rewards over Time
    fig, axes = plt.subplots(nrows=len(episodes), ncols=1, figsize=(14, 7*len(episodes)))
    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        # Create a colormap to color the curr_price line based on reward
        reward_colormap = plt.cm.get_cmap('RdYlBu_r')
        # Normalize the rewards for coloring
        norm = plt.Normalize(episode_data['raw_reward'].min(), episode_data['raw_reward'].max())
        # Plot the actions (price_lower and price_upper) as shaded regions
        axes[i].fill_between(episode_data['step_count'], episode_data['price_lower'], episode_data['price_upper'], color='gray', alpha=0.5, label='Action Range (price_lower to price_upper)')
        # Plot curr_price with color-coded reward
        points = axes[i].scatter(episode_data['step_count'], episode_data['curr_price'], c=episode_data['raw_reward'], cmap=reward_colormap, norm=norm, s=20, label='Current Price (Color-coded by Reward)')
        axes[i].set_title(f'Episode {episode} - Agent Actions and Color-coded Rewards over Time')
        axes[i].set_xlabel('Step Count')
        axes[i].set_ylabel('Price')
        axes[i].legend(loc='upper left')
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

    # Other Suggested Plots
    fig, axes = plt.subplots(nrows=len(episodes)*3, ncols=1, figsize=(14, 7*len(episodes)*3))
    for i, episode in enumerate(episodes):
        episode_data = data_df[data_df['episode'] == episode]
        # Reward Distribution
        axes[3*i].hist(episode_data['raw_reward'], bins=50, color='purple', alpha=0.7)
        axes[3*i].set_title(f'Episode {episode} - Reward Distribution')
        # Action Range Variability
        action_range = episode_data['price_upper'] - episode_data['price_lower']
        axes[3*i+1].plot(episode_data['step_count'], action_range, label='Action Range (price_upper - price_lower)', color='orange')
        axes[3*i+1].set_title(f'Episode {episode} - Action Range Variability Over Time')
        # Reward Rolling Average
        rolling_avg_reward = episode_data['raw_reward'].rolling(window=5).mean()
        axes[3*i+2].plot(episode_data['step_count'], rolling_avg_reward, label='Rolling Average Reward', color='red')
        axes[3*i+2].set_title(f'Episode {episode} - Reward Rolling Average Over Time')
    plt.tight_layout()
    plt.show()

# Using the function

# Define a function to plot the data episode-wise
def plot_episode_data(data_df):
    
    episodes = data_df['episode'].unique()

    # Iterate through each episode
    for episode in episodes:
        episode_data = data_df[data_df['episode'] == episode]

        # Plot curr_price over step_count
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(episode_data['step_count'], episode_data['curr_price'], label='Current Price', color='blue')
        ax1.fill_between(episode_data['step_count'], episode_data['price_lower'], episode_data['price_upper'], color='gray', alpha=0.3, label='Action Range (price_lower to price_upper)')
        ax1.set_xlabel('Step Count')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a secondary y-axis for rewards
        ax2 = ax1.twinx()
        ax2.plot(episode_data['step_count'], episode_data['raw_reward'], label='Reward', color='red', linestyle='--')
        ax2.set_ylabel('Reward', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Title and show plot
        plt.title(f'Agent Actions and Rewards over Time - Episode {episode}')
        fig.tight_layout()
        plt.show()

        # Color-coding curr_price based on reward magnitude
        fig, ax = plt.subplots(figsize=(14, 7))
        reward_colormap = plt.cm.get_cmap('RdYlBu_r')
        norm = plt.Normalize(episode_data['raw_reward'].min(), episode_data['raw_reward'].max())
        ax.fill_between(episode_data['step_count'], episode_data['price_lower'], episode_data['price_upper'], color='gray', alpha=0.5, label='Action Range (price_lower to price_upper)')
        points = ax.scatter(episode_data['step_count'], episode_data['curr_price'], c=episode_data['raw_reward'], cmap=reward_colormap, norm=norm, s=20, label='Current Price (Color-coded by Reward)')
        cbar = plt.colorbar(points, ax=ax)
        cbar.set_label('Reward Value')
        ax.set_xlabel('Step Count')
        ax.set_ylabel('Price')
        ax.set_title(f'Agent Actions and Color-coded Rewards over Time - Episode {episode}')
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        # Plotting the distribution of raw rewards
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.hist(episode_data['raw_reward'], bins=50, color='blue', alpha=0.7)
        plt.title(f'Distribution of Raw Rewards (raw_reward) - Episode {episode}')
        plt.xlabel('raw_reward')
        plt.ylabel('Frequency')
        plt.subplot(1, 2, 2)
        plt.hist(episode_data['scaled_reward'], bins=50, color='green', alpha=0.7)
        plt.title(f'Distribution of Scaled Rewards (reward) - Episode {episode}')
        plt.xlabel('scaled_reward')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

        # Plotting cumulative reward and episode-wise average reward
        episode_data['cumulative_reward'] = episode_data['raw_reward'].cumsum()
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(episode_data['step_count'], episode_data['cumulative_reward'], label='Cumulative Reward', color='blue')
        ax.set_title(f'Cumulative Reward Over Time - Episode {episode}')
        ax.set_xlabel('Step Count')
        ax.set_ylabel('Cumulative Reward')
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Plot the fee income and impermanent loss over time
        plt.figure(figsize=(15, 6))
        plt.plot(episode_data['fee_earned'], label='Fee Income', color='orange')
        plt.plot(episode_data['impermanent_loss'], label='Impermanent Loss', color='black', linestyle='--')
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.title(f'Fee Income and Impermanent Loss Over Time - Episode {episode}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Creating the other suggested plots
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 15))
        axes[0].hist(episode_data['raw_reward'], bins=50, color='purple', alpha=0.7)
        axes[0].set_title(f'Reward Distribution - Episode {episode}')
        axes[0].set_xlabel('Reward Value')
        axes[0].set_ylabel('Frequency')
        action_range = episode_data['price_upper'] - episode_data['price_lower']
        axes[1].plot(episode_data['step_count'], action_range, label='Action Range (price_upper - price_lower)', color='orange')
        axes[1].set_title(f'Action Range Variability Over Time - Episode {episode}')
        axes[1].set_xlabel('Step Count')
        axes[1].set_ylabel('Action Range')
        axes[1].legend()
        rolling_avg_reward = episode_data['raw_reward'].rolling(window=5).mean()
        axes[2].plot(episode_data['step_count'], rolling_avg_reward, label='Rolling Average Reward', color='red')
        axes[2].set_title(f'Reward Rolling Average Over Time - Episode {episode}')
        axes[2].set_xlabel('Step Count')
        axes[2].set_ylabel('Average Reward')
        axes[2].legend()
        plt.tight_layout()
        plt.show()