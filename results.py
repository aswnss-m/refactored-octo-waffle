import matplotlib.pyplot as plt
import pandas as pd

# Load the data
train_data = pd.read_csv('training_data.csv')
eval_data = pd.read_csv('evaluation_data.csv')

# Plot training rewards for each agent
agents = train_data['agent_id'].unique()
for agent in agents:
    agent_data = train_data[train_data['agent_id'] == agent]
    plt.figure(figsize=(10, 5))
    plt.plot(agent_data['step'], agent_data['reward'], label=f'Reward for Agent {agent}')
    plt.title(f'Training Rewards per Step for Agent {agent}')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(agent_data['step'], agent_data['headway'], label=f'Headway for Agent {agent}')
    plt.title(f'Training Headways per Step for Agent {agent}')
    plt.xlabel('Step')
    plt.ylabel('Headway')
    plt.legend()
    plt.show()

# Evaluation plots can be similarly adjusted to reflect per-agent data
