from platooning_env import PlatooningEnv
import numpy as np

# Create environment instance
env = PlatooningEnv()

# Initialize environment after a certain number of steps
env.initialize_after_steps(21)

# Define the number of episodes
NUM_EPISODES = 5  # For example, run 10 episodes

headway_data_per_agent = {}  # Dictionary to store headway data for each agent

# Run episodes
for episode in range(NUM_EPISODES):
    # Reset environment for new episode
    observations = env.reset()
    done = False  # Initialize done flag for episode termination

    # Run episode until termination
    while not done:
        # Select random actions for each agent
        actions = {agent: np.random.randint(0, 3) for agent in env.agents}  # Assuming 3 discrete actions
        # Execute actions and get next observations, rewards, termination flag, and additional info
        next_observations, rewards, done, info = env.step(actions)
    
    # Append headway details of this episode for each agent
    for agent in env.agents:
        if agent not in headway_data_per_agent:
            headway_data_per_agent[agent] = []
        # Check the type of headway data before accessing it
        for headway in env.headway_details:
            if isinstance(headway, dict) and agent in headway:
                headway_data_per_agent[agent].append(headway[agent])
            else:
                # Handle cases where headway data is missing or invalid
                # For example, append a placeholder value or skip this data point
                headway_data_per_agent[agent].append(np.nan)

# Compute statistics of headway for each agent
for agent, headway_data in headway_data_per_agent.items():
    mean_headway = np.nanmean(headway_data)
    median_headway = np.nanmedian(headway_data)
    std_headway = np.nanstd(headway_data)
    max_headway = np.nanmax(headway_data)
    min_headway = np.nanmin(headway_data)

    print(f"Agent: {agent}")
    print("Mean Headway:", mean_headway)
    print("Median Headway:", median_headway)
    print("Standard Deviation of Headway:", std_headway)
    print("Maximum Headway:", max_headway)
    print("Minimum Headway:", min_headway)

# Save headway data
print("Total number of headway details:", sum(len(headway_data) for headway_data in headway_data_per_agent.values()))
env.save_headway_to_csv("headway_data.csv")
env.save_headway_plot()
# Close environment
env.close()
