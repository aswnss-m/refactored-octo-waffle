from platooning_env import PlatooningEnv

# Create environment instance
env = PlatooningEnv()

# Initialize environment after a certain number of steps
env.initialize_after_steps(21)

# Define the number of episodes
NUM_EPISODES = 10  # For example, run 10 episodes

# Run episodes
for episode in range(NUM_EPISODES):
    # Reset environment for new episode
    observation = env.reset()
    done = False  # Initialize done flag for episode termination
    total_reward = 0  # Initialize total reward for the episode

    # Run episode until termination
    while not done:
        # Select random actions for each agent
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        # Execute actions and get next observation, reward, termination flag, and additional info
        next_observation, reward, done, truncated, info = env.step(actions)
        # Accumulate total reward
        total_reward += sum(reward.values())

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Save headway data
env.save_headway_to_csv("headway_data.csv")
env.save_headway_plot("headway_plot.png")
# Close environment
env.close()
