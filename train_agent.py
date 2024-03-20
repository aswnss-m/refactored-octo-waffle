import torch as th
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PlatooningEnv import PlatooningEnv  # Import the environment we created earlier
from PlatooningEnvAcc import PlatooningEnvAcc  # Import the environment we created earlier
# Create a function to define and train the agent
def train_agent():
    # Create the environment
    env = PlatooningEnv()

    # Wrap the environment in a vectorized environment
    env = DummyVecEnv([lambda: env])

    # Define and train the PPO agent
    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)

    # Get the underlying single environment
    single_env = env.envs[0]

    headway_df = pd.DataFrame(single_env.headway_details, columns=["Headway"])

    # Plot the line plot
    plt.plot(headway_df.index, headway_df["Headway"])
    plt.xlabel("Time Step")
    plt.ylabel("Headway")
    plt.title("Headway over Time")
    plt.show()
    # Close the environment
    env.close()

# Train the agent
train_agent()
