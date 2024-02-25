import gymnasium as gym
from stable_baselines3 import PPO
import os

# MODEL_DIRS = "./models/PPO"
# LOGS = "./logs"

# if not os.path.exists(MODEL_DIRS):
#     os.makedirs(MODEL_DIRS)
# if not os.path.exists(LOGS):
#     os.makedirs(LOGS)

env = gym.make('LunarLander-v2')
env.reset()

model = PPO("MlpPolicy",env,verbose=1)
model.learn(total_timesteps=100)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        env.render()
        obs,reward,done,info = env.step(env.action_space.sample())


env.close()