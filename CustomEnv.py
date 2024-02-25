import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci
from sumo import checkBinary


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        ...
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        print("The options passed to reset function : ",options)
        sumo_binary = checkBinary('sumo') #use sumo-gui for gui

        # Start SUMO as a subprocess and connect with TraCI
        traci.start([sumo_binary, "-c", "./maps/singlelane/singlelane.sumocfg",
                 "--tripinfo-output", "tripinfo.xml"])
        # run the simulation a 20 steps to load the car into scene
        for i in range(5):
            traci.simulationStep()
        return observation, info

    def render(self):
        ...

    def close(self):
        traci.close()