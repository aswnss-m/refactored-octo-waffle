import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci
from sumolib import checkBinary

"""
GLOBAL VARIABLES
"""
MAX_HEADWAY = 20  # its in meters
MIN_HEADWAY = 10  # its in meters
N_DISCRETE_ACTIONS = 3
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.99

"""
Helper functions
"""


def leader_exists(follower):
    (id, headway) = traci.vehicle.getLeader(follower)
    if id:
        return id, headway
    else:
        return None, None


def change_speed(veh_id, new_speed):
    traci.vehicle.setSpeed(veh_id, new_speed)


class TraciEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(TraciEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(4,), dtype=np.float32)

        # Initialize Q-table
        self.q_table = np.zeros((N_DISCRETE_ACTIONS, N_DISCRETE_ACTIONS))

        # Start SUMO as a subprocess and connect with TraCI
        self.sumo_binary = checkBinary('sumo')  # use sumo-gui for gui
        traci.start([self.sumo_binary, "-c", "./maps/singlelane/singlelane.sumocfg",
                     "--tripinfo-output", "tripinfo.xml"])
        # run the simulation a 20 steps to load the car into scene
        for i in range(5):
            traci.simulationStep()

        # initial observation will include follower cars details
        self.vehicle_ids = traci.vehicle.getIDList()  # get the ids of the vehicles
        self.actorVehicle = self.vehicle_ids[1]  # Changed index to 0 for the first vehicle
        self.actorVehicle_speed = traci.vehicle.getSpeed(self.actorVehicle)
        self.leader_id, self.headway = leader_exists(self.actorVehicle)
        self.observation = (self.actorVehicle, self.actorVehicle_speed, self.leader_id, self.headway)

    def step(self, action):
        traci.simulationStep()  # take next step

        # Check if the leader exists
        leader_id, headway = leader_exists(self.actorVehicle)

        if leader_id is not None:
            # Calculate reward based on headway
            if headway > MAX_HEADWAY:
                reward = 1
            elif headway < MIN_HEADWAY:
                reward = -1
            else:
                reward = 0

            # Update Q-table
            current_state = self.observation
            next_state = (self.actorVehicle, self.actorVehicle_speed, leader_id, headway)
            self.q_table[current_state][action] += LEARNING_RATE * (
                    reward + DISCOUNT_FACTOR * np.max(self.q_table[next_state]) -
                    self.q_table[current_state][action])

            # Exploration vs. exploitation
            if np.random.rand() < EXPLORATION_RATE:
                action = np.random.randint(0, N_DISCRETE_ACTIONS)

            # Execute the chosen action
            if action == 0:  # Increase speed
                change_speed(self.actorVehicle, self.actorVehicle_speed + 1)
            elif action == 1:  # Decrease speed
                change_speed(self.actorVehicle, self.actorVehicle_speed - 1)
            # Else, maintain speed

            # Update exploration rate
            EXPLORATION_RATE *= EXPLORATION_DECAY

        info = {}
        return self.observation, reward, self.done, self.truncated, info

    def reset(self, seed=None, options=None):
        self.done = False
        print("The options passed to reset function: ", options)

        # run the simulation a 20 steps to load the car into scene
        for i in range(5):
            traci.simulationStep()

        # initial observation will include follower cars details
        self.vehicle_ids = traci.vehicle.getIDList()  # get the ids of the vehicles
        self.actorVehicle = self.vehicle_ids[1]  # Changed index to 0 for the first vehicle
        self.actorVehicle_speed = traci.vehicle.getSpeed(self.actorVehicle)
        self.leader_id, self.headway = leader_exists(self.actorVehicle)
        self.observation = [self.actorVehicle, self.actorVehicle_speed, self.leader_id, self.headway]
        print(self.observation)

        return np.array(self.observation).astype(np.float32), {}  # reward, done, info cant be included

    def render(self):
        pass

    def close(self):
        traci.close()