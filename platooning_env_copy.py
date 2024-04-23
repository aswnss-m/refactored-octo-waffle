import numpy as np
from pettingzoo.utils import AECEnv
from sumolib import checkBinary
import pandas as pd
import matplotlib.pyplot as plt
import traci

TOTAL_STEPS = 1000

class PlatooningEnv(AECEnv):
    """
    Custom PettingZoo environment for vehicle platooning using SUMO simulation.
    """
    def __init__(self, sumo_cfg_path="./maps/cross/cross_ext.sumocfg"):
        """
        Initialize PlatooningEnv.

        Args:
            sumo_cfg_path (str): Path to the SUMO configuration file.
        """
        super().__init__()
        self.sumo_binary = checkBinary('sumo-gui')
        self.sumo_cfg_path = sumo_cfg_path
        self._start_sumo()

        self.agents = []
        self.leader = None
        self.follower = None
        self.STEPS = 0
        self.headway_details = []
        self.initialized = False

    def _start_sumo(self):
        """
        Start SUMO simulation.
        """
        try:
            traci.start([self.sumo_binary, "-c", self.sumo_cfg_path, "--tripinfo-output", "tripinfo.xml"])
            print("SUMO simulation started successfully.")
        except traci.exceptions.FatalTraCIError:
            raise RuntimeError("Failed to start SUMO simulation")

    def _stop_sumo(self):
        """
        Stop SUMO simulation.
        """
        traci.close()

    def initialize_after_steps(self, num_steps=20):
        """
        Initialize the environment after a certain number of steps.
        """
        for _ in range(num_steps):
            traci.simulationStep()
            self.STEPS += 1

        self.initialized = True
        self.agents = traci.vehicle.getIDList()

    def observe(self, agent):
        """
        Obtain observation for the agent.

        Args:
            agent (str): Agent ID.

        Returns:
            numpy.array: Observation for the agent.
        """
        try:
            current_speed_follower = traci.vehicle.getSpeed(agent)
            leader_info = traci.vehicle.getLeader(agent)
            if leader_info is not None:
                current_headway = leader_info[1]
                self.headway_details.append(current_headway)
            else:
                current_headway = 100
            return np.array([current_speed_follower, current_headway], dtype=np.float32)
        except traci.exceptions.TraCIException as e:
            print(f"Error retrieving observation for agent {agent}: {e}")
            return np.array([0, 0], dtype=np.float32)

    def step(self, action):
        """
        Execute a step in the environment.

        Args:
            action (dict): Dictionary mapping agent IDs to their corresponding actions.

        Returns:
            tuple: Tuple containing observations, rewards, done flag, and info.
        """
        if not self.initialized:
            return {agent: self.observe(agent) for agent in self.agents}, {agent: 0 for agent in self.agents}, False, {}

        self.STEPS += 1

        for agent, act in action.items():
            try:
                traci.vehicle.setSpeed(agent, 20 if act == 0 else (10 if act == 1 else 15))
            except traci.exceptions.TraCIException as e:
                print(f"Error executing action for agent {agent}: {e}")

        traci.simulationStep()

        observations = {agent: self.observe(agent) for agent in self.agents}

        # Record headway details for all agents
        headways = {}
        for agent in self.agents:
            leader_info = traci.vehicle.getLeader(agent)
            if leader_info is not None:
                _, current_headway = leader_info
                headways[agent] = current_headway
            else:
                headways[agent] = -1  # Placeholder value for missing leader
        self.headway_details.append(headways)

        # Compute rewards
        rewards = {}
        for agent in self.agents:
            try:
                current_speed_follower = traci.vehicle.getSpeed(agent)
                leader_info = traci.vehicle.getLeader(agent)
                if leader_info is not None:
                    _, current_headway = leader_info
                    leader_speed = traci.vehicle.getSpeed(leader_info[0])
                    # Reward for maintaining safe headway and speed
                    if current_headway < 10:
                        rewards[agent] = -10  # Give a negative reward for dangerously low headway
                    elif current_headway < 15:
                        rewards[agent] = 10  # Give a high positive reward for maintaining safe headway
                    elif current_headway <= 20:
                        rewards[agent] = 5  # Give a positive reward for maintaining safe headway
                    else:
                        rewards[agent] = -5  # Penalize for excessive headway
                    """"
                    if current_headway >= 10 and current_headway <= 20:
                        rewards[agent] = 0.1 * leader_speed  # Reward for maintaining safe headway
                    else:
                        rewards[agent] = -1  # Penalize for unsafe headway
                    # Penalize for exceeding speed limit
                    if current_speed_follower > 20:
                        rewards[agent] -= 0.5 * (current_speed_follower - 20)
                    # Additional reward for maintaining platooning speed
                    rewards[agent] += 0.1 * (20 - abs(current_speed_follower - 20))
                else:
                    rewards[agent] = -1  # Penalize for not having a leader
                    """
            except traci.exceptions.TraCIException as e:
                print(f"Error computing rewards for agent {agent}: {e}")

        done = self.STEPS >= TOTAL_STEPS
        info = {}

        return observations, rewards, done, info

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            dict: Initial observation.
        """
        self.STEPS = 0
        self.initialized = False
        self._stop_sumo()
        self._start_sumo()

        self.headway_details = []

        self.initialize_after_steps()

        return {agent: self.observe(agent) for agent in self.agents}

    def close(self):
        """
        Close the environment.
        """
        self._stop_sumo()

    def save_headway_to_csv(self, filename="headway_data.csv"):
        """
        Save headway data to a CSV file.

        Args:
            filename (str): Name of the CSV file.
        """
        # Filter out numerical headway values and keep only dictionaries
        headway_dicts = [entry for entry in self.headway_details if isinstance(entry, dict)]
        # Convert list of dictionaries to DataFrame
        headway_df = pd.DataFrame(headway_dicts)
        # Save DataFrame to CSV
        headway_df.to_csv(filename, index=False)

    def save_headway_plot(self, filename="headway_plot"):
        """
        Save a plot of headway data for each agent separately to a file.

        Args:
            filename (str): Base name of the file to save the plots.
        """
        headway_df = pd.read_csv("headway_data.csv")  # Read headway data from CSV

        # Iterate over each agent's headway data
        for column in headway_df.columns:
            if column != "ev_0":  # Exclude the ev_0 column
                plt.figure(figsize=(15, 5))
                plt.plot(headway_df.index, headway_df[column])
                plt.xlabel("Time Step")
                plt.ylabel("Headway")
                plt.title(f"Headway of Agent {column} over Time")
                plt.savefig(f"{filename}_{column}.png")  # Save plot to a separate image file
                plt.close()  # Close the plot to release memory
