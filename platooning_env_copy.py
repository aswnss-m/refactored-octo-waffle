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
        self.agents = []  # Initialize agents attribute
        self.initialized = False  # Initialize the initialized attribute

    def _start_sumo(self):
        """
        Start SUMO simulation.
        """
        try:
            traci.start([self.sumo_binary, "-c", self.sumo_cfg_path, "--tripinfo-output", "tripinfo.xml"])
            print("SUMO simulation started successfully.")
            self.initialized = True  # Set initialized to True after starting SUMO
        except traci.exceptions.FatalTraCIError:
            raise RuntimeError("Failed to start SUMO simulation")

    def _stop_sumo(self):
        """
        Stop SUMO simulation.
        """
        traci.close()

    def observe(self, agents):
        """
        Obtain observations for the agents that are present in the simulation.

        Args:
            agents (list): List of agent IDs.

        Returns:
            dict: Dictionary containing observations for each agent.
        """
        observations = {}

        try:
            for agent in agents:
                if traci.vehicle.getIDList().count(agent) > 0:
                    current_speed_follower = traci.vehicle.getSpeed(agent)
                    leader_info = traci.vehicle.getLeader(agent)
                    if leader_info is not None:
                        current_headway = leader_info[1]
                    else:
                        current_headway = 100  # Set a default value for headway if leader is not found
                    observations[agent] = np.array([current_speed_follower, current_headway], dtype=np.float32)
                else:
                    # Vehicle not present in the simulation, skip observation
                    pass
        except traci.exceptions.TraCIException as e:
            print(f"Error retrieving observations for agents: {e}")
            observations = {agent: np.array([0, 0], dtype=np.float32) for agent in agents}

        return observations

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

        observations = {}
        rewards = {}

        # Get list of vehicles present in the simulation
        present_vehicles = traci.vehicle.getIDList()

        for agent, act in action.items():
            try:
                if present_vehicles.count(agent) > 0:
                    traci.vehicle.setSpeed(agent, 20 if act == 0 else (10 if act == 1 else 15))
                else:
                    print(f"Vehicle {agent} is not known. Skipping action execution.")
            except traci.exceptions.TraCIException as e:
                print(f"Error executing action for agent {agent}: {e}")

        traci.simulationStep()

        for agent in present_vehicles:
            try:
                current_speed_follower = traci.vehicle.getSpeed(agent)
                leader_info = traci.vehicle.getLeader(agent)
                if leader_info is not None:
                    _, current_headway = leader_info
                    leader_speed = traci.vehicle.getSpeed(leader_info[0])
                    # Reward for maintaining safe headway and speed
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
                    current_headway = 100  # Set a default value for headway if leader is not found
                    rewards[agent] = -1  # Penalize for not having a leader
            except traci.exceptions.TraCIException as e:
                print(f"Error processing agent {agent}: {e}")

            observations[agent] = np.array([current_speed_follower, current_headway], dtype=np.float32)

        done = self.STEPS >= TOTAL_STEPS or not traci.simulation.getMinExpectedNumber() > 0

        if done:
            self.close()  # Close the environment if simulation is done

        return observations, rewards, done, {}

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            dict: Initial observation.
        """
        self.STEPS = 0
        self._start_sumo()

        # Wait until all vehicles are initialized in the simulation
        while traci.simulation.getMinExpectedNumber() > len(traci.vehicle.getIDList()):
            traci.simulationStep()

        self.agents = traci.vehicle.getIDList()

        self.headway_details = []

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
