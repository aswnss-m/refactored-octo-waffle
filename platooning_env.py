import numpy as np
import traci
from sumolib import checkBinary
import pandas as pd
import matplotlib.pyplot as plt
from pettingzoo import AECEnv

TOTAL_STEPS = 5000

class PlatooningEnv(AECEnv):
    """
    Custom AEC environment for vehicle platooning using SUMO simulation.
    """
    def __init__(self, sumo_cfg_path="./maps/jesus/jesus.sumocfg"):
        """
        Initialize PlatooningEnv.

        Args:
            sumo_cfg_path (str): Path to the SUMO configuration file.
        """
        super().__init__()
        self.sumo_binary = checkBinary('sumo')
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

    def step(self, action_dict):
        """
        Execute a step in the environment.

        Args:
            action_dict (dict): Dictionary mapping agent IDs to their corresponding actions.

        Returns:
            tuple: Tuple containing observations, rewards, done flag, and additional info.
        """
        if not self.initialized:
            return self.observe(self.agents), {agent: 0 for agent in self.agents}, {agent: False for agent in self.agents}, {}

        self.STEPS += 1

        for agent, act in action_dict.items():
            try:
                traci.vehicle.setSpeed(agent, 20 if act == 0 else (10 if act == 1 else 15))
            except traci.exceptions.TraCIException as e:
                print(f"Error executing action for agent {agent}: {e}")

        traci.simulationStep()

        rewards = {}
        for agent in self.agents:
            current_headway = 0
            try:
                leader_info = traci.vehicle.getLeader(agent)
                if leader_info is not None:
                    _, current_headway = leader_info
                rewards[agent] = 0
                if current_headway < 10:
                    rewards[agent] -= 2
                elif current_headway > 20:
                    rewards[agent] -= 10
                else:
                    rewards[agent] += 5
                self.headway_details.append(current_headway)
            except traci.exceptions.TraCIException as e:
                print(f"Error computing rewards for agent {agent}: {e}")

        done = {agent: self.STEPS >= TOTAL_STEPS for agent in self.agents}
        info = {}

        return self.observe(self.agents), rewards, done, info

    def observe(self, agents):
        """
        Obtain observations for the agents.

        Args:
            agents (list): List of agent IDs.

        Returns:
            dict: Dictionary containing observations for each agent.
        """
        try:
            observations = {}
            for agent in agents:
                current_speed_follower = traci.vehicle.getSpeed(agent)
                leader_info = traci.vehicle.getLeader(agent)
                if leader_info is not None:
                    current_headway = leader_info[1]
                else:
                    current_headway = 100
                observations[agent] = np.array([current_speed_follower, current_headway], dtype=np.float32)
            return observations
        except traci.exceptions.TraCIException as e:
            print(f"Error retrieving observation for agents: {e}")
            return {agent: np.array([0, 0], dtype=np.float32) for agent in agents}

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

        return self.observe(self.agents)

    def close(self):
        """
        Close the environment.
        """
        self._stop_sumo()

    def get_headway_statistics(self):
        """
        Compute statistics of headway data.

        Returns:
            tuple: Mean, median, and standard deviation of headway.
        """
        headway_df = pd.DataFrame(self.headway_details, columns=["Headway"])
        mean_headway = headway_df["Headway"].mean()
        median_headway = headway_df["Headway"].median()
        std_headway = headway_df["Headway"].std()
        return mean_headway, median_headway, std_headway

    def save_headway_plot(self, filename="headway_plot.png"):
        """
        Save a plot of headway data to a file.

        Args:
            filename (str): Name of the file to save the plot.
        """
        headway_df = pd.DataFrame(self.headway_details, columns=["Headway"])
        plt.figure(figsize=(15, 5))
        plt.plot(headway_df.index, headway_df["Headway"])
        plt.xlabel("Time Step")
        plt.ylabel("Headway")
        plt.title("Headway over Time")
        plt.savefig(filename)

    def save_headway_to_csv(self, filename="headway_data.csv"):
        """
        Save headway data to a CSV file.

        Args:
            filename (str): Name of the CSV file.
        """
        headway_df = pd.DataFrame(self.headway_details, columns=["Headway"])
        headway_df.to_csv(filename, index=False)
