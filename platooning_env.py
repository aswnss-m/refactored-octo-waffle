import numpy as np
from pettingzoo.utils import AECEnv
from sumolib import checkBinary
import pandas as pd
import matplotlib.pyplot as plt
import traci
import traci.exceptions

TOTAL_STEPS = 10000

class PlatooningEnv(AECEnv):
    def __init__(self, sumo_cfg_path="./maps/cross/cross_ext.sumocfg"):
        super().__init__()
        self.sumo_binary = checkBinary('sumo')
        self.sumo_cfg_path = sumo_cfg_path
        self._start_sumo()

        self.agents = {}
        self.STEPS = 0
        self.initialized = False

    def _start_sumo(self):
        try:
            traci.start([self.sumo_binary, "-c", self.sumo_cfg_path, "--tripinfo-output", "tripinfo.xml"])
            print("SUMO simulation started successfully.")
        except traci.exceptions.FatalTraCIError:
            raise RuntimeError("Failed to start SUMO simulation")

    def _stop_sumo(self):
        traci.close()

    def initialize_after_steps(self, num_steps=20):
        for _ in range(num_steps):
            traci.simulationStep()
            self.STEPS += 1

        self.initialized = True
        self.agents = {agent: self.observe(agent) for agent in traci.vehicle.getIDList()}

    def observe(self, agent):
        try:
            # Check if agent is present in the simulation
            if agent not in traci.vehicle.getIDList():
                return None  # If not present, return None for observation
                
            current_speed_follower = traci.vehicle.getSpeed(agent)
            leader_info = traci.vehicle.getLeader(agent)
            if leader_info is not None:
                current_headway = leader_info[1]
            else:
                current_headway = 100
            return np.array([current_speed_follower, current_headway], dtype=np.float32)
        except traci.exceptions.TraCIException as e:
            print(f"Error retrieving observation for agent {agent}: {e}")
            return None

    def step(self, action):
        if not self.initialized:
            return {agent: self.observe(agent) for agent in self.agents}, {agent: 0 for agent in self.agents}, False, {}

        self.STEPS += 1

        # Inside the step method, after updating the simulation
        for agent, act in action.items():
            try:
                # Check if agent is present in the simulation
                if agent not in traci.vehicle.getIDList():
                    continue  # Skip processing for agents not present in simulation
                    
                # Update the simulation
                traci.vehicle.setSpeed(agent, 20 if act == 0 else (10 if act == 1 else 15))
                
                # Check if the agent is following a leader
                leader_info = traci.vehicle.getLeader(agent)
                is_following_leader = leader_info is not None
                
                if is_following_leader:
                    # Get the ID of the leader
                    leader_id, _ = leader_info
                    # Set the color of the follower vehicle to blue if it's following a leader
                    traci.vehicle.setColor(agent, (0, 0, 255))  # Set color to blue (RGB format)
                else:
                    # Reset color to default if the vehicle is not following a leader
                    traci.vehicle.setColor(agent, (255, 255, 255))  # Set color to white (RGB format)
                        
            except traci.exceptions.TraCIException as e:
                print(f"Error executing action for agent {agent}: {e}")

        traci.simulationStep()

        # Update the agent dictionary based on the current vehicles in the simulation
        self.agents = {agent: self.observe(agent) for agent in traci.vehicle.getIDList()}

        observations = {agent: self.observe(agent) for agent in self.agents.keys() if self.agents[agent] is not None}


        # Record headway details for all agents
        headways = {}
        for agent in self.agents:
            try:
                leader_info = traci.vehicle.getLeader(agent)
                if leader_info is not None:
                    _, current_headway = leader_info
                    headways[agent] = current_headway
                else:
                    headways[agent] = -1  # Placeholder value for missing leader
            except traci.exceptions.TraCIException as e:
                print(e)
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
            except traci.exceptions.TraCIException as e:
                print(f"Error computing rewards for agent {agent}: {e}")

        done = self.STEPS >= TOTAL_STEPS
        info = {}

        return observations, rewards, done, info

    def reset(self):
        self.STEPS = 0
        self.initialized = False
        self._stop_sumo()
        self._start_sumo()

        self.headway_details = []

        self.initialize_after_steps()

        return {agent: self.observe(agent) for agent in self.agents}

    def close(self):
        self._stop_sumo()

    def save_headway_to_csv(self, filename="headway_data.csv"):
        # Filter out numerical headway values and keep only dictionaries
        headway_dicts = [entry for entry in self.headway_details if isinstance(entry, dict)]
        # Convert list of dictionaries to DataFrame
        headway_df = pd.DataFrame(headway_dicts)
        # Save DataFrame to CSV
        headway_df.to_csv(filename, index=False)

    def save_headway_plot(self, filename="headway_plot"):
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
