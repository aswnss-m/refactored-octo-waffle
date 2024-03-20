import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci
from sumolib import checkBinary

# Global Variables
TOTAL_STEPS = 3000

class PlatooningEnvAcc(gym.Env):
    def __init__(self):
        super(PlatooningEnvAcc, self).__init__()

        # Initialize step count
        self.STEPS = 0

        # Initialize headway details list
        self.headway_details = []

        # Define action space: 0 - accelerate, 1 - decelerate, 2 - maintain speed
        self.action_space = spaces.Discrete(3)

        # Define observation space: [follower_speed, headway_distance]
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([50, 50]), dtype=np.float32)

        # Start SUMO simulation
        self.sumo_binary = checkBinary('sumo')
        traci.start([self.sumo_binary, "-c", "./maps/singlelane/singlelane.sumocfg", "--tripinfo-output", "tripinfo.xml"])

        # Run simulation for initialization
        for _ in range(20):
            traci.simulationStep()
            self.STEPS += 1

    def step(self, action):
        traci.simulationStep()
        self.STEPS += 1

        # Execute action
        if action == 0:
            traci.vehicle.setAcceleration(self.follower, 20,10)
        elif action == 1:
            traci.vehicle.setAcceleration(self.follower, 10,10)
        else:
            traci.vehicle.setAcceleration(self.follower, 0,10)


        # Compute reward
        score = 0
        current_speed_follower = traci.vehicle.getSpeed(self.follower)
        current_headway = 0  # Initialize with a default value

        leader_info = traci.vehicle.getLeader(self.follower)
        if leader_info is not None:
            leader_id, current_headway = leader_info
            if current_headway < 10:
                score -= 1
            elif current_headway > 20:
                score -= 10
            else:
                score += 5
            # Append headway details
            self.headway_details.append(current_headway)
        else:
            # If there's no leader, consider it a bad situation
            score += 1

        # Update observation
        observation = np.array([current_speed_follower, current_headway], dtype=np.float32)

        # Check termination condition
        done = self.STEPS >= TOTAL_STEPS

        # Check if episode is truncated
        truncated = False

        # Additional info
        info = {}

        return observation, score, done, truncated, info

    def reset(self, seed=None):
        self.STEPS = 0
        traci.close()
        traci.start([self.sumo_binary, "-c", "./maps/singlelane/singlelane.sumocfg", "--tripinfo-output", "tripinfo.xml"])

        # Clear headway details
        self.headway_details = []

        # Run simulation for initialization
        for _ in range(20):
            traci.simulationStep()
            self.STEPS += 1

        # Get vehicle IDs
        vehicles = traci.vehicle.getIDList()

        # Find leader and follower
        for i, vehicle_id in enumerate(vehicles):
            if traci.vehicle.getLeader(vehicle_id) is None:
                self.leader = vehicle_id
                self.follower = vehicles[(i + 1) % len(vehicles)]
                break

        # Initial observation
        current_speed_follower = traci.vehicle.getSpeed(self.follower)
        current_headway = traci.vehicle.getLeader(self.follower)[1]
        observation = np.array([current_speed_follower, current_headway], dtype=np.float32)

        return observation

    def render(self, mode='human'):
        pass

    def close(self):
        traci.close()