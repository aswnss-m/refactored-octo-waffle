import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci
from sumolib import checkBinary

### GLOBAL VARIABLES ###
TOTAL_STEPS = 3000

### HELPER FUNCTIONS ###
def leader_exists(follower):
    leader_info = traci.vehicle.getLeader(follower)
    if leader_info is not None:
        leader_id, headway = leader_info
        return leader_id, headway
    else:
        return None, None


class TraciEnv(gym.Env):
    def __init__(self):
        print("Inside init")
        super(TraciEnv, self).__init__()

        self.STEPS = 0
        self.action_space = spaces.Discrete(3)  # accelerate, decelerate, maintain
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([50, 50]), dtype=np.float32)

        self.sumo_binary = checkBinary('sumo')
        traci.start([self.sumo_binary, "-c", "./maps/singlelane/singlelane.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        for i in range(20):
            traci.simulationStep()
            print("initial steps ",i)
            self.STEPS += 1
        
    def step(self, action):
        print("taking a step")
        traci.simulationStep()
        score = 0

        if action == 0:
            traci.vehicle.setSpeed(self.follower, 20)
        elif action == 1:
            traci.vehicle.setSpeed(self.follower, 10)
        else:
            pass

        current_speed_follower = traci.vehicle.getSpeed(self.follower)
        current_headway = traci.vehicle.getLeader(self.follower)[1]

        if current_headway < 10:
            score -= 2
        elif current_headway > 20:
            score -= 10
        else:
            score += 5
        
        self.STEPS += 1
        self.done = self.STEPS >= TOTAL_STEPS
        self.reward = score
        observation = np.array([current_speed_follower, current_headway], dtype=np.float32)
        print("Observation: ", observation)

        return observation, self.reward, self.done, False,{} #obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        print("resetting")
        self.done = False
        self.STEPS = 0
        traci.close()
        traci.start([self.sumo_binary, "-c", "./maps/singlelane/singlelane.sumocfg", "--tripinfo-output", "tripinfo.xml"])

        for i in range(20):
            print("resetting steps ",i)
            traci.simulationStep()
            self.STEPS += 1
        
        self.vehicles = traci.vehicle.getIDList()
        print("Vehicles: ", self.vehicles)

        if traci.vehicle.getLeader(self.vehicles[0]) is None:
            self.leader = self.vehicles[0]
            self.follower = self.vehicles[1]
        else:
            self.leader = self.vehicles[1]
            self.follower = self.vehicles[0]

        current_speed_follower = traci.vehicle.getSpeed(self.follower)
        current_headway = traci.vehicle.getLeader(self.follower)[1]
        print("Current Speed: ", current_speed_follower)
        print("Current Headway: ", current_headway)
        self.observation = np.array([current_speed_follower, current_headway], dtype=np.float32)

        return self.observation,{}

    def render(self, mode='human'):
        pass

    def close(self):
        print("Closing")
        traci.close()
