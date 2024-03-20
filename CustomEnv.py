import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci
from sumolib import checkBinary

"""
GLOBAL VARIABLES
"""
MAX_HEADWAY = 20 #its in meters
MIN_HEADWAY = 10 #its in meters
N_DISCRETE_ACTIONS = 4

"""
Helper functions
"""

def leader_exists(follower):
    (id, headway) = traci.vehicle.getLeader(follower)
    if leader_info:
        return id,headway
    else:
        return None, None

N_DISCRETE_ACTIONS = 4 # [ increase speed , decrease speed , maintain speed , change lane]

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
                                            shape=(4,), dtype=np.uint8)

    def step(self, action):

        traci.simulationStep() #take nextstep
        """
        What here supposed to happen is 
        algorithm
        ----------
        1. check if the leader exists using self.leaderExitsts
        2. if leaderExists is None , then skip to step 7, else run step 3
        3. check the headway of the follower and leader using self.headway
        4. if the headway is greater than MAX_HEADWAY increase the speed of the actorVehicle 
        5. if the headway is less than MIN_HEADWAY decrease the speed of the actorVehicle
        6. if the headway is between the MIN and MAX then maintain speed
        7. run next step
        """
        leader_id, headway = leader_exists(self.actorVehicle)
        if leader_id is not None:
            if headway > MAX_HEADWAY:
                change_speed(self.actorVehicle, self.actorVehicle_speed + 1)
            elif headway < MIN_HEADWAY:
                change_speed(self.actorVehicle, self.actorVehicle_speed - 1)
            # Else maintain speed

        info = {}
        return self.observation, self.reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):
        self.done = False
        print("The options passed to reset function : ",options)
        self.sumo_binary = checkBinary('sumo') #use sumo-gui for gui

        # Start SUMO as a subprocess and connect with TraCI
        traci.start([self.sumo_binary, "-c", "./maps/singlelane/singlelane.sumocfg",
                 "--tripinfo-output", "tripinfo.xml"])
        
        # run the simulation a 20 steps to load the car into scene
        for i in range(5):
            traci.simulationStep()

        #initial observation will include follower cars details
        
        self.vehicle_ids = traci.vehicle.getIDList()  # get the ids of the vehicles
        self.actorVehicle = self.vehicle_ids[1]  # Changed index to 0 for the first vehicle
        self.actorVehicle_speed = traci.vehicle.getSpeed(self.actorVehicle)
        self.leader_id, self.headway = leader_exists(self.actorVehicle)
        self.observation = [self.actorVehicle, self.actorVehicle_speed, self.leader_id, self.headway]  # Corrected variable names
        self.observation = [self.actorVehicle,self.actorVehicle_speed, self.leader_id, self.headway] # [actorVehicle_id,actorVehicle_speed , egoVehicle_id, headway]
    
        return self.observation #reward , done, info cant be included

    def render(self):
        pass
    def close(self):
        traci.close()