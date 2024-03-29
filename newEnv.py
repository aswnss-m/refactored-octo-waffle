import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci
from sumolib import checkBinary

# Attempt to connect to a running SUMO instance
# try:
#     traci.connect(port=4001)
#     print("Successfully connected to SUMO.")
# except Exception as e:
#     print(f"Failed to connect to SUMO: {e}")

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

print("Phase 1")

def leader_exists(follower):
    leader_info = traci.vehicle.getLeader(follower)
    print(follower)
    if leader_info is not None:
        # Assuming leader_info is a tuple where the first element is the leader's ID
        # and the second element is the secure gap. Adjust this according to the actual structure.
        leader_id, secure_gap = leader_info
        print("Leader_ID: ", leader_id, "  Secure_Gap: ", secure_gap)
        return leader_id, secure_gap
    else:
        print("No Leader Found!")
        return None, None


def change_speed(veh_id, new_speed):
    traci.vehicle.setSpeed(veh_id, new_speed)
    print("Speed changed")

print("Phase 2")

class TraciEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    print("Metadata created")

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
        for i in range(25):
            traci.simulationStep()

        # initial observation will include follower cars details
        self.vehicle_ids = traci.vehicle.getIDList()  # get the ids of the vehicles
        self.actorVehicle = self.vehicle_ids[0]  # Changed index to 0 for the first vehicle
        self.actorVehicle_speed = traci.vehicle.getSpeed(self.actorVehicle)
        self.leader_id, self.headway = leader_exists(self.actorVehicle)
        self.observation = (self.actorVehicle, self.actorVehicle_speed, self.leader_id, self.headway)

        print("Initialized")

    def step(self, action):
        traci.simulationStep()  # take next step

        # Initialize reward to a default value
        reward = 0

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
        observation = np.array(self.observation).astype(np.float32)

        print("Step function")

        return observation, reward, self.done, False, info

    def reset(self, seed=None, options=None):
        self.done = False

        # run the simulation a 20 steps to load the car into scene
        for i in range(20):
            traci.simulationStep()

        # initial observation will include follower cars details
        self.vehicle_ids = traci.vehicle.getIDList() # get the ids of the vehicles
        self.actorVehicle = self.vehicle_ids[0]
        self.actorVehicle_speed = traci.vehicle.getSpeed(self.actorVehicle)
        self.leader_id, self.headway = leader_exists(self.actorVehicle)

        vehicle_id_to_int = {veh_id: i for i, veh_id in enumerate(self.vehicle_ids)}
        actor_vehicle_int = vehicle_id_to_int[self.actorVehicle]
        leader_id_int = vehicle_id_to_int[self.leader_id] if self.leader_id is not None else -1

        if self.headway is None:
            self.headway = 0 # or any other default value that makes sense in your context

        self.observation = [float(actor_vehicle_int), float(self.actorVehicle_speed), float(leader_id_int), float(self.headway)]

        if self.headway > 500.0:
            self.headway = 500.0
        elif self.headway < -500.0:
            self.headway = -500.0

        print("Reset function")

        return np.array(self.observation).astype(np.float32), {}

    def render(self):
        print("Render function")
        pass

    def close(self):
        traci.close()
        print("Close function")

print("Phase 3")


 # self.q_table = np.zeros((N_DISCRETE_ACTIONS,N_DISCRETE_ACTIONS))
  # for i in range(20):
        #     traci.simulationStep()
        
        # self.vehicles = traci.vehicle.getIDList()
        
        # if traci.vehicle.getLeader(self.vehicles[0]) is None:
        #     self.leader = self.vehicles[0]
        #     self.follower = self.vehicles[1]
        # else:
        #     self.leader = self.vehicles[1]
        #     self.follower = self.vehicles[0]
        
        # current_speed_follower = traci.vehicle.getSpeed(self.follower)
        # current_headway = traci.vehicle.getLeader(self.follower)[1]
        # self.observation = (current_speed_follower, current_headway)