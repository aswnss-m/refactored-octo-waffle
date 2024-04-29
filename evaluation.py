import numpy as np
import torch
from platooning_env import PlatooningEnv
from pet1 import QNetwork
from pet1 import device


# Load the model
model = QNetwork(state_size=2, action_size=3, seed=0)  # Ensure these parameters match those used during training
model.load_state_dict(torch.load('./best_dqn_platooning_model.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode

print("Model loaded and ready for evaluation!")

# Create environment instance
env = PlatooningEnv()
env.initialize_after_steps(21)

# Evaluation Loop
total_reward = 0
observations = env.reset()
done = False
while not done:
    actions = {agent_id: np.argmax(model(torch.from_numpy(observations[agent_id]).float().unsqueeze(0).to(device)).detach().cpu().numpy()) for agent_id in observations}
    print(actions)
    next_observations, rewards, done, _ = env.step(actions)
    # print(rewards)
    total_reward += sum(rewards.values())
    observations = next_observations

print(f"Total Reward from Evaluation: {total_reward}")

# Close environment
env.close()
