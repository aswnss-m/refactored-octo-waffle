import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
from platooning_env import PlatooningEnv
from torch.utils.tensorboard import SummaryWriter

# Define hyperparameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
NUM_EPISODES = 6        # number of episodes
EPS_START = 1.0         # Initial epsilon
EPS_END = 0.01          # Minimum epsilon
EPS_DECAY = 0.995       # Epsilon decay rate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.epsilon = EPS_START
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        self.global_step = 0  # Initialize global step for TensorBoard

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state):
        self.global_step += 1  # Increment global step
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log the loss
        writer.add_scalar('Loss', loss.item(), self.global_step)

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

if __name__ == "__main__":
    writer = SummaryWriter('runs/DQN_Plattooning')
    env = PlatooningEnv()
    env.initialize_after_steps(21)
    agent = DQNAgent(state_size=2, action_size=3, seed=0)

    # # Optional: Add model graph to TensorBoard
    # random_state_example = np.random.rand(1, 2)  # Adjust dimensions as needed
    # writer.add_graph(agent.qnetwork_local, torch.from_numpy(random_state_example).to(device))

    for episode in range(NUM_EPISODES):
        observations = env.reset()
        done = False
        total_reward = 0
        while not done:
            actions = {agent_id: agent.act(observations[agent_id]) for agent_id in observations}
            next_observations, rewards, done, _ = env.step(actions)

            for agent_id in observations:
                state = observations[agent_id]
                action = actions[agent_id]
                reward = rewards.get(agent_id, 0)
                next_state = next_observations.get(agent_id)
                if next_state is not None:
                    agent.step(state, action, reward, next_state, done)
                    total_reward += reward

        # Log episode metrics
        writer.add_scalar('Total Reward/Episode', total_reward, episode)
        writer.add_scalar('Epsilon/Episode', agent.epsilon, episode)

        agent.epsilon = max(EPS_END, EPS_START * np.exp(-EPS_DECAY * episode))
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {agent.epsilon}")

    torch.save(agent.qnetwork_local.state_dict(), 'dqn_platooning_model.pth')
    print("Model saved successfully!")

    env.save_headway_to_csv("headway_data.csv")
    env.save_headway_plot()

    env.close()
    writer.close()
