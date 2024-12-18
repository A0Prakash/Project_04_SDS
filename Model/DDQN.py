import torch
import torch.nn as nn
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class SwerveNN(nn.Module):
    def __init__(self, input_dims, action_dims, freeze=False):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dims)  # Output means for each action dimension
        )
        
        # Second head for standard deviation of actions
        self.std_network = nn.Sequential(
            nn.Linear(256, action_dims),
            nn.Softplus()  # Ensures positive standard deviation
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        if freeze:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, state):
        features = self.network[:-1](state)
        action_means = self.network[-1](features)
        action_stds = self.std_network(features)
        return action_means, action_stds

class SwerveAgent:
    def __init__(self, input_dims, action_dims):
        self.action_dims = action_dims
        self.learn_step_counter = 0
        
        # BASE PARAMETERS
        self.lr = 0.0003
        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_decay = 0.999975
        self.eps_min = 0.1
        self.batch_size = 64
        self.sync_network_rate = 10_000
        
        # Network and target network creation
        self.online_network = SwerveNN(input_dims, action_dims)
        self.target_network = SwerveNN(input_dims, action_dims, freeze=True)
        
        # Optimizing the network
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.value_loss = nn.MSELoss()
        
        # Replay buffer setup
        replay_buffer_capacity = 100_000
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)
        
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            # Random actions between -1 and 1
            return np.random.uniform(-1, 1, self.action_dims)
        
        with torch.no_grad():
            observation = torch.tensor(np.array(observation), dtype=torch.float32) \
                .unsqueeze(0) \
                .to(self.online_network.device)
            
            action_means, action_stds = self.online_network(observation)
            
            # Sample from normal distribution
            actions = torch.normal(action_means, action_stds)
            # Clip actions to [-1, 1]
            actions = torch.clamp(actions, -1, 1)
            
            return actions.squeeze().cpu().numpy()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)
    
    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
            "state": torch.tensor(np.array(state), dtype=torch.float32),
            "action": torch.tensor(np.array(action), dtype=torch.float32),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32),
            "done": torch.tensor(done, dtype=torch.float32)
        }, batch_size=[]))
        
    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
            
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.sync_networks()
        self.optimizer.zero_grad()
        
        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)
        keys = ("state", "action", "reward", "next_state", "done")
        states, actions, rewards, next_states, dones = [samples[key] for key in keys]
        
        # Get current Q values
        current_q_means, current_q_stds = self.online_network(states)
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_means, next_q_stds = self.target_network(next_states)
            next_q_values = next_q_means + next_q_stds * torch.randn_like(next_q_stds)
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # Compute loss
        value_loss = self.value_loss(current_q_means, target_q_values)
        
        # Add regularization for standard deviations
        std_loss = 0.01 * current_q_stds.mean()  # Penalize large standard deviations
        
        total_loss = value_loss + std_loss
        total_loss.backward()
        self.optimizer.step()
        
        self.learn_step_counter += 1
        self.decay_epsilon()
        
    def save_model(self, path):
        torch.save({
            'online_network': self.online_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.online_network.load_state_dict(checkpoint['online_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
