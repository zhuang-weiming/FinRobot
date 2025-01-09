import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output between -1 and 1
        )
        
    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64
        self.gamma = 0.99
        
        self.update_target_networks(tau=1.0)  # Hard copy weights
        
    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).squeeze(0).cpu().numpy()
            action += np.random.normal(0, noise, size=action.shape)
            return np.clip(action, -1, 1)
    
    def update(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample random batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        # Convert to numpy arrays first for better performance
        states = np.array([s for s, _, _, _, _ in batch])
        actions = np.array([a for _, a, _, _, _ in batch])
        rewards = np.array([r for _, _, r, _, _ in batch])
        next_states = np.array([s_ for _, _, _, s_, _ in batch])
        dones = np.array([d for _, _, _, _, d in batch])
        
        # Convert to tensors
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.FloatTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # Compute target Q value
        with torch.no_grad():
            next_actions = self.target_actor(next_state_batch)
            target_q = self.target_critic(next_state_batch, next_actions)
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q
            
        # Update critic
        current_q = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.update_target_networks()
    
    def update_target_networks(self, tau: float = 0.005):
        """Soft update target networks"""
        for target, source in zip(self.target_actor.parameters(), self.actor.parameters()):
            target.data.copy_(tau * source.data + (1 - tau) * target.data)
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.copy_(tau * source.data + (1 - tau) * target.data)
            
    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
        }, path)
        
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic']) 