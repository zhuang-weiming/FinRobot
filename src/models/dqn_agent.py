import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import logging

class LSTMBlock(nn.Module):
    """带残差连接的LSTM块"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        residual = self.proj(x)
        out, _ = self.lstm(x)
        return self.layer_norm(out + residual)

class DQNNetwork(nn.Module):
    def __init__(self, input_shape, action_dim):
        super().__init__()
        self.feature_dim = input_shape[0]
        
        # 简化网络结构，移除LSTM
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, action_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """智能权重初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    nn.init.xavier_normal_(param, gain=1.0)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.net(x)

class DQNAgent:
    """DQN交易代理"""
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32
    ):
        # 确保状态维度格式正确
        self.state_dim = (state_dim,) if isinstance(state_dim, int) else state_dim
        logging.info(f"Initializing DQNAgent with state_dim: {self.state_dim}")
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.q_network = DQNNetwork(self.state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(self.state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            eps=1e-5
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # 使用简单的 deque 替代 PrioritizedReplayBuffer
        self.memory = deque(maxlen=memory_size)
        
        # 损失函数
        self.criterion = nn.SmoothL1Loss()
        
        # 设置为训练模式
        self.q_network.train()
        self.target_network.eval()
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作"""
        try:
            # 探索：随机动作
            if training and random.random() < self.epsilon:
                return np.array([random.uniform(-1, 1)], dtype=np.float32)
            
            # 利用：使用Q网络
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                q_values = self.q_network(state_tensor)
                
                # 确保动作在[-1,1]范围内
                action = torch.clamp(torch.tanh(q_values), min=-1.0, max=1.0)
                
                # 转换为 numpy 数组并确保类型正确
                action_np = action.cpu().numpy().astype(np.float32)
                
                # 验证动作是否有效
                if np.any(np.isnan(action_np)):
                    # 如果出现 NaN，返回一个安全的默认动作
                    logging.warning("Q-network produced NaN action, using default action")
                    return np.array([0.0], dtype=np.float32)
                
                return action_np
                
        except Exception as e:
            logging.error(f"Error in choose_action: {str(e)}")
            # 发生错误时返回一个安全的默认动作
            return np.array([0.0], dtype=np.float32)
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为 tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前 Q 值
        current_q = self.q_network(states)
        
        # 计算目标 Q 值（使用双 Q 学习）
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards + (1 - dones) * self.gamma * next_q.squeeze()
        
        # 计算 TD 误差
        td_error = torch.abs(current_q.squeeze() - target_q).detach()
        
        # 计算加权损失
        loss = (self.criterion(current_q.squeeze(), target_q)).mean()
        
        # 优化器步骤
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度监控
        total_grad_norm = self._monitor_gradients()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # 更新学习率
        self.scheduler.step(loss)
        
        # 软更新目标网络
        self._soft_update_target_network()
        
        return loss.item(), total_grad_norm
    
    def _monitor_gradients(self):
        """监控梯度状态"""
        total_grad_norm = 0.0
        for name, p in self.q_network.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_grad_norm += param_norm.item() ** 2
                
                # 记录每个参数的梯度范数
                if param_norm > 1e3 or param_norm < 1e-5:
                    logging.warning(f"Gradient issue in {name}: {param_norm:.2e}")
        
        total_grad_norm = total_grad_norm ** 0.5
        
        # 梯度状态警告
        if total_grad_norm > 1e3:
            logging.warning(f"Gradient explosion: {total_grad_norm:.2f}")
        elif total_grad_norm < 1e-5:
            logging.warning(f"Gradient vanishing: {total_grad_norm:.2e}")
        
        return total_grad_norm
    
    def _soft_update_target_network(self, tau=0.01):
        """软更新目标网络"""
        for target_param, local_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

    def update(self, batch):
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        # 其他代码保持不变... 