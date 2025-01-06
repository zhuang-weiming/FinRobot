import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

logger = logging.getLogger(__name__)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        # 初始化权重
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

class DDPGAgent:
    def __init__(self, env, gamma=0.99, tau=0.005, lr_actor=1e-4, lr_critic=1e-3):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 改进的经验回放缓冲区
        self.batch_size = 64
        self.buffer_size = 10000
        self.replay_buffer = []
        self.priorities = np.ones(self.buffer_size)  # 初始化优先级数组
        self.alpha = 0.6  # 优先级采样参数
        self.beta = 0.4  # 重要性采样参数
        self.beta_increment = 0.001
        
        # 探索策略参数
        self.noise_std = 0.2
        self.noise_decay = 0.995
        self.min_noise = 0.01
        
        # 训练参数
        self.target_update_freq = 100  # 目标网络更新频率
        self.grad_clip = 1.0  # 梯度裁剪阈值
        self.update_step = 0  # 更新计数器
        
    def predict(self, state, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        if not deterministic:
            action += np.random.normal(0, 0.1, size=action.shape)
        return np.clip(action, -1, 1)
    
    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            while True:
                action = self.predict(state, deterministic=False)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.append((state, action, reward, next_state, done))
                
                if len(self.replay_buffer) > self.buffer_size:
                    self.replay_buffer.pop(0)
                
                if len(self.replay_buffer) >= self.batch_size:
                    self.update()
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            logger.info(f"Episode {episode + 1}, Reward: {episode_reward}")
    
    def update(self):
        # 计算优先级
        if len(self.priorities) != len(self.replay_buffer):
            self.priorities = np.ones(len(self.replay_buffer))
            
        # 重要性采样
        probs = np.array(self.priorities[:len(self.replay_buffer)]) ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probs, replace=False)
        weights = (len(self.replay_buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = [self.replay_buffer[i] for i in indices]
        weights = torch.FloatTensor(weights).unsqueeze(1)
        
        states = torch.FloatTensor(np.array([t[0] for t in batch]))
        actions = torch.FloatTensor(np.array([t[1] for t in batch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in batch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch]))
        dones = torch.FloatTensor(np.array([t[4] for t in batch])).unsqueeze(1)
        
        # 更新critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        current_q_values = self.critic(states, actions)
        td_errors = (current_q_values - target_q_values).abs().detach().numpy()
        
        # 更新优先级
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = error.mean() + 1e-5
            
        critic_loss = (weights * (current_q_values - target_q_values).pow(2)).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        
        # 更新actor
        actor_loss = -(weights * self.critic(states, self.actor(states))).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        
        # 更新目标网络
        if self.update_step % self.target_update_freq == 0:
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.update_step += 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        self.noise_std = max(self.min_noise, self.noise_std * self.noise_decay)

def create_and_train_agent(env_train, batch_size, buffer_size, learning_rate, net_arch, total_timesteps):
    try:
        agent = DDPGAgent(env_train)
        agent.batch_size = batch_size
        agent.buffer_size = buffer_size
        agent.actor_optimizer.param_groups[0]['lr'] = learning_rate
        agent.critic_optimizer.param_groups[0]['lr'] = learning_rate
        agent.train(num_episodes=total_timesteps // 1000)
        logger.info("Agent训练完成")
        return agent
    except Exception as e:
        logger.error(f"Agent训练失败: {str(e)}")
        raise

def predict_with_agent(agent, environment, deterministic=True):
    """使用训练好的智能体进行预测。"""
    try:
        # 获取符合Stable-Baselines3接口的环境对象
        env, initial_state = environment.get_sb_env()
        
        account_values = []
        actions = []
        prices = []

        # 获取测试环境的原始日期索引
        if hasattr(env, 'df'):
            test_dates = env.df['date'].unique()
        else:
            test_dates = np.arange(10)  # 默认10天预测

        # 重置环境并获取初始状态
        state = initial_state
        account_values.append(env.get_account_value())

        for i in range(len(test_dates)):
            action = agent.predict(state, deterministic=deterministic)
            actions.append(action)
            
            # 执行一步操作
            state, reward, terminated, info = env.step(action)
            
            # 记录账户价值
            account_values.append(env.get_account_value())
            
            # 从info字典中获取当前价格
            if info and 'current_price' in info:
                prices.append(info['current_price'])
            else:
                prices.append(np.nan)
                logger.warning("无法从info字典中获取价格信息")

            if terminated:
                break

        # 确保数组长度一致
        min_length = min(len(test_dates), len(account_values), len(prices), len(actions))
        
        # 创建包含日期和价格的 DataFrame
        df_results = pd.DataFrame({
            'date': test_dates[:min_length],
            'current_price': prices[:min_length] if len(prices) > 0 else account_values[:min_length],
            'account_value': account_values[:min_length]
        })
        
        # 创建包含动作的DataFrame
        df_actions = pd.DataFrame({
            'date': test_dates[:min_length],
            'action': actions[:min_length]
        })
        
        # 设置日期索引
        df_results.set_index('date', inplace=True)
        df_actions.set_index('date', inplace=True)
        
        # 验证数据
        if 'current_price' not in df_results.columns:
            raise ValueError("无法获取价格信息，请检查环境实现")
        
        return df_results, df_actions
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise

if __name__ == '__main__':
    # 示例环境 (需要根据实际情况创建)
    import gym
    from stable_baselines3.common.env_util import make_vec_env
    env_train = make_vec_env('CartPole-v1', n_envs=1)

    # 示例参数
    batch_size = 64
    buffer_size = 10000
    learning_rate = 0.001
    net_arch = [64, 64]
    total_timesteps = 1000

    trained_agent = create_and_train_agent(env_train, batch_size, buffer_size, learning_rate, net_arch, total_timesteps)
    print("Agent trained successfully.")

    # 示例预测 (需要创建合适的交易环境)
    # class DummyEnv(gym.Env):
    #     def __init__(self):
    #         super(DummyEnv, self).__init__()
    #         self.action_space = gym.spaces.Discrete(2)
    #         self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
    #     def step(self, action):
    #         return self.observation_space.sample(), 1, True, {}
    #
    # dummy_env = DummyEnv()
    # df_account_value, df_actions = predict_with_agent(trained_agent, dummy_env)
    # print(df_account_value)
    # print(df_actions)
