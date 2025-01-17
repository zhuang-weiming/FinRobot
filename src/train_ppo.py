from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    StopTrainingOnRewardThreshold
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy

from data_loader import StockDataLoader
from environment.environment_trading import StockTradingEnvironment, RewardScaler
from utils.config_manager import ConfigManager

import torch.nn as nn
import torch
import gymnasium as gym
import numpy as np
import time
import traceback
import os
import pandas as pd
import yaml

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        n_input = np.prod(observation_space.shape)
        
        self.feature_net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.feature_net(observations.float().reshape(observations.shape[0], -1))

class ImprovedEarlyStoppingCallback(BaseCallback):
    """改进的早停回调"""
    def __init__(
        self,
        check_freq: int = 1000,
        patience: int = 5,
        min_improvement: float = 0.01,
        reward_threshold: float = 1000.0,
        std_threshold: float = 2.0,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.min_improvement = min_improvement
        self.reward_threshold = reward_threshold
        self.std_threshold = std_threshold
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.reward_history = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 获取最近的奖励
            rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            
            self.reward_history.append(mean_reward)
            
            # 1. 检查奖励改善
            improvement = (mean_reward - self.best_mean_reward) / (abs(self.best_mean_reward) + 1e-8)
            
            # 2. 检查奖励波动性
            is_unstable = std_reward > self.std_threshold * np.mean(np.abs(rewards))
            
            # 3. 检查奖励发散
            is_diverging = mean_reward > self.reward_threshold
            
            if self.verbose > 0:
                print(f"\nEarly Stopping Check:")
                print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
                print(f"Improvement: {improvement:.2%}")
                print(f"Stability: {'Unstable' if is_unstable else 'Stable'}")
            
            if improvement > self.min_improvement and not is_unstable:
                self.best_mean_reward = mean_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            # 停止条件
            if (self.no_improvement_count >= self.patience or 
                is_unstable or 
                is_diverging):
                if self.verbose > 0:
                    print("Early stopping triggered!")
                    if self.no_improvement_count >= self.patience:
                        print("Reason: No improvement")
                    elif is_unstable:
                        print("Reason: Training unstable")
                    else:
                        print("Reason: Rewards diverging")
                return False
                
        return True

# 添加自定义网络架构
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 添加观察值归一化
        self.normalize_observations = True
        self.running_mean = torch.zeros(self.observation_space.shape[-1])
        self.running_var = torch.ones(self.observation_space.shape[-1])
        self.count = 0
        
        # 添加dropout和批归一化
        self.shared_net = nn.Sequential(
            nn.Linear(self.features_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 归一化观察值
        if self.normalize_observations:
            with torch.no_grad():
                if self.training:
                    self.count += obs.shape[0]
                    batch_mean = obs.mean(dim=0)
                    batch_var = obs.var(dim=0, unbiased=False)
                    
                    # 更新运行时统计
                    delta = batch_mean - self.running_mean
                    self.running_mean += delta * obs.shape[0] / self.count
                    self.running_var = (self.running_var * (self.count - obs.shape[0]) + 
                                      batch_var * obs.shape[0]) / self.count
                
                # 标准化
                obs = (obs - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
        
        return super().forward(obs)

# 改进学习率调度
def improved_lr_schedule(progress_remaining: float) -> float:
    """Improved learning rate schedule with longer warm-up and slower decay"""
    initial_lr = 5e-5  # 降低初始学习率
    min_lr = 1e-6
    warmup_fraction = 0.3  # 增加预热期
    
    if progress_remaining > 0.7:  # 更长的预热期
        return initial_lr * ((1.0 - progress_remaining) / warmup_fraction)
    else:
        progress = (0.7 - progress_remaining) / 0.7
        return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * progress)) / 2

def create_env(df: pd.DataFrame, training: bool = True) -> gym.Env:
    """创建交易环境"""
    # 创建配置对象
    config = ConfigManager.DEFAULT_CONFIG
    
    # 创建环境实例
    env = StockTradingEnvironment(
        df=df,
        config=config,  # 通过config传递参数
        training=training
    )
    return env

def train_ppo_model(env: gym.Env, config: dict) -> PPO:
    """训练PPO模型"""
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['training']['learning_rate'],
        n_steps=config['training']['n_steps'],
        batch_size=config['training']['batch_size'],
        policy_kwargs={"net_arch": config['training']['network']['hidden_sizes']},
        verbose=1
    )
    
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            progress_bar=True
        )
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()
    
    return model

def evaluate_model(model, env, num_episodes=50):
    """Enhanced evaluation with better error handling"""
    results = {
        'episode_returns': [],
        'portfolio_values': [],
        'trades': []
    }
    
    # 获取实际环境（从向量化环境中）
    if hasattr(env, 'envs'):
        actual_env = env.envs[0]
    else:
        actual_env = env
    
    for episode in range(num_episodes):
        # 修复 reset() 的返回值解包
        obs = env.reset()  # 直接接收返回值
        if isinstance(obs, tuple):  # 兼容新旧版本的 gym
            obs = obs[0]  # 如果是元组，取第一个值
            
        done = False
        portfolio_values = []
        episode_trades = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # 修复 step() 的返回值解包
            step_result = env.step(action)
            if len(step_result) == 5:  # 新版本 gym
                obs, reward, done, truncated, info = step_result
                done = done or truncated
            else:  # 旧版本 gym
                obs, reward, done, info = step_result
            
            # 修复: 处理向量化环境的info格式
            if isinstance(info, dict):
                portfolio_value = info.get('portfolio_value', 0)
                trades = info.get('trades', [])
            else:  # VecEnv返回的是info列表
                portfolio_value = info[0].get('portfolio_value', 0)
                trades = info[0].get('trades', [])
                
            portfolio_values.append(portfolio_value)
            episode_trades.extend(trades)
        
        # 计算每个episode的收益率
        if len(portfolio_values) > 1:  # 确保有足够的数据
            episode_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
            results['episode_returns'].append(episode_return)
            results['portfolio_values'].append(portfolio_values)
            results['trades'].append(episode_trades)
    
    print("\nEvaluation Results:")
    
    # 安全计算统计数据
    if results['episode_returns']:
        mean_return = np.mean(results['episode_returns'])
        std_return = np.std(results['episode_returns'])
        best_return = max(results['episode_returns'])
        worst_return = min(results['episode_returns'])
        
        print(f"Mean Return: {mean_return:.2f}% ± {std_return:.2f}%")
        print(f"Best Return: {best_return:.2f}%")
        print(f"Worst Return: {worst_return:.2f}%")
    else:
        print("Warning: No valid returns to report")
    
    # 安全计算交易统计
    total_trades = sum(len(episode_trades) for episode_trades in results['trades'])
    print(f"Total Trades: {total_trades}")
    
    if total_trades > 0:
        # 计算平均仓位大小
        all_trades = [trade for episode_trades in results['trades'] for trade in episode_trades]
        avg_position_size = np.mean([abs(trade.get('size', 0)) for trade in all_trades])
        print(f"Average Position Size: {avg_position_size:.2f}")
        
        # 安全获取max_position
        max_position = getattr(actual_env, 'max_position', 1.0)
        print(f"Position Utilization: {avg_position_size/max_position:.2%}")
    else:
        print("Warning: No trades executed during evaluation")
    
    return results

# 添加学习率调度
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class RewardMonitorCallback(BaseCallback):
    def __init__(self, check_freq: int = 1000, min_reward: float = -100.0):
        super().__init__()
        self.check_freq = check_freq
        self.min_reward = min_reward
        self.rewards = []
        self.raw_rewards = []  # 添加原始奖励跟踪
        
    def _on_step(self) -> bool:
        reward = float(self.locals['rewards'][0])
        
        # 记录并检查原始奖励
        if hasattr(self.training_env.envs[0], 'last_raw_reward'):
            raw_reward = self.training_env.envs[0].last_raw_reward
            self.raw_rewards.append(raw_reward)
            
        if not np.isfinite(reward):
            print(f"Warning: Invalid reward detected: {reward}")
            return True
            
        self.rewards.append(reward)
        
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.rewards[-100:])
            mean_raw_reward = np.mean(self.raw_rewards[-100:]) if self.raw_rewards else 0
            print(f"Current mean reward: {mean_reward:.2f} (raw: {mean_raw_reward:.2f})")
            
            if abs(mean_reward) > 1e6:  # 添加异常值检测
                print(f"Warning: Reward value too large")
                return False
                
        return True

class EarlyStoppingCallback(BaseCallback):
    """早停回调"""
    def __init__(
        self,
        check_freq: int = 1000,
        patience: int = 5,
        min_improvement: float = 0.01,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 获取最近的平均奖励
            x, y = self.model.ep_info_buffer.get()
            mean_reward = np.mean(y)
            
            # 检查是否有显著改善
            improvement = (mean_reward - self.best_mean_reward) / abs(self.best_mean_reward)
            
            if improvement > self.min_improvement:
                self.best_mean_reward = mean_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                
            if self.verbose > 0:
                print(f"Current mean reward: {mean_reward:.2f}")
                print(f"Best mean reward: {self.best_mean_reward:.2f}")
                print(f"No improvement count: {self.no_improvement_count}")
            
            # 如果连续多次没有显著改善，停止训练
            if self.no_improvement_count >= self.patience:
                if self.verbose > 0:
                    print("Early stopping triggered!")
                return False
                
        return True

def main():
    try:
        # 1. 加载配置
        config = ConfigManager.load_config()
        if not ConfigManager.validate_config(config):
            print("Invalid configuration, using default config")
            config = ConfigManager.DEFAULT_CONFIG
        
        # 2. 创建必要的目录
        for path in config['paths'].values():
            os.makedirs(path, exist_ok=True)
        
        # 3. 加载数据
        loader = StockDataLoader(config['data']['stock_code'])
        print("Loading and preprocessing data...")
        train_df, test_df = loader.load_and_split_data(
            start_date=config['data']['start_date'],
            end_date=config['data']['end_date'],
            train_ratio=config['data']['train_ratio']
        )
        print("Data preprocessing completed successfully")
        
        # 4. 创建环境
        print("Creating environments...")
        # 训练环境
        train_env = create_env(train_df, training=True)
        train_env = RewardScaler(train_env)  # 添加奖励缩放
        train_env = Monitor(train_env, filename="./logs/train_monitor.csv")
        train_env = DummyVecEnv([lambda: train_env])
        
        # 评估环境
        eval_env = create_env(test_df, training=False)
        eval_env = RewardScaler(eval_env)  # 评估环境也添加奖励缩放
        eval_env = Monitor(eval_env, filename="./logs/eval_monitor.csv")
        eval_env = DummyVecEnv([lambda: eval_env])
        
        # 5. 添加回调
        callbacks = [
            EvalCallback(
                eval_env,
                best_model_save_path="./best_model/",
                log_path="./logs/",
                eval_freq=5000,
                n_eval_episodes=20,
                deterministic=True,
                verbose=1
            ),
            CheckpointCallback(
                save_freq=20000,
                save_path="./checkpoints/",
                name_prefix="ppo_stock_model",
                verbose=1
            ),
            # 添加早停回调
            ImprovedEarlyStoppingCallback(
                check_freq=2048,
                patience=10,
                min_improvement=0.01,
                reward_threshold=1e4,
                std_threshold=2.0,
                verbose=1
            )
        ]
        
        # 6. 训练模型
        print("\nTraining Configuration:")
        print(f"- Total timesteps: {config['training']['total_timesteps']}")
        print(f"- Batch size: {config['training']['batch_size']}")
        print(f"- Learning rate: {config['training']['learning_rate']}")
        print(f"- Network architecture: {config['training']['network']['hidden_sizes']}")
        print(f"- Target KL: {config['training']['target_kl']}")
        
        print("\nStarting training...")
        model = train_ppo_model(train_env, config)
        
        # 7. 评估模型
        print("\nEvaluating model...")
        evaluate_model(model, eval_env)
        
        # 8. 保存最终模型
        model.save("./best_model/final_model")
        print("\nTraining and evaluation completed successfully")
        
        return model, train_env, eval_env
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, env, eval_env = main()
