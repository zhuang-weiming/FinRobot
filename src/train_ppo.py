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
from environment.environment_trading import StockTradingEnvironment

import torch.nn as nn
import torch
import gymnasium as gym
import numpy as np
import time
import traceback
import os

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
    def __init__(self):
        super().__init__()
        self.patience = 100          # 减少耐心值
        self.min_improvement = 0.01  # 增加改进阈值

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

def make_env(df, is_training=True):
    """Create a wrapped environment"""
    def _init():
        env = StockTradingEnvironment(df, training=is_training)
        env = Monitor(
            env,
            filename=None,
            allow_early_resets=True
        )
        return env
    return _init

def train_ppo_model():
    try:
        # 创建必要的目录
        os.makedirs("./best_model", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        os.makedirs("./checkpoints", exist_ok=True)
        os.makedirs("./tensorboard_logs", exist_ok=True)
        
        # Load and split data
        loader = StockDataLoader('300059')
        print("Loading and preprocessing data...")
        train_df, test_df = loader.load_and_split_data(
            start_date='20200102',
            end_date='20241231',
            train_ratio=0.8
        )
        print("Data preprocessing completed successfully")
        
        # Create environments with Monitor wrapper
        print("Creating training environment...")
        env = DummyVecEnv([make_env(train_df, True)])
        print("Creating evaluation environment...")
        eval_env = DummyVecEnv([make_env(test_df, False)])
        
        # 优化的网络架构
        policy_kwargs = dict(
            net_arch=dict(
                pi=[128, 128, 64],    # 保持深度，减少宽度
                vf=[128, 128, 64]     # 同样的结构用于值函数
            ),
            activation_fn=nn.ReLU,
            log_std_init=-2.0,
            ortho_init=True,
            # 移除不支持的参数
            # normalize_observations=True,
            # normalize_returns=True
        )
        
        # 优化的模型配置
        model = PPO(
            CustomActorCriticPolicy,
            env,
            learning_rate=3e-5,          # 降低学习率
            n_steps=2048,                # 减小步长
            batch_size=128,              # 减小批量
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,             # 减小裁剪范围
            clip_range_vf=0.1,          # 添加值函数裁剪
            ent_coef=0.005,             # 降低熵系数
            max_grad_norm=0.5,          # 添加梯度裁剪
            target_kl=0.01,             # 添加KL散度目标
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[128, 128, 64],   # 简化策略网络
                    vf=[128, 128, 64]    # 简化值函数网络
                )
            )
        )
        
        print("\nTraining Configuration:")
        print("- Total timesteps:", 500_000)  # 增加训练步数
        print("- Batch size:", 256)
        print("- Learning rate:", 3e-4)
        print("- Network architecture:", "256-256-128")
        print("- Target KL:", 0.015)
        print("\nStarting training...")
        
        # 修改回调配置
        callbacks = [
            EvalCallback(
                eval_env,
                best_model_save_path="./best_model/",
                log_path="./logs/",
                eval_freq=5000,
                n_eval_episodes=20,  # 增加评估轮数
                deterministic=True,
                verbose=1
            ),
            CheckpointCallback(
                save_freq=20000,    # 增加保存频率
                save_path="./checkpoints/",
                name_prefix="ppo_stock_model",
                verbose=1
            ),
            RewardMonitorCallback(
                check_freq=1000,
                min_reward=-5.0,
                window_size=200     # 增加窗口大小
            ),
            ImprovedEarlyStoppingCallback(
                check_freq=2000,
                patience=150,
                min_evals=300,
                min_improvement=0.000005,
                verbose=1
            )
        ]
        
        # 使用 CallbackList 包装所有回调
        callback = CallbackList(callbacks)
        
        # 训练模型
        try:
            model.learn(
                total_timesteps=500_000,
                callback=callback,
                progress_bar=True,
                log_interval=10
            )
            print("\nTraining completed successfully")
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            traceback.print_exc()
            return None, None, None
            
        return model, env, eval_env
        
    except Exception as e:
        print(f"Error in training setup: {str(e)}")
        traceback.print_exc()
        return None, None, None

def evaluate_model(model, env, num_episodes=20):
    """Enhanced evaluation"""
    metrics = {
        'returns': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'win_rates': [],
        'profit_factors': [],
        'recovery_factors': []
    }
    
    for episode in range(num_episodes):
        # ... 运行episode ...
        
        # 计算更多指标
        metrics['profit_factors'].append(
            sum([r for r in returns if r > 0]) / abs(sum([r for r in returns if r < 0]))
        )
        metrics['recovery_factors'].append(
            total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        )
    
    # 输出详细评估结果
    print("\nDetailed Evaluation Results:")
    for metric, values in metrics.items():
        print(f"{metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")

# 添加学习率调度
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class RewardMonitorCallback(BaseCallback):
    """自定义回调来监控奖励"""
    def __init__(self, check_freq: int = 1000, min_reward: float = -5.0, window_size: int = 100):
        super().__init__()
        self.check_freq = check_freq
        self.min_reward = min_reward
        self.window_size = window_size
        self.rewards = []

    def _on_step(self) -> bool:
        if len(self.rewards) >= self.window_size:
            self.rewards.pop(0)
        self.rewards.append(float(self.locals['rewards'][0]))
        
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.rewards)
            if mean_reward < self.min_reward:
                print(f"\nStopping training due to low mean reward: {mean_reward:.2f}")
                return False
        return True

if __name__ == "__main__":
    model, env, eval_env = train_ppo_model()  # 获取所有返回的对象
