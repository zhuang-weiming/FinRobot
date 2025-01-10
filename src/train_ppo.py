from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from data_loader import StockDataLoader
from environment.environment_trading import StockTradingEnvironment
import torch.nn as nn
import torch
import gym
import numpy as np

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        # 获取输入维度
        self.lookback_window = observation_space.shape[0]
        self.n_features = observation_space.shape[1]
        
        # 特征标准化层
        self.layer_norm = nn.LayerNorm(self.n_features)
        
        # 简单的前馈网络
        self.net = nn.Sequential(
            nn.Linear(self.n_features * self.lookback_window, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, features_dim),
            nn.Tanh()  # 使用 tanh 确保输出范围在 [-1, 1]
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 确保输入是浮点型
        observations = observations.float()
        
        # 应用层标准化
        observations = self.layer_norm(observations)
        
        # 展平输入
        batch_size = observations.size(0)
        flattened = observations.view(batch_size, -1)
        
        # 通过网络
        features = self.net(flattened)
        
        return features

def train_ppo_model():
    try:
        # Load and split data
        loader = StockDataLoader('601788')
        train_df, test_df = loader.load_and_split_data(
            start_date='20200102',
            end_date='20231229',
            train_ratio=0.8
        )
        
        # Create environment
        env = DummyVecEnv([lambda: StockTradingEnvironment(train_df)])
        eval_env = DummyVecEnv([lambda: StockTradingEnvironment(test_df)])
        
        # Initialize model
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=0.0001,
            n_steps=1024,
            batch_size=64,  # 进一步减小批量大小
            n_epochs=5,     # 减少训练轮数
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005, # 减小熵系数
            max_grad_norm=0.3, # 进一步限制梯度
            tensorboard_log="./ppo_stock_tensorboard/",
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[32, 32],  # 使用更小的网络
                    vf=[32, 32]
                ),
                activation_fn=nn.ReLU,
                features_extractor_class=CustomFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=32),
                optimizer_class=torch.optim.AdamW,
                optimizer_kwargs=dict(
                    weight_decay=0.001  # 减小权重衰减
                )
            )
        )
        
        # 创建回调函数
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./best_model/',
            log_path='./logs/',
            eval_freq=2500,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )
        
        # 训练模型
        total_timesteps = 200000
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # 保存模型
        model.save("ppo_stock_model")
        
        # 评估模型
        mean_reward = evaluate_model(model, eval_env)
        
        return model, env
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        raise

def evaluate_model(model, env, num_episodes=5):
    """Evaluate the trained model"""
    rewards = []
    for _ in range(num_episodes):
        obs = env.reset()[0]
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            obs = step_result[0]
            reward = step_result[1]
            done = step_result[2]
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            
        rewards.append(episode_reward)
    
    mean_reward = sum(rewards) / len(rewards)
    print(f"\nMean evaluation reward: {mean_reward:.2f}")
    return mean_reward

if __name__ == "__main__":
    model, env = train_ppo_model()
