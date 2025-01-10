from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from data_loader import StockDataLoader
from environment.environment_trading import StockTradingEnvironment
import torch.nn as nn
import numpy as np

def train_ppo_model():
    try:
        # Load and split data
        loader = StockDataLoader('601788')  # 光大证券的股票代码
        train_df, test_df = loader.load_and_split_data(
            start_date='20200102',
            end_date='20231229',
            train_ratio=0.8
        )
        
        # Create environment with training data
        env = DummyVecEnv([lambda: StockTradingEnvironment(train_df)])
        eval_env = DummyVecEnv([lambda: StockTradingEnvironment(test_df)])
        
        # Initialize model with improved parameters for A股特点
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=0.0001,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            max_grad_norm=0.5,
            tensorboard_log="./ppo_stock_tensorboard/",
            policy_kwargs=dict(
                # 使用更深的网络
                net_arch=dict(
                    pi=[256, 256, 128, 64],  # 加深策略网络
                    vf=[256, 256, 128, 64]   # 加深价值网络
                ),
                # 添加激活函数
                activation_fn=nn.ReLU
            )
        )
        
        # 创建回调函数用于评估和保存最佳模型
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./best_model/',
            log_path='./logs/',
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # 训练模型
        total_timesteps = 1000000
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # 保存最终模型
        model.save("ppo_stock_model")
        
        # 评估最终模型
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
            # DummyVecEnv returns (obs, reward, done, info) instead of (obs, reward, terminated, truncated, info)
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
