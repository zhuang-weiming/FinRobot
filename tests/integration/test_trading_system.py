import unittest
import pandas as pd
import numpy as np
from src.data.data_loader import StockDataLoader
from src.environment.environment_trading import StockTradingEnvironment
from src.models.dqn_agent import DQNAgent  # 假设我们使用 DQN 作为交易代理
import logging

class TestTradingSystem(unittest.TestCase):
    """交易系统集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.config = {
            'stock_code': '000001.SH',  # 只测试上证指数
            'start_date': '20200101',    # 从2020年开始
            'end_date': '20240229',      # 到2024年2月（当前最新数据）
            'train_ratio': 0.8,          # 80%用于训练
            'initial_balance': 1_000_000.0,
            'transaction_fee': 0.0003,    # 降低交易成本到0.03%更符合指数基金实际情况
            'reward_scaling': 1.0
        }
        
        # 初始化数据加载器
        cls.data_loader = StockDataLoader(
            stock_code=cls.config['stock_code'],
            config=cls.config
        )
        
        logging.info(f"Loading data from {cls.config['start_date']} to {cls.config['end_date']}")
    
    def setUp(self):
        """每个测试用例初始化"""
        # 加载数据
        self.train_data, self.test_data = self.data_loader.load_and_split_data(
            start_date=self.config['start_date'],
            end_date=self.config['end_date'],
            train_ratio=self.config['train_ratio']
        )
        
        # 创建训练环境
        self.train_env = StockTradingEnvironment(
            df=self.train_data,
            initial_balance=self.config['initial_balance'],
            transaction_fee=self.config['transaction_fee'],
            reward_scaling=self.config['reward_scaling']
        )
        
        # 创建测试环境
        self.test_env = StockTradingEnvironment(
            df=self.test_data,
            initial_balance=self.config['initial_balance'],
            transaction_fee=self.config['transaction_fee'],
            reward_scaling=self.config['reward_scaling']
        )
        
        # 获取状态维度
        state_dim = self.train_env.observation_space.shape
        logging.info(f"Setting up test with state_dim: {state_dim}")
        
        # 创建代理
        self.agent = DQNAgent(
            state_dim=state_dim,  # 直接使用环境的观察空间形状
            action_dim=1,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=32
        )
    
    def test_complete_trading_cycle(self):
        """测试完整交易周期"""
        # 1. 训练阶段
        n_episodes = 10  # 实际训练时应该更多
        max_steps = len(self.train_data) - self.train_env.lookback_window
        
        for episode in range(n_episodes):
            state, _ = self.train_env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # 选择动作
                action = self.agent.choose_action(state)
                
                # 执行动作
                next_state, reward, done, _, info = self.train_env.step(action)
                
                # 存储经验
                self.agent.remember(state, action, reward, next_state, done)
                
                # 训练代理
                if len(self.agent.memory) > self.agent.batch_size:
                    self.agent.train()
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward:.2f}")
        
        # 2. 测试阶段
        state, _ = self.test_env.reset()
        total_reward = 0
        total_trades = 0
        
        while True:
            # 选择动作
            action = self.agent.choose_action(state, training=False)
            
            # 执行动作
            next_state, reward, done, _, info = self.test_env.step(action)
            
            total_reward += reward
            if info.get('trades'):
                total_trades += 1
            
            state = next_state
            
            if done:
                break
        
        # 3. 验证结果
        final_portfolio = self.test_env.total_value
        returns = (final_portfolio / self.config['initial_balance'] - 1) * 100
        max_drawdown = self.test_env.max_drawdown * 100
        
        print(f"\nTest Results:")
        print(f"Final Portfolio Value: {final_portfolio:,.2f}")
        print(f"Total Return: {returns:.2f}%")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Total Trades: {total_trades}")
        
        # 基本验证
        self.assertGreater(len(self.agent.memory), 0)  # 确认有交易记录
        self.assertIsInstance(final_portfolio, float)  # 确认有最终价值
        self.assertGreater(total_trades, 0)  # 确认执行了交易

if __name__ == '__main__':
    unittest.main() 