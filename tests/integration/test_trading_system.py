import unittest
import numpy as np
import pandas as pd
from src.data.data_loader import StockDataLoader
from src.environment.environment_trading import StockTradingEnvironment
from src.models.dqn_agent import DQNAgent

class TestTradingSystem(unittest.TestCase):
    """交易系统集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.config = {
            'stock_code': '000001.SH',  # 上证指数
            'start_date': '20200101',    # 从2020年开始
            'end_date': '20240229',      # 到2024年2月
            'train_ratio': 0.8,          # 80%用于训练
            'initial_balance': 1_000_000.0,
            'transaction_cost': 0.0003,    # 修改这里：统一使用 transaction_cost
            'reward_scaling': 1.0
        }
        
        # 加载数据
        cls.data_loader = StockDataLoader(cls.config['stock_code'], cls.config)
        cls.train_data, cls.test_data = cls.data_loader.load_and_split_data(
            cls.config['start_date'],
            cls.config['end_date'],
            cls.config['train_ratio']
        )
        
        # 创建环境
        cls.env = StockTradingEnvironment(
            df=cls.train_data,
            initial_balance=cls.config['initial_balance'],
            transaction_cost=cls.config['transaction_cost']  # 修改这里：使用 transaction_cost
        )
        
        # 创建代理
        cls.agent = DQNAgent(
            state_dim=cls.env.observation_space.shape[0],
            action_dim=1,
            learning_rate=1e-4
        )
    
    def test_data_environment_agent_interaction(self):
        """测试数据、环境和代理的交互"""
        # 1. 验证数据加载
        self.assertIsInstance(self.train_data, pd.DataFrame)
        self.assertGreater(len(self.train_data), 0)
        required_columns = ['close', 'volume', 'ma5', 'ma20']
        for col in required_columns:
            self.assertIn(col, self.train_data.columns)
        
        # 2. 验证环境初始化
        state, info = self.env.reset()
        self.assertEqual(self.env.cash, self.config['initial_balance'])
        self.assertEqual(self.env.position, 0)
        self.assertEqual(len(self.env.trades), 0)
        
        # 3. 验证代理动作
        action = self.agent.choose_action(state)
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape, (1,))
        self.assertTrue(-1 <= action[0] <= 1)
    
    def test_trading_cycle_with_stop_loss(self):
        """测试完整交易周期（包含止损）"""
        state, _ = self.env.reset()
        self.env.stop_loss_threshold = 0.05  # 设置5%止损
        
        # 1. 执行买入
        buy_action = np.array([1.0])  # 满仓买入
        next_state, reward, done, _, info = self.env.step(buy_action)
        
        # 验证买入结果
        self.assertGreater(self.env.position, 0)
        self.assertLess(self.env.cash, self.config['initial_balance'])
        
        # 2. 模拟价格大幅下跌触发止损
        original_price = self.env.df.iloc[self.env.current_step]['close']
        self.env.df.iloc[self.env.current_step:self.env.current_step+5, self.env.df.columns.get_loc('close')] = original_price * 0.94
        self.env.df.iloc[self.env.current_step:self.env.current_step+5, self.env.df.columns.get_loc('low')] = original_price * 0.93
        
        # 执行持仓动作，应该触发止损
        for _ in range(3):  # 多执行几步以确保触发止损
            next_state, reward, done, _, info = self.env.step(np.array([0.0]))
            if done:
                break
        
        # 验证止损结果
        self.assertTrue(done, "Stop loss should trigger when price drops below threshold")
        self.assertAlmostEqual(self.env.position, 0, delta=0.0001)
    
    def test_model_persistence(self):
        """测试模型保存和加载"""
        # 1. 训练模型
        state, _ = self.env.reset()
        for _ in range(100):
            action = self.agent.choose_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.agent.remember(state, action, reward, next_state, done)
            if len(self.agent.memory) > self.agent.batch_size:
                self.agent.train()
            state = next_state
            if done:
                break
        
        # 2. 保存模型
        save_path = 'test_model.pth'
        self.agent.save(save_path)
        
        # 3. 创建新代理并加载模型
        new_agent = DQNAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=1
        )
        new_agent.load(save_path)
        
        # 4. 验证行为一致性
        state, _ = self.env.reset()
        action1 = self.agent.choose_action(state, training=False)
        action2 = new_agent.choose_action(state, training=False)
        np.testing.assert_array_almost_equal(action1, action2)
    
    def test_reward_mechanism(self):
        """测试奖励机制"""
        state, _ = self.env.reset()
        
        # 调整期望值的计算
        initial_cash = self.env.cash
        initial_price = self.env.df.iloc[self.env.current_step]['close']
        
        # 买入操作
        buy_action = np.array([0.5])  # 使用一半资金买入
        trade_value = initial_cash * 0.5
        fee = trade_value * self.env.transaction_cost
        expected_position = (trade_value - fee) / initial_price
        
        _, reward1, _, _, _ = self.env.step(buy_action)
        
        # 验证交易结果，增加容差
        self.assertAlmostEqual(
            self.env.position, 
            expected_position,
            delta=expected_position * 0.05,  # 允许5%的误差
            msg=f"Position {self.env.position} should be close to expected {expected_position}"
        )

if __name__ == '__main__':
    unittest.main() 