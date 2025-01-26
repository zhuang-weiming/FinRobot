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
        self.assertEqual(len(self.env.trades), 1)
        self.assertEqual(self.env.trades[-1]['type'], 'buy')
        
        # 2. 模拟价格下跌触发止损
        original_price = self.env.df.loc[self.env.current_step, 'close']
        self.env.df.loc[self.env.current_step, 'close'] = original_price * 0.94
        
        # 执行任意动作，应该触发止损
        next_state, reward, done, _, info = self.env.step(np.array([0.0]))
        
        # 验证止损结果
        self.assertTrue(done)
        self.assertEqual(self.env.position, 0)
        self.assertGreater(self.env.cash, 0)
        self.assertEqual(self.env.trades[-1]['type'], 'close')
    
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
        
        # 1. 测试持有现金的奖励
        action = np.array([0.0])  # 不进行交易
        _, reward, _, _, _ = self.env.step(action)
        self.assertIsInstance(reward, float)
        
        # 2. 测试盈利交易的奖励
        # 准备一段连续上涨的价格数据
        state, _ = self.env.reset()
        start_step = self.env.current_step
        base_price = 3000.0  # 设置基准价格
        
        # 创建连续上涨的价格序列
        for i in range(10):  # 设置10个时间步
            price = base_price * (1 + i * 0.01)  # 每步上涨1%
            self.env.df.loc[start_step + i, 'close'] = price
            self.env.df.loc[start_step + i, 'open'] = price * 0.999
            self.env.df.loc[start_step + i, 'high'] = price * 1.001
            self.env.df.loc[start_step + i, 'low'] = price * 0.998
        
        # 买入前状态
        initial_cash = self.env.cash
        initial_position = self.env.position
        initial_price = self.env.df.loc[self.env.current_step, 'close']
        print(f"\nInitial state:")
        print(f"Cash: {initial_cash:.2f}")
        print(f"Position: {initial_position:.2f}")
        print(f"Price: {initial_price:.2f}")
        
        # 计算预期的交易结果
        trade_value = initial_cash * 0.5  # 使用一半资金
        fee = trade_value * self.env.transaction_cost
        actual_trade_value = trade_value - fee
        expected_position = actual_trade_value / initial_price
        expected_cash = initial_cash - trade_value - fee  # 修改这里：正确计算剩余现金
        
        print(f"\nExpected trade results:")
        print(f"Trade value: {trade_value:.2f}")
        print(f"Fee: {fee:.2f}")
        print(f"Expected position: {expected_position:.2f}")
        print(f"Expected cash: {expected_cash:.2f}")
        
        # 买入
        buy_action = np.array([0.5])  # 使用一半资金买入
        _, reward1, _, _, _ = self.env.step(buy_action)
        
        # 买入后状态
        post_buy_cash = self.env.cash
        post_buy_position = self.env.position
        post_buy_price = self.env.df.loc[self.env.current_step, 'close']
        post_buy_total_value = self.env.total_value  # 保存买入后的总价值
        
        print(f"\nPost buy state:")
        print(f"Cash: {post_buy_cash:.2f}")
        print(f"Position: {post_buy_position:.2f}")
        print(f"Price: {post_buy_price:.2f}")
        print(f"Position value: {self.env.position_value:.2f}")
        print(f"Total value: {post_buy_total_value:.2f}")
        
        # 验证基本交易逻辑
        self.assertGreater(post_buy_position, 0, "Should have positive position after buy")
        self.assertLess(post_buy_cash, initial_cash, "Cash should decrease after buy")
        self.assertAlmostEqual(post_buy_position, expected_position, delta=expected_position*0.01, 
                              msg=f"Position {post_buy_position} should be close to expected {expected_position}")
        self.assertAlmostEqual(post_buy_cash, expected_cash, delta=abs(expected_cash*0.01), 
                              msg=f"Cash {post_buy_cash} should be close to expected {expected_cash}")
        
        # 验证持仓价值
        expected_position_value = post_buy_position * post_buy_price
        actual_position_value = self.env.position_value
        self.assertAlmostEqual(actual_position_value, expected_position_value, delta=abs(expected_position_value*0.01),
                              msg=f"Position value {actual_position_value} should be close to expected {expected_position_value}")
        
        # 等待一步，验证价格上涨导致的价值变化
        _, reward2, _, _, _ = self.env.step(np.array([0.0]))
        
        # 持仓期间状态
        hold_cash = self.env.cash
        hold_position = self.env.position
        hold_price = self.env.df.loc[self.env.current_step, 'close']
        hold_position_value = self.env.position_value
        hold_total_value = self.env.total_value
        
        print(f"\nHolding state:")
        print(f"Cash: {hold_cash:.2f}")
        print(f"Position: {hold_position:.2f}")
        print(f"Price: {hold_price:.2f}")
        print(f"Position value: {hold_position_value:.2f}")
        print(f"Total value: {hold_total_value:.2f}")
        print(f"Reward: {reward2:.2f}")
        
        # 验证价格上涨导致的价值变化
        self.assertEqual(hold_position, post_buy_position, "Position should not change while holding")
        self.assertEqual(hold_cash, post_buy_cash, "Cash should not change while holding")
        expected_hold_position_value = hold_position * hold_price
        self.assertAlmostEqual(hold_position_value, expected_hold_position_value, delta=abs(expected_hold_position_value*0.01),
                              msg=f"Hold position value {hold_position_value} should be close to expected {expected_hold_position_value}")
        
        # 验证总价值上涨
        self.assertGreater(hold_total_value, post_buy_total_value, "Total value should increase from post buy")
        
        # 卖出
        _, reward3, _, _, _ = self.env.step(np.array([-0.5]))
        
        # 卖出后状态
        final_cash = self.env.cash
        final_position = self.env.position
        final_price = self.env.df.loc[self.env.current_step, 'close']
        print(f"\nFinal state:")
        print(f"Cash: {final_cash:.2f}")
        print(f"Position: {final_position:.2f}")
        print(f"Price: {final_price:.2f}")
        print(f"Position value: {self.env.position_value:.2f}")
        print(f"Total value: {self.env.total_value:.2f}")
        
        # 验证卖出逻辑
        self.assertLess(final_position, post_buy_position, "Position should decrease after sell")
        self.assertGreater(final_cash, post_buy_cash, "Cash should increase after sell")

if __name__ == '__main__':
    unittest.main() 